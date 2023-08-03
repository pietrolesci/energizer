from dataclasses import dataclass
from typing import Optional, Union

import numpy as np
from tqdm.auto import tqdm

from energizer.enums import Interval, RunningStage


@dataclass
class Tracker:
    current: int
    max: Optional[int]

    def __post_init__(self) -> None:
        self.total = self.current
        self.progress_bar = None

    def max_reached(self) -> bool:
        return self.max is not None and self.current >= self.max
    
    def remaining(self) -> int:
        if self.max is None:
            raise ValueError("This tracker does not have a max.")
        return self.max - self.current

    def increment(self) -> None:
        self.current += 1
        self.total += 1
        if self.progress_bar is not None:
            self.progress_bar.update(1)

    def reset(self) -> None:
        self.current = 0
        if self.progress_bar is not None:
            self.progress_bar.reset(total=self.max)

    def make_progress_bar(self) -> Optional[tqdm]:
        pass

    def terminate_progress_bar(self) -> None:
        if self.progress_bar is not None:
            self.progress_bar.set_postfix_str("Done!", refresh=True)
            self.progress_bar.refresh()

    def close_progress_bar(self) -> None:
        self.terminate_progress_bar()
        if self.progress_bar is not None:
            self.progress_bar.clear()
            self.progress_bar.close()


@dataclass
class EpochTracker(Tracker):
    def make_progress_bar(self) -> Optional[tqdm]:
        self.progress_bar = tqdm(
            total=self.max,
            desc="Completed epochs",
            dynamic_ncols=True,
            leave=True,
        )

@dataclass
class StepTracker(Tracker):
    def make_progress_bar(self) -> Optional[tqdm]:
        self.progress_bar = tqdm(
            total=self.max,
            desc="Optimisation steps",
            dynamic_ncols=True,
            leave=True,
        )


@dataclass
class StageTracker(Tracker):
    stage: str

    def make_progress_bar(self) -> Optional[tqdm]:
        desc = f"Epoch {self.total}".strip() if self.stage == RunningStage.TRAIN else f"{self.stage.title()}"
        self.progress_bar = tqdm(
            total=self.max,
            desc=desc,
            dynamic_ncols=True,
            leave=True,
        )


@dataclass
class ProgressTracker:
    def __post_init__(self) -> None:
        # trackers
        self.epoch_tracker = EpochTracker(current=0, max=None)
        self.step_tracker = StepTracker(current=0, max=None)
        self.train_tracker = StageTracker(stage=RunningStage.TRAIN, current=0, max=None)
        self.validation_tracker = StageTracker(stage=RunningStage.VALIDATION, current=0, max=None)
        self.test_tracker = StageTracker(stage=RunningStage.TEST, current=0, max=None)

        # states
        self.stop_training: bool = False
        self.log_interval: int = 1
        self.enable_progress_bar: bool = True
        self.current_stage: Optional[RunningStage] = None
        self.has_validation: bool = False
        self.validate_every_n: Optional[int] = None
        self.validation_interval: Optional[str] = None

    def setup(self, stage: Union[str, RunningStage], log_interval: int, enable_progress_bar: bool, **kwargs) -> None:
        """Do all the math here and create progress bars for every stage."""

        self.log_interval = log_interval
        self.enable_progress_bar = enable_progress_bar

        if stage == RunningStage.TRAIN:
            self.setup_fit(**kwargs)  # type: ignore
        else:
            self.setup_eval(stage, **kwargs)  # type: ignore

        self.make_progress_bars(stage)

    def setup_fit(
        self,
        max_epochs: Optional[int],
        min_epochs: Optional[int],
        max_steps: Optional[int],
        min_steps: Optional[int],
        gradient_accumulation_steps: Optional[int],
        validation_freq: str,
        num_train_batches: int,
        num_validation_batches: int,
        limit_train_batches: Optional[int] = None,
        limit_validation_batches: Optional[int] = None,
    ) -> None:
        
        # checks
        assert max_epochs or max_steps, "`max_epochs` or `max_steps` must be specified."
        
        num_accumulation_steps = gradient_accumulation_steps or 1
        assert num_accumulation_steps > 0, f"`gradient_accumulation_steps > 0` or None, not {gradient_accumulation_steps}."
            
        # limit batches  
        max_train_batches = int(min(num_train_batches, limit_train_batches or float("Inf")))
        max_validation_batches = int(min(num_validation_batches, limit_validation_batches or float("Inf")))
        if max_train_batches < 1:
            self.has_validation = False
            self.stop_training = True
            return
                
        # define the number of training steps
        max_train_steps = float("Inf")
        if max_epochs is not None:
            max_train_steps = int(np.floor(max_train_batches * max_epochs / num_accumulation_steps))
        if max_steps is not None:
            max_train_steps = min(max_steps, max_train_steps)

        min_train_steps = -1
        if min_epochs is not None:
            min_train_steps = int(np.floor(max_train_batches * min_epochs / num_accumulation_steps))
        if min_steps is not None:
            min_train_steps = max(min_steps, min_train_steps)

        # now we have the number of total steps to perform
        total_steps = max(max_train_steps, min_train_steps)
        total_epochs = int(np.ceil(total_steps * num_accumulation_steps / max_train_batches))

        self.step_tracker.max = total_steps  # type: ignore
        self.epoch_tracker.max = total_epochs
        self.train_tracker.max = max_train_batches
        self.gradient_accumulation_steps = num_accumulation_steps
        self.stop_training = False

        # validation schedule
        if validation_freq is not None and max_validation_batches > 0:
            every_n, interval = validation_freq.split(":")
            every_n = int(every_n)
            assert interval in list(Interval)
            assert every_n > 0
            
            self.has_validation = True
            self.validate_every_n = every_n
            self.validation_interval = interval
        else:
            self.has_validation = False

        self.validation_tracker.max = max_validation_batches
                                
    def setup_eval(self, stage: Union[str, RunningStage], num_batches: int, limit_batches: Optional[int]) -> None:
        getattr(self, f"{stage}_tracker").max = int(min(num_batches, limit_batches or float("Inf")))

    """Properties"""

    @property
    def global_step(self) -> int:
        return self.step_tracker.total

    @property
    def global_batch(self) -> int:
        return self.get_stage_tracker().total

    @property
    def global_epoch(self) -> int:
        return self.epoch_tracker.total if self.is_fitting else 0

    @property
    def safe_global_epoch(self) -> int:
        if self.current_stage == RunningStage.VALIDATION:
            return self.train_tracker.total
        return self.global_epoch
    
    @property
    def is_fitting(self) -> bool:
        return self.current_stage in (RunningStage.TRAIN, RunningStage.VALIDATION)

    @property
    def is_accumulating(self) -> bool:
        return (self.global_batch + 1) % self.gradient_accumulation_steps != 0
    
    @property
    def is_done(self) -> bool:
        """Whether a stage is done."""
        return (
            self.get_stage_tracker().max_reached() 
            or self.current_stage == RunningStage.TRAIN and self.stop_training
            # or (self.epoch_tracker.remaining() <= 1 and self.gradient_accumulation_steps > self.train_tracker.remaining())
        )
    
    @property
    def is_fit_done(self) -> bool:
        return self.epoch_tracker.max_reached() or self.stop_training

    @property
    def should_log(self) -> bool:
        return (self.global_batch + 1) % self.log_interval == 0
    
    @property
    def should_validate(self) -> bool:
        if not self.has_validation:
            return False
        
        def _check(iter: int)-> bool:
            return (iter + 1) % self.validate_every_n == 0  # type: ignore
        
        should_validate = False
        if self.validation_interval == Interval.EPOCH:
            should_validate = _check(self.global_epoch) and self.is_done

        elif self.validation_interval == Interval.BATCH:
            should_validate = _check(self.global_batch) and not self.is_done # type: ignore
        
        elif self.validation_interval == Interval.STEP:        
            should_validate = _check(self.global_step) and not self.is_done and ((self.global_batch * self.gradient_accumulation_steps) % (self.global_step + 1) == 0)# type: ignore
        
        else:
            raise NotImplementedError

        return should_validate
        

                
    """Methods"""

    def start_fit(self) -> None:
        self.epoch_tracker.reset()
        self.step_tracker.reset()

    def start(self, stage: Union[str, RunningStage]) -> None:
        """Make progress bars and reset the counters of stage trackers."""
        self.current_stage = stage  # type: ignore

        tracker = self.get_stage_tracker()

        # TODO: take a look at this
        # # last epoch is shorter if not enough batches to make an update ("drop last")
        # if stage == RunningStage.TRAIN and self.gradient_accumulation_steps > 1 and self.epoch_tracker.remaining() <= 1:
        #     tracker.max = int(np.floor(tracker.max / self.gradient_accumulation_steps)) * self.gradient_accumulation_steps   # type: ignore

        tracker.reset()
        
        if tracker.progress_bar is not None:
            tracker.progress_bar.set_postfix_str("")

        if self.train_tracker.progress_bar is not None:
            if self.current_stage == RunningStage.TRAIN:
                self.train_tracker.progress_bar.set_description(f"Epoch {self.epoch_tracker.current}")
            elif self.current_stage == RunningStage.VALIDATION:
                self.train_tracker.progress_bar.set_postfix_str("Validating")

    def end(self) -> None:
        """Close progress bars of stage tracker when testing or re-attach training when validating."""
        if not self.is_fitting:
            return self.get_stage_tracker().close_progress_bar()

        self.get_stage_tracker().terminate_progress_bar()
        if self.current_stage == RunningStage.VALIDATION:
            self.current_stage = RunningStage.TRAIN  # reattach training
            if self.train_tracker.progress_bar is not None:
                self.train_tracker.progress_bar.set_postfix_str("")

    def end_fit(self) -> None:
        """Close progress bars."""
        self.step_tracker.close_progress_bar()
        self.epoch_tracker.close_progress_bar()
        self.train_tracker.close_progress_bar()
        self.validation_tracker.close_progress_bar()

    def increment(self) -> None:
        """Increment stage trackers."""
        self.get_stage_tracker().increment()

    def increment_epoch(self) -> None:
        self.epoch_tracker.increment()

    def increment_step(self) -> None:
        self.step_tracker.increment()

    """Helpers"""

    def set_stop_training(self, value: bool) -> None:
        self.stop_training = value

    def get_stage_tracker(self) -> StageTracker:
        return getattr(self, f"{self.current_stage}_tracker")
    
    def make_progress_bars(self, stage: Union[str, RunningStage]) -> None:
        if not self.enable_progress_bar:
            return

        if stage in (RunningStage.TRAIN, RunningStage.VALIDATION):
            self.step_tracker.make_progress_bar()
            self.epoch_tracker.make_progress_bar()
            self.train_tracker.make_progress_bar()
            if self.has_validation:
                self.validation_tracker.make_progress_bar()
        else:
            getattr(self, f"{stage}_tracker").make_progress_bar()