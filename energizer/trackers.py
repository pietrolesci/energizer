from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
from tqdm.auto import tqdm

from energizer.enums import RunningStage


@dataclass
class Tracker:
    current: int
    max: Optional[int]

    def __post_init__(self) -> None:
        self.total = self.current
        self.progress_bar = None

    def max_reached(self) -> bool:
        return self.max is not None and self.current >= self.max

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
        self.epoch_tracker = EpochTracker(current=0, max=None)
        self.step_tracker = Tracker(current=0, max=None)

        self.train_tracker = StageTracker(stage=RunningStage.TRAIN, current=0, max=None)
        self.validation_tracker = StageTracker(stage=RunningStage.VALIDATION, current=0, max=None)
        self.test_tracker = StageTracker(stage=RunningStage.TEST, current=0, max=None)

        self.stop_training: bool = False
        self.log_interval: int = 1
        self.enable_progress_bar: bool = True

        self.steps_to_validate: List[int] = []
        self.has_validation: bool = False
        self.current_stage: Optional[RunningStage] = None

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
        min_steps: Optional[int],
        num_train_batches: int,
        num_validation_batches: int,
        num_validation_per_epoch: int,
        limit_train_batches: Optional[int] = None,
        limit_validation_batches: Optional[int] = None,
    ) -> None:
        self.stop_training = False
        self.has_validation = num_validation_batches > 0

        # limit batches
        max_train_batches = int(min(num_train_batches, limit_train_batches or float("Inf")))
        max_validation_batches = int(min(num_validation_batches, limit_validation_batches or float("Inf")))

        if max_train_batches < 1:
            return

        # convert steps into max_epochs
        if min_steps is not None:
            max_epochs_for_num_steps = int(np.ceil(min_steps / max_train_batches))
            if max_epochs is None or max_epochs < max_epochs_for_num_steps:
                # if we do not have enough batches across epochs, adjust epoch number
                max_epochs = max_epochs_for_num_steps

        # validation interval
        if max_validation_batches > 0 and num_validation_per_epoch > 0 and max_train_batches > num_validation_per_epoch:
            self.steps_to_validate = np.linspace(
                max_train_batches / num_validation_per_epoch, max_train_batches, num_validation_per_epoch, dtype=int
            ).tolist()[:-1]

        self.epoch_tracker.max = max_epochs
        self.train_tracker.max = max_train_batches
        self.validation_tracker.max = max_validation_batches

    def setup_eval(self, stage: Union[str, RunningStage], num_batches: int, limit_batches: Optional[int]) -> None:
        getattr(self, f"{stage}_tracker").max = int(min(num_batches, limit_batches or float("Inf")))

    def make_progress_bars(self, stage: Union[str, RunningStage]) -> None:
        if not self.enable_progress_bar:
            return

        if stage in (RunningStage.TRAIN, RunningStage.VALIDATION):
            self.epoch_tracker.make_progress_bar()
            self.train_tracker.make_progress_bar()
            if self.has_validation:
                self.validation_tracker.make_progress_bar()
        else:
            getattr(self, f"{stage}_tracker").make_progress_bar()

    """Properties"""

    @property
    def is_fitting(self) -> bool:
        return self.current_stage in (RunningStage.TRAIN, RunningStage.VALIDATION)

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

    def should_log(self) -> bool:
        return (self.global_batch + 1) % self.log_interval == 0

    def should_validate(self) -> bool:
        return self.has_validation and (self.is_done() or self.train_tracker.current in self.steps_to_validate)

    """Outer loops"""

    def is_fit_done(self) -> bool:
        return self.epoch_tracker.max_reached() or self.stop_training

    def start_fit(self) -> None:
        self.epoch_tracker.reset()
        self.step_tracker.reset()

    def end_fit(self) -> None:
        self.epoch_tracker.close_progress_bar()
        self.train_tracker.close_progress_bar()
        self.validation_tracker.close_progress_bar()

    def increment_epoch(self) -> None:
        self.epoch_tracker.increment()

    def increment_step(self) -> None:
        self.step_tracker.increment()

    """Stage trackers"""

    def is_done(self) -> bool:
        return self.get_stage_tracker().max_reached() or (
            self.current_stage == RunningStage.TRAIN and self.stop_training
        )

    def start(self, stage: Union[str, RunningStage]) -> None:
        """Make progress bars and reset the counters."""
        self.current_stage = stage

        tracker = self.get_stage_tracker()
        tracker.reset()
        if tracker.progress_bar is not None:
            tracker.progress_bar.set_postfix_str("")

        if self.train_tracker.progress_bar is not None:
            if self.current_stage == RunningStage.TRAIN:
                self.train_tracker.progress_bar.set_description(f"Epoch {self.epoch_tracker.current}")
            elif self.current_stage == RunningStage.VALIDATION:
                self.train_tracker.progress_bar.set_postfix_str("Validating")

    def end(self) -> None:
        if not self.is_fitting:
            return self.get_stage_tracker().close_progress_bar()

        self.get_stage_tracker().terminate_progress_bar()
        if self.current_stage == RunningStage.VALIDATION:
            self.current_stage = RunningStage.TRAIN  # reattach training
            if self.train_tracker.progress_bar is not None:
                self.train_tracker.progress_bar.set_postfix_str("")

    def increment(self) -> None:
        self.get_stage_tracker().increment()

    """Helpers"""

    def set_stop_training(self, value: bool) -> None:
        self.stop_training = value

    def get_stage_tracker(self) -> StageTracker:
        return getattr(self, f"{self.current_stage}_tracker")


@dataclass
class RoundTracker(Tracker):
    def make_progress_bar(self) -> Optional[tqdm]:
        self.progress_bar = tqdm(
            total=self.max,
            desc="Completed rounds",
            dynamic_ncols=True,
            leave=True,
            colour="#32a852",
        )

    def increment(self) -> None:
        self.current += 1
        self.total += 1
        if self.progress_bar is not None and self.current > 0:
            self.progress_bar.update(1)


@dataclass
class BudgetTracker(Tracker):
    query_size: int

    def increment(self, n_labelled: Optional[int] = None) -> None:
        n_labelled = n_labelled or self.query_size
        self.current += n_labelled
        self.total += n_labelled

    def max_reached(self) -> bool:
        return self.max is not None and self.max < (self.query_size + self.total)

    def get_remaining_budget(self) -> int:
        return self.max - self.current if self.max is not None else int(float("Inf"))


@dataclass
class ActiveProgressTracker(ProgressTracker):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.round_tracker = RoundTracker(current=0, max=None)
        self.pool_tracker = StageTracker(stage=RunningStage.POOL, current=0, max=None)

        self.run_on_pool = False
        self.has_test = False

    def setup_active(
        self,
        max_rounds: Optional[int],
        max_budget: int,
        query_size: int,
        initial_budget: int,
        run_on_pool: bool,
        has_test: bool,
        has_validation: bool,
        log_interval: int,
        enable_progress_bar: bool,
    ) -> None:
        """Create progress bars."""

        assert max_budget > initial_budget, ValueError("`max_budget` must be bigger than `initial_budget`.")

        self.log_interval = log_interval
        self.enable_progress_bar = enable_progress_bar

        self.has_validation = has_validation
        self.run_on_pool = run_on_pool
        self.has_test = has_test

        max_rounds_per_budget = int(np.ceil((max_budget - initial_budget) / query_size))
        if max_rounds is None or max_rounds > max_rounds_per_budget:
            max_rounds = max_rounds_per_budget

        self.round_tracker.reset()
        self.round_tracker.max = max_rounds + 1  # type: ignore
        self.budget_tracker = BudgetTracker(
            max=max_budget,
            current=initial_budget,
            query_size=query_size,
        )

        if self.enable_progress_bar:
            self.round_tracker.make_progress_bar()
            self.epoch_tracker.make_progress_bar()
            self.train_tracker.make_progress_bar()
            if self.has_validation:
                self.validation_tracker.make_progress_bar()
            if self.has_test:
                self.test_tracker.make_progress_bar()
            if self.run_on_pool:
                self.pool_tracker.make_progress_bar()

    """Properties"""

    @property
    def is_last_round(self) -> bool:
        return self.round_tracker.current >= (self.round_tracker.max - 1)  # type: ignore

    @property
    def global_round(self) -> int:
        return self.round_tracker.total

    @property
    def global_budget(self) -> int:
        return self.budget_tracker.total

    @property
    def safe_global_epoch(self) -> int:
        return (
            self.global_round
            if self.current_stage in (RunningStage.TEST, RunningStage.POOL)
            else super().safe_global_epoch
        )

    """Super outer loops"""

    def is_active_fit_done(self) -> bool:
        return self.round_tracker.max_reached()

    def end_active_fit(self) -> None:
        self.round_tracker.close_progress_bar()
        self.epoch_tracker.close_progress_bar()
        self.train_tracker.close_progress_bar()
        self.validation_tracker.close_progress_bar()
        self.test_tracker.close_progress_bar()
        self.pool_tracker.close_progress_bar()

    def increment_round(self) -> None:
        self.round_tracker.increment()

    def increment_budget(self, n_labelled: Optional[int] = None) -> None:
        self.budget_tracker.increment(n_labelled)

    """Outer loops"""

    def start_fit(self) -> None:
        super().start_fit()
        if self.enable_progress_bar:
            self.epoch_tracker.progress_bar.set_postfix_str("")

    def end_fit(self) -> None:
        self.epoch_tracker.terminate_progress_bar()
        self.train_tracker.terminate_progress_bar()
        self.validation_tracker.terminate_progress_bar()

    """Stage trackers"""

    def end(self) -> None:
        self.get_stage_tracker().terminate_progress_bar()
        if self.current_stage == RunningStage.VALIDATION:
            self.current_stage = RunningStage.TRAIN  # reattach training
