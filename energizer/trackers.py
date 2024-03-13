from dataclasses import dataclass

import numpy as np
from lightning_utilities.core.rank_zero import rank_zero_info
from tqdm.auto import tqdm

from energizer.enums import Interval, RunningStage


@dataclass
class Tracker:
    current: int
    max: int | None

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

    def make_progress_bar(self) -> tqdm | None:
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
    def make_progress_bar(self) -> tqdm | None:
        self.progress_bar = tqdm(total=self.max, desc="Completed epochs", dynamic_ncols=True, leave=True)


@dataclass
class StepTracker(Tracker):
    def make_progress_bar(self) -> tqdm | None:
        self.progress_bar = tqdm(total=self.max, desc="Optimisation steps", dynamic_ncols=True, leave=True)


@dataclass
class StageTracker(Tracker):
    stage: str

    def make_progress_bar(self) -> tqdm | None:
        desc = f"Epoch {self.total}".strip() if self.stage == RunningStage.TRAIN else f"{self.stage.title()}"
        self.progress_bar = tqdm(total=self.max, desc=desc, dynamic_ncols=True, leave=True)


@dataclass
class ProgressTracker:
    """Progress tracker for training, testing, and validation.

    There are two types of trackers: counters and idx_trackers. The idea is that a counter "counts" stuff.
    For example, optimisation steps needs to be a counter, it does not make sense to say "step 0". The same
    holds for batch counters as we want to know how many batches we have seen so far. However, we also allow
    for epoch_idx and batch_idx. These are mere identifier for a specific epoch or batch. It makes sense to
    say "In epoch 0 this or that happened". For now, this division makes sense. Might come back to this and
    change it in the future.
    """

    def __post_init__(self) -> None:
        # trackers
        self.epoch_idx_tracker = EpochTracker(current=0, max=None)
        self.step_counter = StepTracker(current=0, max=None)
        self.train_batch_counter = StageTracker(stage=RunningStage.TRAIN, current=0, max=None)
        self.validation_batch_counter = StageTracker(stage=RunningStage.VALIDATION, current=0, max=None)
        self.test_batch_counter = StageTracker(stage=RunningStage.TEST, current=0, max=None)

        # states
        self.stop_training: bool = False
        self.log_interval: int = 1
        self.enable_progress_bar: bool = True
        self.current_stage: RunningStage | None = None

        # validation logic
        self.has_validation: bool = False
        self.validate_every_n: int | None = None
        self.validation_interval: str | None = None
        self.validate_on_epoch_end: bool = False
        self._last_epoch_num_steps: int = 0
        self._xepoch_set: bool = False

    def setup(self, stage: str | RunningStage, log_interval: int, enable_progress_bar: bool, **kwargs) -> None:
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
        max_epochs: int | None,
        min_epochs: int | None,
        max_steps: int | None,
        min_steps: int | None,
        gradient_accumulation_steps: int | None,
        validation_freq: str | None,
        num_train_batches: int,
        num_validation_batches: int,
        limit_train_batches: int | None = None,
        limit_validation_batches: int | None = None,
    ) -> None:
        # checks
        assert max_epochs or max_steps, "`max_epochs` or `max_steps` must be specified."

        num_accumulation_steps = gradient_accumulation_steps or 1
        assert (
            num_accumulation_steps > 0
        ), f"`gradient_accumulation_steps > 0` or None, not {gradient_accumulation_steps}."

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

        self.step_counter.max = total_steps  # type: ignore
        self.epoch_idx_tracker.max = total_epochs
        self.train_batch_counter.max = max_train_batches
        self.gradient_accumulation_steps = num_accumulation_steps
        self.stop_training = False

        # validation schedule
        if validation_freq is not None and max_validation_batches > 0:
            if validation_freq.endswith("+"):
                self.validate_on_epoch_end = True
                validation_freq = validation_freq.removesuffix("+")

            if "xepoch" in validation_freq:
                # automatically compute number of times per epoch
                times_per_epoch = int(validation_freq.split("x")[0])
                assert times_per_epoch > 0
                every_n = int(np.floor(np.ceil(total_steps / total_epochs) / times_per_epoch))
                # print(f"{total_steps=}\n{total_epochs=}\n{every_n=}")
                interval = Interval.STEP
                self._xepoch_set = True
                msg = f"Validating {times_per_epoch} times per epoch"
            else:
                every_n, interval = validation_freq.split(":")
                every_n = int(every_n)
                assert every_n > 0
                assert interval in list(Interval)
                msg = f"Validating every {every_n} {interval}"

            if self.validate_on_epoch_end and interval != Interval.EPOCH:
                msg += ". You passed `+` so will always validate on epoch end"

            rank_zero_info(msg)
            self.has_validation = True
            self.validate_every_n = every_n
            self.validation_interval = interval
            self.validation_batch_counter.max = max_validation_batches
        else:
            self.has_validation = False

    def setup_eval(self, stage: str | RunningStage, num_batches: int, limit_batches: int | None) -> None:
        getattr(self, f"{stage}_batch_counter").max = int(min(num_batches, limit_batches or float("Inf")))

    """Properties"""

    @property
    def global_step(self) -> int:
        return self.step_counter.total

    @property
    def global_batch(self) -> int:
        return self.get_batch_counter().total

    @property
    def global_epoch_idx(self) -> int:
        return self.epoch_idx_tracker.total if self.is_fitting else 0

    @property
    def safe_global_epoch_idx(self) -> int:
        if self.current_stage == RunningStage.VALIDATION:
            return self.step_counter.total
        return self.global_epoch_idx

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
            self.get_batch_counter().max_reached() or self.current_stage == RunningStage.TRAIN and self.stop_training
            # or (self.epoch_idx_tracker.remaining() <= 1 and self.gradient_accumulation_steps > self.train_batch_counter.remaining())  # noqa: E501
        )

    @property
    def is_fit_done(self) -> bool:
        return self.epoch_idx_tracker.max_reached() or self.stop_training

    @property
    def should_log(self) -> bool:
        return (self.global_batch + 1) % self.log_interval == 0

    @property
    def should_validate(self) -> bool:
        if not self.has_validation:
            return False

        def _check(iter: int) -> bool:
            return (iter + 1) % self.validate_every_n == 0  # type: ignore

        should_validate = False

        if self.validation_interval == Interval.EPOCH:
            should_validate = _check(self.global_epoch_idx) and self.is_done

        elif self.validation_interval == Interval.BATCH:
            should_validate = _check(self.global_batch) and not self.is_done  # type: ignore

        elif self.validation_interval == Interval.STEP:
            # this makes sure that when we pass xepoch we exactly validate the same number of times per epoch
            # if we use `self.global_step` some epochs mights have more validations
            step = self.global_step - self._last_epoch_num_steps if self._xepoch_set else self.global_step
            should_validate = (
                (_check(step) and not self.is_done and not self.is_accumulating)
                or (self.validate_on_epoch_end and self.is_done)
                # this check avoids validating on epoch end when the steps happens to be at the end of the epoch
                # not _check(self.global_step)
            )

        else:
            raise NotImplementedError

        # if should_validate and self.train_batch_counter.progress_bar is not None:
        #     self.train_batch_counter.progress_bar.set_postfix_str(
        #         f"{self.global_epoch_idx=} {self.global_step=} {self.is_done=}"
        #     )

        return should_validate

    """Methods"""

    def start_fit(self) -> None:
        self.epoch_idx_tracker.reset()
        self.step_counter.reset()
        self._last_epoch_num_steps = 0

    def start(self, stage: str | RunningStage) -> None:
        """Make progress bars and reset the counters of stage trackers."""
        self.current_stage = stage  # type: ignore

        tracker = self.get_batch_counter()

        # TODO: take a look at this
        # # last epoch is shorter if not enough batches to make an update ("drop last")
        # if stage == RunningStage.TRAIN and self.gradient_accumulation_steps > 1 and self.epoch_idx_tracker.remaining() <= 1:  # noqa: E501
        #     tracker.max = int(np.floor(tracker.max / self.gradient_accumulation_steps)) * self.gradient_accumulation_steps   # noqa: E501

        tracker.reset()

        if tracker.progress_bar is not None:
            tracker.progress_bar.set_postfix_str("")

        if self.train_batch_counter.progress_bar is not None:
            if self.current_stage == RunningStage.TRAIN:
                self.train_batch_counter.progress_bar.set_description(f"Epoch {self.epoch_idx_tracker.current}")
            elif self.current_stage == RunningStage.VALIDATION:
                self.train_batch_counter.progress_bar.set_postfix_str("Validating")

    def end(self) -> None:
        """Close progress bars of stage tracker when testing or re-attach training when validating."""

        # NOTE: when testing we directly close the progress bar when we are done
        if not self.is_fitting:
            return self.get_batch_counter().close_progress_bar()

        self.get_batch_counter().terminate_progress_bar()

        # NOTE: if this is the end of the validation stage we need to reattach the training tracker
        if self.current_stage == RunningStage.VALIDATION:
            self.current_stage = RunningStage.TRAIN
            if self.train_batch_counter.progress_bar is not None:
                self.train_batch_counter.progress_bar.set_postfix_str("")

    def end_fit(self) -> None:
        """Close progress bars."""
        self.step_counter.close_progress_bar()
        self.epoch_idx_tracker.close_progress_bar()
        self.train_batch_counter.close_progress_bar()
        self.validation_batch_counter.close_progress_bar()

    def increment(self) -> None:
        """Increment stage trackers."""
        self.get_batch_counter().increment()

    def increment_epoch_idx(self) -> None:
        self.epoch_idx_tracker.increment()
        self._last_epoch_num_steps += self.global_step

    def increment_step(self) -> None:
        self.step_counter.increment()
        if self.step_counter.max_reached():
            self.stop_training = True

    """Helpers"""

    def set_stop_training(self, value: bool) -> None:
        self.stop_training = value

    def get_batch_counter(self) -> StageTracker:
        return getattr(self, f"{self.current_stage}_batch_counter")

    def make_progress_bars(self, stage: str | RunningStage) -> None:
        if not self.enable_progress_bar:
            return

        if stage in (RunningStage.TRAIN, RunningStage.VALIDATION):
            self.step_counter.make_progress_bar()
            self.epoch_idx_tracker.make_progress_bar()
            self.train_batch_counter.make_progress_bar()
            if self.has_validation:
                self.validation_batch_counter.make_progress_bar()
        else:
            getattr(self, f"{stage}_batch_counter").make_progress_bar()
