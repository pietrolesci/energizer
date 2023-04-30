from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from tqdm.auto import tqdm

from energizer.enums import RunningStage


@dataclass
class Tracker:
    max: Optional[int] = None
    total: int = 0
    current: int = 0
    progress_bar: Optional[tqdm] = None

    def max_reached(self) -> bool:
        """If a max is not set, it will never stop."""
        return self.max is not None and self.current >= self.max

    def reset(self) -> None:
        self.current = 0
        if self.progress_bar is not None:
            self.progress_bar.reset(total=self.max)

    def increment(self) -> None:
        self.current += 1
        self.total += 1
        if self.progress_bar is not None:
            self.progress_bar.update(1)

    def make_progress_bar(self) -> None:
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
    def make_progress_bar(self) -> None:
        self.progress_bar = tqdm(
            total=self.max,
            desc="Completed epochs",
            dynamic_ncols=True,
            leave=True,
        )


@dataclass
class StageTracker(Tracker):
    stage: str = None

    def make_progress_bar(self) -> None:
        desc = f"Epoch {self.total}".strip() if self.stage == RunningStage.TRAIN else f"{self.stage.title()}"
        self.progress_bar = tqdm(
            total=self.max,
            desc=desc,
            dynamic_ncols=True,
            leave=True,
        )


@dataclass
class ProgressTracker:
    epoch_tracker: EpochTracker = field(default_factory=lambda: EpochTracker())
    step_tracker: Tracker = field(default_factory=lambda: Tracker())

    # stage trackers
    train_tracker: StageTracker = field(default_factory=lambda: StageTracker(stage=RunningStage.TRAIN))
    validation_tracker: StageTracker = field(default_factory=lambda: StageTracker(stage=RunningStage.VALIDATION))
    test_tracker: StageTracker = field(default_factory=lambda: StageTracker(stage=RunningStage.TEST))

    validation_interval: Optional[List[int]] = None
    stop_training: bool = False
    has_validation: bool = False
    log_interval: int = 1
    enable_progress_bar: bool = True
    current_stage: RunningStage = None

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
        if self.is_fitting:
            return self.epoch_tracker.total
        return 0

    @property
    def safe_global_epoch(self) -> int:
        if self.current_stage == RunningStage.VALIDATION:
            return self.train_tracker.total
        return self.global_epoch

    def should_log(self) -> bool:
        return (self.global_batch + 1) % self.log_interval == 0

    def should_validate(self) -> bool:
        return (
            self.validation_tracker.max is not None
            and self.has_validation
            and (self.is_done() or self.train_tracker.current in self.validation_interval)
        )

    """Outer loops"""

    def is_fit_done(self) -> bool:
        return self.epoch_tracker.max_reached() or self.stop_training or self.step_tracker.max_reached()

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

    def start(self, stage: RunningStage) -> None:
        """Make progress bars and reset the counters."""
        self.current_stage = stage
        self.get_stage_tracker().reset()

        if self.get_stage_tracker().progress_bar is not None:
            self.get_stage_tracker().progress_bar.set_postfix_str("")

            if self.current_stage == RunningStage.TRAIN:
                self.train_tracker.progress_bar.set_description(f"Epoch {self.epoch_tracker.current}")

            if self.current_stage == RunningStage.VALIDATION:
                self.train_tracker.progress_bar.set_postfix_str("Validating")

    def end(self) -> None:
        if not self.is_fitting:
            return getattr(self, f"{self.current_stage}_tracker").close_progress_bar()

        self.get_stage_tracker().terminate_progress_bar()
        if self.current_stage == RunningStage.VALIDATION:
            self.current_stage = RunningStage.TRAIN  # reattach training
            if self.enable_progress_bar:
                self.train_tracker.progress_bar.set_postfix_str("")

    def increment(self) -> None:
        self.get_stage_tracker().increment()

    """Setup"""

    def setup(self, stage: RunningStage, **kwargs) -> None:
        """Do all the math here and create progress bars."""
        self.log_interval = kwargs.pop("log_interval", 1)
        self.enable_progress_bar = kwargs.pop("enable_progress_bar", True)

        if stage == RunningStage.TRAIN:
            self.has_validation = kwargs.get("num_validation_batches", 0) > 0
            self._setup_fit(**kwargs)
        else:
            limit_batches = kwargs.get("limit_batches") or float("Inf")
            getattr(self, f"{stage}_tracker").max = min(kwargs.get("num_batches"), limit_batches)

        self.make_progress_bars(stage)

    def make_progress_bars(self, stage: RunningStage) -> None:
        if not self.enable_progress_bar:
            return

        if stage in (RunningStage.TRAIN, RunningStage.VALIDATION):
            self.epoch_tracker.make_progress_bar()
            self.train_tracker.make_progress_bar()
            if self.has_validation:
                self.validation_tracker.make_progress_bar()
        else:
            getattr(self, f"{stage}_tracker").make_progress_bar()

    """Helpers"""

    def set_stop_training(self, value: bool) -> None:
        self.stop_training = value

    def get_stage_tracker(self) -> StageTracker:
        return getattr(self, f"{self.current_stage}_tracker")

    def _setup_fit(
        self,
        max_epochs: Optional[int],
        min_steps: Optional[int],
        num_train_batches: int,
        num_validation_batches: int,
        limit_train_batches: Optional[int] = None,
        limit_validation_batches: Optional[int] = None,
        validation_interval: Optional[int] = True,
    ) -> None:
        self.stop_training = False

        assert max_epochs is not None or min_steps is not None, "`max_epochs` or `min_steps` must be passed."

        # train: limit batches
        max_train_batches = min(num_train_batches, limit_train_batches or float("Inf"))

        # train: epochs and steps
        if max_epochs is None:
            max_epochs = np.ceil(min_steps / max_train_batches)
        if min_steps is not None:
            max_epochs_for_num_steps = int(np.ceil(min_steps / max_train_batches))
            if max_epochs < max_epochs_for_num_steps:
                # if we do not have enough batches across epochs, adjust epoch number
                max_epochs = max_epochs_for_num_steps
            else:
                # if we have enough batches to cover the num steps, do nothing
                min_steps = None

        # validation: limit batches and validation interval
        max_validation_batches = min(num_validation_batches, limit_validation_batches or float("Inf"))
        if (
            max_validation_batches is not None
            and validation_interval is not None
            and max_train_batches > validation_interval
        ):
            validation_interval = np.linspace(
                max_train_batches / validation_interval, max_train_batches, validation_interval, dtype=int
            ).tolist()[:-1]
        else:
            validation_interval = []

        self.epoch_tracker.max = max_epochs
        self.step_tracker.max = min_steps
        self.train_tracker.max = max_train_batches
        self.validation_tracker.max = max_validation_batches
        self.validation_interval = validation_interval


@dataclass
class RoundTracker(Tracker):
    current: int = 0
    total: int = 0

    def make_progress_bar(self) -> None:
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
    query_size: int = None

    def increment(self, n_labelled: Optional[int] = None) -> None:
        n_labelled = n_labelled or self.query_size
        self.current += n_labelled
        self.total += n_labelled

    def set_initial_budget(self, initial_budget: int) -> None:
        self.current = initial_budget
        self.total = initial_budget

    def max_reached(self) -> bool:
        if self.max is not None:
            return self.max < (self.query_size + self.total)

    @property
    def remaining_budget(self) -> Optional[int]:
        return self.max - self.current if self.max is not None else None


@dataclass
class ActiveProgressTracker(ProgressTracker):
    round_tracker: RoundTracker = field(default_factory=lambda: RoundTracker())
    budget_tracker: BudgetTracker = field(default_factory=lambda: BudgetTracker())
    pool_tracker: StageTracker = field(default_factory=lambda: StageTracker(stage=RunningStage.POOL))

    has_pool: bool = False
    has_test: bool = False

    """Properties"""

    @property
    def is_last_round(self) -> bool:
        return self.round_tracker.current >= (self.round_tracker.max - 1)

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

    """Helpers"""

    def setup(
        self,
        max_rounds: Optional[int],
        max_budget: Optional[int],
        query_size: int,
        initial_budget: int,
        has_pool: bool,
        has_test: bool,
        has_validation: bool,
        log_interval: Optional[int] = 1,
        enable_progress_bar: Optional[bool] = True,
    ) -> None:
        """Create progress bars."""

        self.log_interval = log_interval
        self.enable_progress_bar = enable_progress_bar
        self.has_validation = has_validation
        self.has_pool = has_pool
        self.has_test = has_test

        self.round_tracker.reset()
        self.budget_tracker.reset()
        if max_budget is not None:
            assert max_budget - initial_budget > 0, ValueError("`max_budget` must be bigger than `initial_budget`.")
            if (max_budget - initial_budget) > 0:
                max_rounds = min(max_rounds, math.ceil((max_budget - initial_budget) / query_size))
        self.round_tracker.max = max_rounds + 1
        self.budget_tracker = BudgetTracker(
            max=max_budget, total=initial_budget, current=initial_budget, query_size=query_size
        )

        if self.enable_progress_bar:
            self.round_tracker.make_progress_bar()
            self.epoch_tracker.make_progress_bar()
            self.train_tracker.make_progress_bar()
            if self.has_validation:
                self.validation_tracker.make_progress_bar()
            if self.has_test:
                self.test_tracker.make_progress_bar()
            if self.has_pool:
                self.pool_tracker.make_progress_bar()

    def setup_round_tracking(self, **kwargs) -> None:
        self._setup_fit(
            max_epochs=kwargs.get("max_epochs"),
            min_steps=kwargs.get("min_steps"),
            num_train_batches=kwargs.get("num_train_batches"),
            num_validation_batches=kwargs.get("num_validation_batches"),
            limit_train_batches=kwargs.get("limit_train_batches"),
            limit_validation_batches=kwargs.get("limit_validation_batches"),
            validation_interval=kwargs.get("validation_interval"),
        )
        for stage in (RunningStage.TEST, RunningStage.POOL):
            if getattr(self, f"has_{stage}"):
                limit_batches = kwargs.get(f"limit_{stage}_batches") or float("Inf")
                getattr(self, f"{stage}_tracker").max = min(kwargs.get(f"num_{stage}_batches"), limit_batches)
