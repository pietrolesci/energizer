from dataclasses import dataclass
from typing import Optional

import numpy as np
from tqdm.auto import tqdm

from energizer.active_learning.datastores.base import ActiveDataStore
from energizer.enums import RunningStage
from energizer.trackers import ProgressTracker, StageTracker, Tracker


@dataclass
class RoundTracker(Tracker):
    def reset(self) -> None:
        self.total = 0
        return super().reset()

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
    query_size: int

    def reset(self) -> None:
        self.total = 0
        return super().reset()

    def make_progress_bar(self) -> None:
        self.progress_bar = tqdm(
            total=self.max,
            desc="Labelled",
            dynamic_ncols=True,
            leave=True,
            colour="#32a852",
        )
        if self.current > 0:
            self.progress_bar.update(self.current)

    def increment(self, n_labelled: Optional[int] = None) -> None:
        n_labelled = n_labelled or self.query_size
        self.current += n_labelled
        self.total += n_labelled
        if self.progress_bar is not None and self.current > 0:
            self.progress_bar.update(n_labelled)

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
        self.budget_tracker = BudgetTracker(current=0, max=None, query_size=-1)

        self.run_on_pool = False
        self.has_test = False

    def setup_active(
        self,
        max_rounds: Optional[int],
        max_budget: Optional[int],
        query_size: int,
        datastore: ActiveDataStore,
        validation_perc: Optional[float],
        log_interval: int,
        enable_progress_bar: bool,
        run_on_pool: bool,
    ) -> None:
        """Create progress bars."""

        self.log_interval = log_interval
        self.enable_progress_bar = enable_progress_bar
        self.run_on_pool = run_on_pool

        # rounds
        max_budget = int(min(datastore.pool_size(self.global_round), max_budget or float("Inf")))
        initial_budget = datastore.labelled_size(self.global_round)

        assert (
            max_budget is not None or max_rounds is not None
        ), "At least one of `max_rounds` or `max_budget` must be not None."
        assert max_budget > initial_budget, ValueError(f"`{max_budget=}` must be bigger than `{initial_budget=}`.")

        max_rounds = min(int(np.ceil((max_budget - initial_budget) / query_size)), max_rounds or float("Inf"))
        max_budget = (query_size * max_rounds) + initial_budget

        self.has_test = datastore.test_size() is not None and datastore.test_size() > 0
        self.has_validation = (
            datastore.validation_size() is not None and datastore.validation_size() > 0
        ) or validation_perc is not None

        self.round_tracker.reset()
        self.budget_tracker.reset()

        self.round_tracker.max = max_rounds + 1
        self.budget_tracker.query_size = query_size
        self.budget_tracker.max = max_budget
        if initial_budget > 0:
            self.budget_tracker.increment(initial_budget)

        self.make_progress_bars_active()

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

    @property
    def is_active_fit_done(self) -> bool:
        return self.round_tracker.max_reached()

    """Super outer loops"""

    def end_active_fit(self) -> None:
        self.round_tracker.close_progress_bar()
        self.budget_tracker.close_progress_bar()
        self.step_tracker.close_progress_bar()
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
        if self.enable_progress_bar and self.epoch_tracker.progress_bar is not None:
            self.epoch_tracker.progress_bar.set_postfix_str("")

    def end_fit(self) -> None:
        self.step_tracker.terminate_progress_bar()
        self.epoch_tracker.terminate_progress_bar()
        self.train_tracker.terminate_progress_bar()
        self.validation_tracker.terminate_progress_bar()

    """Stage trackers"""

    def end(self) -> None:
        self.get_stage_tracker().terminate_progress_bar()
        if self.current_stage == RunningStage.VALIDATION:
            self.current_stage = RunningStage.TRAIN  # reattach training

    def make_progress_bars_active(self) -> None:
        if self.enable_progress_bar:
            self.round_tracker.make_progress_bar()
            self.budget_tracker.make_progress_bar()
            self.step_tracker.make_progress_bar()
            self.epoch_tracker.make_progress_bar()
            self.train_tracker.make_progress_bar()
            if self.has_validation:
                self.validation_tracker.make_progress_bar()
            if self.has_test:
                self.test_tracker.make_progress_bar()
            if self.run_on_pool:
                self.pool_tracker.make_progress_bar()
