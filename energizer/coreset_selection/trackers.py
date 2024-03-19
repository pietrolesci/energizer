from dataclasses import dataclass

from energizer.enums import RunningStage
from energizer.trackers import ProgressTracker, StageTracker


@dataclass
class CoresetProgressTracker(ProgressTracker):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.pool_batch_counter = StageTracker(stage=RunningStage.POOL, current=0, max=None)

    def setup(
        self,
        log_interval: int,
        enable_progress_bar: bool,
        num_pool_batches: int,
        limit_pool_batches: int | None,
        **fit_kwargs,
    ) -> None:
        self.log_interval = log_interval
        self.enable_progress_bar = enable_progress_bar

        self.setup_fit(
            min_epochs=None,
            max_steps=None,
            min_steps=None,
            validation_freq=None,
            num_validation_batches=0,
            **fit_kwargs,
        )
        self.setup_eval(RunningStage.POOL, num_batches=num_pool_batches, limit_batches=limit_pool_batches)

        self.make_progress_bars()

    def make_progress_bars(self) -> None:
        if self.enable_progress_bar:
            self.step_counter.make_progress_bar()
            self.epoch_idx_tracker.make_progress_bar()
            self.train_batch_counter.make_progress_bar()
            self.pool_batch_counter.make_progress_bar()

    def end_fit(self) -> None:
        """Close progress bars."""
        self.step_counter.close_progress_bar()
        self.epoch_idx_tracker.close_progress_bar()
        self.train_batch_counter.close_progress_bar()
        self.pool_batch_counter.close_progress_bar()

    @property
    def is_last_epoch(self) -> bool:
        return self.epoch_idx_tracker.current >= (self.epoch_idx_tracker.max - 1)  # type: ignore
