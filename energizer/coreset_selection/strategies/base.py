from energizer.coreset_selection.trackers import CoresetProgressTracker
from energizer.datastores import Datastore
from energizer.estimator import Estimator, OptimizationArgs, SchedulerArgs


class CoresetSelectionStrategy(Estimator):
    _tracker: CoresetProgressTracker

    def init_tracker(self) -> None:
        self._tracker = CoresetProgressTracker()

    @property
    def tracker(self) -> CoresetProgressTracker:
        return self._tracker

    """
    Coreset learning loop
    """

    def select_coreset(
        self,
        datastore: Datastore,
        max_epochs: int | None = 3,
        min_epochs: int | None = None,
        max_steps: int | None = None,
        min_steps: int | None = None,
        validation_freq: str | None = "1:epoch",
        gradient_accumulation_steps: int | None = None,
        learning_rate: float | None = None,
        optimizer: str | None = None,
        optimizer_kwargs: dict | OptimizationArgs | None = None,
        scheduler: str | None = None,
        scheduler_kwargs: dict | SchedulerArgs | None = None,
        log_interval: int = 1,
        enable_progress_bar: bool = True,
        limit_train_batches: int | None = None,
        limit_validation_batches: int | None = None,
        limit_pool_batches: int | None = None,
    ) -> list[float]: ...
