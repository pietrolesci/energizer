import time

from src.energizer.callbacks.base import Callback
from src.energizer.enums import RunningStage
from src.energizer.estimator import Estimator


class Timer(Callback):
    def epoch_start(self, stage: RunningStage) -> None:
        setattr(self, f"{stage}_epoch_start_time", time.perf_counter())

    def epoch_end(self, estimator: Estimator, stage: RunningStage) -> None:
        setattr(self, f"{stage}_epoch_end_time", time.perf_counter())
        runtime = getattr(self, f"{stage}_epoch_end_time") - getattr(self, f"{stage}_epoch_start_time")
        estimator.log(f"timer/{stage}_epoch_time", runtime, step=estimator.progress_tracker.safe_global_epoch)

    def batch_start(self, stage: RunningStage) -> None:
        setattr(self, f"{stage}_batch_start_time", time.perf_counter())

    def batch_end(self, estimator: Estimator, stage: RunningStage) -> None:
        setattr(self, f"{stage}_batch_end_time", time.perf_counter())
        runtime = getattr(self, f"{stage}_batch_end_time") - getattr(self, f"{stage}_batch_start_time")
        estimator.log(f"timer/{stage}_batch_time", runtime, step=estimator.progress_tracker.global_batch)

    def on_fit_start(self, *args, **kwargs) -> None:
        self.fit_start = time.perf_counter()

    def on_fit_end(self, estimator: Estimator, *args, **kwargs) -> None:
        self.fit_end = time.perf_counter()
        estimator.log("timer/fit_time", self.fit_end - self.fit_start, step=0)

    """
    Epoch start
    """

    def on_train_epoch_start(self, *args, **kwargs) -> None:
        self.epoch_start(RunningStage.TRAIN)

    def on_validation_epoch_start(self, *args, **kwargs) -> None:
        self.epoch_start(RunningStage.VALIDATION)

    def on_test_epoch_start(self, *args, **kwargs) -> None:
        self.epoch_start(RunningStage.TEST)

    """
    Epoch end
    """

    def on_train_epoch_end(self, estimator: Estimator, *args, **kwargs) -> None:
        self.epoch_end(estimator, RunningStage.TRAIN)

    def on_validation_epoch_end(self, estimator: Estimator, *args, **kwargs) -> None:
        self.epoch_end(estimator, RunningStage.VALIDATION)

    def on_test_epoch_end(self, estimator: Estimator, *args, **kwargs) -> None:
        self.epoch_end(estimator, RunningStage.TEST)

    """
    Batch start
    """

    def on_train_batch_start(self, *args, **kwargs) -> None:
        self.batch_start(RunningStage.TRAIN)

    def on_validation_batch_start(self, *args, **kwargs) -> None:
        self.batch_start(RunningStage.VALIDATION)

    def on_test_batch_start(self, *args, **kwargs) -> None:
        self.batch_start(RunningStage.TEST)

    """
    Batch end
    """

    def on_train_batch_end(self, estimator: Estimator, batch_idx: int, *args, **kwargs) -> None:
        self.batch_end(estimator, RunningStage.TRAIN)

    def on_validation_batch_end(self, estimator: Estimator, batch_idx: int, *args, **kwargs) -> None:
        self.batch_end(estimator, RunningStage.VALIDATION)

    def on_test_batch_end(self, estimator: Estimator, batch_idx: int, *args, **kwargs) -> None:
        self.batch_end(estimator, RunningStage.TEST)
