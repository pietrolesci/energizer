import time
from typing import Any

from lightning.fabric.wrappers import _FabricModule

from energizer.active_learning.strategies.base import ActiveEstimator
from energizer.callbacks.base import Callback
from energizer.callbacks.timer import Timer as _Timer
from energizer.datastores.base import Datastore
from energizer.enums import RunningStage
from energizer.types import BATCH_OUTPUT, EPOCH_OUTPUT, METRIC, ROUND_OUTPUT


class ActiveLearningCallback(Callback):
    r"""
    Abstract base class used to build new callbacks.

    Subclass this class and override any of the relevant hooks
    """

    def on_pool_batch_start(self, estimator: ActiveEstimator, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        return self.on_batch_start(RunningStage.POOL, estimator, model, batch, batch_idx)

    def on_pool_batch_end(
        self, estimator: ActiveEstimator, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        return self.on_batch_end(RunningStage.POOL, estimator, model, output, batch, batch_idx)

    def on_pool_epoch_start(self, estimator: ActiveEstimator, model: _FabricModule) -> None:
        return self.on_epoch_start(RunningStage.POOL, estimator, model)

    def on_pool_epoch_end(
        self, estimator: ActiveEstimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        return self.on_epoch_end(RunningStage.POOL, estimator, model, output, metrics)

    def on_active_fit_start(self, estimator: ActiveEstimator, datastore: Datastore) -> None:
        ...

    def on_active_fit_end(self, estimator: ActiveEstimator, datastore: Datastore, output: Any) -> None:
        ...

    def on_round_start(self, estimator: ActiveEstimator, datastore: Datastore) -> None:
        ...

    def on_round_end(self, estimator: ActiveEstimator, datastore: Datastore, output: ROUND_OUTPUT) -> None:
        ...

    def on_query_start(self, estimator: ActiveEstimator, model: _FabricModule, datastore: Datastore) -> None:
        ...

    def on_query_end(
        self, estimator: ActiveEstimator, model: _FabricModule, datastore: Datastore, indices: list[int]
    ) -> None:
        ...

    def on_label_start(self, estimator: ActiveEstimator, datastore: Datastore) -> None:
        ...

    def on_label_end(self, estimator: ActiveEstimator, datastore: Datastore) -> None:
        ...


class Timer(_Timer, ActiveLearningCallback):
    def on_active_fit_start(self, *args, **kwargs) -> None:
        self.active_fit_start = time.perf_counter()

    def on_active_fit_end(self, estimator: ActiveEstimator, *args, **kwargs) -> None:
        self.active_fit_end = time.perf_counter()
        estimator.fabric.log("timer/active_fit_time", self.active_fit_end - self.active_fit_start, step=0)

    def on_round_start(self, *args, **kwargs) -> None:
        self.round_start = time.perf_counter()

    def on_round_end(self, estimator: ActiveEstimator, *args, **kwargs) -> None:
        self.round_end = time.perf_counter()
        estimator.fabric.log("timer/round_time", self.round_end - self.round_start, step=estimator.tracker.global_round)

    def on_query_start(self, *args, **kwargs) -> None:
        self.query_start = time.perf_counter()

    def on_query_end(self, estimator: ActiveEstimator, *args, **kwargs) -> None:
        self.query_end = time.perf_counter()
        estimator.fabric.log("timer/query_time", self.query_end - self.query_start, step=estimator.tracker.global_round)

    def on_label_start(self, *args, **kwargs) -> None:
        self.label_start = time.perf_counter()

    def on_label_end(self, estimator: ActiveEstimator, *args, **kwargs) -> None:
        self.label_end = time.perf_counter()
        estimator.fabric.log("timer/label_time", self.label_end - self.label_start, step=estimator.tracker.global_round)

    def on_pool_epoch_start(self, *args, **kwargs) -> None:
        self.epoch_start(RunningStage.POOL)

    def on_pool_epoch_end(self, estimator: ActiveEstimator, *args, **kwargs) -> None:
        self.epoch_end(estimator, RunningStage.POOL)

    def on_pool_batch_start(self, *args, **kwargs) -> None:
        self.batch_start(RunningStage.POOL)

    def on_pool_batch_end(self, estimator: ActiveEstimator, *args, **kwargs) -> None:
        self.batch_end(estimator, RunningStage.POOL)
