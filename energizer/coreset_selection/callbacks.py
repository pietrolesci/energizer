import time
from typing import Any

from lightning.fabric.wrappers import _FabricModule

from energizer.callbacks.base import Callback
from energizer.callbacks.timer import Timer as _Timer
from energizer.coreset_selection.strategies.base import CoresetSelectionStrategy
from energizer.enums import RunningStage
from energizer.types import BATCH_OUTPUT, EPOCH_OUTPUT, METRIC


class CoresetSelectionCallback(Callback):
    r"""
    Abstract base class used to build new callbacks.

    Subclass this class and override any of the relevant hooks
    """

    def on_pool_batch_start(
        self, estimator: CoresetSelectionStrategy, model: _FabricModule, batch: Any, batch_idx: int
    ) -> None:
        return self.on_batch_start(RunningStage.POOL, estimator, model, batch, batch_idx)

    def on_pool_batch_end(
        self,
        estimator: CoresetSelectionStrategy,
        model: _FabricModule,
        output: BATCH_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        return self.on_batch_end(RunningStage.POOL, estimator, model, output, batch, batch_idx)

    def on_pool_epoch_start(self, estimator: CoresetSelectionStrategy, model: _FabricModule) -> None:
        return self.on_epoch_start(RunningStage.POOL, estimator, model)

    def on_pool_epoch_end(
        self, estimator: CoresetSelectionStrategy, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        return self.on_epoch_end(RunningStage.POOL, estimator, model, output, metrics)


class Timer(_Timer, CoresetSelectionCallback):
    def on_round_start(self, *args, **kwargs) -> None:
        self.round_start = time.perf_counter()

    def on_pool_epoch_start(self, *args, **kwargs) -> None:
        self.epoch_start(RunningStage.POOL)

    def on_pool_epoch_end(self, estimator: CoresetSelectionStrategy, *args, **kwargs) -> None:
        self.epoch_end(estimator, RunningStage.POOL)

    def on_pool_batch_start(self, *args, **kwargs) -> None:
        self.batch_start(RunningStage.POOL)

    def on_pool_batch_end(self, estimator: CoresetSelectionStrategy, *args, **kwargs) -> None:
        self.batch_end(estimator, RunningStage.POOL)
