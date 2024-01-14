from collections.abc import Callable
from typing import Any, Optional, Union

import numpy as np
from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from energizer.enums import RunningStage
from energizer.estimator import Estimator
from energizer.types import BATCH_OUTPUT, EPOCH_OUTPUT, FIT_OUTPUT, METRIC
from energizer.utilities import move_to_cpu


class Callback:
    r"""
    Abstract base class used to build new callbacks.

    Subclass this class and override any of the relevant hooks
    """

    """
    Fit
    """

    def on_fit_start(self, estimator: Estimator, model: _FabricModule) -> None:
        ...

    def on_fit_end(self, estimator: Estimator, model: _FabricModule, output: list[FIT_OUTPUT]) -> None:
        ...

    """
    Epoch
    """

    def on_epoch_start(
        self, stage: Union[str, RunningStage], estimator: Estimator, model: _FabricModule, **kwargs
    ) -> None:
        ...

    def on_train_epoch_start(self, estimator: Estimator, model: _FabricModule, optimizer: Optimizer) -> None:
        return self.on_epoch_start(RunningStage.TRAIN, estimator, model, optimizer=optimizer)

    def on_validation_epoch_start(self, estimator: Estimator, model: _FabricModule) -> None:
        return self.on_epoch_start(RunningStage.VALIDATION, estimator, model)

    def on_test_epoch_start(self, estimator: Estimator, model: _FabricModule) -> None:
        return self.on_epoch_start(RunningStage.TEST, estimator, model)

    def on_epoch_end(
        self,
        stage: Union[str, RunningStage],
        estimator: Estimator,
        model: _FabricModule,
        output: EPOCH_OUTPUT,
        metrics: METRIC,
    ) -> None:
        ...

    def on_train_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        return self.on_epoch_end(RunningStage.TRAIN, estimator, model, output, metrics)

    def on_validation_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        return self.on_epoch_end(RunningStage.VALIDATION, estimator, model, output, metrics)

    def on_test_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        return self.on_epoch_end(RunningStage.TEST, estimator, model, output, metrics)

    """
    Batch
    """

    def on_batch_start(
        self,
        stage: Union[str, RunningStage],
        estimator: Estimator,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        **kwargs,
    ) -> None:
        ...

    def on_train_batch_start(
        self, estimator: Estimator, model: _FabricModule, batch: Any, batch_idx: int, optimizer: Optimizer
    ) -> None:
        return self.on_batch_start(RunningStage.TRAIN, estimator, model, batch, batch_idx, optimizer=optimizer)

    def on_validation_batch_start(self, estimator: Estimator, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        return self.on_batch_start(RunningStage.VALIDATION, estimator, model, batch, batch_idx)

    def on_test_batch_start(self, estimator: Estimator, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        return self.on_batch_start(RunningStage.TEST, estimator, model, batch, batch_idx)

    def on_batch_end(
        self,
        stage: Union[str, RunningStage],
        estimator: Estimator,
        model: _FabricModule,
        output: BATCH_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        ...

    def on_train_batch_end(
        self, estimator: Estimator, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        return self.on_batch_end(RunningStage.TRAIN, estimator, model, output, batch, batch_idx)

    def on_validation_batch_end(
        self, estimator: Estimator, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        return self.on_batch_end(RunningStage.VALIDATION, estimator, model, output, batch, batch_idx)

    def on_test_batch_end(
        self, estimator: Estimator, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        return self.on_batch_end(RunningStage.TEST, estimator, model, output, batch, batch_idx)

    def on_before_optimizer(self, estimator: Estimator, model: _FabricModule, optimizer: _FabricOptimizer) -> None:
        ...

    def on_after_optimizer(self, estimator: Estimator, model: _FabricModule, optimizer: _FabricOptimizer) -> None:
        ...

    def on_before_scheduler(
        self, estimator: Estimator, model: _FabricModule, optimizer: _FabricOptimizer, scheduler: _LRScheduler
    ) -> None:
        ...

    def on_after_scheduler(
        self, estimator: Estimator, model: _FabricModule, optimizer: _FabricOptimizer, scheduler: _LRScheduler
    ) -> None:
        ...


class CallbackWithMonitor(Callback):
    mode: str
    monitor: str
    mode_dict = {"min": np.less, "max": np.greater}
    optim_dict = {"min": min, "max": max}
    reverse_optim_dict = {"min": max, "max": min}

    @property
    def monitor_op(self) -> Callable:
        return self.mode_dict[self.mode]

    @property
    def optim_op(self) -> Callable:
        return self.optim_dict[self.mode]

    @property
    def reverse_optim_op(self) -> Callable:
        return self.reverse_optim_dict[self.mode]

    def _get_monitor(self, output: Optional[Union[BATCH_OUTPUT, EPOCH_OUTPUT]]) -> float:
        if not isinstance(output, dict) or output is None:
            raise RuntimeError(
                "From `*_step` and `*_epoch_end` method you need to return dict to use ",
                "monitoring Callbacks like EarlyStopping and ModelCheckpoint.",
            )

        monitor = output.get(self.monitor, None)
        if monitor is None:
            raise ValueError(f"`{self.monitor}` is not logged. Currently logged metrics {list(output.keys())}")

        return move_to_cpu(monitor)
