from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer
from torch.optim import Optimizer

from energizer.datastores.base import Datastore
from energizer.enums import RunningStage
from energizer.estimators.active_estimator import ActiveEstimator
from energizer.estimators.estimator import Estimator
from energizer.types import BATCH_OUTPUT, EPOCH_OUTPUT, FIT_OUTPUT, METRIC, ROUND_OUTPUT
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

    def on_fit_end(self, estimator: Estimator, model: _FabricModule, output: List[FIT_OUTPUT]) -> None:
        ...

    """
    Epoch
    """

    def on_epoch_start(self, stage: Union[str, RunningStage], estimator: Estimator, model: _FabricModule, **kwargs) -> None:
        ...

    def on_train_epoch_start(self, estimator: Estimator, model: _FabricModule, optimizer: Optimizer) -> None:
        return self.on_epoch_start(RunningStage.TRAIN, estimator, model, optimizer=optimizer)

    def on_validation_epoch_start(self, estimator: Estimator, model: _FabricModule) -> None:
        return self.on_epoch_start(RunningStage.VALIDATION, estimator, model)

    def on_test_epoch_start(self, estimator: Estimator, model: _FabricModule) -> None:
        return self.on_epoch_start(RunningStage.TEST, estimator, model)

    def on_pool_epoch_start(self, estimator: ActiveEstimator, model: _FabricModule) -> None:
        return self.on_epoch_start(RunningStage.POOL, estimator, model)

    def on_epoch_end(
        self, stage: Union[str, RunningStage], estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
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

    def on_pool_epoch_end(
        self, estimator: ActiveEstimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        return self.on_epoch_end(RunningStage.POOL, estimator, model, output, metrics)

    """
    Batch
    """

    def on_batch_start(
        self, stage: Union[str, RunningStage], estimator: Estimator, model: _FabricModule, batch: Any, batch_idx: int, **kwargs
    ) -> None:
        ...

    def on_train_batch_start(
        self,
        estimator: Estimator,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        optimizer: Optimizer,
    ) -> None:
        return self.on_batch_start(RunningStage.TRAIN, estimator, model, batch, batch_idx, optimizer=optimizer)

    def on_validation_batch_start(self, estimator: Estimator, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        return self.on_batch_start(RunningStage.VALIDATION, estimator, model, batch, batch_idx)

    def on_test_batch_start(self, estimator: Estimator, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        return self.on_batch_start(RunningStage.TEST, estimator, model, batch, batch_idx)

    def on_pool_batch_start(self, estimator: ActiveEstimator, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        return self.on_batch_start(RunningStage.POOL, estimator, model, batch, batch_idx)

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

    def on_pool_batch_end(
        self, estimator: ActiveEstimator, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        return self.on_batch_end(RunningStage.POOL, estimator, model, output, batch, batch_idx)

    """
    Step
    """

    def on_before_optimizer_step(self, estimator: Estimator, model: _FabricModule, optimizer: _FabricOptimizer) -> None:
        ...

    def on_after_optimizer_step(self, estimator: Estimator, model: _FabricModule, optimizer: _FabricOptimizer) -> None:
        ...

    """
    Active Learning
    """

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

    def on_query_end(self, estimator: ActiveEstimator, model: _FabricModule, datastore: Datastore, indices: List[int]) -> None:
        ...

    def on_label_start(self, estimator: ActiveEstimator, datastore: Datastore) -> None:
        ...

    def on_label_end(self, estimator: ActiveEstimator, datastore: Datastore) -> None:
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

    def _get_monitor(self, output: Union[BATCH_OUTPUT, EPOCH_OUTPUT]) -> float:
        if not isinstance(output, Dict):
            raise RuntimeError(
                "From `*_step` and `*_epoch_end` method you need to return dict to use monitoring Callbacks like EarlyStopping and ModelCheckpoint."
            )

        monitor = output.get(self.monitor, None)
        if monitor is None:
            raise ValueError(f"`{self.monitor}` is not logged. Currently logged metrics {list(output.keys())}")

        return move_to_cpu(monitor)
