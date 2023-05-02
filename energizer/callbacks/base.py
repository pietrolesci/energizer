from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer
from torch.optim import Optimizer

from energizer.datastores.base import Datastore
from energizer.estimators.active_estimator import ActiveEstimator
from energizer.estimators.estimator import Estimator
from energizer.types import BATCH_OUTPUT, EPOCH_OUTPUT, FIT_OUTPUT, METRIC, ROUND_OUTPUT
from energizer.utilities import move_to_cpu


class BaseCallback:
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

    def on_train_epoch_start(self, estimator: Estimator, model: _FabricModule, optimizer: Optimizer) -> None:
        ...

    def on_validation_epoch_start(self, estimator: Estimator, model: _FabricModule) -> None:
        ...

    def on_test_epoch_start(self, estimator: Estimator, model: _FabricModule) -> None:
        ...

    def on_train_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        ...

    def on_validation_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        ...

    def on_test_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        ...

    """
    Batch
    """

    def on_train_batch_start(
        self, estimator: Estimator, model: _FabricModule, optimizer: Optimizer, batch: Any, batch_idx: int
    ) -> None:
        ...

    def on_train_batch_end(
        self, estimator: Estimator, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        ...

    def on_validation_batch_start(self, estimator: Estimator, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        ...

    def on_validation_batch_end(
        self, estimator: Estimator, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        ...

    def on_test_batch_start(self, estimator: Estimator, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        ...

    def on_test_batch_end(
        self, estimator: Estimator, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        ...

    def on_before_optimizer_step(self, estimator: Estimator, model: _FabricModule, optimizer: _FabricOptimizer) -> None:
        ...

    def on_after_optimizer_step(self, estimator: Estimator, model: _FabricModule, optimizer: _FabricOptimizer) -> None:
        ...


class ActiveLearningCallbackMixin:
    def on_active_fit_start(self, estimator: ActiveEstimator, datastore: Datastore) -> None:
        ...

    def on_active_fit_end(self, estimator: ActiveEstimator, datastore: Datastore, output: Any) -> None:
        ...

    def on_round_start(self, estimator: ActiveEstimator, datastore: Datastore) -> None:
        ...

    def on_round_end(self, estimator: ActiveEstimator, datastore: Datastore, output: ROUND_OUTPUT) -> None:
        ...

    def on_query_start(self, estimator: ActiveEstimator, model: _FabricModule) -> None:
        ...

    def on_query_end(self, estimator: ActiveEstimator, model: _FabricModule, output) -> None:
        ...

    def on_label_start(self, estimator: ActiveEstimator, datastore: Datastore) -> None:
        ...

    def on_label_end(self, estimator: ActiveEstimator, datastore: Datastore) -> None:
        ...

    def on_pool_batch_start(self, estimator: ActiveEstimator, model: _FabricModule, batch: Any, batch_idx: int) -> None:
        ...

    def on_pool_batch_end(
        self, estimator: ActiveEstimator, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        ...

    def on_pool_epoch_start(self, estimator: ActiveEstimator, model: _FabricModule) -> None:
        ...

    def on_pool_epoch_end(
        self, estimator: ActiveEstimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        ...


class Callback(BaseCallback, ActiveLearningCallbackMixin):
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

    def _get_monitor(self, output: Union[BATCH_OUTPUT, EPOCH_OUTPUT]) -> Optional[float]:
        if not isinstance(output, Dict):
            raise RuntimeError(
                "From `*_step` and `*_epoch_end` method you need to return dict to use monitoring Callbacks like EarlyStopping and ModelCheckpoint."
            )

        monitor = output.get(self.monitor)
        if monitor is None:
            raise ValueError(f"`{self.monitor}` is not logged.")

        return move_to_cpu(monitor)
