from typing import Any, Callable, List, Optional, Union

import numpy as np
from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer
from torch.optim import Optimizer

from src.energizer.enums import OutputKeys
from src.energizer.estimator import Estimator, FitEpochOutput
from src.energizer.types import BATCH_OUTPUT, EPOCH_OUTPUT, METRIC
from src.energizer.utilities import move_to_cpu


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

    def on_fit_end(self, estimator: Estimator, model: _FabricModule, output: List[FitEpochOutput]) -> None:
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


class CallbackWithMonitor(Callback):
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
        if self.monitor is None:
            return

        # look for monitored metric in output or in metrics
        monitor = output.get(self.monitor) or output[OutputKeys.METRICS].get(self.monitor)
        if monitor is None:
            raise ValueError(f"`{self.monitor}` is not logged.")

        return move_to_cpu(monitor)
