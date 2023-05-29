from pathlib import Path
from typing import Any, Optional, Tuple, Union

import numpy as np
import srsly
from lightning.fabric.wrappers import _FabricModule

from energizer.callbacks.base import CallbackWithMonitor
from energizer.enums import Interval, RunningStage
from energizer.estimators.estimator import Estimator
from energizer.types import BATCH_OUTPUT, EPOCH_OUTPUT, METRIC
from energizer.utilities import make_dict_json_serializable


class EarlyStopping(CallbackWithMonitor):
    order_dict = {"min": "<", "max": ">"}

    def __init__(
        self,
        monitor: str,
        stage: Union[str, RunningStage],
        interval: Interval = Interval.EPOCH,
        mode: str = "min",
        min_delta=0.00,
        patience=3,
        stopping_threshold: Optional[float] = None,
        divergence_threshold: Optional[float] = None,
        verbose: bool = True,
    ) -> None:
        super().__init__()
        self.monitor = monitor
        self.stage = stage
        self.interval = interval
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.stopping_threshold = stopping_threshold
        self.divergence_threshold = divergence_threshold
        self.verbose = verbose
        self.dirpath = Path("./.early_stopping.jsonl")

    def _check_stopping_criteria(
        self, output: Union[BATCH_OUTPUT, EPOCH_OUTPUT], step: int
    ) -> Tuple[bool, Union[str, None]]:
        current = self._get_monitor(output)

        should_stop = False
        reason = None
        if not np.isfinite(current):
            should_stop = True
            reason = (
                f"Monitored metric {self.monitor} = {current} is not finite."
                f" Previous best value was {self.best_score:.6f}."
            )
        elif self.stopping_threshold is not None and self.monitor_op(current, self.stopping_threshold):
            should_stop = True
            reason = (
                "Stopping threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.stopping_threshold}."
            )
        elif self.divergence_threshold is not None and self.monitor_op(-current, -self.divergence_threshold):
            should_stop = True
            reason = (
                "Divergence threshold reached:"
                f" {self.monitor} = {current} {self.order_dict[self.mode]} {self.divergence_threshold}."
            )
        elif self.monitor_op(current - self.min_delta, self.best_score):
            should_stop = False
            self.best_score = current
            self.best_step = step
            self.wait_count = 0
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                should_stop = True
                reason = (
                    f"Monitored metric `{self.monitor}` did not improve in the last {self.wait_count} {self.interval}"
                    f"{'es' if self.interval == Interval.BATCH else 's'}."
                )

        return should_stop, reason

    def check(
        self,
        estimator: Estimator,
        output: Union[BATCH_OUTPUT, EPOCH_OUTPUT],
        stage: Union[str, RunningStage],
        interval: Interval,
    ) -> None:
        if (self.stage == stage and self.interval == interval) and estimator.progress_tracker.is_fitting:
            step = (
                estimator.progress_tracker.safe_global_epoch
                if interval == Interval.EPOCH
                else estimator.progress_tracker.global_batch
            )
            should_stop, reason = self._check_stopping_criteria(output, step)
            if should_stop:
                estimator.progress_tracker.set_stop_training(True)
                if self.verbose:
                    _msg = {
                        "best_score": round(self.best_score, 6),
                        "best_step": self.best_step,
                        "stage": stage,
                        "interval": interval,
                        "stopping_step": step,
                        "reason": reason,
                    }
                    srsly.write_jsonl(
                        self.dirpath, [make_dict_json_serializable(_msg)], append=True, append_new_line=False
                    )

    def reset(self) -> None:
        self.wait_count = 0
        self.min_delta *= 1 if self.monitor_op == np.greater else -1
        self.best_score = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_step = 0

    def on_fit_start(self, *args, **kwargs) -> None:
        self.reset()

    def on_train_batch_end(
        self, estimator: Estimator, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        self.check(estimator, output, RunningStage.TRAIN, Interval.BATCH)

    def on_validation_batch_end(
        self, estimator: Estimator, model: _FabricModule, output: BATCH_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        self.check(estimator, output, RunningStage.VALIDATION, Interval.BATCH)

    def on_train_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        self.check(estimator, output, RunningStage.TRAIN, Interval.EPOCH)

    def on_validation_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        self.check(estimator, output, RunningStage.VALIDATION, Interval.EPOCH)
