import os
from pathlib import Path
import shutil
from typing import Any, Dict, Optional, Union

import srsly
from lightning.fabric.wrappers import _FabricModule

from energizer.callbacks.base import CallbackWithMonitor
from energizer.enums import Interval, RunningStage
from energizer.estimator import Estimator
from energizer.types import BATCH_OUTPUT, EPOCH_OUTPUT, METRIC
from energizer.utilities import make_dict_json_serializable
from lightning_utilities.core.rank_zero import rank_zero_info


class ModelCheckpoint(CallbackWithMonitor):
    _best_k_models: Dict[str, float] = {}

    def __init__(
        self,
        dirpath: Union[Path, str],
        monitor: str,
        stage: Union[str, RunningStage],
        mode: str = "min",
        save_last: Optional[bool] = None,
        save_top_k: int = 1,
        verbose: bool = True,
        frequency: str = "1:epoch",
    ) -> None:
        super().__init__()
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.stage = stage
        self.mode = mode
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.verbose = verbose
        self.frequency = frequency

        every_n, interval = self.frequency.split(":")
        every_n = int(every_n)
        assert every_n > 0
        assert interval in list(Interval)

        rank_zero_info(f"Running ModelCheckpoint callback every {every_n} {interval}")
        self.every_n = every_n
        self.interval = interval

    @property
    def best_model_path(self) -> str:
        return self.optim_op(self._best_k_models, key=self._best_k_models.get)

    def on_fit_start(self, *args, **kwargs) -> None:
        # prepare directory
        if self.dirpath.exists():
            # during active learning we do not want to keep checkpoints from previous iterations
            for ckpt_path in self.dirpath.glob("*.pt"):
                ckpt_path.unlink()
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self._best_k_models = {}

    def on_fit_end(self, estimator: Estimator, *args, **kwargs) -> None:
        # load best model
        if self.monitor is not None:
            estimator.load_state_dict(self.dirpath, self.best_model_path)

            if self.verbose:
                logs = {"selected": self.best_model_path, "step": estimator.tracker.safe_global_epoch}
                if hasattr(estimator.tracker, "global_round"):
                    logs["round"] = getattr(estimator.tracker, "global_round")

                srsly.write_jsonl(
                    self.dirpath / "checkpoint_logs.jsonl",
                    [make_dict_json_serializable(logs)],
                    append=True,
                    append_new_line=False,
                )

    """
    Helpers
    """

    def should_checkpoint(self, stage: Union[str, RunningStage], interval: Interval, step_or_epoch: int) -> bool:
        return stage == self.stage and interval == self.interval and (step_or_epoch + 1) % self.every_n == 0

    def on_epoch_end(
        self,
        stage: Union[str, RunningStage],
        estimator: Estimator,
        model: _FabricModule,
        output: EPOCH_OUTPUT,
        metrics: METRIC,
    ) -> None:
        if self.should_checkpoint(stage, Interval.EPOCH, estimator.tracker.global_epoch):
            self.checkpoint(stage, estimator, output)

    def on_batch_end(
        self,
        stage: Union[str, RunningStage],
        estimator: Estimator,
        model: _FabricModule,
        output: BATCH_OUTPUT,
        batch: Any,
        batch_idx: int,
    ) -> None:
        if self.should_checkpoint(stage, Interval.STEP, estimator.tracker.global_step):
            self.checkpoint(stage, estimator, output)

    def checkpoint(
        self, stage: Union[str, RunningStage], estimator: Estimator, output: Union[EPOCH_OUTPUT, BATCH_OUTPUT]
    ) -> None:
        current = self._get_monitor(output)
        if self._check_should_save(stage, current):
            # checkpoint
            name = self._get_name(estimator, current)
            estimator.save_state_dict(self.dirpath, name)
            self._update_best_models(name, current)

            # log
            if self.verbose:
                logs = {"step": estimator.tracker.global_step, **self._best_k_models}
                if hasattr(estimator.tracker, "global_round"):
                    logs["round"] = getattr(estimator.tracker, "global_round", None)
                srsly.write_jsonl(
                    self.dirpath / "checkpoint_logs.jsonl",
                    [make_dict_json_serializable(logs)],
                    append=True,
                    append_new_line=False,
                )

    def _check_should_save(self, stage: Union[str, RunningStage], current: Optional[float]) -> bool:
        should_save = False

        # if you do not monitor it will save every time the stage is finished
        if self.monitor is None or self.stage is None or self.save_top_k is None:
            should_save = True

        # save based on monitored value
        elif self.stage == stage and current is not None:
            # if we still do not have k checkpoints saved
            if len(self._best_k_models) < self.save_top_k:
                should_save = True

            else:
                worst_scores = self.reverse_optim_op(self._best_k_models.values())
                should_save = self.monitor_op(current, worst_scores)

        return should_save

    def _get_name(self, estimator: Estimator, current: Optional[float] = None) -> str:
        # build filename
        name = f"step{estimator.tracker.global_step}"
        if current is not None:
            name += f"_{self.monitor.replace('/', '__')}={current:.6f}"
        name += ".pt"

        return name

    def _update_best_models(self, name: str, current: Optional[float]) -> None:
        if current is not None:
            if self.save_top_k is not None and len(self._best_k_models) >= self.save_top_k:
                worst_ckpt = self.reverse_optim_op(self._best_k_models, key=self._best_k_models.get)
                self._best_k_models.pop(worst_ckpt)
                if (self.dirpath / worst_ckpt).is_dir():
                    shutil.rmtree(self.dirpath / worst_ckpt)
                else:
                    os.remove(self.dirpath / worst_ckpt)
            self._best_k_models[name] = current
