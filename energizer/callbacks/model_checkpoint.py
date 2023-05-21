import os
from pathlib import Path
from typing import Dict, Optional, Union

import srsly
from lightning.fabric.wrappers import _FabricModule
import shutil
from energizer.callbacks.base import CallbackWithMonitor
from energizer.enums import RunningStage
from energizer.estimators.estimator import Estimator
from energizer.types import EPOCH_OUTPUT, METRIC
from energizer.utilities import make_dict_json_serializable


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
    ):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.monitor = monitor
        self.stage = stage
        self.mode = mode
        self.save_last = save_last
        self.save_top_k = save_top_k
        self.verbose = verbose

    @property
    def best_model_path(self) -> str:
        return self.optim_op(self._best_k_models, key=self._best_k_models.get)

    def on_train_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        self.epoch_end(estimator, output, RunningStage.TRAIN)

    def on_validation_epoch_end(
        self, estimator: Estimator, model: _FabricModule, output: EPOCH_OUTPUT, metrics: METRIC
    ) -> None:
        self.epoch_end(estimator, output, RunningStage.VALIDATION)

    def on_fit_start(self, *args, **kwargs) -> None:
        # prepare directory
        if self.dirpath.exists():
            # during active learning we do not want to keep checkpoints from previous iterations
            for i in self.dirpath.glob("*.pt"):
                os.remove(i.absolute())
            shutil.rmtree(self.dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self._best_k_models = {}

    def on_fit_end(self, estimator: Estimator, *args, **kwargs) -> None:
        if self.monitor is not None:
            estimator.load_state_dict(self.dirpath, self.best_model_path)

            if self.verbose:
                logs = {
                    "selected": self.best_model_path,
                    "step": estimator.progress_tracker.safe_global_epoch,
                }
                if hasattr(estimator.progress_tracker, "global_round"):
                    logs["round"] = getattr(estimator.progress_tracker, "global_round")

                srsly.write_jsonl(
                    self.dirpath / "checkpoint_logs.jsonl",
                    [make_dict_json_serializable(logs)],
                    append=True,
                    append_new_line=False,
                )

    """
    Helpers
    """

    def epoch_end(self, estimator: Estimator, output: EPOCH_OUTPUT, stage: Union[str, RunningStage]) -> None:
        if stage != self.stage:
            return

        current = self._get_monitor(output)

        if self._check_should_save(stage, current):
            # checkpoint
            name = self._get_name(estimator, stage, current)
            estimator.save_state_dict(self.dirpath, name)
            self._update_best_models(name, current)

            # log
            if self.verbose:
                logs = {
                    "stage": stage,
                    "step": estimator.progress_tracker.safe_global_epoch,
                    **self._best_k_models,
                }
                if hasattr(estimator.progress_tracker, "global_round"):
                    logs["round"] = getattr(estimator.progress_tracker, "global_round", None)
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

    def _get_name(self, estimator: Estimator, stage: Union[str, RunningStage], current: Optional[float] = None) -> str:
        # build filename
        step = "step" if stage == RunningStage.VALIDATION else "epoch"
        name = f"{stage}_{step}={estimator.progress_tracker.safe_global_epoch}"
        if current is not None:
            name += f"_{self.monitor}={current}"
        name += ".pt"

        return name

    def _update_best_models(self, name: str, current: Optional[float]) -> None:
        if current is not None:
            if self.save_top_k is not None and len(self._best_k_models) >= self.save_top_k:
                worst_ckpt = self.reverse_optim_op(self._best_k_models, key=self._best_k_models.get)
                self._best_k_models.pop(worst_ckpt)
                os.remove(self.dirpath / worst_ckpt)
            self._best_k_models[name] = current
