import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import torch.nn as nn
import wandb
from lightning.fabric.loggers.logger import Logger, rank_zero_experiment
from lightning.fabric.utilities.logger import _convert_params, _sanitize_callable_params
from lightning.fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn  # type: ignore
from lightning.fabric.utilities.types import _PATH
from wandb.sdk.lib import RunDisabled
from wandb.wandb_run import Run


class WandbLogger(Logger):
    _experiment: Optional[Union[Run, RunDisabled]] = None
    LOGGER_NAME: str = "wandb"

    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        dir: _PATH = ".",
        anonymous: Optional[bool] = None,
        start_method: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()

        # set wandb init arguments
        self._wandb_init: Dict[str, Any] = {
            "project": project or os.environ.get("WANDB_PROJECT", "energizer_logs"),
            "dir": os.fspath(dir) if dir is not None else dir,
            "name": name,
            "resume": "allow",
            "anonymous": ("allow" if anonymous else None),
            **kwargs,
        }
        if start_method is not None:
            self._wandb_init["settings"] = wandb.Settings(start_method=start_method)

        # start wandb run (to create an attach_id for distributed modes)
        wandb.require("service")  # type: ignore
        _ = self.experiment

    def __getstate__(self) -> Dict[str, Any]:
        state = self.__dict__.copy()
        # args needed to reload correct experiment
        if self._experiment is not None:
            state["_id"] = getattr(self._experiment, "id", None)
            state["_attach_id"] = getattr(self._experiment, "_attach_id", None)
            state["_name"] = self._experiment.name

        # cannot be pickled
        state["_experiment"] = None
        return state

    @property
    def logger_name(self) -> str:
        return self.LOGGER_NAME

    @property
    def name(self) -> Optional[str]:
        return self._wandb_init.get("name")

    @property
    def version(self) -> Optional[Union[int, str]]:
        return self._experiment.id if self._experiment else self._wandb_init.get("id")

    @property
    def entity(self) -> Optional[Union[int, str]]:
        return self._experiment.entity if self._experiment else self._wandb_init.get("entity")

    @property
    def project(self) -> Optional[Union[int, str]]:
        return self._experiment.project if self._experiment else self._wandb_init.get("project")

    @property
    def root_dir(self) -> Optional[str]:
        return self._wandb_init.get("dir")

    @property
    def run_id(self) -> Optional[Union[int, str]]:
        return self.version

    @property
    def run_path(self) -> str:
        return f"{self.entity}/{self.project}/{self.run_id}"

    @property
    @rank_zero_experiment
    def experiment(self) -> Union[Run, RunDisabled]:
        if self._experiment is None:
            if self._wandb_init.get("mode", None) == "offline":
                os.environ["WANDB_MODE"] = "dryrun"

            attach_id = getattr(self, "_attach_id", None)
            if wandb.run is not None:
                # wandb process already created in this instance
                rank_zero_warn(
                    "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
                    " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
                )
                self._experiment = wandb.run
            elif attach_id is not None and hasattr(wandb, "_attach"):
                # attach to wandb process referenced
                self._experiment = wandb._attach(attach_id)
            else:
                # create new wandb process
                self._experiment = wandb.init(**self._wandb_init)

                # define default x-axis
                if isinstance(self._experiment, (Run, RunDisabled)) and getattr(
                    self._experiment, "define_metric", None
                ):
                    self._experiment.define_metric("step")
                    self._experiment.define_metric("*", step_metric="step", step_sync=True)

        assert isinstance(self._experiment, (Run, RunDisabled))
        return self._experiment

    def watch(self, model: nn.Module, log: str = "gradients", log_freq: int = 100, log_graph: bool = True) -> None:
        self.experiment.watch(model, log=log, log_freq=log_freq, log_graph=log_graph)

    @rank_zero_only
    def finalize(self, status: str) -> None:
        self.experiment.finish()
        wandb.finish()

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
        params = _convert_params(params)
        params = _sanitize_callable_params(params)
        self.experiment.config.update(params, allow_val_change=True)

    @rank_zero_only
    def log_metrics(self, metrics: Dict, step: int) -> None:
        assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"
        self.experiment.log({**metrics, "step": step})

    def save_to_parquet(self, path: Union[str, Path]) -> None:
        run = wandb.Api().run(self.run_path)
        df = pd.DataFrame(run.scan_history())
        df.to_parquet(path, index=False)

    # @rank_zero_only
    # def log_table(
    #     self,
    #     step: int,
    #     key: str,
    #     columns: Optional[List[str]] = None,
    #     data: Optional[List[List[Any]]] = None,
    #     dataframe: Any = None,
    # ) -> None:
    #     """Log a Table containing any object type (text, image, audio, video, molecule, html, etc).

    #     Can be defined either with `columns` and `data` or with `dataframe`.
    #     """

    #     metrics = {key: wandb.Table(columns=columns, data=data, dataframe=dataframe)}
    #     self.log_metrics(metrics, step)

    # @rank_zero_only
    # def log_text(
    #     self,
    #     key: str,
    #     columns: Optional[List[str]] = None,
    #     data: Optional[List[List[str]]] = None,
    #     dataframe: Any = None,
    #     step: Optional[int] = None,
    # ) -> None:
    #     """Log text as a Table.

    #     Can be defined either with `columns` and `data` or with `dataframe`.
    #     """

    #     self.log_table(key, columns, data, dataframe, step)
