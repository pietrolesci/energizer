# import os
# from argparse import Namespace
# from typing import Any, Dict, List, Mapping, Optional, Union

# import torch.nn as nn
# import logging
# from lighting_fabric.utilities.logger import _add_prefix, _convert_params, _sanitize_callable_params
# from lighting_fabric.utilities.types import _PATH
# from lighting_fabric.utilities.rank_zero import rank_zero_only, rank_zero_warn
# from lighting_fabric.loggers.logger import Logger, rank_zero_experiment

# import wandb
# from wandb.sdk.lib import RunDisabled
# from wandb.wandb_run import Run

# log = logging.getLogger(__name__)


# class WandbLogger(Logger):
#     r"""Log using `Weights and Biases."""

#     LOGGER_JOIN_CHAR = "-"

#     def __init__(
#         self,
#         dir: _PATH = ".",
#         project: Optional[str] = None,
#         name: Optional[str] = None,
#         id: Optional[str] = None,
#         offline: bool = False,
#         anonymous: Optional[bool] = None,
#         prefix: str = "",
#         **kwargs: Any,
#     ) -> None:

#         super().__init__()

#         # paths are processed as strings
#         if dir is not None:
#             dir = os.fspath(dir)

#         project = project or os.environ.get("WANDB_PROJECT", "energizer_logs")

#         # set wandb init arguments
#         self._wandb_init: Dict[str, Any] = {
#             "dir": dir,
#             "project": project,
#             "name": name,
#             "id": id,
#             "resume": "allow",
#             "anonymous": ("allow" if anonymous else None),
#         }
#         self._wandb_init.update(**kwargs)

#         # extract parameters
#         self._project = self._wandb_init.get("project")
#         self._dir = self._wandb_init.get("dir")
#         self._name = self._wandb_init.get("name")
#         self._id = self._wandb_init.get("id")
#         self._offline = offline
#         self._prefix = prefix
#         self._experiment = None

#         # start wandb run (to create an attach_id for distributed modes)
#         wandb.require("service")  # type: ignore
#         _ = self.experiment

#     def __getstate__(self) -> Dict[str, Any]:
#         state = self.__dict__.copy()
#         # args needed to reload correct experiment
#         if self._experiment is not None:
#             state["_id"] = getattr(self._experiment, "id", None)
#             state["_attach_id"] = getattr(self._experiment, "_attach_id", None)
#             state["_name"] = self._experiment.name

#         # cannot be pickled
#         state["_experiment"] = None
#         return state

#     @property
#     def dir(self) -> Optional[str]:
#         return self._dir

#     @property
#     def project(self) -> Optional[str]:
#         return self._project

#     @property
#     def name(self) -> Optional[str]:
#         return self._name

#     @property
#     def id(self) -> Optional[str]:
#         """Gets the id of the experiment.

#         Returns:
#             The id of the experiment if the experiment exists else the id given to the constructor.
#         """
#         # don't create an experiment if we don't have one
#         return self._experiment.id if self._experiment else self._id

#     @property
#     @rank_zero_experiment
#     def experiment(self) -> Union[Run, RunDisabled]:
#         r"""

#         Actual wandb object. To use wandb features in your
#         :class:`~lightning.pytorch.core.module.LightningModule` do the following.

#         Example::

#         .. code-block:: python

#             self.logger.experiment.some_wandb_function()

#         """
#         if self._experiment is None:
#             if self._offline:
#                 os.environ["WANDB_MODE"] = "dryrun"

#             attach_id = getattr(self, "_attach_id", None)
#             if wandb.run is not None:
#                 # wandb process already created in this instance
#                 rank_zero_warn(
#                     "There is a wandb run already in progress and newly created instances of `WandbLogger` will reuse"
#                     " this run. If this is not desired, call `wandb.finish()` before instantiating `WandbLogger`."
#                 )
#                 self._experiment = wandb.run
#             elif attach_id is not None and hasattr(wandb, "_attach"):
#                 # attach to wandb process referenced
#                 self._experiment = wandb._attach(attach_id)
#             else:
#                 # create new wandb process
#                 self._experiment = wandb.init(**self._wandb_init)

#                 # define default x-axis
#                 if isinstance(self._experiment, (Run, RunDisabled)) and getattr(
#                     self._experiment, "define_metric", None
#                 ):
#                     self._experiment.define_metric("trainer/global_step")
#                     self._experiment.define_metric("*", step_metric="trainer/global_step", step_sync=True)

#         assert isinstance(self._experiment, (Run, RunDisabled))
#         return self._experiment

#     def watch(self, model: nn.Module, log: str = "gradients", log_freq: int = 100, log_graph: bool = True) -> None:
#         self.experiment.watch(model, log=log, log_freq=log_freq, log_graph=log_graph)

#     @rank_zero_only
#     def log_hyperparams(self, params: Union[Dict[str, Any], Namespace]) -> None:
#         params = _convert_params(params)
#         params = _sanitize_callable_params(params)
#         self.experiment.config.update(params, allow_val_change=True)

#     @rank_zero_only
#     def log_metrics(self, metrics: Mapping[str, float], step: Optional[int] = None) -> None:
#         assert rank_zero_only.rank == 0, "experiment tried to log from global_rank != 0"

#         metrics = _add_prefix(metrics, self._prefix, self.LOGGER_JOIN_CHAR)
#         if step is not None:
#             self.experiment.log(dict(metrics, **{"step": step}))
#         else:
#             self.experiment.log(metrics)

#     @rank_zero_only
#     def log_table(
#         self,
#         key: str,
#         columns: Optional[List[str]] = None,
#         data: Optional[List[List[Any]]] = None,
#         dataframe: Any = None,
#         step: Optional[int] = None,
#     ) -> None:
#         """Log a Table containing any object type (text, image, audio, video, molecule, html, etc).

#         Can be defined either with `columns` and `data` or with `dataframe`.
#         """

#         metrics = {key: wandb.Table(columns=columns, data=data, dataframe=dataframe)}
#         self.log_metrics(metrics, step)

#     @rank_zero_only
#     def log_text(
#         self,
#         key: str,
#         columns: Optional[List[str]] = None,
#         data: Optional[List[List[str]]] = None,
#         dataframe: Any = None,
#         step: Optional[int] = None,
#     ) -> None:
#         """Log text as a Table.

#         Can be defined either with `columns` and `data` or with `dataframe`.
#         """

#         self.log_table(key, columns, data, dataframe, step)

#     @rank_zero_only
#     def finalize(self, status: str) -> None:
#         ...
