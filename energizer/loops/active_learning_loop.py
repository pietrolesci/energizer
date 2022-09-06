from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import yaml
from pandas import DataFrame
from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.loops.loop import Loop
from pytorch_lightning.loops.utilities import _is_max_limit_reached
from pytorch_lightning.trainer.connectors.logger_connector.result import _ResultCollection
from pytorch_lightning.trainer.progress import Progress
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.apply_func import apply_to_collection
from pytorch_lightning.utilities.types import _EVALUATE_OUTPUT

from energizer.utilities.logger import logger
from energizer.utilities.types import tensor_to_python


@dataclass
class LabellingIterOutputs:
    labelling_iter: Optional[int] = None
    indices: Optional[List[int]] = field(default_factory=list)
    test_outputs: Optional[_EVALUATE_OUTPUT] = field(default_factory=list)
    pool_outputs: Optional[_EVALUATE_OUTPUT] = field(default_factory=list)
    data_stats: Optional[Dict[str, int]] = field(default_factory=dict)


class LabellingOutputsList(list):
    def to_pandas(self) -> DataFrame:
        return DataFrame(
            data=[(out.data_stats["train_size"], *out.test_outputs[0].values()) for out in self],
            columns=("train_size", *self[0].test_outputs[0].keys()),
        )

    def __repr__(self) -> str:
        tab = "    "
        return f"{self.__class__.__name__}(\n{tab}" + f",\n{tab}".join(map(str, self)) + ",\n)"

    def __str__(self) -> str:
        return self.__repr__()

    def append(self, outputs: LabellingIterOutputs) -> None:
        outputs = self._prepare_outputs(outputs)
        return super().append(outputs)

    def _prepare_outputs(self, outputs: LabellingIterOutputs) -> LabellingIterOutputs:
        args = (torch.Tensor, tensor_to_python, "cpu")
        outputs.test_outputs = apply_to_collection(outputs.test_outputs, *args)
        outputs.pool_outputs = apply_to_collection(outputs.pool_outputs, *args)
        return outputs


class ActiveLearningLoop(Loop):
    def __init__(
        self,
        reset_weights: bool,
        test_after_labelling: bool,
        total_budget: int,
        min_labelling_epochs: int,
        max_labelling_epochs: int,
    ):
        super().__init__()
        self.reset_weights = reset_weights
        self.test_after_labelling = test_after_labelling
        self.total_budget = total_budget
        self.min_labelling_epochs = min_labelling_epochs
        self.max_labelling_epochs = max_labelling_epochs

        # labelling iterations tracker
        self.epoch_progress = Progress()

        # pool evaluation loop
        self.pool_loop: Optional[Loop] = None

        # fit after each labelling session
        self.fit_loop: Optional[FitLoop] = None

        # test after labelling and fitting
        self.test_loop: Optional[EvaluationLoop] = None

        self._outputs: LabellingOutputsList = LabellingOutputsList()

    @property
    def can_run_testing(self) -> bool:
        return self.test_after_labelling and self.trainer.datamodule.test_dataloader() is not None

    @property
    def _results(self) -> _ResultCollection:
        if self.trainer.training:
            return self.fit_loop.epoch_loop._results

        if self.trainer.validating:
            return self.fit_loop.epoch_loop.val_loop._results

        if self.trainer.testing:
            return self.test_loop._results

        raise RuntimeError("`ActiveLearningLoop._results` property isn't defined. Accessed outside of scope")

    @property
    def done(self) -> bool:
        """Evaluates when to leave the loop."""
        # TODO: add stopping condition based on satisfactory level of a
        # performance metric. Potentially do this via a callback

        # stop if no more data to label are available
        stop_no_data_to_label = not (
            self.trainer.datamodule.has_unlabelled_data and self.trainer.query_size <= self.trainer.datamodule.pool_size
        )

        # stop if total budget is labelled
        stop_total_budget = _is_max_limit_reached(self.trainer.datamodule.train_size, self.total_budget)

        stop_epochs = _is_max_limit_reached(self.epoch_progress.current.processed, self.max_labelling_epochs)
        if stop_epochs:
            # in case they are not equal, override so `trainer.current_epoch` has the expected value
            self.epoch_progress.current.completed = self.epoch_progress.current.processed

        return stop_no_data_to_label or stop_total_budget or stop_epochs

    @property
    def skip(self) -> bool:
        return self.done

    def connect(self, pool_loop: Loop, fit_loop: FitLoop, test_loop: EvaluationLoop) -> None:
        self.pool_loop = pool_loop
        self.fit_loop = fit_loop
        self.test_loop = test_loop

    def reset(self) -> None:
        """Resets the internal state of this loop."""
        # this is exactly as in the `FitLoop`
        if self.restarting:
            self.epoch_progress.reset_on_restart()

    def on_run_start(self) -> None:
        """Store the original weights of the model."""
        self.epoch_progress.current.completed = self.epoch_progress.current.processed

        # make a copy of the initial state of the underlying model
        # in `trainer.active_fit` the underlying model is passed,
        # ie `self._run(model.model, ...)``
        if self.reset_weights:
            self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

        # set query_strategy as lightning_module to access hooks
        self.trainer.set_lightning_module(use_query_strategy=True)

        self._results.to(device=self.trainer.lightning_module.device)
        self.trainer._call_callback_hooks("on_active_learning_start")
        self.trainer._call_lightning_module_hook("on_active_learning_start")

    def on_advance_start(self) -> None:
        self.epoch_progress.increment_ready()

        # TODO: check this connector
        self.trainer._logger_connector.on_epoch_start()

        self.trainer._call_callback_hooks("on_labelling_epoch_start")
        self.trainer._call_lightning_module_hook("on_labelling_epoch_start")

        self.epoch_progress.increment_started()

        self._print_separator_line()

    def advance(self) -> None:  # type: ignore[override]
        """Runs the active learning loop: training, testing, and pool evaluation."""
        # TODO: add `with self.trainer.profiler.profile("<some_step>"):`

        outputs = LabellingIterOutputs(
            labelling_iter=self.epoch_progress.current.completed,
            data_stats=self.trainer.datamodule.get_statistics(),
        )

        if self.trainer.datamodule.has_labelled_data:
            self._reset_fitting()
            self.fit_loop.run()

        if self.can_run_testing:
            # TODO: check that test_after_labelling and test_dl is present
            # TODO: check it's using the last checkpoint
            self._reset_testing()
            outputs.test_outputs = self.test_loop.run()
            self.test_loop._print_results(outputs.test_outputs, "test")

        if self.trainer.datamodule.has_unlabelled_data:
            self._reset_evaluating_pool()
            outputs.pool_outputs, indices = self.pool_loop.run()
            outputs.indices = indices
            logger.info(f"Queried {len(indices)} instance")
            self.label_datamodule(indices)

        self._outputs.append(outputs)

    def on_advance_end(self) -> None:
        """Used to restore the original weights and the optimizers and schedulers states."""
        # TODO: check this connector
        self.trainer._logger_connector.epoch_end_reached()

        self.epoch_progress.increment_processed()

        self.trainer._call_callback_hooks("on_labelling_epoch_end")
        self.trainer._call_lightning_module_hook("on_labelling_epoch_end")

        # TODO: check this connector
        self.trainer._logger_connector.on_epoch_end()

        self.epoch_progress.increment_completed()

        # if fault tolerant is enabled and process has been notified, exit.
        self.trainer._exit_gracefully_on_signal()

    def on_run_end(self) -> List[LabellingIterOutputs]:

        # enforce training at the end of acquisitions
        self._print_separator_line(is_last=True)
        outputs = LabellingIterOutputs(
            labelling_iter=self.epoch_progress.current.completed,
            data_stats=self.trainer.datamodule.get_statistics(),
        )
        self._reset_fitting()
        self.fit_loop.run()
        if self.can_run_testing:
            self._reset_testing()
            outputs.test_outputs = self.test_loop.run()
            self.test_loop._print_results(outputs.test_outputs, "test")

        self._outputs.append(outputs)
        outputs, self._outputs = self._outputs, []

        self.trainer.set_lightning_module(use_query_strategy=True)
        self.trainer._call_callback_hooks("on_active_learning_end")
        self.trainer._call_lightning_module_hook("on_active_learning_end")

        return outputs

    def _reset_fitting(self) -> None:
        # reset counters so that fit_loop is run again otherwise it does not run
        self.fit_loop.epoch_progress.reset()

        # puts the underlying module as "trainer.lightning_module"
        self.trainer.set_lightning_module(use_query_strategy=False)

        # maybe reset the weights
        if self.reset_weights:
            self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
            logger.debug(f"{self.trainer.lightning_module.__class__.__name__} " "state dict has been re-initialized")

        # resets the train/val dataloaders
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()

        # set up the states and enable grads
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True
        self.trainer.lightning_module.train()
        torch.set_grad_enabled(True)

    def _reset_testing(self) -> None:
        self.trainer.set_lightning_module(use_query_strategy=False)
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True
        self.trainer.lightning_module.eval()
        torch.set_grad_enabled(False)

    def _reset_evaluating_pool(self) -> None:
        self.trainer.set_lightning_module(use_query_strategy=True)
        self.trainer.reset_pool_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True
        self.trainer.lightning_module.eval()
        torch.set_grad_enabled(False)

    def label_datamodule(self, indices: List[int]) -> Dict[str, int]:
        """This method changes the state of the underlying dataset."""
        self.trainer.datamodule.label(indices)
        logger.info(f"Annotated {len(indices)} instances")
        logger.info("New data statistics\n" f"{yaml.dump(self.trainer.datamodule.get_statistics())}")

    """
    Helpers
    """

    def _print_separator_line(self, is_last: bool = False) -> None:
        if is_last:
            print("Last fit_loop".center(72, "-"))
            return
        print(f"Labelling Iteration {self.epoch_progress.current.completed}".center(72, "-"), flush=True)
