from copy import deepcopy
from typing import List, Optional

import torch
from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.loops.loop import Loop
from pytorch_lightning.loops.utilities import _is_max_limit_reached
from pytorch_lightning.trainer.connectors.logger_connector.result import _ResultCollection
from pytorch_lightning.trainer.progress import Progress
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.distributed import rank_zero_info, rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from energizer.loops.pool_loop import PoolEvaluationLoop


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
        self.min_labelling_epochs = min_labelling_epochs
        self.max_labelling_epochs = max_labelling_epochs
        self.reset_weights = reset_weights
        self.total_budget = total_budget
        self.test_after_labelling = test_after_labelling

        # labelling iterations tracker
        self.epoch_progress = Progress()

        # pool evaluation loop
        self.pool_loop: Optional[PoolEvaluationLoop] = None

        # fit after each labelling session
        self.fit_loop: Optional[FitLoop] = None

        # test after labelling and fitting
        self.test_loop: Optional[EvaluationLoop] = None

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
        # `processed` is increased before `on_train_epoch_end`, the hook where checkpoints are typically saved.
        # we use it here because the checkpoint data won't have `completed` increased yet
        has_labelled_data = (
            not self.trainer.datamodule.has_unlabelled_data
            or self.trainer.query_size > self.trainer.datamodule.pool_size
        )
        stop_total_budget = _is_max_limit_reached(self.trainer.datamodule.labelled_size, self.total_budget)
        stop_epochs = _is_max_limit_reached(self.epoch_progress.current.processed, self.max_labelling_epochs)
        if stop_epochs:
            # in case they are not equal, override so `trainer.current_epoch` has the expected value
            self.epoch_progress.current.completed = self.epoch_progress.current.processed

        self.trainer.should_stop = False

        return stop_epochs or has_labelled_data or stop_total_budget

    @property
    def skip(self) -> bool:
        return self.done

    def connect(self, pool_loop: PoolEvaluationLoop, fit_loop: FitLoop, test_loop: EvaluationLoop) -> None:
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

        # make a copy of the initial state of the model to reset the weights
        if self.reset_weights:
            self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

        self._results.to(device=self.trainer.lightning_module.device)

        self.trainer._call_callback_hooks("on_active_learning_start")
        self.trainer._call_lightning_module_hook("on_active_learning_start")

    def on_advance_start(self) -> None:

        # TODO: check if this is needed
        # reset outputs here instead of in `reset` as they are not accumulated between epochs
        self._outputs = []

        self.epoch_progress.increment_ready()

        # TODO: check this connector
        self.trainer._logger_connector.on_epoch_start()

        self.trainer._call_callback_hooks("on_labelling_epoch_start")
        self.trainer._call_lightning_module_hook("on_labelling_epoch_start")

        self.epoch_progress.increment_started()

    def advance(self) -> None:  # type: ignore[override]
        """Runs the active learning loop: training, testing, and pool evaluation."""
        if self.trainer.datamodule.has_labelled_data:
            self._reset_fitting()  # required to reset the tracking stage
            self.fit_loop.run()
            # reset counters so that epochs always starts from 0
            self.fit_loop.epoch_progress.reset()

            if self.test_after_labelling:
                self._reset_testing()  # required to reset the tracking stage
                self.test_loop.run()

        if self.trainer.datamodule.has_unlabelled_data:
            self._reset_evaluating_pool()  # requires to reset the tracking stage.
            _, indices = self.pool_loop.run()
            self.labelling_loop(indices)

    def on_advance_end(self) -> None:
        """Used to restore the original weights and the optimizers and schedulers states."""
        # TODO: check this connector
        self.trainer._logger_connector.epoch_end_reached()

        if self.reset_weights:
            self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)

        self.epoch_progress.increment_processed()

        self.trainer._call_callback_hooks("on_labelling_epoch_end")
        self.trainer._call_lightning_module_hook("on_labelling_epoch_end")

        # TODO: check this connector
        self.trainer._logger_connector.on_epoch_end()

        self.epoch_progress.increment_completed()

        # if fault tolerant is enabled and process has been notified, exit.
        self.trainer._exit_gracefully_on_signal()

    def on_run_end(self) -> None:
        # enforce training at the end of acquisitions
        self._reset_fitting()
        self.fit_loop.run()

        self.trainer._call_callback_hooks("on_active_learning_end")
        self.trainer._call_lightning_module_hook("on_active_learning_end")

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True
        self.trainer.lightning_module.train()
        torch.set_grad_enabled(True)

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True
        self.trainer.lightning_module.eval()
        torch.set_grad_enabled(False)

    def _reset_evaluating_pool(self) -> None:
        self.trainer.reset_pool_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True
        self.trainer.lightning_module.eval()
        torch.set_grad_enabled(False)

    @rank_zero_only
    def labelling_loop(self, indices: List[int]) -> None:
        """This method changes the state of the underlying dataset."""
        self.trainer.datamodule.label(indices)
