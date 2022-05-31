from copy import deepcopy
from typing import Any, List

from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.distributed import rank_zero_only


class ActiveLearningLoop(FitLoop):
    def __init__(
        self,
        reset_weights: bool,
        test_after_labelling: bool,
        total_budget: int,
        min_epochs: int,
        max_epochs: int,
    ):
        super().__init__(min_epochs=min_epochs, max_epochs=max_epochs)
        self.reset_weights = reset_weights
        self.total_budget = total_budget
        self.test_after_labelling = test_after_labelling

    @property
    def done(self) -> bool:
        """Check if we are done.

        The loop terminates when one of these conditions is satisfied:

        - The pool dataloader has no more unlabelled data

        - The total labelling budget has been reached

        - The maximum number of epochs have been run

        - The `query_size` is bigger than the available instance to label
        """
        return (
            not self.trainer.datamodule.has_unlabelled_data
            or (self.total_budget > 0 and self.trainer.datamodule.labelled_size >= self.total_budget)
            or self.progress.current.completed >= self.max_epochs
            or self.trainer.lightning_module.query_size > self.trainer.datamodule.pool_size
        )

    def reset(self) -> None:
        """Nothing to reset in this loop."""
        pass

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Store the original weights of the model."""

        # split train dataset in train and pool folds
        self.trainer.datamodule.setup_folds()

        # make a copy of the initial state of the model to reset the weights
        if self.reset_weights:
            self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting, testing, and pool evaluation."""
        if self.trainer.datamodule.has_labelled_data:
            self._reset_fitting()  # requires to reset the tracking stage.
            self.trainer.active_fit_loop.run()

            if self.test_after_labelling:
                self._reset_testing(is_on_pool=False)  # requires to reset the tracking stage.
                self.trainer.active_test_loop.run()

        if self.trainer.datamodule.has_unlabelled_data:
            self._reset_testing(is_on_pool=True)  # requires to reset the tracking stage.
            self.trainer.pool_loop.run()
            indices = self.trainer.lightning_module.accumulation_metric.compute().tolist()
            self.labelling_loop(indices)

    def on_advance_end(self) -> None:
        """Used to restore the original weights and the optimizers and schedulers states."""
        if self.reset_weights:
            self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
            self.trainer.accelerator.setup(self.trainer)  # TODO: do I need this?

    def on_run_end(self):
        # make sure we are not on pool when active_fit ends
        self.trainer.lightning_module.is_on_pool = False

        # enforce training at the end of acquisitions
        self._reset_fitting()
        self.fit_loop.run()

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.current_epoch = 0
        self.global_step = 0
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self, is_on_pool: bool = False) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.datamodule.is_on_pool = is_on_pool
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    @rank_zero_only
    def labelling_loop(self, indices: List[int]):
        self.trainer.datamodule.label(indices)
