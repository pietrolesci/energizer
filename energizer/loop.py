from copy import copy, deepcopy
from typing import Any, List

from pytorch_lightning.loggers.base import LightningLoggerBase, LoggerCollection
from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.distributed import rank_zero_info, rank_zero_only
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from energizer.datamodule import ActiveDataModule


class ActiveLearningLoop(Loop):
    def __init__(
        self,
        n_epochs_between_labelling: int,
        reset_weights: bool,
        total_budget: int,
        test_after_labelling: bool,
        fit_loop: FitLoop,
    ):
        self.n_epochs_between_labelling = n_epochs_between_labelling
        self.reset_weights = reset_weights
        self.total_budget = total_budget
        self.test_after_labelling = test_after_labelling
        self.fit_loop = fit_loop

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
        if not isinstance(self.trainer.datamodule, ActiveDataModule):
            raise MisconfigurationException(
                "Expected the trainer to have an instance of the `ActiveDataModule` "
                f"not {type(self.trainer.datamodule)}. Active learning requires `ActiveDataModule`."
            )

        # split train dataset in train and pool folds
        self.trainer.datamodule.setup_folds()

        # make a copy of the initial state of the model to reset the weights
        if self.reset_weights:
            self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())

        # FitLoop set the total number of iters the active learning loop must run
        if self.total_budget > 0:
            self.max_epochs = self.total_budget * self.n_epochs_between_labelling
        else:
            self.max_epochs = len(self.trainer.datamodule.pool_fold)

        # set the total number of epochs of the fit loop to run within one active learning iter
        if self.fit_loop.max_epochs != self.n_epochs_between_labelling:
            rank_zero_info(
                f"You specified `max_epochs={self.fit_loop.max_epochs}` in the `Trainer`"
                f"but `n_epochs_between_labelling={self.n_epochs_between_labelling}`."
                f"Therefore the `fit_loop`, within one active learning loop, will run {self.n_epochs_between_labelling} times."
            )
        self.fit_loop.max_epochs = self.n_epochs_between_labelling

        # attach the loop for pool
        # TODO: is it the correct way to instantiate the pool_loop?
        self.pool_loop = copy(self.trainer.test_loop)
        self.pool_loop.verbose = False
        self.trainer.test_loop.verbose = False
        self.trainer.validate_loop.verbose = False

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        pass

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting, testing, and pool evaluation."""
        if self.trainer.datamodule.has_labelled_data:
            self._reset_fitting()  # requires to reset the tracking stage.
            self.fit_loop.run()

            if self.test_after_labelling:
                print("Running test")
                self._reset_testing(is_on_pool=False)  # requires to reset the tracking stage.
                self.trainer.test_loop.run()

        if self.trainer.datamodule.has_unlabelled_data:
            self._reset_testing(is_on_pool=True)  # requires to reset the tracking stage.
            self.pool_loop.run()
            indices = self.trainer.lightning_module.accumulation_metric.compute().tolist()
            self.labelling_loop(indices)

    def on_advance_end(self) -> None:
        """Used to restore the original weights and the optimizers and schedulers states."""
        if self.reset_weights:
            self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
            self.trainer.accelerator.setup(self.trainer)

    def on_run_end(self):
        # make sure we are not on pool when active_fit ends
        self.trainer.lightning_module.is_on_pool = False

        # enforce training at the end of acquisitions
        self._reset_fitting()
        self.fit_loop.run()

        # TODO: need to reset properly - reset normal fit_loop
        self._reset_fitting()
        self.trainer.fit_loop = self.trainer.fit_loop.fit_loop
        # self.epoch_loop.global_step = 0
        # self.epoch_progress.current.processed = 0

    def _reset_fitting(self) -> None:
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.current_epoch = 0
        self.global_step = 0
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self, is_on_pool: bool = False) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.lightning_module.is_on_pool = is_on_pool
        self.trainer.datamodule.is_on_pool = is_on_pool
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        """Calls attributes not present in the `ActiveLearningLoop` from the underlying `FitLoop`."""
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    @rank_zero_only
    def labelling_loop(self, indices: List[int]):
        self.trainer.datamodule.label(indices)
