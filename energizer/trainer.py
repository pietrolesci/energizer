import logging
from typing import Optional, Union

import torch
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning import Trainer as Trainer_pl
from pytorch_lightning.loops import EvaluationLoop, FitLoop, PredictionLoop, TrainingEpochLoop
from pytorch_lightning.trainer.states import TrainerFn, TrainerStatus
from pytorch_lightning.trainer.trainer import _determine_batch_limits
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.seed import isolate_rng
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from energizer.data.datamodule import ActiveDataModule
from energizer.loops.active_learning_loop import ActiveLearningLoop
from energizer.loops.pool_loop import PoolEvaluationLoop
from energizer.mixin.base import Learner
from energizer.utilities.connectors import DataConnector, PoolRunningStage
from energizer.utilities.trainer_utils import patch_callbacks

log = logging.getLogger(__name__)


class Trainer(Trainer_pl):
    def __init__(
        self,
        query_size: int = 2,
        reset_weights: Optional[bool] = True,
        test_after_labelling: Optional[bool] = False,
        total_budget: Optional[int] = -1,
        min_labelling_epochs: Optional[int] = 0,
        max_labelling_epochs: Optional[int] = -1,
        limit_pool_batches: Optional[Union[int, float]] = None,
        **kwargs,
    ) -> None:

        # initialize lightning trainer
        super().__init__(**kwargs)

        # register inputs
        self.query_size = query_size
        self.reset_weights = reset_weights
        self.test_after_labelling = test_after_labelling
        self.total_budget = total_budget
        self.min_labelling_epochs = min_labelling_epochs
        self.max_labelling_epochs = max_labelling_epochs
        self.limit_pool_batches = limit_pool_batches

        # used to decide which `_run_train()` method to run`
        self.active_fitting: bool = False

        # overwrite data_connector in trainer
        self._data_connector = DataConnector(self, self._data_connector.multiple_trainloader_mode)
        self._data_connector.on_trainer_init(
            val_check_interval=self.val_check_interval,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            reload_dataloaders_every_n_epochs=self.reload_dataloaders_every_n_epochs,
        )

        # pool loop
        pool_loop = PoolEvaluationLoop(self.query_size, verbose=False)

        # fit after each labelling session
        fit_loop = FitLoop(min_epochs=self.fit_loop.min_epochs, max_epochs=self.fit_loop.max_epochs)
        training_epoch_loop = TrainingEpochLoop(min_steps=self.fit_loop.min_steps, max_steps=self.fit_loop.max_steps)
        fit_loop.connect(epoch_loop=training_epoch_loop)

        # test after labelling and fitting
        test_loop = EvaluationLoop(verbose=False)

        # main active learning loop
        active_learning_loop = ActiveLearningLoop(
            reset_weights=self.reset_weights,
            total_budget=self.total_budget,
            test_after_labelling=self.test_after_labelling,
            min_labelling_epochs=self.min_labelling_epochs,
            max_labelling_epochs=self.max_labelling_epochs,
        )
        active_learning_loop.connect(pool_loop=pool_loop, fit_loop=fit_loop, test_loop=test_loop)
        self.active_learning_loop = active_learning_loop  # also attaches the trainer

        # this needed to be patched
        self._extend_setup_on_init()
        self._extend_init_debugging_flags()

    """
    Add states, properties, and methods
    """

    @property
    def active_fitting(self) -> bool:
        return self._active_fitting

    @active_fitting.setter
    def active_fitting(self, active_fitting: bool) -> None:
        self._active_fitting = active_fitting

    @property
    def active_learning_loop(self) -> FitLoop:
        return self._active_learning_loop

    @active_learning_loop.setter
    def active_learning_loop(self, loop: ActiveLearningLoop):
        """Attach a custom active_learning_loop to this Trainer."""
        loop.trainer = self
        self._active_learning_loop = loop

    def active_fit(
        self,
        model: Learner,
        train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        test_dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional[LightningDataModule] = None,
        ckpt_path: Optional[str] = None,
    ) -> None:
        self.strategy.model = model
        self._call_and_handle_interrupt(
            self._active_fit_impl, model, train_dataloaders, val_dataloaders, test_dataloaders, datamodule, ckpt_path
        )

    def _active_fit_impl(
        self,
        model: Learner,
        train_dataloaders: Optional[Union[TRAIN_DATALOADERS, LightningDataModule]] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        test_dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional[LightningDataModule] = None,
        ckpt_path: Optional[str] = None,
    ) -> None:

        # patch progress bar and add pool-related hooks
        _old_callbacks = self.callbacks
        self.callbacks = patch_callbacks(self.callbacks)

        # set states as in the original `_fit_impl`
        self.active_fitting = True
        # TODO: some of these are duplicated in the `reset_{fitting, test}` methods in `ActiveLearningLoop`
        Trainer_pl._log_api_event("fit")
        log.detail(f"{self.__class__.__name__}: trainer active_fit stage")
        self.state.fn = TrainerFn.FITTING
        self.state.status = TrainerStatus.RUNNING
        self.training = True
        self._last_train_dl_reload_epoch = float("-inf")
        self._last_val_dl_reload_epoch = float("-inf")
        self._last_test_dl_reload_epoch = float("-inf")

        # check inputs
        if not isinstance(model, Learner):
            raise MisconfigurationException("model must be a `Learner` not a `LightningModule`.")

        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(train_dataloaders, LightningDataModule):
            datamodule = train_dataloaders
            train_dataloaders = None
        is_dataloaders = train_dataloaders is not None or val_dataloaders is not None or test_dataloaders is not None
        is_datamodule = datamodule is not None

        # if you supply a datamodule you can't supply train_dataloader or val_dataloaders
        if is_dataloaders and is_datamodule:
            raise MisconfigurationException(
                "You cannot pass `train_dataloader` or `val_dataloaders` or ",
                "`test_dataloaders` to `trainer.active_fit(datamodule=...)`",
            )

        elif is_dataloaders or (is_datamodule and not isinstance(datamodule, ActiveDataModule)):
            datamodule = ActiveDataModule(
                train_dataloader=train_dataloaders,
                val_dataloaders=val_dataloaders,
                test_dataloaders=test_dataloaders,
                datamodule=datamodule,
            )

        if self.test_after_labelling and getattr(datamodule, "test_dataloader", None) is None:
            raise MisconfigurationException(
                "You specified `test_after_labelling=True` but no test_dataloader was provided."
            )

        # links data to the trainer
        self._data_connector.attach_data(model, datamodule=datamodule)

        # TODO: ckpt_path only in v2.0
        ckpt_path = ckpt_path or self.resume_from_checkpoint
        self._ckpt_path = self.__set_ckpt_path(
            ckpt_path, model_provided=True, model_connected=self.lightning_module is not None
        )

        results = self._run(model, ckpt_path=self.ckpt_path)

        assert self.state.stopped
        self.training = False
        self.active_fitting = False

        # reset original callbacks
        self.callbacks = _old_callbacks

        return results

    def reset_pool_dataloader(self, model: Optional[LightningModule] = None) -> None:
        """Resets the pool dataloader and determines the number of batches.

        This method is exactly the same as `trainer.reset_test_dataloader`.

        Args:
            model: The ``LightningModule`` if called outside of the trainer scope.
        """
        # source = self._data_connector._pool_dataloader_source
        pl_module = self.lightning_module or model
        has_step = is_overridden("pool_step", pl_module, Learner)
        enable_pool = self.limit_pool_batches > 0
        if has_step and enable_pool:
            self.num_pool_batches, self.pool_dataloaders = self._data_connector._reset_eval_dataloader(
                PoolRunningStage.POOL, model=pl_module
            )

    """
    Extend methods that need to be called in the `__init__`
    """

    def _extend_init_debugging_flags(self) -> None:
        """This method extends the original `trainer._init_debugging_flags` method."""
        if self.fast_dev_run:
            num_batches = int(self.fast_dev_run)
            self.limit_pool_batches = num_batches
            self.fit_loop.max_steps = num_batches
            self.fit_loop.max_epochs = 1

        self.limit_pool_batches = _determine_batch_limits(self.limit_pool_batches, "limit_pool_batches")

    def _extend_setup_on_init(self) -> None:
        """This method extends the original `trainer._setup_on_init` method."""
        self.num_pool_batches = []
        self.pool_dataloaders = None

    """
    Patch `_run_train` implementation
    """

    def _run_train(self) -> None:
        """Method that depending on `self.active_fitting` selects which loop to run.

        If `self.active_fitting is False` it runs the usual `fit_loop`, otherwise
        runs the `active_learning_loop`.
        """
        if not self.active_fitting:
            # run normal fit_loop
            return super()._run_train()

        # NOTE: this exactly mimics what they do in `super()._run_train()`
        self._pre_training_routine()

        with isolate_rng():
            self._run_sanity_check()

        # enable train mode
        self.model.train()
        torch.set_grad_enabled(True)

        # TODO: apparently here it's too late to attach a trainer
        self.active_learning_loop.trainer = self
        with torch.autograd.set_detect_anomaly(self._detect_anomaly):
            self.active_learning_loop.run()

    """
    Dispatch properties calls to the right `FitLoop` depending on whether
    we are `active_fitting` or simply training
    """

    @property
    def global_step(self) -> int:
        """The number of optimizer steps taken (does not reset each epoch).

        This includes multiple optimizers and TBPTT steps (if enabled).
        """
        if self.active_fitting:
            return self.active_learning_loop.fit_loop.epoch_loop.global_step
        return self.fit_loop.epoch_loop.global_step

    @property
    def current_epoch(self) -> int:
        """The current epoch, updated after the epoch end hooks are run."""
        if self.active_fitting:
            return self.active_learning_loop.fit_loop.epoch_progress.current.completed
        return self.fit_loop.epoch_progress.current.completed

    @property
    def max_epochs(self) -> int:
        if self.active_fitting:
            return self.active_learning_loop.fit_loop.max_epochs
        return self.fit_loop.max_epochs

    @property
    def min_epochs(self) -> int:
        if self.active_fitting:
            return self.active_learning_loop.fit_loop.min_epochs
        return self.fit_loop.min_epochs

    @property
    def max_steps(self) -> int:
        if self.active_fitting:
            return self.active_learning_loop.fit_loop.max_steps
        return self.fit_loop.max_steps

    @property
    def min_steps(self) -> Optional[int]:
        if self.active_fitting:
            return self.active_learning_loop.fit_loop.min_steps
        return self.fit_loop.min_steps

    @property
    def is_last_batch(self) -> bool:
        """Whether trainer is executing the last batch."""
        if self.active_fitting:
            return self.active_learning_loop.fit_loop.epoch_loop.batch_progress.is_last_batch
        return self.fit_loop.epoch_loop.batch_progress.is_last_batch

    @property
    def _evaluation_loop(self) -> EvaluationLoop:

        if self.state.fn in (TrainerFn.FITTING, TrainerFn.TUNING):
            return (
                self.fit_loop.epoch_loop.val_loop
                if not self.active_fitting
                else self.active_learning_loop.fit_loop.epoch_loop.val_loop
            )

        if self.state.fn == TrainerFn.VALIDATING:
            return (
                self.validate_loop
                if not self.active_fitting
                else self.active_learning_loop.fit_loop.epoch_loop.val_loop
            )

        if self.state.fn == TrainerFn.TESTING:
            return self.test_loop if not self.active_fitting else self.active_learning_loop.test_loop

        raise RuntimeError("The `Trainer._evaluation_loop` property isn't defined. Accessed outside of scope")

    @property
    def _active_loop(self) -> Optional[Union[FitLoop, EvaluationLoop, PredictionLoop]]:
        """Returns the currently active loop based on the `Trainer`'s state."""

        if self.training:
            return self.fit_loop if not self.active_fitting else self.active_learning_loop

        if self.sanity_checking or self.evaluating:
            # this resolves the `active_fitting` condition internally
            return self._evaluation_loop

        if self.predicting:
            return self.predict_loop

    # @property
    # def _results(self):
    #     active_loop = self._active_loop
    #     print(self.state.fn, self.active_fitting, self.sanity_checking)
    #     print("HERE", active_loop)
    #     print("HERE", active_loop.trainer)
    #     if active_loop is not None:
    #         return active_loop._results
