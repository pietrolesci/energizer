import logging
from typing import List, Optional, Union

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.progress.base import ProgressBarBase
from pytorch_lightning.loops import EvaluationLoop, FitLoop, PredictionLoop, TrainingEpochLoop
from pytorch_lightning.trainer.connectors.data_connector import DataConnector as DataConnector_pl
from pytorch_lightning.trainer.connectors.data_connector import _DataLoaderSource
from pytorch_lightning.trainer.states import TrainerFn, TrainerStatus
from pytorch_lightning.trainer.trainer import _determine_batch_limits
from pytorch_lightning.utilities import LightningEnum
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.seed import isolate_rng
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from energizer.callbacks.tqdm_progress import TQDMProgressBarActiveLearning
from energizer.data.datamodule import ActiveDataModule
from energizer.loops.active_learning_loop import ActiveLearningLoop, LabellingIterOutputs
from energizer.mixin.hooks import CallBackActiveLearningHooks
from energizer.query_strategies.base import BaseQueryStrategy
from energizer.utilities.logger import logger

"""
Preliminary components needed to add support for `pool_dataloader`
and pool-related hooks
"""


class PoolRunningStage(LightningEnum):
    POOL = "pool"

    @property
    def evaluating(self) -> bool:
        return True

    @property
    def dataloader_prefix(self) -> Optional[str]:
        return self.value


class DataConnector(DataConnector_pl):
    def __init__(self, trainer: "pl.Trainer", multiple_trainloader_mode: str = "max_size_cycle"):
        super().__init__(trainer, multiple_trainloader_mode)
        self._pool_dataloader_source = _DataLoaderSource(None, "")

    def attach_datamodule(
        self, model: "pl.LightningModule", datamodule: Optional["pl.LightningDataModule"] = None
    ) -> None:
        # do the usual
        super().attach_datamodule(model, datamodule)

        # and attach the pool dataloader if the user has passed an ActiveDataModule
        if isinstance(datamodule, ActiveDataModule):
            self._pool_dataloader_source = _DataLoaderSource(datamodule, "pool_dataloader")


def patch_callbacks(callbacks: List[Callback]) -> List[Callback]:
    def add_pool_hooks(callback: Callback) -> Callback:
        hook_names = [m for m in dir(CallBackActiveLearningHooks) if not m.startswith("_")]
        for name in hook_names:
            if not hasattr(callback, name):
                setattr(callback, name, getattr(CallBackActiveLearningHooks, name))
        return callback

    new_callbacks = []
    for c in callbacks:
        if isinstance(c, ProgressBarBase):
            prog_bar = TQDMProgressBarActiveLearning(process_position=c.process_position, refresh_rate=c.refresh_rate)
            prog_bar = add_pool_hooks(prog_bar)
            new_callbacks.append(prog_bar)
        else:
            new_callbacks.append(add_pool_hooks(c))

    return new_callbacks


"""
New trainer implementation
"""


class Trainer(pl.Trainer):
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
        active_learning_loop.connect(pool_loop=None, fit_loop=fit_loop, test_loop=test_loop)
        self.active_learning_loop = active_learning_loop  # also attaches the trainer

        # this needed to be patched
        self._extend_setup_on_init()
        self._extend_init_debugging_flags()

    """
    Properties
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

    @property
    def query_strategy(self) -> BaseQueryStrategy:
        return self._query_strategy

    @query_strategy.setter
    def query_strategy(self, model: BaseQueryStrategy):
        """Attach a custom query_strategy to this Trainer."""
        self._query_strategy = model

    """
    New `active_fit` method
    """

    def active_fit(
        self,
        model: BaseQueryStrategy,
        train_dataloaders: Optional[Union[TRAIN_DATALOADERS, "pl.LightningDataModule"]] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        test_dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional["pl.LightningDataModule"] = None,
        ckpt_path: Optional[str] = None,
    ) -> List[LabellingIterOutputs]:
        # set up model reference
        self.query_strategy = model
        self.query_strategy.query_size = self.query_size
        self.active_learning_loop.pool_loop = self.query_strategy.pool_loop

        """
        NOTE: this creates the `self.lightning_module` attribute on the trainer
        it is set in the `fit` method, but we do not set it here and let the
        loop components do it        
        """
        # self.strategy.model = model

        return self._call_and_handle_interrupt(
            self._active_fit_impl,
            self.query_strategy,
            train_dataloaders,
            val_dataloaders,
            test_dataloaders,
            datamodule,
            ckpt_path,
        )

    def _active_fit_impl(
        self,
        model: BaseQueryStrategy,
        train_dataloaders: Optional[Union[TRAIN_DATALOADERS, "pl.LightningDataModule"]] = None,
        val_dataloaders: Optional[EVAL_DATALOADERS] = None,
        test_dataloaders: Optional[EVAL_DATALOADERS] = None,
        datamodule: Optional["pl.LightningDataModule"] = None,
        ckpt_path: Optional[str] = None,
    ) -> None:

        # patch progress bar and add pool-related hooks
        _old_callbacks = self.callbacks
        self.callbacks = patch_callbacks(self.callbacks)

        # set states as in the original `_fit_impl`
        self.active_fitting = True
        # TODO: some of these are duplicated in the `reset_{fitting, test}` methods in `ActiveLearningLoop`
        pl.Trainer._log_api_event("fit")
        logger.detail(f"{self.__class__.__name__}: trainer active_fit stage")
        self.state.fn = TrainerFn.FITTING
        self.state.status = TrainerStatus.RUNNING
        self.training = True
        self._last_train_dl_reload_epoch = float("-inf")
        self._last_val_dl_reload_epoch = float("-inf")
        self._last_test_dl_reload_epoch = float("-inf")

        # # check inputs
        # if not isinstance(model, BaseQueryStrategy):
        #     raise MisconfigurationException("model must be a `BaseQueryStrategy` not a `LightningModule`.")

        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(train_dataloaders, pl.LightningDataModule):
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

        # NOTE: pass the underlying `LightningModule` to `_run` so that it finds
        # the `training_step` method etc
        results = self._run(model.model, ckpt_path=self.ckpt_path)

        assert self.state.stopped
        self.training = False
        self.active_fitting = False

        # reset original callbacks
        self.callbacks = _old_callbacks

        return results

    def reset_pool_dataloader(self, model: Optional["pl.LightningModule"] = None) -> None:
        """Resets the pool dataloader and determines the number of batches.

        This method is exactly the same as `trainer.reset_test_dataloader`.

        Args:
            model: The ``LightningModule`` if called outside of the trainer scope.
        """
        # source = self._data_connector._pool_dataloader_source
        pl_module = self.lightning_module or model
        # has_step = is_overridden("pool_step", pl_module, BaseQueryStrategy)
        enable_pool = self.limit_pool_batches > 0
        # if has_step and enable_pool:
        if enable_pool:
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

    def _run_train(self) -> List[LabellingIterOutputs]:
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

        self.active_learning_loop.trainer = self
        with torch.autograd.set_detect_anomaly(self._detect_anomaly):
            results = self.active_learning_loop.run()

        return results

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

    def set_lightning_module(self, use_query_strategy: bool = False) -> None:
        if use_query_strategy:
            # self.strategy.model = self.query_strategy
            self.strategy.connect(self.query_strategy)
            logger.info(f"Using `{self.lightning_module.__class__.__name__}`")
        else:
            # self.strategy.model = self.query_strategy.model
            self.strategy.connect(self.query_strategy.model)
            logger.info(f"Using underlying `{self.lightning_module.__class__.__name__}`")
        self.strategy.model_to_device()

    @property
    def using_query_strategy_as_lightning_module(self) -> bool:
        return isinstance(self.lightning_module, BaseQueryStrategy)

    # @property
    # def _results(self):
    #     active_loop = self._active_loop
    #     print(self.state.fn, self.active_fitting, self.sanity_checking)
    #     print("HERE", active_loop)
    #     print("HERE", active_loop.trainer)
    #     if active_loop is not None:
    #         return active_loop._results
