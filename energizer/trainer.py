import logging
from typing import Optional, Union

import torch
from pytorch_lightning import LightningDataModule
from pytorch_lightning import Trainer as Trainer_pl
from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.loops.epoch.training_epoch_loop import TrainingEpochLoop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn, TrainerStatus
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.seed import isolate_rng
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from energizer.data.datamodule import ActiveDataModule
from energizer.learners.base import Learner
from energizer.loops.active_loop import ActiveLearningLoop
from energizer.loops.pool_loop import PoolEvaluationLoop
from energizer.utilities.trainer import patch_callbacks

log = logging.getLogger(__name__)


class Trainer(Trainer_pl):
    def __init__(
        self,
        query_size: int = 2,
        total_budget: int = -1,
        min_labelling_iters: int = 0,
        max_labelling_iters: int = 1000,
        reset_weights: bool = True,
        test_after_labelling: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.query_size = query_size
        self.total_budget = total_budget
        self.reset_weights = reset_weights
        self.min_labelling_iters = min_labelling_iters
        self.max_labelling_iters = max_labelling_iters
        self.test_after_labelling = test_after_labelling

        self._active_fitting: bool = False

        # Intialize rest of the trainer
        self.callbacks = patch_callbacks(self.callbacks)

        # pool evaluation loop
        pool_loop = PoolEvaluationLoop(self.query_size)

        # fit after each labelling session
        active_fit_loop = FitLoop(min_epochs=self.min_epochs, max_epochs=self.max_epochs)
        training_epoch_loop = TrainingEpochLoop(min_steps=self.min_steps, max_steps=self.max_steps)
        active_fit_loop.connect(epoch_loop=training_epoch_loop)

        # test after labelling and fitting
        active_test_loop = EvaluationLoop(verbose=False)  # leave it here in case I need to pick args from init

        # root loop
        self.active_learning_loop = ActiveLearningLoop(
            reset_weights=self.reset_weights,
            total_budget=self.total_budget,
            test_after_labelling=self.test_after_labelling,
            min_labelling_iters=self.min_labelling_iters,
            max_labelling_iters=self.max_labelling_iters,
        )

        # connect with children loops
        self.active_learning_loop.connect(
            pool_loop=pool_loop, active_fit_loop=active_fit_loop, active_test_loop=active_test_loop
        )

    @property
    def active_fitting(self) -> bool:
        return self._active_fitting

    @active_fitting.setter
    def active_fitting(self, active_fitting: bool) -> None:
        self._active_fitting = active_fitting

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
        Trainer._log_api_event("fit")
        log.detail(f"{self.__class__.__name__}: trainer fit stage")
        self.active_fitting = True
        self.state.fn = TrainerFn.FITTING
        self.state.status = TrainerStatus.RUNNING
        self.training = True
        self._last_train_dl_reload_epoch = float("-inf")
        self._last_val_dl_reload_epoch = float("-inf")
        self._last_test_dl_reload_epoch = float("-inf")

        if not isinstance(model, Learner):
            raise MisconfigurationException("model must be a `Learner` not a `LightningModule`.")

        # if a datamodule comes in as the second arg, then fix it for the user
        if isinstance(train_dataloaders, LightningDataModule):
            datamodule = train_dataloaders
            train_dataloaders = None

        is_dataloaders = train_dataloaders is not None or val_dataloaders is not None or test_dataloaders is not None
        is_datamodule = datamodule is not None

        # If you supply a datamodule you can't supply train_dataloader or val_dataloaders
        if is_dataloaders and is_datamodule:
            raise MisconfigurationException(
                "You cannot pass `train_dataloader` or `val_dataloaders` or `test_dataloaders` to `trainer.active_fit(datamodule=...)`"
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

        return results

    def _run_train(self) -> None:
        if not self.active_fitting:
            # run normal fit_loop
            return super()._run_train()

        self._pre_training_routine()

        with isolate_rng():
            self._run_sanity_check()

        # enable train mode
        self.model.train()
        torch.set_grad_enabled(True)

        self.active_learning_loop.trainer = self
        with torch.autograd.set_detect_anomaly(self._detect_anomaly):
            self.active_learning_loop.run()
