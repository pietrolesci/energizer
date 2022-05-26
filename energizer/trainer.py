import sys
from argparse import ArgumentParser
from gc import callbacks
from typing import Optional

from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning import Trainer as Trainer_pl
from pytorch_lightning.callbacks.progress.base import ProgressBarBase
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm, TQDMProgressBar
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from torch.utils.data import DataLoader

from energizer.datamodule import ActiveDataModule
from energizer.loop import ActiveLearningLoop
from energizer.strategies.strategies import EnergizerStrategy


class TQDMProgressBarPool(TQDMProgressBar):
    @property
    def test_description(self) -> str:
        if self.trainer.lightning_module.is_on_pool:
            return "Pool Evaluation"
        return super().test_description


class Trainer(Trainer_pl):
    def __init__(
        self,
        query_size: int = 2,
        total_budget: int = -1,
        reset_weights: bool = True,
        n_epochs_between_labelling: int = 1,
        test_after_labelling: bool = True,
        *args,
        **kwargs,
    ) -> None:
        self.query_size = query_size
        self.total_budget = total_budget
        self.reset_weights = reset_weights
        self.n_epochs_between_labelling = n_epochs_between_labelling
        self.test_after_labelling = test_after_labelling

        # Intialize rest of the trainer
        super().__init__(*args, **kwargs)
        self._patch_progress_bar()

        # TODO: create EvaluationLoop() here and fitloop
        # active_fit_loop = FitLoop()
        # pool_loop = EvaluationLoop()
        # test_after_fit_loop = EvaluationLoop()

    def active_fit(
        self,
        model: EnergizerStrategy,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[DataLoader] = None,
        test_dataloaders: Optional[DataLoader] = None,
        datamodule: Optional[LightningDataModule] = None,
    ) -> None:

        # overwrite standard fit loop
        self.fit_loop = ActiveLearningLoop(
            n_epochs_between_labelling=self.n_epochs_between_labelling,
            reset_weights=self.reset_weights,
            total_budget=self.total_budget,
            test_after_labelling=self.test_after_labelling,
            fit_loop=self.fit_loop,
        )

        # construct active learning pool and train data splits
        if train_dataloader is not None or datamodule is not None and not isinstance(datamodule, ActiveDataModule):
            datamodule = ActiveDataModule(
                train_dataloader=train_dataloader,
                val_dataloaders=val_dataloaders,
                test_dataloaders=test_dataloaders,
                datamodule=datamodule,
            )

        if self.test_after_labelling and getattr(datamodule, "test_dataloader", None) is None:
            raise MisconfigurationException(
                "You specified `test_after_labelling=True` but no test_dataloader was provided."
            )

        # run loop
        self.fit(model, datamodule=datamodule)

        # TODO: restore original fit loop
        # self.fit_loop = self.fit_loop.fit_loop
        # self.test_loop = self.fit_loop.test_loop

    @classmethod
    def add_argparse_args(cls, parent_parser: ArgumentParser, **kwargs) -> ArgumentParser:
        """Alter the argparser to also include the new arguments"""
        parser = super().add_argparse_args(parent_parser, **kwargs)
        parser.add_argument("--num_folds", type=int, default=5)
        parser.add_argument("--shuffle", type=bool, default=False)
        parser.add_argument("--stratified", type=bool, default=False)
        return parser

    def _patch_progress_bar(self):
        callbacks = []
        for c in self.callbacks:
            if not isinstance(c, ProgressBarBase):
                callbacks.append(c)
            else:
                callbacks.append(TQDMProgressBarPool(process_position=c.process_position, refresh_rate=c.refresh_rate))
        self.callbacks = callbacks
