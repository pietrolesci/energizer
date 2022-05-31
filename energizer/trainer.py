from typing import Optional, List

from pytorch_lightning import LightningDataModule
from pytorch_lightning import Trainer as Trainer_pl
from pytorch_lightning.callbacks.progress.base import ProgressBarBase
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.loops.epoch.training_epoch_loop import TrainingEpochLoop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from torch.utils.data import DataLoader

from energizer.data.datamodule import ActiveDataModule
from energizer.loops.pool_loop import PoolEvaluationLoop
from energizer.loops.active_loop import ActiveLearningLoop
from energizer.learners.base import Learner


class TQDMProgressBarPool(TQDMProgressBar):
    @property
    def test_description(self) -> str:
        if hasattr(self.trainer.lightning_module, "is_on_pool") and self.trainer.lightning_module.is_on_pool:
            return "Pool Evaluation"
        return super().test_description


def patch_progress_bar(trainer: Trainer_pl) -> List:
    callbacks = []
    for c in trainer.callbacks:
        if not isinstance(c, ProgressBarBase):
            callbacks.append(c)
        else:
            callbacks.append(TQDMProgressBarPool(process_position=c.process_position, refresh_rate=c.refresh_rate))
    return callbacks


class Trainer(Trainer_pl):
    def __init__(
        self,
        query_size: int = 2,
        total_budget: int = -1,
        min_labelling_iters: int = 0,
        max_labelling_iters: int = 1000,
        reset_weights: bool = True,
        n_epochs_between_labelling: int = 1,
        test_after_labelling: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.query_size = query_size
        self.total_budget = total_budget
        self.reset_weights = reset_weights
        self.n_epochs_between_labelling = n_epochs_between_labelling
        self.min_labelling_iters = min_labelling_iters
        self.max_labelling_iters = max_labelling_iters
        self.test_after_labelling = test_after_labelling

        # Intialize rest of the trainer
        self.callbacks = patch_progress_bar(self)

        # pool evaluation loop
        self.pool_loop = PoolEvaluationLoop(self.query_size)

        # fit after each labelling session
        active_fit_loop = FitLoop(min_epochs=self.min_epochs, max_epochs=n_epochs_between_labelling)
        training_epoch_loop = TrainingEpochLoop(min_steps=self.min_steps, max_steps=self.max_steps)
        active_fit_loop.connect(epoch_loop=training_epoch_loop)
        self.active_fit_loop = active_fit_loop

        # test after labelling and fitting
        self.active_test_loop = EvaluationLoop(verbose=False)

        self.active_learning_loop = ActiveLearningLoop(
            reset_weights=self.reset_weights,
            total_budget=self.total_budget,
            test_after_labelling=self.test_after_labelling,
            min_epochs=self.min_labelling_iters,
            max_epochs=self.max_labelling_iters,
        )

    def active_fit(
        self,
        model: Learner,
        train_dataloader: Optional[DataLoader] = None,
        val_dataloaders: Optional[DataLoader] = None,
        test_dataloaders: Optional[DataLoader] = None,
        datamodule: Optional[LightningDataModule] = None,
    ) -> None:

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
        self.active_learning_loop.run()
