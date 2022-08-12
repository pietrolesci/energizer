"""Forward hooks to underlying Lightning module."""

from typing import Any, Optional

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.types import STEP_OUTPUT


class ModelHooks:
    """Hooks to be used in LightningModule."""

    """
    New hooks
    """

    def on_pool_start(self) -> None:
        pass

    def on_pool_end(self) -> None:
        pass

    def on_pool_epoch_start(self) -> None:
        pass

    def on_pool_epoch_end(self) -> None:
        pass

    def on_pool_model_train(self) -> None:
        pass

    def on_pool_model_eval(self) -> None:
        pass

    def on_active_learning_start(self) -> None:
        pass

    def on_active_learning_end(self) -> None:
        pass

    def on_labelling_epoch_start(self) -> None:
        pass

    def on_labelling_epoch_end(self) -> None:
        pass

    def on_pool_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        pass

    def on_pool_batch_end(
        self, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        pass

    def on_pool_dataloader(self, *_: Any):
        """To deprecate when PL deprecates the similar ones.

        This is used by PL in trainer._data_connector._request_dataloader
        """


class CallBackActiveLearningHooks:
    @staticmethod
    def on_pool_start(trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the pool evaluation begins."""

    @staticmethod
    def on_pool_end(trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the pool evaluation ends."""

    @staticmethod
    def on_pool_epoch_start(trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the pool evaluation epoch begins."""

    @staticmethod
    def on_pool_epoch_end(trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the pool evaluation epoch ends."""

    @staticmethod
    def on_active_learning_start(trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when active learning starts."""
        pass
    @staticmethod
    def on_active_learning_end(trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when active learning ends."""
        pass
    @staticmethod
    def on_labelling_epoch_start(trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the active learning (labelling) epoch starts."""
        pass
    @staticmethod
    def on_labelling_epoch_end(trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the pool evaluation ends."""
        pass

    @staticmethod
    def on_pool_batch_start(
        trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the active learning (labelling) epoch begins."""

    @staticmethod
    def on_pool_batch_end(
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> None:
        """Called when the pool evaluation batch ends."""
