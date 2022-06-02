from typing import Any, List, Optional

from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.progress.base import ProgressBarBase
from pytorch_lightning.callbacks.progress.tqdm_progress import TQDMProgressBar
from pytorch_lightning.utilities.types import STEP_OUTPUT


class TQDMProgressBarPool(TQDMProgressBar):
    # TODO: this is not working I can still see Testing
    @property
    def test_description(self) -> str:
        if hasattr(self.trainer.datamodule, "is_on_pool") and self.trainer.datamodule.is_on_pool:
            return "Pool Evaluation"
        return super().test_description


class CallBackPoolHooks:
    @staticmethod
    def on_pool_batch_start(
        trainer: Trainer, pl_module: LightningModule, batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        """Called when the test batch begins."""

    @staticmethod
    def on_pool_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: Optional[int] = None,
    ) -> None:
        """Called when the test batch ends."""

    @staticmethod
    def on_pool_epoch_start(trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the test epoch begins."""

    @staticmethod
    def on_pool_epoch_end(trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the test epoch ends."""

    @staticmethod
    def on_pool_start(trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the test begins."""

    @staticmethod
    def on_pool_end(trainer: Trainer, pl_module: LightningModule) -> None:
        """Called when the test ends."""

    @staticmethod
    def _add_pool_hooks(callback: Callback) -> Callback:
        callback.on_pool_batch_start = CallBackPoolHooks.on_pool_batch_start
        callback.on_pool_batch_end = CallBackPoolHooks.on_pool_batch_end
        callback.on_pool_epoch_start = CallBackPoolHooks.on_pool_epoch_start
        callback.on_pool_epoch_end = CallBackPoolHooks.on_pool_epoch_end
        callback.on_pool_start = CallBackPoolHooks.on_pool_start
        callback.on_pool_end = CallBackPoolHooks.on_pool_end

        return callback


def patch_callbacks(callbacks: List[Callback]) -> List[Callback]:
    new_callbacks = []
    for c in callbacks:
        if isinstance(c, ProgressBarBase):
            prog_bar = TQDMProgressBarPool(process_position=c.process_position, refresh_rate=c.refresh_rate)
            prog_bar = CallBackPoolHooks._add_pool_hooks(prog_bar)
            new_callbacks.append(prog_bar)
        else:
            new_callbacks.append(CallBackPoolHooks._add_pool_hooks(c))
    
    return new_callbacks
