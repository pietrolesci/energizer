"""Forward hooks to underlying Lightning module."""

from typing import Any, Dict, List, Optional

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch.optim.optimizer import Optimizer


class ModelHooks:
    """Hooks to be used in LightningModule."""

    def on_fit_start(self) -> None:
        return self.adapter.on_fit_start()

    def on_fit_end(self) -> None:
        return self.adapter.on_fit_end()

    def on_train_start(self) -> None:
        return self.adapter.on_train_start()

    def on_train_end(self) -> None:
        return self.adapter.on_train_end()

    def on_validation_start(self) -> None:
        return self.adapter.on_validation_start()

    def on_validation_end(self) -> None:
        return self.adapter.on_validation_end()

    def on_test_start(self) -> None:
        return self.adapter.on_test_start()

    def on_test_end(self) -> None:
        return self.adapter.on_test_end()

    def on_predict_start(self) -> None:
        return self.adapter.on_predict_start()

    def on_predict_end(self) -> None:
        return self.adapter.on_predict_end()

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> Optional[int]:
        return self.adapter.on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        return self.adapter.on_train_batch_end(outputs, batch, batch_idx)

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return self.adapter.on_validation_batch_start(batch, batch_idx, dataloader_idx)

    def on_validation_batch_end(
        self, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        return self.adapter.on_validation_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return self.adapter.on_test_batch_start(batch, batch_idx, dataloader_idx)

    def on_test_batch_end(
        self, outputs: Optional[STEP_OUTPUT], batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        return self.adapter.on_test_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def on_predict_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return self.adapter.on_predict_batch_start(batch, batch_idx, dataloader_idx)

    def on_predict_batch_end(self, outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        return self.adapter.on_predict_batch_end(outputs, batch, batch_idx, dataloader_idx)

    def on_validation_model_eval(self) -> None:
        return self.adapter.on_validation_model_eval()

    def on_validation_model_train(self) -> None:
        return self.adapter.on_validation_model_train()

    def on_test_model_train(self) -> None:
        return self.adapter.on_test_model_train()

    def on_test_model_eval(self) -> None:
        return self.adapter.on_test_model_eval()

    def on_predict_model_eval(self) -> None:
        return self.adapter.on_predict_model_eval()

    def on_train_epoch_start(self) -> None:
        return self.adapter.on_train_epoch_start()

    def on_train_epoch_end(self) -> None:
        return self.adapter.on_train_epoch_end()

    def on_validation_epoch_start(self) -> None:
        return self.adapter.on_validation_epoch_start()

    def on_validation_epoch_end(self) -> None:
        return self.adapter.on_validation_epoch_end()

    def on_test_epoch_start(self) -> None:
        return self.adapter.on_test_epoch_start()

    def on_test_epoch_end(self) -> None:
        return self.adapter.on_test_epoch_end()

    def on_predict_epoch_start(self) -> None:
        return self.adapter.on_predict_epoch_start()

    def on_predict_epoch_end(self, results: List[Any]) -> None:
        return self.adapter.on_predict_epoch_end(results)

    def on_before_zero_grad(self, optimizer: Optimizer) -> None:
        return self.adapter.on_before_zero_grad(optimizer)

    def on_before_backward(self, loss: torch.Tensor) -> None:
        return self.adapter.on_before_backward(loss)

    def on_after_backward(self) -> None:
        return self.adapter.on_after_backward()

    def on_before_optimizer_step(self, optimizer: Optimizer, optimizer_idx: int) -> None:
        return self.adapter.on_before_optimizer_step(optimizer, optimizer_idx)

    def configure_sharded_model(self) -> None:
        return self.adapter.configure_sharded_model()


class DataHooks:
    """Hooks to be used for data related stuff."""

    def prepare_data(self) -> None:
        return self.adapter.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        return self.adapter.setup(stage)

    def teardown(self, stage: Optional[str] = None) -> None:
        return self.adapter.teardown(stage)

    def transfer_batch_to_device(self, batch: Any, device: torch.device, dataloader_idx: int) -> Any:
        return self.adapter.transfer_batch_to_device(batch, device, dataloader_idx)

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return self.adapter.on_before_batch_transfer(batch, dataloader_idx)

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        return self.adapter.on_after_batch_transfer(batch, dataloader_idx)


class CheckpointHooks:
    """Hooks to be used with Checkpointing."""

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        return self.adapter.on_load_checkpoint(checkpoint)

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        return self.adapter.on_save_checkpoint(checkpoint)
