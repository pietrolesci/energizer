from typing import Any, Optional, Tuple

import torch
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor

from energizer.inference.inference_modules import EnergizerInference


class EnergizerStrategy(LightningModule):
    def __init__(self, inference_module: EnergizerInference) -> None:
        super().__init__()
        self.inference_module = inference_module
        self.trainer: Optional[Trainer] = None
        self.query_size: Optional[int] = None

        self._counter: Optional[int] = None
        self.values: Optional[Tensor] = None
        self.indices: Optional[Tensor] = None

    def connect(self, module: LightningModule, query_size: int) -> None:
        """Deferred initialization of the strategy."""
        self.inference_module.connect(module)
        self.trainer = module.trainer
        self.query_size = query_size

    def forward(self, *args, **kwargs) -> Any:
        return self.inference_module(*args, **kwargs)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        # print("INF - PREDICTING")

        with torch.inference_mode():
            logits = self(batch)
            batch_size = logits.shape[0]

            scores = self.objective(logits).flatten()

            # print("\tBATCH STARTING:", flush=True)
            # print("\tScores:", scores, flush=True)
            # print("\tBatch_size:", batch_size, flush=True)
            # print("\tBatch_idx:", batch_idx, flush=True)

            values, indices = self.optimize_objective(scores, min(batch_size, self.query_size))

            # print("\tIndices Batch:", indices.tolist(), flush=True)
            # print("\tInstance Counter:", self._counter, flush=True)
            # print("\tInstance Progress:", self.batch_progress, flush=True)

            indices = self._batch_to_pool(indices, batch_size)
            # print("\tIndices Batch to Pool:", indices.tolist(), flush=True)
            # print("\tAll indeces:", self.indices.tolist(), flush=True)

            # print("\tUpdated All indeces:", self.indices.tolist(), "\n", flush=True)
            # self.batch_progress.increment_completed()

        return indices, values

    def test_step_end(self, output_results) -> None:
        indices, values = output_results
        self._update(indices, values)

    def reset(self) -> None:
        self._counter = 0
        self.values = torch.zeros(self.query_size, dtype=torch.float32, device=self.device, requires_grad=False)
        self.indices = -torch.ones(self.query_size, dtype=torch.int64, device=self.device, requires_grad=False)

    def objective(self, logits: Tensor) -> Tensor:
        raise NotImplementedError

    def optimize_objective(self, scores: Tensor, query_size: Optional[int] = 1) -> Tuple[Tensor, Tensor]:
        return torch.topk(scores, query_size, dim=0)

    def _update(self, indices: Tensor, values: Tensor) -> None:
        all_values = torch.cat([self.values, values], dim=0)
        all_indices = torch.cat([self.indices, indices], dim=0)

        new_values, idx = self.optimize_objective(all_values, self.query_size)
        self.values.copy_(new_values)  # type: ignore
        self.indices.copy_(all_indices[idx])  # type: ignore

    def _batch_to_pool(self, indices: Tensor, batch_size: int) -> Tensor:
        indices += self._counter
        self._counter += batch_size  # type: ignore
        return indices

    def _call_lightning_module_hook(self, hook_name, *args, **kwargs):
        fn = getattr(self.inference_module.module, hook_name, None)
        if fn:
            fn(*args, **kwargs)

    def _call_callback_hook(self, hook_name, *args, **kwargs):
        for callback in self.trainer.callbacks:
            fn = getattr(callback, hook_name, None)
            if callable(fn):
                fn(self, *args, **kwargs)

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self._call_lightning_module_hook("on_pool_batch_start", batch, batch_idx, dataloader_idx)
        self._call_callback_hook("on_pool_batch_start", batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        self._call_lightning_module_hook("on_pool_batch_end", batch, batch_idx, dataloader_idx)
        self._call_callback_hook("on_pool_batch_end", batch, batch_idx, dataloader_idx)

    def on_test_epoch_start(self) -> None:
        self._call_lightning_module_hook("on_pool_epoch_start")
        self._call_callback_hook("on_pool_epoch_start")

    def on_test_epoch_end(self) -> None:
        self._call_lightning_module_hook("on_pool_epoch_end")
        self._call_callback_hook("on_pool_epoch_end")

    def on_test_start(self) -> None:
        self._call_lightning_module_hook("on_pool_start")
        self._call_callback_hook("on_pool_start")

    def on_test_end(self) -> None:
        self._call_lightning_module_hook("on_pool_end")
        self._call_callback_hook("on_pool_end")

    def on_test_model_eval(self) -> None:
        self._call_lightning_module_hook("on_pool_model_eval")
        self._call_callback_hook("on_pool_model_eval")

    def on_test_model_train(self) -> None:
        self._call_lightning_module_hook("on_pool_model_train")
        self._call_callback_hook("on_pool_model_train")
