from typing import Any, Optional

import torch
from pytorch_lightning import LightningModule
from torch import Tensor, nn

from energizer.strategies.hooks import CheckpointHooks, DataHooks, ModelHooks
from energizer.strategies.utilities import ModuleWrapperBase, patch_dropout_layers


# class Adapter(ModelHooks, DataHooks, CheckpointHooks, ModuleWrapperBase, LightningModule):
class Adapter(ModuleWrapperBase, LightningModule):
    # TODO: is there a better way to automatically forward methods?

    def __init__(self, adapter: LightningModule):
        super().__init__()
        self.adapter = adapter

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def training_step(self, *args, **kwargs) -> Any:
        return LightningModule.training_step(self.adapter, *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> None:
        return LightningModule.validation_step(self.adapter, *args, **kwargs)

    def test_step(self, *args, **kwargs) -> None:
        return LightningModule.test_step(self.adapter, *args, **kwargs)

    def pool_step(self, *args, **kwargs) -> None:
        return self.adapter.pool_step(self.adapter, *args, **kwargs)

    def training_epoch_end(self, outputs: Any) -> None:
        return LightningModule.training_epoch_end(self.adapter, outputs)

    def validation_epoch_end(self, outputs: Any) -> None:
        return LightningModule.validation_epoch_end(self.adapter, outputs)

    def test_epoch_end(self, outputs: Any) -> None:
        return LightningModule.test_epoch_end(self.adapter, outputs)

    def configure_optimizers(self):
        return self.adapter.configure_optimizers()


class Deterministic(Adapter):
    def forward(self, *args, **kwargs) -> Any:
        return self.adapter(*args, **kwargs)


class MCDropout(Adapter):
    """Implements the MCDropout inference method in [PUT REFERENCES]."""

    def __init__(
        self,
        adapter: nn.Module,
        num_inference_iters: int = 10,
        consistent: bool = False,
        inplace: bool = False,
        prob: Optional[float] = None,
    ) -> None:
        """Instantiates a new adapter (same as `adapter`) with patched dropout layers.

        The patched such that they are active even during evaluation.

        Args:
            num_inference_iters (int): The number of forward passes to perform.
            consistent (bool): If True, it uses the consistent version of dropout that fixes the mask across batches.
            prob (float): If specified, this changes the dropout probability of all layers to `prob`. If `None` the
                dropout probability is the same as the original layer. Must be 0 <= prob <= 1.
            inplace (bool): Whether to modify the adapter in place or return a copy of the adapter.
        """
        self.num_inference_iters = num_inference_iters
        self.consistent = consistent
        self.prob = prob
        self.inplace = inplace
        adapter = patch_dropout_layers(
            module=adapter,
            prob=prob,
            consistent=consistent,
            inplace=inplace,
        )
        super().__init__(adapter)

    def forward(self, *args, **kwargs) -> Tensor:
        """Performs `num_inference_iters` forward passes using the underlying adapter and keeping the dropout layers active..

        Returns:
            A tensor of dimension `(B: batch_size, C: num_classes, S: num_samples)`.
        """
        out = []
        for _ in range(self.num_inference_iters):
            out.append(self.adapter(*args, **kwargs))  # type: ignore
        # expects a shape [num_samples, num_classes, num_iterations]
        return torch.stack(out).permute((1, 2, 0))
