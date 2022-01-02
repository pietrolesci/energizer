from typing import Any, Optional

import torch
from pytorch_lightning import LightningModule
from torch.functional import Tensor

from energizer.inference.utils import patch_dropout_layers


class EnergizerInference(LightningModule):
    """A LightningModule that wraps an existing LightningModule and changes its forward."""

    def __init__(self) -> None:
        super().__init__()
        self.module: Optional[LightningModule] = None

    def forward(self, *args, **kwargs) -> Any:
        """Must be redefined to implement the forward pass of the specific inference module."""
        raise NotImplementedError

    def connect(self, module: LightningModule) -> None:
        """Deferred initialization of the inference module.

        Wrap original model with the inference module. A module cannot be passed directly in
        the inference module constructor.
        """
        self.module = module


class Deterministic(EnergizerInference):
    """Deterministic inference method that uses the forward pass of the underlying module."""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, *args, **kwargs) -> Any:
        """Simply calls the forward method of the underlying module. Here for consistency."""
        return self.module(*args, **kwargs)  # type: ignore


class MCDropout(EnergizerInference):
    """Implements the MCDropout inference method in [PUT REFERENCES]."""

    def __init__(
        self,
        num_inference_iters: int = 10,
        consistent: bool = False,
        prob: Optional[float] = None,
        inplace: bool = True,
    ) -> None:
        """Instantiates a new module (same as `module`) with patched dropout layers.

        The patched such that they are active even during evaluation.

        Args:
            num_inference_iters (int): The number of forward passes to perform.
            consistent (bool): If True, it uses the consistent version of dropout that fixes the mask across batches.
            prob (float): If specified, this changes the dropout probability of all layers to `prob`. If `None` the
                dropout probability is the same as the original layer. Must be 0 <= prob <= 1.
            inplace (bool): Whether to modify the module in place or return a copy of the module.
        """
        super().__init__()
        self.num_inference_iters = num_inference_iters
        self.consistent = consistent
        self.prob = prob
        self.inplace = inplace

    def connect(self, module: LightningModule) -> None:
        """Provide the original module to the inference module.

        It calls the `connect` method of the base class and, additionally, it patches the dropout layers of the original
        module so that they remain active even during evaluation.
        """
        super().connect(module)
        patch_dropout_layers(
            module=self.module,
            prob=self.prob,
            consistent=self.consistent,
            inplace=self.inplace,
            num_inference_iters=self.num_inference_iters,
        )

    def forward(self, *args, **kwargs) -> Tensor:
        """Performs `num_inference_iters` forward passes using the underlying module and keeping the dropout layers active..

        Returns:
            A tensor of dimension `(B: batch_size, C: num_classes, S: num_samples)`.
        """
        out = []
        for _ in range(self.num_inference_iters):
            out.append(self.module(*args, **kwargs))  # type: ignore

        # expects a shape [num_samples, num_classes, num_iterations]
        return torch.stack(out).permute((1, 2, 0))
