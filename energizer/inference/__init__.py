from typing import Any, Optional

import torch
from pytorch_lightning import LightningModule
from torch.functional import Tensor

from energizer.inference.utils import patch_dropout_layers


class EnergizerInference(LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.module: Optional[LightningModule] = None

    def forward(self, *args, **kwargs) -> Any:
        raise NotImplementedError

    def connect(self, module: LightningModule) -> None:
        self.module = module


class Deterministic(EnergizerInference):
    """Deterministic inference method that uses the forward pass of the underlying module."""

    def __init__(self) -> None:
        """
        Args:
            module (LightningModule): The LightningModule to use for inference.
        """
        super().__init__()

    def forward(self, *args, **kwargs) -> Any:
        return self.module(*args, **kwargs)  # type: ignore


class MCDropout(EnergizerInference):
    """Implements the MCDropout inference method in [PUT REFERENCES]."""

    def __init__(self, num_inference_iters: int, consistent: bool = False, prob: Optional[float] = None) -> None:
        """Instantiates a new module that is the same as `module` but with the dropout layers patched such that they are
        active even during evaluation.

        Args:
            module (LightingModule): The LightningModule to patch. It must contain at least one dropout layer.
            num_inference_iters (int): The number of forward passes to perform.
            consistent (bool): If True, it uses the consistent version of dropout that fixes the mask across batches.
            prob (float): If specified, it is used as the dropout probabilies of the patched layers.
                Must be 0 < prob < 1.
        """
        super().__init__()
        self.num_inference_iters = num_inference_iters
        self.consistent = consistent
        self.prob = prob

    def connect(self, module: LightningModule) -> None:
        super().connect(module)
        patch_dropout_layers(self.module, self.consistent, self.prob)

    def forward(self, *args, **kwargs) -> Tensor:
        out = []
        for _ in range(self.num_inference_iters):
            out.append(self.module(*args, **kwargs))  # type: ignore

        # expects a shape [num_samples, num_classes, num_iterations]
        return torch.stack(out).permute((1, 2, 0))
