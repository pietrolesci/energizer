from typing import Any, Optional

import torch
from pytorch_lightning import LightningModule
from torch import Tensor

from energizer.learners.hooks import ModelHooks
from energizer.utilities.mcdropout_utils import patch_dropout_layers


class PostInitCaller(type):
    """Used to call `setup` automatically after `__init__`"""

    def __call__(cls, *args, **kwargs):
        """Called when you call MyNewClass()"""
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


class Learner(LightningModule, ModelHooks, metaclass=PostInitCaller):
    def __init__(self):
        super().__init__()

    def __post_init__(self) -> None:
        pass

    def pool_step(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def pool_step_end(self, *args, **kwargs) -> Any:
        pass

    def pool_epoch_end(self, *args, **kwargs) -> Optional[Any]:
        pass

    def _forward(self, *args, **kwargs) -> Any:
        return self.forward(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        return self._forward(*args, **kwargs)


class DeterministicMixin(Learner):
    pass


class MCDropoutMixin(Learner):
    """Implements the MCDropout inference method in [PUT REFERENCES]."""

    def __init__(
        self,
        num_inference_iters: Optional[int] = 10,
        consistent: Optional[bool] = False,
        prob: Optional[float] = None,
    ) -> None:
        """Instantiates a new learner (same as `learner`) with patched dropout layers.

        The patched such that they are active even during evaluation.

        Args:
            num_inference_iters (int): The number of forward passes to perform.
            consistent (bool): If True, it uses the consistent version of dropout that fixes the mask across batches.
            prob (float): If specified, this changes the dropout probability of all layers to `prob`. If `None` the
                dropout probability is the same as the original layer. Must be 0 <= prob <= 1.
            inplace (bool): Whether to modify the learner in place or return a copy of the learner.
        """
        self.num_inference_iters = num_inference_iters
        self.consistent = consistent
        self.prob = prob
        super().__init__()

    def __post_init__(self) -> None:
        patch_dropout_layers(
            module=self,
            prob=self.prob,
            consistent=self.consistent,
            inplace=True,
        )

    def _forward(self, *args, **kwargs) -> Tensor:
        """Performs `num_inference_iters` forward passes using the underlying learner and keeping the dropout layers active..

        Returns:
            A tensor of dimension `(B: batch_size, C: num_classes, S: num_samples)`.
        """
        out = []
        for _ in range(self.num_inference_iters):
            out.append(self.forward(*args, **kwargs))  # type: ignore
        # expects shape [num_samples, num_classes, num_iterations]
        return torch.stack(out).permute((1, 2, 0))
