from typing import Any, Callable, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor, nn
from torchmetrics import Metric

import energizer.strategies.functional as F
from energizer.strategies.hooks import CheckpointHooks, DataHooks, ModelHooks
from energizer.strategies.inference import Learner
from energizer.strategies.utilities import ModuleWrapperBase, patch_dropout_layers


class AccumulateTopK(Metric):
    def __init__(
        self,
        k: int,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(compute_on_step, dist_sync_on_step, process_group, dist_sync_fn)
        self.k = k
        self.add_state("topk_scores", torch.tensor([float("-inf")] * self.k))
        self.add_state("indices", torch.ones(self.k, dtype=torch.int64).neg())
        self.add_state("size", torch.tensor(0, dtype=torch.int64))

    def update(self, scores: Tensor) -> None:
        batch_size = scores.numel()

        # indices with respect to the pool dataset and scores
        current_indices = torch.arange(self.size, self.size + batch_size, 1)

        # compute topk comparing current batch with the states
        all_scores = torch.cat([self.topk_scores, scores], dim=0)
        all_indices = torch.cat([self.indices, current_indices], dim=0)

        # aggregation
        topk_scores, idx = torch.topk(all_scores, k=min(self.k, batch_size))

        self.topk_scores.copy_(topk_scores)
        self.indices.copy_(all_indices[idx])
        self.size += batch_size

        print("current batch_size:", batch_size)
        print("current_indices:", current_indices)
        print("size:", self.size)
        print("all_scores:", all_scores)
        print("all_indices:", all_indices)
        print("top_scores:", topk_scores)
        print("top_indices:", all_indices[idx], "\n")

    def compute(self) -> Tensor:
        print("compute indices:", self.indices)
        return self.indices


class Learner(ModelHooks, DataHooks, CheckpointHooks, ModuleWrapperBase, LightningModule):
    def __init__(self, learner: LightningModule, query_size: int):
        super().__init__()
        self.learner = learner
        self.query_size = query_size
        self.accumulation_metric = AccumulateTopK(self.query_size)

    def forward(self, *args, **kwargs) -> Any:
        return self.learner(*args, **kwargs)

    def training_step(self, *args, **kwargs) -> Any:
        return LightningModule.training_step(self.learner, *args, **kwargs)

    def validation_step(self, *args, **kwargs) -> None:
        return LightningModule.validation_step(self.learner, *args, **kwargs)

    def test_step(self, *args, **kwargs) -> None:
        return LightningModule.test_step(self.learner, *args, **kwargs)

    def pool_step(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def training_epoch_end(self, outputs: Any) -> None:
        return LightningModule.training_epoch_end(self.learner, outputs)

    def validation_epoch_end(self, outputs: Any) -> None:
        return LightningModule.validation_epoch_end(self.learner, outputs)

    def test_epoch_end(self, outputs: Any) -> None:
        return LightningModule.test_epoch_end(self.learner, outputs)

    def pool_step_end(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        pass

    def configure_optimizers(self):
        return self.learner.configure_optimizers()


class Deterministic(Learner):
    pass


class MCDropout(Learner):
    """Implements the MCDropout inference method in [PUT REFERENCES]."""

    def __init__(
        self,
        learner: nn.Module,
        query_size: int,
        num_inference_iters: int = 10,
        consistent: bool = False,
        inplace: bool = False,
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
        self.inplace = inplace
        learner = patch_dropout_layers(
            module=learner,
            prob=prob,
            consistent=consistent,
            inplace=inplace,
        )
        super().__init__(learner, query_size)

    def forward(self, *args, **kwargs) -> Tensor:
        """Performs `num_inference_iters` forward passes using the underlying learner and keeping the dropout layers active..

        Returns:
            A tensor of dimension `(B: batch_size, C: num_classes, S: num_samples)`.
        """
        out = []
        for _ in range(self.num_inference_iters):
            out.append(self.learner(*args, **kwargs))  # type: ignore
        # expects a shape [num_samples, num_classes, num_iterations]
        return torch.stack(out).permute((1, 2, 0))
