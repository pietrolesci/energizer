from typing import Any, Optional, Tuple, Union, Callable

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from energizer.inference import EnergizerInference

from torchmetrics import Metric


class BaseTopK(Metric):
    def __init__(self, k: int, compute_on_step: bool = True, dist_sync_on_step: bool = False, process_group: Optional[Any] = None, dist_sync_fn: Callable = None) -> None:
        super().__init__(compute_on_step, dist_sync_on_step, process_group, dist_sync_fn)
        self.k = k
        self.add_state("topk_scores", torch.tensor([float("-inf")] * self.k))
        self.add_state("indices", torch.ones(self.k, dtype=torch.int64).neg())
        self.add_state("size", torch.tensor(0, dtype=torch.int64))

    def score(self, logits: Tensor) -> Tensor:
        raise NotImplementedError

    def update(self, scores: Tensor) -> None:        
        batch_size = scores.numel()

        # indices with respect to the pool dataset
        current_indices = torch.arange(self.size, self.size + batch_size, 1)
        
        # compute topk comparing current batch with the states
        values = torch.cat([self.topk_scores, scores], dim=0)
        indices = torch.cat([self.indices, current_indices], dim=0)
        
        # aggregation
        topk_scores, idx = torch.topk(values, k=self.k)

        self.topk_scores.copy_(topk_scores)
        self.indices.copy_(indices[idx])
        self.size += batch_size

    def compute(self) -> Tensor:
        return self.indices


class EntropyTopK(BaseTopK):
    def score(self, logits: Tensor) -> Tensor:
        return super().score(logits)
    



class EnergizerStrategy(LightningModule):
    """Base class for a strategy that is a thin wrapper around `LightningModule`.

    It defines the `pool_*` methods and hooks that the user can redefine.
    Since the `pool_loop` is an `evaluation_loop` under the hood, this class calls
    the `pool_*` and `on_pool_*` methods in the respective `test_*` and `on_test_*` methods.
    This is necessary in order for the lightning `evaluation_loop` to pick up and run the
    methods. On the other hand, the user is able to deal with the `pool_*` and `on_pool_*`
    methods which makes the API cleaner.

    One important feature of this class is that the scores computed on the pool need not be
    kept in memory, but they are computed per each batch and a state is kept in the strategy
    class. On every batch, the state is updated and contains the maximum/minimum (depending on
    the strategy) scores seen so far. The states have dimension equal to the `query_size`.
    Therefore, the memory overhead is negligible in many cases, compared to other implementations
    that first compute the scores on the entire pool and then optimize them and extrarct the indices.
    """

    def __init__(self, module: LightningModule, inference: Union[str, EnergizerInference]) -> None:
        """Initializes a strategy.

        Args:
            module (torch.nn.Module): An inference module that modifies the forward behaviour
                of the underlying module.
        """
        super().__init__()
        self.inference = inference
        self.module = self.inference.prepare(module)
        
        self.query_size: Optional[int] = None
        self.topk = TopK(self.k)

    def forward(self, *args, **kwargs) -> Any:
        """Calls the forward method of the inference module."""
        return self.inference(self.module, *args, **kwargs)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Tuple[Tensor, Tensor, int]:
        """Main entry point to define a custom strategy.

        Since the `pool_loop` is a lightning `evalutation_loop`, in particular at `test_loop`, this method will be
        called inplace of the usual `test_step` method.

        This method should implement the logic to apply to each batch in the pool dataset and must return a tuple
        where the first element is a tensor with the actual scores and the second element is a tensor with the
        relative indices with respect to the batch.

        This method simply performs the following steps

        ```python
        # compute logits
        logits = self(batch)

        # compute scores
        logits = self.on_before_objective(logits)
        scores = self.objective(logits)
        scores = self.on_after_objective(scores)

        # optimize objective
        values, indices = self.optimize_objective(scores, query_size)

        # finally returns the tuple (values, indices)
        ```
        """
        logits = self(batch)
        scores = self.score(logits)
        return self.topk(scores)

    def score(self, logits: Tensor) -> Tensor:
        """Must be redefined with the specific active learning objective, aka the acquisition function.

        Args:
            logits (Tensor): The output of the inference module forward pass.

        Returns:
            A tensor containing the scores that have to be optimized. By default, this tensor
            is then flatten by the `on_after_objective` method.
        """
        raise NotImplementedError
