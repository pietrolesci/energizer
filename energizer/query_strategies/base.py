from typing import Any, Optional, List
from abc import ABC, abstractmethod
import torch
from pytorch_lightning import LightningModule
from torch import Tensor

from energizer.mixin.hooks import ModelHooks
from energizer.mixin.mcdropout_utils import patch_dropout_layers
from energizer.loops.pool_loop import PoolEvaluationLoop, PoolNoEvalLoop
from energizer.loops.topk_accumulator import AccumulateTopK
import numpy as np


class PostInitCaller(type):
    """Used to call `setup` automatically after `__init__`."""

    def __call__(cls, *args, **kwargs):
        """Called when you call `MyNewClass()`."""
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


class BaseQueryStrategy(LightningModule, ModelHooks, metaclass=PostInitCaller):

    def __init__(self, model: LightningModule) -> None:
        super().__init__()
        self.model = model

    @property
    def pool_loop(self) -> PoolEvaluationLoop:
        return self._pool_loop

    @pool_loop.setter
    def pool_loop(self, pool_loop) -> None:
        self._pool_loop = pool_loop

    def query(self) -> List[int]:
        """Queries instances from the unlabeled pool.
        
        A query selects instances from the unlabeled pool.
        """
        pass

    def _forward(self, *args, **kwargs) -> Any:
        return self.model.forward(*args, **kwargs)

    def __call__(self, *args, **kwargs) -> Any:
        return self._forward(*args, **kwargs)
    
    def __post_init__(self) -> None:
        pass

    @property
    def query_size(self) -> int:
        return self._query_size

    @query_size.setter
    def query_size(self, query_size: int) -> None:
        self._query_size = query_size


class RandomStrategy(BaseQueryStrategy):

    def __post_init__(self) -> None:
        self.pool_loop = PoolNoEvalLoop()

    def query(self) -> List[int]:
        pool_size = self.trainer.datamodule.pool_size
        return np.random.randint(low=0, high=pool_size, size=self.query_size).tolist()


class AccumulatorStrategy(BaseQueryStrategy):

    def __post_init__(self) -> None:
        self.pool_loop = PoolEvaluationLoop(verbose=False)

    def on_pool_epoch_start(self) -> None:
        self.accumulator.reset()

    def pool_step(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def pool_step_end(self, *args, **kwargs) -> Any:
        pass

    def pool_epoch_end(self, *args, **kwargs) -> Optional[Any]:
        pass

    def query(self) -> List[int]:
        return self.accumulator.compute().cpu().tolist()

    @property
    def query_size(self) -> int:
        return self._query_size

    @property
    def accumulator(self) -> AccumulateTopK:
        return self._accumulator        

    @accumulator.setter
    def accumulator(self, accumulator: AccumulateTopK) -> None:
        self._accumulator = accumulator   

    @query_size.setter
    def query_size(self, query_size: int) -> None:
        self._query_size = query_size
        self.accumulator = AccumulateTopK(query_size)
        self.pool_loop.query_size = query_size

    


# class DeterministicMixin(Learner):
#     pass


# class MCDropoutMixin(Learner):
#     """Implements the MCDropout inference method in [PUT REFERENCES]."""

#     def __init__(
#         self,
#         model: LightningModule,
#         num_inference_iters: Optional[int] = 10,
#         consistent: Optional[bool] = False,
#         prob: Optional[float] = None,
#     ) -> None:
#         """Instantiates a new learner (same as `learner`) with patched dropout layers.

#         The patched such that they are active even during evaluation.

#         Args:
#             num_inference_iters (int): The number of forward passes to perform.
#             consistent (bool): If True, it uses the consistent version of dropout that fixes the mask across batches.
#             prob (float): If specified, this changes the dropout probability of all layers to `prob`. If `None` the
#                 dropout probability is the same as the original layer. Must be 0 <= prob <= 1.
#             inplace (bool): Whether to modify the learner in place or return a copy of the learner.
#         """
#         self.num_inference_iters = num_inference_iters
#         self.consistent = consistent
#         self.prob = prob
#         super().__init__(model)

#     def __post_init__(self) -> None:
#         patch_dropout_layers(
#             module=self.model,
#             prob=self.prob,
#             consistent=self.consistent,
#             inplace=True,
#         )

#     def _forward(self, *args, **kwargs) -> Tensor:
#         """Performs `num_inference_iters` forward passes using the underlying learner and keeping the dropout layers active..

#         Returns:
#             A tensor of dimension `(B: batch_size, C: num_classes, S: num_samples)`.
#         """
#         out = []
#         for _ in range(self.num_inference_iters):
#             out.append(self.model.forward(*args, **kwargs))  # type: ignore
#         # expects shape [num_samples, num_classes, num_iterations]
#         return torch.stack(out).permute((1, 2, 0))
