from copy import deepcopy
from typing import Any, List, Optional, Callable

import numpy
import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loops.loop import Loop
from pytorch_lightning.utilities.model_helpers import is_overridden
from torch import Tensor

from energizer.loops.pool_loops import AccumulateTopK, PoolEvaluationLoop, PoolNoEvaluationLoop
from energizer.query_strategies.hooks import ModelHooks
from energizer.utilities.mcdropout import local_seed, replace_dropout_layers
from energizer.utilities.types import BATCH_TYPE, MODEL_INPUT


def identity(x: Any) -> Any:
    return x


class PostInitCaller(type):
    """Used to call `setup` automatically after `__init__`."""

    def __call__(cls, *args, **kwargs):
        """Called when you call `MyNewClass()`."""
        obj = type.__call__(cls, *args, **kwargs)
        obj.__post_init__()
        return obj


class BaseQueryStrategy(LightningModule, ModelHooks, metaclass=PostInitCaller):
    def __init__(self, model: LightningModule, get_inputs_from_batch_fn: Optional[Callable] = None) -> None:
        super().__init__()
        self.model = deepcopy(model)
        self.get_inputs_from_batch_fn = get_inputs_from_batch_fn if get_inputs_from_batch_fn else identity

    def __post_init__(self) -> None:
        raise NotImplementedError("You need to attach a pool loop.")

    def __call__(self, *args, **kwargs) -> Any:
        return self._forward(*args, **kwargs)

    @property
    def pool_loop(self) -> Loop:
        return self._pool_loop

    @pool_loop.setter
    def pool_loop(self, pool_loop) -> None:
        self._pool_loop = pool_loop

    @property
    def query_size(self) -> int:
        return self._query_size

    @query_size.setter
    def query_size(self, query_size: int) -> None:
        self._query_size = query_size

    def query(self) -> List[int]:
        """Queries instances from the unlabeled pool.

        A query selects instances from the unlabeled pool.
        """
        raise NotImplementedError("You need to define how the pool is queried.")

    def _forward(self, *args, **kwargs) -> Any:
        return self.model.forward(*args, **kwargs)

    def get_inputs_from_batch(self, batch: BATCH_TYPE) -> MODEL_INPUT:
        return self.get_inputs_from_batch_fn(batch)


class NoAccumulatorStrategy(BaseQueryStrategy):
    def __post_init__(self) -> None:
        self.pool_loop = PoolNoEvaluationLoop()


class AccumulatorStrategy(BaseQueryStrategy):
    def __post_init__(self) -> None:
        self.pool_loop = PoolEvaluationLoop(verbose=False)

    @property
    def query_size(self) -> int:
        return self._query_size

    @query_size.setter
    def query_size(self, query_size: int) -> None:
        self._query_size = query_size
        self.pool_loop.epoch_loop.accumulator = AccumulateTopK(query_size)

    def query(self) -> List[int]:
        return self.pool_loop.epoch_loop.accumulator.compute().cpu().tolist()

    def on_pool_epoch_start(self) -> None:
        pass

    def pool_step(self, batch: MODEL_INPUT, batch_idx: int, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def pool_step_end(self, *args, **kwargs) -> Any:
        pass

    def pool_epoch_end(self, *args, **kwargs) -> Optional[Any]:
        pass


class MCAccumulatorStrategy(AccumulatorStrategy):
    """Implements the MCDropout inference method in [PUT REFERENCES]."""

    def __init__(
        self,
        model: LightningModule,
        num_inference_iters: Optional[int] = 10,
        consistent: Optional[bool] = False,
        prob: Optional[float] = None,
        seeds: Optional[List[int]] = None,
        **kwargs,
    ) -> None:
        """Instantiates a new learner (same as `learner`) with patched dropout layers.

        The patched such that they are active even during evaluation.

        When `consistent=True`, it is possible to guarantee that the dropout mask is "consistent"
        during validation and testing. That is, if `num_inference_iters=10`, the same 10 masks
        will be used for each batch.

        One of the advantages of the `Energizer` library is that it implements this feature without requiring
        to keep in memory `num_inference_iters` copies of the mask of each dropout layer in a model.
        Instead, with wrap the forward pass into a `local_seed` context manager that uses the same seeds.

        It roughly works as follows:

        ```python
        seeds = [1, 2, 3]

        outputs = []
        for seed in seeds:
            with local_seed(seed):
                out = model(inputs)
            outputs.append(out)
        ```

        Args:
            model (LightningModule): The model to use for inference.
            num_inference_iters (int): The number of forward passes to perform.
            consistent (bool): If True, it uses the consistent version of dropout that fixes the mask across batches.
            prob (float): If specified, this changes the dropout probability of all layers to `prob`. If `None` the
                dropout probability is the same as the original layer. Must be 0 <= prob <= 1.
            seeds: (List[int]): List of seeds to be used when `consistent` is True.
                It must have length equal to `num_inference_iters`
        """
        self.num_inference_iters = num_inference_iters
        self.consistent = consistent
        self.prob = prob

        if seeds is not None:
            assert num_inference_iters == len(seeds), ValueError(  # type: ignore
                f"For consistent dropout to work, the list of `seeds` ({len(seeds)}) ",  # type: ignore
                f"needs to have length equal to `num_inference_iters ({num_inference_iters})",
            )
        else:
            seeds = numpy.random.randint(0, 10_000, size=num_inference_iters).tolist()
        self.seeds = seeds

        super().__init__(model, **kwargs)

    def __post_init__(self) -> None:
        super().__post_init__()
        replace_dropout_layers(
            module=self.model,
            prob=self.prob,
            consistent=self.consistent,
            inplace=True,
        )

    def _forward(self, *args, **kwargs) -> Tensor:
        """Performs multiple forward passes using the underlying model while keeping the dropout active.

        Returns:
            A tensor of dimension `(B: batch_size, C: num_classes, S: num_samples)`.
        """
        if self.consistent:
            outputs = self._consistent_forward(*args, **kwargs)
        else:
            outputs = self._normal_forward(*args, **kwargs)

        # expects shape [num_samples, num_classes, num_iterations]
        return torch.stack(outputs).permute((1, 2, 0))

    def _consistent_forward(self, *args, **kwargs) -> List[Tensor]:
        outputs = []
        for seed in range(self.seeds):
            with local_seed(seed):
                out = self.model.forward(*args, **kwargs)  # type: ignore
            outputs.append(out)

    def _normal_forward(self, *args, **kwargs) -> List[Tensor]:
        outputs = []
        for _ in range(self.num_inference_iters):
            out = self.model.forward(*args, **kwargs)  # type: ignore
            outputs.append(out)
