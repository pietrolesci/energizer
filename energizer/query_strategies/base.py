from typing import Any, List, Optional

import numpy as np
from pytorch_lightning import LightningModule
from pytorch_lightning.loops.loop import Loop
from torch import Tensor

from energizer.loops.pool_loops import AccumulateTopK, PoolEvaluationLoop, PoolNoEvaluationLoop
from energizer.query_strategies.hooks import ModelHooks


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

    def __post_init__(self) -> None:
        pass

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
        pass

    def _forward(self, *args, **kwargs) -> Any:
        return self.model.forward(*args, **kwargs)


class RandomStrategy(BaseQueryStrategy):
    def __post_init__(self) -> None:
        self.pool_loop = PoolNoEvaluationLoop()

    def query(self) -> List[int]:
        pool_size = self.trainer.datamodule.pool_size
        return np.random.randint(low=0, high=pool_size, size=self.query_size).tolist()


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

    def pool_step(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError

    def pool_step_end(self, *args, **kwargs) -> Any:
        pass

    def pool_epoch_end(self, *args, **kwargs) -> Optional[Any]:
        pass


# class RandomArchorPointsStrategy(BaseQueryStrategy):
