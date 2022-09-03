from typing import Any, List, Optional, Tuple

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
        NotImplementedError

    def _forward(self, *args, **kwargs) -> Any:
        return self.model.forward(*args, **kwargs)

    def get_inputs_from_batch(self, batch: Any) -> Any:
        return batch


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


class RandomArchorPointsStrategy(BaseQueryStrategy):
    def __init__(self, model: LightningModule, n_anchors: int) -> None:
        super().__init__(model)
        self.n_anchors = n_anchors

    def __post_init__(self) -> None:
        self.pool_loop = PoolNoEvaluationLoop()

    def query(self) -> List[int]:

        assert self.trainer.datamodule.is_synced
    
        if not self.trainer.datamodule.has_labelled_data:
            pool_size = self.trainer.datamodule.pool_size
            indices = np.random.randint(low=0, high=pool_size, size=self.query_size).tolist()
        else:
            # indices wrt train_set
            anchors_indices = self.query_archors()
            
            # NOTE: faiss pads wiwth `-1` when there are less entries than k
            scores, indices = self.search_anchors(anchors_indices, self.query_size)
            
            # across all retrieved compute the one with the highest inner-product
            ids = scores.flatten().argsort()[::-1][:self.query_size]
            indices = indices.flatten()[ids].tolist()

        return indices
    
    def query_archors(self) -> List[int]:
        train_size = self.trainer.datamodule.train_size
        return np.random.randint(low=0, high=train_size, size=self.n_anchors).tolist()

    def get_search_query_from_batch(self, batch: Any) -> Tensor:
        return batch

    def search_anchors(self, indices: List[int], k: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.trainer.datamodule.search_anchors(indices, k)



