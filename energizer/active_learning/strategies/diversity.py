from abc import ABC, abstractmethod
from typing import Any, Callable, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from lightning.fabric.wrappers import _FabricModule
from numpy.random import RandomState
from sklearn.utils.validation import check_random_state

from energizer.active_learning.datastores.base import ActiveDatastore
from energizer.active_learning.registries import CLUSTERING_FUNCTIONS
from energizer.active_learning.strategies.base import ActiveEstimator, PoolBasedMixin
from energizer.enums import OutputKeys, SpecialKeys
from energizer.types import METRIC


class DiversityBasedStrategy(ABC, ActiveEstimator):
    """This does not run on pool.

    Here for now, but usually even diversity-based require running on the pool.
    """

    rng: RandomState

    def __init__(self, *args, seed: int = 42, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.rng = check_random_state(seed)  # reproducibility

    def run_query(
        self,
        model: _FabricModule,
        datastore: ActiveDatastore,
        query_size: int,
        **kwargs,
    ) -> List[int]:
        embeddings_and_ids = self.get_embeddings_and_ids(model, datastore, query_size, **kwargs)
        if embeddings_and_ids is None:
            return []
        else:
            embeddings, ids = embeddings_and_ids
        return self.select_from_embeddings(model, datastore, query_size, embeddings, ids, **kwargs)

    @abstractmethod
    def get_embeddings_and_ids(
        self,
        model: _FabricModule,
        datastore: ActiveDatastore,
        query_size: int,
        **kwargs,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        # NOTE: Always need the ids because you might not return the entire pool
        ...

    @abstractmethod
    def select_from_embeddings(
        self,
        model: _FabricModule,
        datastore: ActiveDatastore,
        query_size: int,
        embeddings: np.ndarray,
        ids: np.ndarray,
        **kwargs,
    ) -> List[int]:
        ...


class DiversityBasedStrategyWithPool(PoolBasedMixin, DiversityBasedStrategy):
    POOL_OUTPUT_KEY: OutputKeys = OutputKeys.EMBEDDINGS

    def get_embeddings_and_ids(
        self,
        model: _FabricModule,
        datastore: ActiveDatastore,
        query_size: int,
        **kwargs,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        pool_loader = self.get_pool_loader(datastore, **kwargs)
        if pool_loader is not None and len(pool_loader.dataset or []) > query_size:  # type: ignore
            # enough instances
            out = self.run_pool_evaluation(model, pool_loader)
            return out[self.POOL_OUTPUT_KEY], out[SpecialKeys.ID]


class ClusteringMixin:
    def __init__(
        self,
        *args,
        clustering_fn: Literal["kmeans_sampling", "kmeans_silhouette_sampling", "kmeans_pp_sampling"],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.clustering_fn = CLUSTERING_FUNCTIONS[clustering_fn]

    def select_from_embeddings(
        self,
        model: _FabricModule,
        datastore: ActiveDatastore,
        query_size: int,
        embeddings: np.ndarray,
        ids: np.ndarray,
        **kwargs,
    ) -> List[int]:
        center_ids = self.clustering_fn(embeddings, query_size, rng=self.rng)  # type: ignore
        return ids[center_ids].tolist()


class EmbeddingClustering(ClusteringMixin, DiversityBasedStrategy):
    ...


class PoolBasedEmbeddingClustering(ClusteringMixin, DiversityBasedStrategyWithPool):
    @abstractmethod
    def pool_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: Optional[METRIC],
    ) -> torch.Tensor:
        """This needs to return the embedded batch."""
        ...


# class GreedyCoreset(DiversityBasedStrategyWithPool):
#     def __init__(self, *args, distance_metric: Literal["euclidean", "cosine"], normalize: bool = True, batch_size: int = 100, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.distance_metric = distance_metric
#         self.normalize = normalize
#         self.batch_size = batch_size
