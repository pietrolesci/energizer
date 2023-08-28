from typing import Callable, Dict, List, Optional, Union

import numpy as np
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule
from numpy.random import RandomState
from sklearn.utils import check_random_state

from energizer.active_learning.datastores.base import ActiveDataStore, ActiveDataStoreWithIndex
from energizer.active_learning.registries import CLUSTERING_FUNCTIONS
from energizer.active_learning.strategies.diversity import DiversitySamplingMixin
from energizer.active_learning.strategies.two_stage import BaseSubsetStrategy, SEALSStrategy, RandomSubsetStrategy
from energizer.active_learning.strategies.uncertainty import UncertaintyBasedStrategy


class RandomSubset(RandomSubsetStrategy):
    """Ertekin et al. (2007) `Learning on the border: Active learning in imbalanced data classification`.

    Two stages:

        1. Random subset of the pool

        2. Uncertainty sampling
    """

    def __init__(self, *args, score_fn: str, subpool_size: int, seed: int = 42, **kwargs) -> None:
        base_strategy = UncertaintyBasedStrategy(*args, score_fn=score_fn, **kwargs)
        super().__init__(base_strategy, subpool_size, seed)

    def select_pool_subset(
        self, model: _FabricModule, loader: _FabricDataLoader, datastore: ActiveDataStore, **kwargs
    ) -> List[int]:
        subpool_size = min(datastore.pool_size(), self.subpool_size)
        return datastore.sample_from_pool(size=subpool_size, random_state=self.rng)


class Tyrogue(DiversitySamplingMixin, BaseSubsetStrategy):
    """Maekawa et al. (2022) `Low-Resource Interactive Active Labeling for Fine-tuning Language Models`.

    Three-step strategy that:

        1. Gets random subset from the pool

        2. Gets a further subset by diversity sampling

        3. Runs uncertainty sampling on the selected subset of the pool
    """

    clustering_rng: RandomState

    _r_factor: int
    _clustering_fn: Callable

    def __init__(
        self,
        *args,
        score_fn: Union[str, Callable],
        subpool_size: int,
        seed: int = 42,
        r_factor: int,
        clustering_algorithm: str = "kmeans",
        clustering_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        """Strategy that runs uncertainty sampling on a random subset of the pool.

        Args:
            subpool_size (int): Size of the first random subset. The subpool size is determined by `r_factor * query_size`.
            seed (int): Random seed for the subset selection.
        """
        base_strategy = UncertaintyBasedStrategy(*args, score_fn=score_fn, **kwargs)
        super().__init__(base_strategy, subpool_size, seed)

        self._r_factor = r_factor
        self._clustering_fn = CLUSTERING_FUNCTIONS[clustering_algorithm]
        self.clustering_kwargs = clustering_kwargs or {}
        self.clustering_rng = check_random_state(self.clustering_kwargs.get("seed", self.seed))

    @property
    def r_factor(self) -> int:
        return self._r_factor

    @property
    def clustering_fn(self) -> Callable:
        return self._clustering_fn

    def select_pool_subset(
        self, model: _FabricModule, loader: _FabricDataLoader, datastore: ActiveDataStoreWithIndex, **kwargs
    ) -> List[int]:
        subpool_size = min(datastore.pool_size(), self.subpool_size)
        random_subpool_ids = datastore.sample_from_pool(size=subpool_size, random_state=self.rng)

        embeddings = self.get_embeddings(model, loader, datastore, random_subpool_ids=random_subpool_ids)

        query_size: int = kwargs["query_size"]
        num_clusters = query_size * self.r_factor

        return self.select_from_embeddings(embeddings, num_clusters=num_clusters)

    def get_embeddings(
        self, model: _FabricModule, loader: _FabricDataLoader, datastore: ActiveDataStoreWithIndex, **kwargs
    ) -> np.ndarray:
        random_subpool_ids: List[int] = kwargs["random_subpool_ids"]
        return datastore.get_pool_embeddings(random_subpool_ids)

    def select_from_embeddings(self, embeddings: np.ndarray, **kwargs) -> List[int]:
        num_clusters: int = kwargs["num_clusters"]
        return self.clustering_fn(embeddings, num_clusters, rng=self.clustering_rng, **self.clustering_kwargs)


class SEALS(SEALSStrategy):
    """Published instantiation of the SEALS strategy with uncertainty sampling."""

    def __init__(
        self,
        *args,
        score_fn: Union[str, Callable],
        subpool_size: int,
        seed: int = 42,
        num_neighbours: int,
        max_search_size: Optional[int] = None,
        **kwargs,
    ) -> None:
        base_strategy = UncertaintyBasedStrategy(*args, score_fn=score_fn, **kwargs)
        super().__init__(
            base_strategy=base_strategy,
            subpool_size=subpool_size,
            seed=seed,
            num_neighbours=num_neighbours,
            max_search_size=max_search_size,
        )
