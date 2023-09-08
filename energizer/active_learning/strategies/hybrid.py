from typing import Callable, Dict, List, Optional, Union

import numpy as np
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule
from numpy.random import RandomState
from sklearn.utils import check_random_state

from energizer.active_learning.datastores.base import ActiveDataStoreWithIndex
from energizer.active_learning.registries import CLUSTERING_FUNCTIONS
from energizer.active_learning.strategies.diversity import DiversitySamplingMixin
from energizer.active_learning.strategies.uncertainty import UncertaintyBasedStrategy


class Tyrogue(DiversitySamplingMixin, UncertaintyBasedStrategy):
    """Maekawa et al. (2022) `Low-Resource Interactive Active Labeling for Fine-tuning Language Models`.

    Three-step strategy that:

        1. Gets random subset from the pool (this is not done here!!!)

        2. Gets a further subset by diversity sampling

        3. Runs uncertainty sampling on the selected subset of the pool

    If you want to reproduce their results you need to add random subsampling by creating a two_stage
    strategy instance with `RandomSubsetStrategy`.
    """

    clustering_rng: RandomState

    _r_factor: int
    _clustering_fn: Callable

    def __init__(
        self,
        *args,
        score_fn: Union[str, Callable],
        seed: int = 42,
        r_factor: int,
        clustering_algorithm: str = "kmeans",
        clustering_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, score_fn=score_fn, seed=seed, **kwargs)
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

        embeddings = self._get_embeddings(datastore, **kwargs)

        query_size: int = kwargs["query_size"]
        num_clusters = query_size * self.r_factor

        return self.select_from_embeddings(embeddings, num_clusters=num_clusters)

    def select_from_embeddings(self, embeddings: np.ndarray, **kwargs) -> List[int]:
        num_clusters: int = kwargs["num_clusters"]
        return self.clustering_fn(embeddings, num_clusters, rng=self.clustering_rng, **self.clustering_kwargs)

    def _get_embeddings(self, datastore: ActiveDataStoreWithIndex, **kwargs) -> np.ndarray:
        pool_ids = kwargs.get("pool_ids", None) or datastore.get_pool_ids()
        return datastore.get_pool_embeddings(pool_ids)
