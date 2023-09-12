from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

import numpy as np
from lightning.fabric.wrappers import _FabricModule
from numpy.random import RandomState
from sklearn.utils.validation import check_random_state

from energizer.active_learning.datastores.base import ActiveDataStore
from energizer.active_learning.strategies.base import ActiveEstimator


class DiversityBasedStrategy(ABC, ActiveEstimator):
    rng: RandomState

    def __init__(self, *args, seed: int = 42, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.rng = check_random_state(seed)  # reproducibility

    def run_query(
        self,
        model: _FabricModule,
        datastore: ActiveDataStore,
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
        datastore: ActiveDataStore,
        query_size: int,
        **kwargs,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        # NOTE: Always need the ids because you might not return the entire pool
        ...

    @abstractmethod
    def select_from_embeddings(
        self,
        model: _FabricModule,
        datastore: ActiveDataStore,
        query_size: int,
        embeddings: np.ndarray,
        ids: np.ndarray,
        **kwargs,
    ) -> List[int]:
        ...
