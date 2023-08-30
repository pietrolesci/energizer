from typing import Any, List, Optional

import numpy as np
import pandas as pd
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule
from numpy.random import RandomState
from sklearn.utils import check_random_state

from energizer.active_learning.datastores.base import ActiveDataStore, ActiveDataStoreWithIndex
from energizer.active_learning.strategies.base import ActiveEstimator
from energizer.enums import SpecialKeys


class BaseSubsetStrategy(ActiveEstimator):
    """These strategies are applied in conjunction with a base query strategy. If the size of the pool
    falls below the given `k`, this implementation will not select a subset anymore and will just delegate
    to the base strategy instead.
    """

    rng: RandomState
    _subpool_size: int
    _base_strategy: ActiveEstimator

    def __init__(self, base_strategy: ActiveEstimator, subpool_size: int, seed: int = 42) -> None:
        """Strategy that runs uncertainty sampling on a random subset of the pool.

        Args:
            subpool_size (int): Size of the subset.
            seed (int): Random seed for the subset selection.
        """
        self.seed = seed
        self.rng = check_random_state(seed)
        self._subpool_size = subpool_size
        self._base_strategy = base_strategy

    @property
    def subpool_size(self) -> int:
        return self._subpool_size

    @property
    def base_strategy(self) -> ActiveEstimator:
        return self._base_strategy

    def run_query(
        self,
        model: _FabricModule,
        loader: _FabricDataLoader,
        datastore: ActiveDataStore,
        query_size: int,
    ) -> List[int]:
        if datastore.pool_size() > self.subpool_size:
            subpool_ids = self.select_pool_subset(model, loader, datastore, query_size=query_size)
            loader = self._get_subpool_loader_by_ids(datastore, subpool_ids)

        return self.base_strategy.run_query(model, loader, datastore, query_size)

    def select_pool_subset(
        self, model: _FabricModule, loader: _FabricDataLoader, datastore: ActiveDataStore, **kwargs
    ) -> List[int]:
        raise NotImplementedError

    def _get_subpool_loader_by_ids(self, datastore: ActiveDataStore, subpool_ids: List[int]) -> _FabricDataLoader:
        pool_loader = self.configure_dataloader(datastore.pool_loader(with_indices=subpool_ids))  # type: ignore
        self.tracker.pool_tracker.max = len(pool_loader)  # type: ignore
        return pool_loader  # type: ignore

    def __getattr__(self, attr: str) -> Any:
        if attr not in self.__dict__:
            return getattr(self.base_strategy, attr)
        return getattr(self, attr)


class RandomSubsetStrategy(BaseSubsetStrategy):
    def select_pool_subset(
        self, model: _FabricModule, loader: _FabricDataLoader, datastore: ActiveDataStore, **kwargs
    ) -> List[int]:
        subpool_size = min(datastore.pool_size(), self.subpool_size)
        return datastore.sample_from_pool(size=subpool_size, random_state=self.rng)


class BaseSubsetWithSearchStrategy(BaseSubsetStrategy):
    def __init__(self, *args, num_neighbours: int, max_search_size: Optional[int] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_neighbours = num_neighbours
        self.max_search_size = max_search_size

    def select_pool_subset(
        self, model: _FabricModule, loader: _FabricDataLoader, datastore: ActiveDataStoreWithIndex, **kwargs
    ) -> List[int]:
        query_size: int = kwargs["query_size"]

        # SELECT QUERIES
        search_query_ids = self.select_search_query(model, loader, datastore, query_size=query_size)

        if len(search_query_ids) == 0:
            # if cold-starting there is no training embedding, fall-back to random sampling
            return datastore.sample_from_pool(size=query_size, random_state=self.rng)

        search_query_embeddings = self.get_query_embeddings(datastore, search_query_ids)

        # SEARCH
        candidate_df = self.search_pool(datastore, search_query_embeddings, search_query_ids)

        # USE RESULTS TO SUBSET POOL
        return self.get_subpool_from_search_results(candidate_df, datastore)

    def select_search_query(
        self, model: _FabricModule, loader: _FabricDataLoader, datastore: ActiveDataStoreWithIndex, **kwargs
    ) -> List[int]:
        raise NotImplementedError

    def get_query_embeddings(self, datastore: ActiveDataStoreWithIndex, search_query_ids: List[int]) -> np.ndarray:
        raise NotImplementedError

    def search_pool(
        self, datastore: ActiveDataStoreWithIndex, search_query_embeddings: np.ndarray, search_query_ids: List[int]
    ) -> pd.DataFrame:
        num_neighbours = self.num_neighbours
        if self.max_search_size is not None:
            # NOTE: this can create noise in the experimentss
            num_neighbours = int(
                min(self.num_neighbours, np.floor(self.max_search_size / search_query_embeddings.shape[0]))
            )

        ids, dists = datastore.search(query=search_query_embeddings, query_size=num_neighbours, query_in_set=False)
        candidate_df = pd.DataFrame(
            {
                SpecialKeys.ID: ids.flatten(),
                "dists": dists.flatten(),
                "search_query_uid": np.repeat(search_query_ids, ids.shape[1], axis=0).flatten(),
            }
        )

        return candidate_df

    def get_subpool_from_search_results(
        self, candidate_df: pd.DataFrame, datastore: ActiveDataStoreWithIndex
    ) -> List[int]:
        raise NotImplementedError


class SEALSStrategy(BaseSubsetWithSearchStrategy):
    """Colemann et al. (2020) `Similarity Search for Efficient Active Learning and Search of Rare Concepts`."""

    to_search: List[int] = []
    subpool_ids: List[int] = []

    def select_pool_subset(
        self, model: _FabricModule, loader: _FabricDataLoader, datastore: ActiveDataStoreWithIndex, **kwargs
    ) -> List[int]:
        selected_ids = super().select_pool_subset(model, loader, datastore, **kwargs)
        self.to_search += selected_ids
        self.subpool_ids = [i for i in self.subpool_ids if i not in selected_ids]
        return selected_ids

    def select_search_query(
        self, model: _FabricModule, loader: _FabricDataLoader, datastore: ActiveDataStoreWithIndex, **kwargs
    ) -> List[int]:
        if len(self.to_search) < 1:
            return datastore.get_train_ids()
        return self.to_search

    def get_subpool_from_search_results(
        self, candidate_df: pd.DataFrame, datastore: ActiveDataStoreWithIndex
    ) -> List[int]:
        self.subpool_ids += candidate_df[SpecialKeys.ID].unique().tolist()
        return list(set(self.subpool_ids))

    def get_query_embeddings(self, datastore: ActiveDataStoreWithIndex, search_query_ids: List[int]) -> np.ndarray:
        return datastore.get_train_embeddings(search_query_ids)
