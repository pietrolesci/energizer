from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from lightning.fabric.wrappers import _FabricModule
from numpy.random import RandomState
from sklearn.utils import check_random_state

from energizer.active_learning.datastores.base import ActiveDatastore, ActiveDatastoreWithIndex
from energizer.active_learning.strategies.base import ActiveLearningStrategy
from energizer.enums import SpecialKeys


class BaseSubsetStrategy(ABC, ActiveLearningStrategy):
    """These strategies are applied in conjunction with a base query strategy. If the size of the pool
    falls below the given `k`, this implementation will not select a subset anymore and will just delegate
    to the base strategy instead.
    """

    rng: RandomState
    _subpool_size: int
    _base_strategy: ActiveLearningStrategy

    def __init__(self, base_strategy: ActiveLearningStrategy, subpool_size: int, seed: int = 42) -> None:
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
    def base_strategy(self) -> ActiveLearningStrategy:
        return self._base_strategy

    def run_query(self, model: _FabricModule, datastore: ActiveDatastore, query_size: int, **kwargs) -> list[int]:
        if datastore.pool_size() > self.subpool_size:
            subpool_ids = self.select_pool_subset(model, datastore, query_size, **kwargs)
            kwargs["subpool_ids"] = subpool_ids

        return self.base_strategy.run_query(model, datastore, query_size, **kwargs)

    @abstractmethod
    def select_pool_subset(
        self, model: _FabricModule, datastore: ActiveDatastore, query_size: int, **kwargs
    ) -> list[int]:
        ...

    def __getattr__(self, attr: str) -> Any:
        if attr not in self.__dict__:
            return getattr(self.base_strategy, attr)
        return getattr(self, attr)


class RandomSubsetStrategy(BaseSubsetStrategy):
    def select_pool_subset(
        self, model: _FabricModule, datastore: ActiveDatastore, query_size: int, **kwargs
    ) -> list[int]:
        subpool_size = min(datastore.pool_size(), self.subpool_size)
        return datastore.sample_from_pool(size=subpool_size, random_state=self.rng)


class BaseSubsetWithSearchStrategy(BaseSubsetStrategy):
    def __init__(self, *args, num_neighbours: int, max_search_size: int | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_neighbours = num_neighbours
        self.max_search_size = max_search_size

    def select_pool_subset(
        self, model: _FabricModule, datastore: ActiveDatastoreWithIndex, query_size: int, **kwargs
    ) -> list[int]:
        # SELECT QUERIES
        search_query_ids = self.select_search_query(model, datastore, query_size, **kwargs)

        if len(search_query_ids) == 0:
            # if cold-starting there is no training embedding, fall-back to random sampling
            return datastore.sample_from_pool(size=query_size, random_state=self.rng)

        search_query_embeddings = self.get_query_embeddings(datastore, search_query_ids)

        # SEARCH
        candidate_df = self.search_pool(datastore, search_query_embeddings, search_query_ids)

        # USE RESULTS TO SUBSET POOL
        return self.get_subpool_ids_from_search_results(candidate_df, datastore)

    def search_pool(
        self, datastore: ActiveDatastoreWithIndex, search_query_embeddings: np.ndarray, search_query_ids: list[int]
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

    @abstractmethod
    def select_search_query(
        self, model: _FabricModule, datastore: ActiveDatastore, query_size: int, **kwargs
    ) -> list[int]:
        ...

    @abstractmethod
    def get_query_embeddings(self, datastore: ActiveDatastoreWithIndex, search_query_ids: list[int]) -> np.ndarray:
        ...

    @abstractmethod
    def get_subpool_ids_from_search_results(
        self, candidate_df: pd.DataFrame, datastore: ActiveDatastoreWithIndex
    ) -> list[int]:
        ...


class SEALSStrategy(BaseSubsetWithSearchStrategy):
    """Colemann et al. (2020) `Similarity Search for Efficient Active Learning and Search of Rare Concepts`."""

    to_search: list[int] = []
    subpool_ids: list[int] = []

    def run_query(self, model: _FabricModule, datastore: ActiveDatastore, query_size: int, **kwargs) -> list[int]:
        annotated_ids = super().run_query(model, datastore, query_size, **kwargs)

        # in the next round we only need to search the newly labelled data
        self.to_search = annotated_ids

        # make sure to remove instances that might have been annotated
        self.subpool_ids = [i for i in self.subpool_ids if i not in annotated_ids]
        return annotated_ids

    def select_search_query(
        self, model: _FabricModule, datastore: ActiveDatastore, query_size: int, **kwargs
    ) -> list[int]:
        if len(self.to_search) < 1:
            return datastore.get_train_ids()
        # we only search the newly labelled data at the previous round
        return self.to_search

    def get_query_embeddings(self, datastore: ActiveDatastoreWithIndex, search_query_ids: list[int]) -> np.ndarray:
        # queries are always from the training set
        return datastore.get_train_embeddings(search_query_ids)

    def get_subpool_ids_from_search_results(
        self, candidate_df: pd.DataFrame, datastore: ActiveDatastoreWithIndex
    ) -> list[int]:
        # we return all unique instances from the pool that are neighbours of the training set
        selected_ids = candidate_df[SpecialKeys.ID].unique().tolist()

        # in this round we only used the newly labelled data as search queries
        # so we need to add the previously queried samples too!
        self.subpool_ids = list(set(selected_ids + self.subpool_ids))
        return self.subpool_ids
