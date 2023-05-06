from typing import List
from h11 import Data
import numpy as np

# https://scikit-learn.org/stable/developers/develop.html#random-numbers
from sklearn.utils.validation import check_random_state

from energizer.datastores.base import Datastore
from energizer.estimators.active_estimator import ActiveEstimator


class RandomStrategy(ActiveEstimator):
    def __init__(self, *args, seed: int = 42, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.rng = check_random_state(seed)  # reproducibility

    def run_query(self, *_, datastore: Datastore, query_size: int) -> List[int]:
        return datastore.sample_from_pool(size=query_size, mode="uniform", random_state=self.rng)


class RandomStrategySEALS(RandomStrategy):
    def __init__(self, *args, num_neighbours: int = 100, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_neighbours = num_neighbours
        self.to_search = []
        self.pool_subset_ids = []

    def run_query(self, *_, datastore: Datastore, query_size: int) -> List[int]:
        
        if len(self.to_search) == 0:
            self.to_search = datastore.get_train_ids()  # type: ignore

        with_indices = None
        if len(self.to_search) > 0:
                
            # get the embeddings of the labelled instances
            train_embeddings = datastore.get_embeddings(self.to_search)  # type: ignore

            # get neighbours of training instances from the pool
            nn_ids, _ = datastore.search(  # type: ignore
                query=train_embeddings, query_size=self.num_neighbours, query_in_set=False
            )
            nn_ids = np.unique(np.concatenate(nn_ids).flatten()).tolist()
            
            with_indices = list(set(nn_ids + self.pool_subset_ids))
            self.pool_subset_ids = with_indices

        annotated_ids = datastore.sample_from_pool(size=query_size, mode="uniform", random_state=self.rng, with_indices=with_indices)
        self.to_search = annotated_ids  # to search in the next round
        
        return annotated_ids