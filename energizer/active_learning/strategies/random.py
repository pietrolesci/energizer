from typing import List

from lightning.fabric.wrappers import _FabricModule
from numpy.random import RandomState

# https://scikit-learn.org/stable/developers/develop.html#random-numbers
from sklearn.utils.validation import check_random_state

from energizer.active_learning.datastores.base import ActiveDataStore
from energizer.active_learning.strategies.base import ActiveEstimator


class RandomStrategy(ActiveEstimator):
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
        return datastore.sample_from_pool(size=query_size, random_state=self.rng)
