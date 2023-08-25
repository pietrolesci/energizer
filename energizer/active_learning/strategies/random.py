from typing import List

# https://scikit-learn.org/stable/developers/develop.html#random-numbers
from sklearn.utils.validation import check_random_state

from energizer.active_learning.strategies.base import ActiveEstimator
from energizer.active_learning.datastores.base import ActiveDataStore
from lightning.fabric.wrappers import _FabricModule, _FabricDataLoader


class RandomStrategy(ActiveEstimator):
    def __init__(self, *args, seed: int = 42, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.rng = check_random_state(seed)  # reproducibility

    def run_query(
        self, model: _FabricModule, loader: _FabricDataLoader, query_size: int, datastore: ActiveDataStore
    ) -> List[int]:
        return datastore.sample_from_pool(size=query_size, mode="uniform", random_state=self.rng)
