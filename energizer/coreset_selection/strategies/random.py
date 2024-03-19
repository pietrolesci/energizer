from lightning.fabric.wrappers import _FabricModule
from numpy.random import RandomState

# https://scikit-learn.org/stable/developers/develop.html#random-numbers
from sklearn.utils.validation import check_random_state

from energizer.coreset_selection.strategies.base import CoresetSelectionStrategy
from energizer.datastores import Datastore


class RandomStrategy(CoresetSelectionStrategy):
    rng: RandomState

    def __init__(self, *args, seed: int = 42, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.rng = check_random_state(seed)  # reproducibility

    def run_query(self, model: _FabricModule, datastore: Datastore, query_size: int, **kwargs) -> list[int]:
        return datastore.sample_from_pool(size=query_size, random_state=self.rng)
