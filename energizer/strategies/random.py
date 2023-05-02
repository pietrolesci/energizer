from typing import List, Optional

# https://scikit-learn.org/stable/developers/develop.html#random-numbers
from sklearn.utils.validation import check_random_state

from energizer.datastores.base import Datastore
from energizer.estimators.active_estimator import ActiveEstimator


class RandomStrategy(ActiveEstimator):
    def __init__(self, *args, seed: Optional[int], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.rng = check_random_state(seed)  # reproducibility

    def run_query(self, *_, datastore: Datastore, query_size: int) -> List[int]:
        return datastore.sample_from_pool(size=query_size, mode="uniform", random_state=self.rng)
