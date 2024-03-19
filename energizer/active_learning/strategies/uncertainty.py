from collections.abc import Callable

from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule
from numpy.random import RandomState
from sklearn.utils.validation import check_random_state

from energizer.active_learning.datastores.base import ActiveDatastore
from energizer.active_learning.registries import SCORING_FUNCTIONS
from energizer.active_learning.strategies.base import ActiveLearningStrategy, PoolBasedMixin
from energizer.enums import OutputKeys, SpecialKeys


class UncertaintyBasedStrategy(PoolBasedMixin, ActiveLearningStrategy):
    # gets the ABC class from PoolBasedMixin
    rng: RandomState
    _score_fn: Callable
    POOL_OUTPUT_KEY: OutputKeys = OutputKeys.SCORES

    def __init__(self, *args, score_fn: str | Callable, seed: int = 42, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.rng = check_random_state(seed)  # reproducibility
        self._score_fn = score_fn if isinstance(score_fn, Callable) else SCORING_FUNCTIONS[score_fn]

    @property
    def score_fn(self) -> Callable:
        return self._score_fn

    def run_query(self, model: _FabricModule, datastore: ActiveDatastore, query_size: int, **kwargs) -> list[int]:
        pool_loader = self.get_pool_loader(datastore, **kwargs)
        if pool_loader is None or len(pool_loader.dataset or []) <= query_size:  # type: ignore
            # not enough instances
            return []
        return self.compute_most_uncertain(model, pool_loader, query_size)

    def compute_most_uncertain(self, model: _FabricModule, loader: _FabricDataLoader, query_size: int) -> list[int]:
        # run evaluation
        out = self.run_pool_evaluation(model, loader)
        scores = out[self.POOL_OUTPUT_KEY]
        ids = out[SpecialKeys.ID]

        # topk
        topk_ids = scores.argsort()[-query_size:]
        return ids[topk_ids].tolist()
