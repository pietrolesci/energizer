from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule
from numpy.random import RandomState
from sklearn.utils.validation import check_random_state

from energizer.active_learning.datastores.base import ActiveDataStore
from energizer.active_learning.registries import CLUSTERING_FUNCTIONS
from energizer.active_learning.strategies.base import ActiveEstimator, PoolBasedStrategyMixin
from energizer.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from energizer.types import METRIC
from energizer.utilities import ld_to_dl, move_to_cpu


class DiversityBasedStrategy(ABC, ActiveEstimator):
    """This does not run on pool.

    Here for now, but usually even diversity-based require running on the pool.
    """

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


class DiversityBasedStrategyWithPool(PoolBasedStrategyMixin, DiversityBasedStrategy):
    def get_embeddings_and_ids(
        self,
        model: _FabricModule,
        datastore: ActiveDataStore,
        query_size: int,
        **kwargs,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        pool_loader = self.get_pool_loader(datastore, **kwargs)
        if pool_loader is not None and len(pool_loader.dataset or []) > query_size:  # type: ignore
            # enough instances
            return self.compute_pool_embeddings(model, pool_loader)

    def compute_pool_embeddings(self, model: _FabricModule, loader: _FabricDataLoader) -> Tuple[np.ndarray, np.ndarray]:
        # NOTE: this is similar to `UncertaintyBasedStrategy.compute_most_uncertain`
        out: List[Dict] = self.run_evaluation(model, loader, RunningStage.POOL)  # type: ignore
        _out = ld_to_dl(out)

        embs = np.concatenate(_out[OutputKeys.EMBEDDINGS])
        uids = np.concatenate(_out[SpecialKeys.ID])

        return embs, uids

    def evaluation_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: Optional[METRIC],
        stage: Union[str, RunningStage],
    ) -> Dict:
        if stage != RunningStage.POOL:
            return super().evaluation_step(model, batch, batch_idx, loss_fn, metrics, stage)  # type: ignore

        # keep IDs here in case user messes up in the function definition
        ids = batch[InputKeys.ON_CPU][SpecialKeys.ID]
        pool_out = self.pool_step(model, batch, batch_idx, loss_fn, metrics)

        assert isinstance(pool_out, torch.Tensor), "`pool_step` must return the gradient tensor`."

        # enforce that we always return a dict here
        return {OutputKeys.EMBEDDINGS: move_to_cpu(pool_out), SpecialKeys.ID: ids}


class EmbeddingClustering(DiversityBasedStrategyWithPool):
    def __init__(
        self,
        *args,
        clustering_fn: Literal["kmeans_sampling", "kmeans_silhouette_sampling", "kmeans_pp_sampling"],
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.clustering_fn = CLUSTERING_FUNCTIONS[clustering_fn]

    def select_from_embeddings(
        self,
        model: _FabricModule,
        datastore: ActiveDataStore,
        query_size: int,
        embeddings: np.ndarray,
        ids: np.ndarray,
        **kwargs,
    ) -> List[int]:
        center_ids = self.clustering_fn(embeddings, query_size, rng=self.rng)
        return ids[center_ids].tolist()

    def pool_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: Optional[METRIC],
    ) -> torch.Tensor:
        return self.get_model_embeddings(model, batch)

    @abstractmethod
    def get_model_embeddings(self, model: _FabricModule, batch: Any) -> torch.Tensor:
        ...


# class GreedyCoreset(DiversityBasedStrategyWithPool):
#     def __init__(self, *args, distance_metric: Literal["euclidean", "cosine"], normalize: bool = True, batch_size: int = 100, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.distance_metric = distance_metric
#         self.normalize = normalize
#         self.batch_size = batch_size
