from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from lightning.fabric.wrappers import _FabricModule
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.nn.functional import one_hot

from energizer.active_learning.clustering_utilities import kmeans_pp_sampling
from energizer.active_learning.datastores.base import ActiveDataStore, ActiveDataStoreWithIndex
from energizer.active_learning.registries import CLUSTERING_FUNCTIONS
from energizer.active_learning.strategies.diversity import DiversityBasedStrategy, DiversityBasedStrategyWithPool
from energizer.active_learning.strategies.uncertainty import UncertaintyBasedStrategy
from energizer.types import METRIC


class Tyrogue(DiversityBasedStrategy, UncertaintyBasedStrategy):
    """Maekawa et al. (2022) `Low-Resource Interactive Active Labeling for Fine-tuning Language Models`.

    Three-step strategy that:

        1. Gets random subset from the pool (this is not done here!!!)

        2. Gets a further subset by diversity sampling

        3. Runs uncertainty sampling on the selected subset of the pool

    If you want to reproduce their results you need to add random subsampling by creating a two_stage
    strategy instance with `RandomSubsetStrategy`.
    """

    clustering_rng: RandomState

    _r_factor: int
    _clustering_fn: Callable

    def __init__(
        self,
        *args,
        score_fn: Union[str, Callable],
        seed: int = 42,
        r_factor: int,
        clustering_algorithm: str = "kmeans",
        clustering_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, score_fn=score_fn, seed=seed, **kwargs)
        self._r_factor = r_factor
        self._clustering_fn = CLUSTERING_FUNCTIONS[clustering_algorithm]
        self.clustering_kwargs = clustering_kwargs or {}
        self.clustering_rng = check_random_state(self.clustering_kwargs.get("seed", self.seed))

    @property
    def r_factor(self) -> int:
        return self._r_factor

    @property
    def clustering_fn(self) -> Callable:
        return self._clustering_fn

    def run_query(
        self,
        model: _FabricModule,
        datastore: ActiveDataStoreWithIndex,
        query_size: int,
        **kwargs,
    ) -> List[int]:
        # === DIVERSITY === #
        embeddings_and_ids = self.get_embeddings_and_ids(model, datastore, query_size, **kwargs)
        if embeddings_and_ids is None:
            return []
        else:
            embeddings, ids = embeddings_and_ids

        subpool_ids = self.select_from_embeddings(model, datastore, query_size, embeddings, ids, **kwargs)

        # === UNCERTAINTY === #
        pool_loader = self.get_pool_loader(datastore, subpool_ids=subpool_ids, **kwargs)
        if pool_loader is None or len(pool_loader.dataset or []) <= query_size:  # type: ignore
            # not enough instances
            return []
        return self.compute_most_uncertain(model, pool_loader, query_size)

    def get_embeddings_and_ids(
        self,
        model: _FabricModule,
        datastore: ActiveDataStoreWithIndex,
        query_size: int,
        **kwargs,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        pool_ids = kwargs.get("subpool_ids", None) or datastore.get_pool_ids()
        return datastore.get_pool_embeddings(pool_ids), np.array(pool_ids)

    def select_from_embeddings(
        self,
        model: _FabricModule,
        datastore: ActiveDataStoreWithIndex,
        query_size: int,
        embeddings: np.ndarray,
        ids: np.ndarray,
        **kwargs,
    ) -> List[int]:
        num_clusters = query_size * self.r_factor

        centers_ids = self.clustering_fn(embeddings, num_clusters, rng=self.clustering_rng, **self.clustering_kwargs)

        return ids[centers_ids].tolist()


class BADGE(DiversityBasedStrategyWithPool):
    def select_from_embeddings(
        self,
        model: _FabricModule,
        datastore: ActiveDataStore,
        query_size: int,
        embeddings: np.ndarray,
        ids: np.ndarray,
        **kwargs,
    ) -> List[int]:
        # k-means++ sampling
        center_ids = kmeans_pp_sampling(embeddings, query_size, rng=self.rng)
        return ids[center_ids].tolist()

    def pool_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: Optional[METRIC],
    ) -> torch.Tensor:
        r"""Return the loss gradient with respect to the penultimate layer of the model.

        Uses the analytical form from the paper

            $g(x)_i = ( f(x; \theta)_i - \mathbf{1}(\hat{y} = i) ) h(x; W)$

        Refs for the implementation:
        https://github.com/forest-snow/alps/blob/3c7ef2c98249fc975a897b27f275695f97d5b7a9/src/sample.py#L65
        """
        penultimate_layer_out = self.get_penultimate_layer_out(model, batch)
        logits = self.get_logits_from_penultimate_layer_out(model, penultimate_layer_out)
        batch_size, num_classes = logits.size()

        # compute scales
        probs = logits.softmax(dim=-1)
        preds_oh = one_hot(probs.argmax(dim=-1), num_classes=num_classes)
        scales = probs - preds_oh

        # multiply
        grads_3d = torch.einsum("bi,bj->bij", scales, penultimate_layer_out)
        return grads_3d.view(batch_size, -1)  # (batch_size,)

    def get_penultimate_layer_out(self, model: _FabricModule, batch: Any) -> torch.Tensor:
        raise NotImplementedError("Either implement `get_penultimate_layer_out` of `pool_step` directly.")

    def get_logits_from_penultimate_layer_out(
        self, model: _FabricModule, penultimate_layer_out: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError("Either implement `get_logits_from_penultimate_layer_out` of `pool_step` directly.")


# class ContrastiveActiveLearning(DiversityBasedStrategy, UncertaintyBasedStrategy):

#     train_embeddings
#     scores = []
#     for p in pool:
#         ids = knn(p, train_embeddings)
#         train_instances = train_embeddings[ids,:]

#         p_probs = model(p)
#         kls = []
#         for batch in train_instances:
#             probs = model(batch)
#             kls += KL(p_probs, probs)

#         scores.append(kls.mean())

#     topk_ids = scores.argmax()
#     return pool[topk_ids]
