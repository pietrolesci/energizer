from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule
from numpy.random import RandomState
from sklearn.utils.validation import check_random_state
from torch.nn.functional import one_hot

from energizer.active_learning.clustering_utilities import kmeans_pp_sampling
from energizer.active_learning.datastores.base import ActiveDataStore
from energizer.active_learning.strategies.base import ActiveEstimator, PoolBasedStrategyMixin
from energizer.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from energizer.types import METRIC
from energizer.utilities import ld_to_dl, move_to_cpu


class DiversityBasedStrategy(ABC, ActiveEstimator):
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


class BADGE(PoolBasedStrategyMixin, DiversityBasedStrategy):
    def get_embeddings_and_ids(
        self,
        model: _FabricModule,
        datastore: ActiveDataStore,
        query_size: int,
        **kwargs,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        pool_loader = self.get_pool_loader(datastore, **kwargs)
        if pool_loader is not None and len(pool_loader or []) > query_size:
            # enough instances
            return self.compute_gradient_embeddings(model, pool_loader)

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

    def compute_gradient_embeddings(
        self, model: _FabricModule, loader: _FabricDataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        # NOTE: this is similar to `UncertaintyBasedStrategy.compute_most_uncertain`
        out: List[Dict] = self.run_evaluation(model, loader, RunningStage.POOL)  # type: ignore
        _out = ld_to_dl(out)

        grads = np.concatenate(_out[OutputKeys.GRAD])
        uids = np.concatenate(_out[SpecialKeys.ID])

        return grads, uids

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
        return {
            OutputKeys.GRAD: move_to_cpu(pool_out),
            SpecialKeys.ID: ids,
        }  # enforce that we always return a dict here

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

    @abstractmethod
    def get_penultimate_layer_out(self, model: _FabricModule, batch: Any) -> torch.Tensor:
        ...

    @abstractmethod
    def get_logits_from_penultimate_layer_out(
        self, model: _FabricModule, penultimate_layer_out: torch.Tensor
    ) -> torch.Tensor:
        ...
