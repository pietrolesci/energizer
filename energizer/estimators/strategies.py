from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule
from sklearn.utils.validation import (
    check_random_state,  # https://scikit-learn.org/stable/developers/develop.html#random-numbers
)
from torch.utils.data import DataLoader

from energizer.datastores.base import ActiveDataModule
from energizer.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from energizer.estimators.active_estimator import ActiveEstimator
from energizer.registries import SCORING_FUNCTIONS
from energizer.types import BATCH_OUTPUT, METRIC
from energizer.utilities import ld_to_dl


class RandomStrategy(ActiveEstimator):
    def __init__(self, *args, seed: Optional[int], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.seed = seed
        self.rng = check_random_state(seed)  # reproducibility

    def run_query(self, *_, active_datamodule: ActiveDataModule, query_size: int) -> List[int]:
        pool_indices = active_datamodule.pool_indices()
        return self.rng.choice(pool_indices, size=query_size, replace=False).tolist()


class UncertaintyBasedStrategy(ActiveEstimator):
    _scoring_fn_registry = SCORING_FUNCTIONS

    def __init__(self, *args, score_fn: Union[str, Callable], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.score_fn = score_fn if isinstance(score_fn, Callable) else self._scoring_fn_registry[score_fn]

    def _compute_most_uncertain(
        self, model: _FabricModule, pool_loader: _FabricDataLoader, query_size: int
    ) -> List[int]:
        # calls the evaluation step that we override
        output = self.run_evaluation(model, pool_loader, RunningStage.POOL)

        # if user does not aggregate, try to automatically convert List[Dict] to Dict[List]
        if not isinstance(output, Dict):
            try:
                output = ld_to_dl(output)

                assert (
                    OutputKeys.SCORES in output
                ), f"The pool output must include the {OutputKeys.SCORES}. Consider updating `pool_step`."
                for key in (OutputKeys.SCORES, SpecialKeys.ID):
                    if isinstance(output[key], List):
                        output[key] = np.concatenate(output.pop(key))
            except:
                raise ValueError(
                    f"The pool output must be a Dict[List], not {type(output)}, consider updating "
                    "`pool_step` and `pool_epoch_end`. Automatic conversion failed."
                )

        # compute topk
        topk_ids = output[OutputKeys.SCORES].argsort()[-query_size:]

        return output[SpecialKeys.ID][topk_ids].tolist()

    def run_query(self, model: _FabricModule, active_datamodule: ActiveDataModule, query_size: int) -> List[int]:
        pool_loader = self.configure_dataloader(active_datamodule.pool_loader())
        return self._compute_most_uncertain(model, pool_loader, query_size)

    def evaluation_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: Optional[METRIC],
        stage: RunningStage,
    ) -> Union[BATCH_OUTPUT, Dict]:
        if stage != RunningStage.POOL:
            return super().evaluation_step(model, batch, batch_idx, loss_fn, metrics, stage)

        ids = batch[InputKeys.ON_CPU][SpecialKeys.ID]
        pool_out = self.pool_step(model, batch, batch_idx, metrics)

        if isinstance(pool_out, torch.Tensor):
            pool_out = {OutputKeys.SCORES: pool_out}
        else:
            assert isinstance(pool_out, dict) and OutputKeys.SCORES in pool_out, (
                "In `pool_step` you must return a Tensor with the scores per each element in the batch "
                f"or a Dict with a '{OutputKeys.SCORES}' key and the Tensor of scores as the value."
            )

        pool_out[SpecialKeys.ID] = np.array(ids)

        return pool_out

    def pool_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        metrics: Optional[METRIC] = None,
    ) -> BATCH_OUTPUT:
        raise NotImplementedError

    def pool_epoch_end(self, output: List[Dict[str, Any]], metrics: Optional[METRIC]) -> Dict[str, List[Any]]:
        return output


class SimilaritySearchStrategy(RandomStrategy):
    MAX_QUERY_SIZE = 50_000

    def __init__(self, *args, inverse: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inverse = inverse

    def run_query(self, model: _FabricModule, active_datamodule: ActiveDataModule, query_size: int) -> List[int]:
        train_embeddings = active_datamodule.get_train_embeddings()

        if not train_embeddings.size > 0:
            # random sample
            return super().run_query(model, active_datamodule=active_datamodule, query_size=query_size)

        train_size = int(train_embeddings.shape[0])
        ids, dists = active_datamodule.search_index(
            train_embeddings,
            query_size=query_size
            if (train_size * query_size) < self.MAX_QUERY_SIZE
            else max(int(train_size / self.MAX_QUERY_SIZE), 1),
            query_in_set=False,
        )

        # remove duplicates and order from smaller to larger distance
        ids, dists = ids.flatten(), dists.flatten()
        _, unique_ids = np.unique(ids, return_index=True)
        ids, dists = ids[unique_ids], dists[unique_ids]

        dists = dists.argsort()[-query_size:] if self.inverse else dists.argsort()[:query_size]
        return ids[dists].tolist()


class SimilaritySearchStrategyWithUncertainty(UncertaintyBasedStrategy):
    def run_query(self, model: _FabricModule, active_datamodule: ActiveDataModule, query_size: int) -> List[int]:
        # TODO: fix when no initial budget
        train_dataset = active_datamodule.train_loader().dataset
        train_loader = active_datamodule.get_loader(RunningStage.POOL, train_dataset)
        self.progress_tracker.pool_tracker.max = len(train_loader)

        train_loader = self.configure_dataloader(train_loader)
        most_uncertain_train_ids = self._compute_most_uncertain(model, train_loader, query_size)

        train_embeddings = active_datamodule.get_embeddings(most_uncertain_train_ids)
        ids, dists = active_datamodule.search_index(train_embeddings, query_size=query_size, query_in_set=False)

        # remove duplicates and order from smaller to larger distance
        ids, dists = ids.flatten(), dists.flatten()
        _, unique_ids = np.unique(ids, return_index=True)
        ids, dists = ids[unique_ids], dists[unique_ids]
        return ids[dists.argsort()[:query_size]].tolist()


"""
Pool subsampling mixins
"""


class RandomPoolSubsamplingMixin:
    subsampling_size: Union[int, float] = None
    subsampling_rng: int = None

    def get_pool_loader(self, active_datamodule: ActiveDataModule) -> DataLoader:
        pool_indices = active_datamodule.pool_indices
        if isinstance(self.subsampling_size, int):
            pool_size = min(self.subsampling_size, len(pool_indices))
        else:
            pool_size = int(self.subsampling_size * len(pool_indices))

        subset_indices = self.subsampling_rng.choice(pool_indices, size=pool_size)
        return active_datamodule.pool_loader(subset_indices)


class SEALSMixin:
    num_neighbours: int

    def get_pool_loader(self, active_datamodule: ActiveDataModule) -> DataLoader:
        # get the embeddings of the instances not labelled
        train_embeddings = active_datamodule.get_train_embeddings()

        # get neighbours of training instances from the pool
        subset_indices, _ = active_datamodule.index.search_index(
            query=train_embeddings, query_size=self.num_neighbours, query_in_set=False
        )
        subset_indices = np.unique(subset_indices.flatten()).tolist()

        return active_datamodule.pool_loader(subset_indices)


"""
Combined strategies
"""


class RandomSubsamplingRandomStrategy(RandomPoolSubsamplingMixin, RandomStrategy):
    def __init__(self, subsampling_size: Union[int, float], subsampling_seed: int, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.subsampling_size = subsampling_size
        if isinstance(subsampling_size, float):
            assert 0 < subsampling_size <= 1

        self.subsampling_seed = subsampling_seed
        self.subsampling_rng = check_random_state(subsampling_seed)  # reproducibility


class SEALSRandomStrategy(SEALSMixin, RandomStrategy):
    def __init__(self, num_neighbours: int, *args, **kwargs) -> None:
        super().__init__(self, *args, **kwargs)
        self.num_neighbours = num_neighbours
