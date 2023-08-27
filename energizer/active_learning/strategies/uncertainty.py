from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule

from energizer.active_learning.datastores.base import ActiveDataStore
from energizer.active_learning.registries import SCORING_FUNCTIONS
from energizer.active_learning.strategies.base import ActiveEstimator, PoolBasedStrategyMixin
from energizer.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from energizer.types import BATCH_OUTPUT, METRIC
from energizer.utilities import ld_to_dl


class UncertaintyBasedStrategy(PoolBasedStrategyMixin, ActiveEstimator):
    score_fn: Callable

    def __init__(self, *args, score_fn: Union[str, Callable], **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.score_fn = score_fn if isinstance(score_fn, Callable) else SCORING_FUNCTIONS[score_fn]

    def run_query(
        self,
        model: _FabricModule,
        loader: _FabricDataLoader,
        datastore: ActiveDataStore,
        query_size: int,
    ) -> List[int]:
        return self.compute_most_uncertain(model, loader, query_size)

    def compute_most_uncertain(self, model: _FabricModule, loader: _FabricDataLoader, query_size: int) -> List[int]:
        # calls the pool_step and pool_epoch_end that we override
        out: List[Dict] = self.run_evaluation(model, loader, RunningStage.POOL)  # type: ignore
        _out = ld_to_dl(out)
        scores = np.concatenate(_out[OutputKeys.SCORES])
        ids = np.concatenate(_out[SpecialKeys.ID])

        # compute topk
        topk_ids = scores.argsort()[-query_size:]
        return ids[topk_ids].tolist()

    def evaluation_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: Optional[METRIC],
        stage: Union[str, RunningStage],
    ) -> Union[Dict, BATCH_OUTPUT]:
        if stage != RunningStage.POOL:
            return super().evaluation_step(model, batch, batch_idx, loss_fn, metrics, stage)  # type: ignore

        # keep IDs here in case user messes up in the function definition
        ids = batch[InputKeys.ON_CPU][SpecialKeys.ID]
        pool_out = self.pool_step(model, batch, batch_idx, loss_fn, metrics)

        if isinstance(pool_out, torch.Tensor):
            pool_out = {OutputKeys.SCORES: pool_out}
        else:
            assert isinstance(pool_out, dict) and OutputKeys.SCORES in pool_out, (
                "In `pool_step` you must return a Tensor with the scores per each element in the batch "
                f"or a Dict with a '{OutputKeys.SCORES}' key and the Tensor of scores as the value."
            )

        pool_out[SpecialKeys.ID] = ids  # type: ignore

        return pool_out  # enforce that we always return a dict here
