from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule

from energizer.active_learning.datastores.base import ActiveDataStore
from energizer.active_learning.registries import SCORING_FUNCTIONS
from energizer.active_learning.strategies.base import ActiveEstimator
from energizer.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from energizer.types import BATCH_OUTPUT, METRIC
from energizer.utilities import ld_to_dl


class DiversitySamplingMixin:
    def get_embeddings(
        self,
        model: _FabricModule,
        loader: _FabricDataLoader,
        datastore: ActiveDataStore,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError

    def select_from_embeddings(self, embeddings: np.ndarray, **kwargs) -> List[int]:
        raise NotImplementedError


class DiversityBasedStrategy(DiversitySamplingMixin, ActiveEstimator):
    def run_query(
        self,
        model: _FabricModule,
        loader: _FabricDataLoader,
        datastore: ActiveDataStore,
        query_size: int,
    ) -> List[int]:
        embeddings = self.get_embeddings(model, loader, datastore)
        return self.select_from_embeddings(embeddings)


# class BadgeStrategy(DiversityBasedStrategy):
