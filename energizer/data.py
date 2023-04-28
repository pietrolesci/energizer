# Here we define DataModule that work with HuggingFace DATASETs.
# We assume that each dataset is already processed and ready for training.
# Think of the DataModule is the last step of the data preparation pipeline.
#
#   download data -> (process data -> prepare data) -> datamodule -> model
#
# That is, the DataModule is only used to feed data to the model during training
# and evaluation.
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple, Union

import hnswlib as hb
import numpy as np
import torch
from lightning.pytorch.core.mixins.hparams_mixin import HyperparametersMixin
from numpy.random import RandomState
from sklearn.utils.validation import check_random_state
from torch.utils.data import DataLoader, RandomSampler, Sampler, SequentialSampler

from src.energizer.enums import RunningStage
from src.energizer.types import DATASET


class DataModule(HyperparametersMixin):
    """DataModule that defines dataloading and indexing logic."""

    _hparams_ignore: List[str] = ["train_dataset", "validation_dataset", "test_dataset"]
    _index: hb.Index = None
    _rng: RandomState

    def __init__(
        self,
        train_dataset: DATASET,
        validation_dataset: Optional[DATASET] = None,
        test_dataset: Optional[DATASET] = None,
        batch_size: Optional[int] = 32,
        eval_batch_size: Optional[int] = 32,
        num_workers: Optional[int] = 0,
        pin_memory: Optional[bool] = True,
        drop_last: Optional[bool] = False,
        persistent_workers: Optional[bool] = False,
        shuffle: Optional[bool] = True,
        seed: Optional[int] = 42,
        replacement: bool = False,
    ) -> None:
        super().__init__()
        self.train_dataset = train_dataset
        self.validation_dataset = validation_dataset
        self.test_dataset = test_dataset

        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        self.shuffle = shuffle
        self.seed = seed
        self.replacement = replacement

        self.save_hyperparameters(ignore=self._hparams_ignore)
        self.reset_rng()
        self.setup()

    def setup(self) -> None:
        pass

    def reset_rng(self) -> None:
        self._rng = check_random_state(self.seed)

    @property
    def index(self) -> Optional[hb.Index]:
        return self._index

    def load_index(self, path: Union[str, Path], embedding_dim: int) -> None:
        p = hb.Index(space="cosine", dim=embedding_dim)
        p.load_index(str(path))
        self._index = p

    def search_index(
        self, query: np.ndarray, query_size: int, query_in_set: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        # retrieve one additional element if the query is in the set we are looking in
        # because the query itself is returned as the most similar element and we need to remove it
        query_size = query_size + 1 if query_in_set else query_size

        indices, distances = self.index.knn_query(data=query, k=query_size)

        if query_in_set:
            # remove the first element retrieved if the query is in the set since it's the element itself
            indices, distances = indices[:, 1:], distances[:, 1:]

        return indices, distances

    def train_loader(self) -> DataLoader:
        return self.get_loader(RunningStage.TRAIN)

    def validation_loader(self) -> Optional[DataLoader]:
        return self.get_loader(RunningStage.VALIDATION)

    def test_loader(self) -> Optional[DataLoader]:
        return self.get_loader(RunningStage.TEST)

    def get_collate_fn(self, stage: Optional[str] = None) -> Optional[Callable]:
        return None

    def get_loader(self, stage: str, dataset: Optional[Iterable] = None) -> Optional[DataLoader]:
        dataset = dataset or getattr(self, f"{stage}_dataset", None)
        if dataset is None:
            return

        batch_size = self.batch_size if stage == RunningStage.TRAIN else self.eval_batch_size
        batch_size = min(batch_size, len(dataset))

        sampler = _get_sampler(
            dataset,
            shuffle=self.shuffle if stage == RunningStage.TRAIN else False,
            replacement=self.replacement,
            seed=self.seed,
        )

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.get_collate_fn(stage),
            drop_last=self.drop_last,
        )

    def show_batch(self, stage: RunningStage = RunningStage.TRAIN) -> Optional[Any]:
        loader = getattr(self, f"{stage}_loader")()
        if loader is not None:
            return next(iter(loader))


"""
Define as globals otherwise pickle complains when running in multi-gpu
"""


def _get_sampler(
    dataset: DATASET,
    shuffle: bool,
    replacement: bool,
    seed: int,
) -> Sampler:
    """Get a sampler optimizer to work with `datasets.Dataset`.

    ref: https://huggingface.co/docs/datasets/use_with_pytorch
    """

    if not shuffle:
        return SequentialSampler(dataset)

    g = torch.Generator()
    g.manual_seed(seed)
    return RandomSampler(dataset, generator=g, replacement=replacement)
