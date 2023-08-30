"""
Here we define the classes that take care of loading the data.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import hnswlib as hb
import numpy as np
import pandas as pd
import srsly
import torch
from datasets import Dataset
from numpy.random import RandomState
from sklearn.utils import check_random_state  # type: ignore
from torch.utils.data import DataLoader, RandomSampler, Sampler, SequentialSampler

from energizer.enums import RunningStage, SpecialKeys
from energizer.types import DATA_SOURCE, DATASET


class BaseDataStore(ABC):
    """General container for data."""

    """
    Data sources
    """
    _train_data: Optional[DATA_SOURCE]
    _validation_data: Optional[DATA_SOURCE]
    _test_data: Optional[DATA_SOURCE]

    def __init__(self) -> None:
        super().__init__()

    """
    Datasets
    """

    @abstractmethod
    def train_dataset(self, *args, **kwargs) -> Optional[DATASET]:
        ...

    @abstractmethod
    def validation_dataset(self, *args, kwargs) -> Optional[DATASET]:
        ...

    @abstractmethod
    def test_dataset(self, *args, kwargs) -> Optional[DATASET]:
        ...

    """
    Loaders
    """

    @abstractmethod
    def train_loader(self, *args, **kwargs) -> Optional[DataLoader]:
        ...

    @abstractmethod
    def validation_loader(self, *args, kwargs) -> Optional[DataLoader]:
        ...

    @abstractmethod
    def test_loader(self, *args, kwargs) -> Optional[DataLoader]:
        ...

    """
    Methods
    """

    @abstractmethod
    def prepare_for_loading(self, *args, **kwargs) -> None:
        ...

    """
    Status
    """

    @abstractmethod
    def train_size(self, *args, **kwargs) -> Optional[int]:
        ...

    @abstractmethod
    def validation_size(self, *args, **kwargs) -> Optional[int]:
        ...

    @abstractmethod
    def test_size(self, *args, **kwargs) -> Optional[int]:
        ...


class Datastore(BaseDataStore):
    """Defines dataloading for training and evaluation."""

    _collate_fn: Optional[Callable]
    _loading_params: Dict[str, Any] = {}
    _rng: RandomState

    def __init__(self, seed: Optional[int] = 42) -> None:
        super().__init__()
        self.seed = seed
        self.reset_rng(seed)

    def prepare_for_loading(
        self,
        batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        persistent_workers: bool = False,
        shuffle: bool = True,
        data_seed: int = 42,
        replacement: bool = False,
    ) -> None:
        self._loading_params = dict(
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            shuffle=shuffle,
            replacement=replacement,
            data_seed=data_seed,
        )

    @property
    def loading_params(self) -> Dict[str, Any]:
        if len(self._loading_params) > 0:
            return self._loading_params
        raise ValueError("You need to `prepare_for_loading`")

    def reset_rng(self, seed: Optional[int]) -> None:
        self._rng = check_random_state(seed)

    def train_loader(self, *args, **kwargs) -> Optional[DataLoader]:
        return self.get_loader(RunningStage.TRAIN, *args, **kwargs)

    def validation_loader(self, *args, **kwargs) -> Optional[DataLoader]:
        return self.get_loader(RunningStage.VALIDATION, *args, **kwargs)

    def test_loader(self, *args, **kwargs) -> Optional[DataLoader]:
        return self.get_loader(RunningStage.TEST, *args, **kwargs)

    def get_loader(self, stage: str, *args, **kwargs) -> Optional[DataLoader]:
        # get dataset
        dataset = getattr(self, f"{stage}_dataset")(*args, **kwargs)
        if dataset is None:
            return

        batch_size = (
            self.loading_params["batch_size"] if stage == RunningStage.TRAIN else self.loading_params["eval_batch_size"]
        )
        batch_size = min(batch_size, len(dataset))

        # sampler
        sampler = _get_sampler(
            dataset,
            shuffle=self.loading_params["shuffle"] if stage == RunningStage.TRAIN else False,
            replacement=self.loading_params["replacement"],
            seed=self.loading_params["data_seed"],
        )

        # put everything together
        return DataLoader(
            dataset=dataset,
            batch_size=self.loading_params["batch_size"],
            sampler=sampler,
            collate_fn=self.get_collate_fn(stage),
            drop_last=self.loading_params["drop_last"],
        )

    def get_collate_fn(self, stage: Optional[str] = None, show_batch: bool = False) -> Optional[Callable]:
        return None

    def show_batch(self, stage: Union[str, RunningStage] = RunningStage.TRAIN, *args, **kwargs) -> Optional[Any]:
        dataset = getattr(self, f"{stage}_dataset")(*args, **kwargs)
        if dataset is not None:
            loader = DataLoader(dataset=dataset, batch_size=1, collate_fn=self.get_collate_fn(stage, show_batch=True))
            return next(iter(loader))

    def train_size(self, *args, **kwargs) -> Optional[int]:
        return self._get_size(RunningStage.TRAIN, *args, **kwargs)

    def validation_size(self, *args, **kwargs) -> Optional[int]:
        return self._get_size(RunningStage.VALIDATION, *args, **kwargs)

    def test_size(self, *args, **kwargs) -> Optional[int]:
        return self._get_size(RunningStage.TEST, *args, **kwargs)

    def _get_size(self, stage: RunningStage, *args, **kwargs) -> Optional[int]:
        dataset = getattr(self, f"{stage}_dataset")(*args, **kwargs)
        if dataset is not None:
            return len(dataset)


class PandasDataStore(Datastore):
    _train_data: Optional[pd.DataFrame]
    _validation_data: Optional[Dataset]
    _test_data: Optional[Dataset]

    def train_dataset(self) -> Optional[Dataset]:
        if self._train_data is not None:
            return Dataset.from_pandas(self._train_data, preserve_index=False)

    def validation_dataset(self) -> Optional[Dataset]:
        if self._validation_data is not None:
            return self._validation_data

    def test_dataset(self) -> Optional[Dataset]:
        if self._test_data is not None:
            return self._test_data

    def get_by_ids(self, ids: List[int]) -> pd.DataFrame:
        assert self._train_data is not None, "To `get_by_ids` you need to specify the train_data."  # type: ignore
        return self._train_data.loc[self._train_data[SpecialKeys.ID].isin(ids)]  # type: ignore


class IndexMixin:

    index: hb.Index = None
    embedding_name: str

    def search(self, query: np.ndarray, query_size: int, query_in_set: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        # retrieve one additional element if the query is in the set we are looking in
        # because the query itself is returned as the most similar element and we need to remove it
        query_size = query_size + 1 if query_in_set else query_size
        indices, distances = self.index.knn_query(data=query, k=query_size)
        if query_in_set:
            # remove the first element retrieved if the query is in the set since it's the element itself
            indices, distances = indices[:, 1:], distances[:, 1:]
        return indices, distances

    def load_index(self, index_path: Union[str, Path], metadata_path: Union[str, Path]) -> None:
        meta: Dict = srsly.read_json(metadata_path)  # type: ignore
        index = hb.Index(space=meta["metric"], dim=meta["dim"])
        index.load_index(str(index_path))
        self.index = index

        # consistency check: data in index must be the same or more
        assert self._train_data is not None  # type: ignore
        assert len(index.get_ids_list()) >= len(self._train_data[SpecialKeys.ID]), "Index is not compatible with data."  # type: ignore

        # if dataset has been downsampled, mask the ids
        if len(index.get_ids_list()) > len(self._train_data[SpecialKeys.ID]):  # type: ignore
            missing_ids = set(index.get_ids_list()).difference(set(self._train_data[SpecialKeys.ID]))  # type: ignore
            self.mask_ids_from_index(list(missing_ids))

    def mask_ids_from_index(self, ids: List[int]) -> None:
        for i in ids:
            self.index.mark_deleted(i)

    def unmask_ids_from_index(self, ids: List[int]) -> None:
        for i in ids:
            self.index.unmark_deleted(i)

    def get_embeddings(self, ids: List[int]) -> np.ndarray:
        return np.stack(self.index.get_items(ids))


class PandasDataStoreWithIndex(IndexMixin, PandasDataStore):
    """DataModule that defines dataloading and indexing logic."""


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
