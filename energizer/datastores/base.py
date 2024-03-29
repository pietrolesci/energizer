"""
Here we define the classes that take care of loading the data.
"""

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from datasets import Dataset
from numpy.random import RandomState
from sklearn.utils import check_random_state  # type: ignore
from torch.utils.data import DataLoader, RandomSampler, Sampler, SequentialSampler

from energizer.datastores.mixins import IndexMixin
from energizer.enums import RunningStage, SpecialKeys
from energizer.types import DATA_SOURCE, DATASET
from energizer.utilities import Args


class BaseDatastore(ABC):
    """General container for data."""

    """
    Data sources
    """
    _train_data: DATA_SOURCE | None
    _validation_data: DATA_SOURCE | None
    _test_data: DATA_SOURCE | None

    def __init__(self) -> None:
        super().__init__()

    """
    Datasets
    """

    @abstractmethod
    def train_dataset(self, *args, **kwargs) -> DATASET | None:
        ...

    @abstractmethod
    def validation_dataset(self, *args, kwargs) -> DATASET | None:
        ...

    @abstractmethod
    def test_dataset(self, *args, kwargs) -> DATASET | None:
        ...

    """
    Loaders
    """

    @abstractmethod
    def train_loader(self, *args, **kwargs) -> DataLoader | None:
        ...

    @abstractmethod
    def validation_loader(self, *args, kwargs) -> DataLoader | None:
        ...

    @abstractmethod
    def test_loader(self, *args, kwargs) -> DataLoader | None:
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

    # @abstractmethod
    # def num_train_batches(self, *args, **kwargs) -> Optional[int]:
    #     ...

    # @abstractmethod
    # def num_validation_batches(self, *args, **kwargs) -> Optional[int]:
    #     ...

    # @abstractmethod
    # def num_test_batches(self, *args, **kwargs) -> Optional[int]:
    #     ...

    @abstractmethod
    def train_size(self, *args, **kwargs) -> int | None:
        ...

    @abstractmethod
    def validation_size(self, *args, **kwargs) -> int | None:
        ...

    @abstractmethod
    def test_size(self, *args, **kwargs) -> int | None:
        ...


@dataclass
class DataloaderArgs(Args):
    batch_size: int
    eval_batch_size: int
    num_workers: int
    pin_memory: bool
    drop_last: bool
    persistent_workers: bool
    shuffle: bool
    replacement: bool
    data_seed: int
    multiprocessing_context: str | None


class Datastore(BaseDatastore):
    """Defines dataloading for training and evaluation."""

    _collate_fn: Callable | None
    _loading_params: DataloaderArgs | None = None
    _rng: RandomState

    def __init__(self, seed: int | None = 42) -> None:
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
        replacement: bool = False,
        data_seed: int = 42,
        multiprocessing_context: str | None = None,
    ) -> None:
        self._loading_params = DataloaderArgs(
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            shuffle=shuffle,
            replacement=replacement,
            data_seed=data_seed,
            multiprocessing_context=multiprocessing_context,
        )

    @property
    def loading_params(self) -> DataloaderArgs:
        assert self._loading_params is not None, ValueError("You need to `prepare_for_loading`")
        return self._loading_params

    def reset_rng(self, seed: int | None) -> None:
        self._rng = check_random_state(seed)

    def get_collate_fn(self, stage: str | None = None, show_batch: bool = False) -> Callable | None:
        return None

    def show_batch(self, stage: str | RunningStage = RunningStage.TRAIN, *args, **kwargs) -> Any | None:
        dataset = getattr(self, f"{stage}_dataset")(*args, **kwargs)
        if dataset is not None:
            loader = DataLoader(dataset=dataset, batch_size=1, collate_fn=self.get_collate_fn(stage, show_batch=True))
            return next(iter(loader))

    def get_loader(self, stage: str, *args, **kwargs) -> DataLoader | None:
        # get dataset
        dataset = getattr(self, f"{stage}_dataset")(*args, **kwargs)
        if dataset is None:
            return

        batch_size = (
            self.loading_params.batch_size if stage == RunningStage.TRAIN else self.loading_params.eval_batch_size
        )
        batch_size = min(batch_size, len(dataset))

        # sampler
        sampler = _get_sampler(
            dataset,
            shuffle=self.loading_params.shuffle if stage == RunningStage.TRAIN else False,
            replacement=self.loading_params.replacement,
            seed=self.loading_params.data_seed,
        )

        # put everything together
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.get_collate_fn(stage),
            drop_last=self.loading_params.drop_last,
            num_workers=min(batch_size, self.loading_params.num_workers),
            pin_memory=self.loading_params.pin_memory,
            persistent_workers=self.loading_params.persistent_workers,
            multiprocessing_context=self.loading_params.multiprocessing_context,
        )

    def _get_size(self, stage: RunningStage, *args, **kwargs) -> int | None:
        dataset = getattr(self, f"{stage}_dataset")(*args, **kwargs)
        if dataset is not None:
            return len(dataset)

    """Abstract methods implementation"""

    def train_loader(self, *args, **kwargs) -> DataLoader | None:
        return self.get_loader(RunningStage.TRAIN, *args, **kwargs)

    def validation_loader(self, *args, **kwargs) -> DataLoader | None:
        return self.get_loader(RunningStage.VALIDATION, *args, **kwargs)

    def test_loader(self, *args, **kwargs) -> DataLoader | None:
        return self.get_loader(RunningStage.TEST, *args, **kwargs)

    def train_size(self, *args, **kwargs) -> int | None:
        return self._get_size(RunningStage.TRAIN, *args, **kwargs)

    def validation_size(self, *args, **kwargs) -> int | None:
        return self._get_size(RunningStage.VALIDATION, *args, **kwargs)

    def test_size(self, *args, **kwargs) -> int | None:
        return self._get_size(RunningStage.TEST, *args, **kwargs)

    def train_dataset(self) -> Dataset | None:
        if self._train_data is not None:
            return self._train_data

    def validation_dataset(self) -> Dataset | None:
        if self._validation_data is not None:
            return self._validation_data

    def test_dataset(self) -> Dataset | None:
        if self._test_data is not None:
            return self._test_data


class PandasDatastore(Datastore):
    _train_data: pd.DataFrame | None
    _validation_data: Dataset | None
    _test_data: Dataset | None

    def train_dataset(self) -> Dataset | None:
        if self._train_data is not None:
            return Dataset.from_pandas(self._train_data, preserve_index=False)

    def get_by_ids(self, ids: list[int]) -> pd.DataFrame:
        assert self._train_data is not None, "To `get_by_ids` you need to specify the train_data."  # type: ignore
        return self._train_data.loc[self._train_data[SpecialKeys.ID].isin(ids)]  # type: ignore


class PandasDatastoreWithIndex(IndexMixin, PandasDatastore):
    """DataModule that defines dataloading and indexing logic."""


"""
Define as globals otherwise pickle complains when running in multi-gpu
"""


def _get_sampler(dataset: DATASET, shuffle: bool, replacement: bool, seed: int) -> Sampler:
    """Get a sampler optimizer to work with `datasets.Dataset`.

    ref: https://huggingface.co/docs/datasets/use_with_pytorch
    """

    if not shuffle:
        return SequentialSampler(dataset)

    g = torch.Generator()
    g.manual_seed(seed)
    return RandomSampler(dataset, generator=g, replacement=replacement)
