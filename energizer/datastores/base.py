from abc import ABC, abstractmethod
from typing import Any, Callable, List, Optional, Union

import torch

# from lightning.pytorch.core.mixins.hparams_mixin import HyperparametersMixin
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader, RandomSampler, Sampler, SequentialSampler

from energizer.enums import RunningStage
from energizer.types import DATASET


class BaseDataStore(ABC):
    """General container for data."""

    def __init__(self) -> None:
        super().__init__()

    """
    Datasets
    """

    @abstractmethod
    def train_dataset(self, *args, **kwargs) -> DATASET:
        ...

    @abstractmethod
    def validation_dataset(self, *args, kwargs) -> DATASET:
        ...

    @abstractmethod
    def test_dataset(self, *args, kwargs) -> DATASET:
        ...

    @abstractmethod
    def pool_dataset(self, *args, kwargs) -> DATASET:
        ...

    """
    Loaders
    """

    @abstractmethod
    def train_loader(self, *args, **kwargs) -> DataLoader:
        ...

    @abstractmethod
    def validation_loader(self, *args, kwargs) -> DataLoader:
        ...

    @abstractmethod
    def test_loader(self, *args, kwargs) -> DataLoader:
        ...

    @abstractmethod
    def pool_loader(self, *args, kwargs) -> DataLoader:
        ...

    """
    Methods
    """

    @abstractmethod
    def label(self, *args, **kwargs) -> int:
        ...

    @abstractmethod
    def sample_from_pool(self, *args, **kwargs) -> List[int]:
        ...

    @abstractmethod
    def prepare_for_loading(self, *args, **kwargs) -> List[int]:
        ...

    """
    Status
    """

    @abstractmethod
    def train_size(self, *args, **kwargs) -> int:
        ...

    @abstractmethod
    def validation_size(self, *args, **kwargs) -> int:
        ...

    @abstractmethod
    def test_size(self, *args, **kwargs) -> int:
        ...

    @abstractmethod
    def pool_size(self, *args, **kwargs) -> int:
        ...

    @abstractmethod
    def labelled_size(self, *args, **kwargs) -> int:
        ...

    @abstractmethod
    def query_size(self, *args, **kwargs) -> int:
        ...

    @abstractmethod
    def total_rounds(self, *args, **kwargs) -> int:
        ...


class Datastore(BaseDataStore):
    """Defines dataloading for training and evaluation."""

    _rng: RandomState

    def __init__(self, seed: int = 42) -> None:
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
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        self.shuffle = shuffle
        self.replacement = replacement
        self.data_seed = data_seed

    def reset_rng(self, seed: int) -> None:
        self._rng = check_random_state(seed)

    def train_loader(self, *args, **kwargs) -> Optional[DataLoader]:
        return self.get_loader(RunningStage.TRAIN, *args, **kwargs)

    def validation_loader(self, *args, **kwargs) -> Optional[DataLoader]:
        return self.get_loader(RunningStage.VALIDATION, *args, **kwargs)

    def test_loader(self, *args, **kwargs) -> Optional[DataLoader]:
        return self.get_loader(RunningStage.TEST, *args, **kwargs)

    def pool_loader(self, *args, **kwargs) -> Optional[DataLoader]:
        return self.get_loader(RunningStage.POOL, *args, **kwargs)

    def get_loader(self, stage: str, *args, **kwargs) -> Optional[DataLoader]:
        # get dataset
        dataset = getattr(self, f"{stage}_dataset")(*args, **kwargs)
        if dataset is None:
            return

        batch_size = self.batch_size if stage == RunningStage.TRAIN else self.eval_batch_size
        batch_size = min(batch_size, len(dataset))

        # sampler
        sampler = _get_sampler(
            dataset,
            shuffle=self.shuffle if stage == RunningStage.TRAIN else False,
            replacement=self.replacement,
            seed=self.data_seed,
        )

        # put everything together
        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=self.get_collate_fn(stage),
            drop_last=self.drop_last,
        )

    def get_collate_fn(self, stage: Optional[str] = None) -> Optional[Callable]:
        return None

    def show_batch(self, stage: Union[str, RunningStage] = RunningStage.TRAIN, *args, **kwargs) -> Optional[Any]:
        batch_size, eval_batch_size, shuffle = self.batch_size, self.eval_batch_size, self.shuffle
        
        self.prepare_for_loading(batch_size=1, eval_batch_size=1, shuffle=False)
        loader = getattr(self, f"{stage}_loader")(*args, **kwargs)

        self.batch_size, self.eval_batch_size, self.shuffle = batch_size, eval_batch_size, shuffle

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
