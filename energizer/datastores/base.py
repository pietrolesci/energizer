from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import torch
from lightning.pytorch.core.mixins.hparams_mixin import HyperparametersMixin
from numpy.random import RandomState
from sklearn.utils import check_random_state
from torch.utils.data import DataLoader, RandomSampler, Sampler, SequentialSampler

from energizer.enums import RunningStage
from energizer.types import DATASET


class BaseDataStore(ABC):
    """General container for data."""

    def __init__(self) -> None:
        super().__init__()

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

    @abstractmethod
    def label(self, *args, **kwargs) -> int:
        ...


class Datastore(BaseDataStore, HyperparametersMixin):
    """Defines dataloading for training and evaluation."""

    _rng: RandomState

    def __init__(
        self,
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
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        self.shuffle = shuffle
        self.seed = seed
        self.replacement = replacement

        self.save_hyperparameters()
        self.reset_rng()

    def reset_rng(self) -> None:
        self._rng = check_random_state(self.seed)

    def train_loader(self, *args, **kwargs) -> DataLoader:
        return self.get_loader(RunningStage.TRAIN, *args, **kwargs)

    def validation_loader(self, *args, **kwargs) -> Optional[DataLoader]:
        return self.get_loader(RunningStage.VALIDATION, *args, **kwargs)

    def test_loader(self, *args, **kwargs) -> Optional[DataLoader]:
        return self.get_loader(RunningStage.TEST, *args, **kwargs)

    def pool_loader(self, *args, **kwargs) -> Optional[DataLoader]:
        return self.get_loader(RunningStage.POOL, *args, **kwargs)

    def get_loader(self, stage: str, *args, **kwargs) -> Optional[DataLoader]:
        fn = getattr(self, f"{stage}_dataset", None)
        if fn is None:
            return
        dataset = fn(*args, **kwargs)

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

    def get_collate_fn(self, stage: Optional[str] = None) -> Optional[Callable]:
        return None

    def show_batch(self, stage: RunningStage = RunningStage.TRAIN, *args, **kwargs) -> Optional[Any]:
        loader = getattr(self, f"{stage}_loader")(*args, **kwargs)
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
