from abc import ABC, abstractmethod
from math import floor
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple, Union

import hnswlib as hb
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from lightning.pytorch.core.mixins.hparams_mixin import HyperparametersMixin
from numpy.random import RandomState
from sklearn.utils import check_random_state, resample
from torch.utils.data import DataLoader, RandomSampler, Sampler, SequentialSampler

from energizer.enums import InputKeys, RunningStage, SpecialKeys
from energizer.types import DATASET


def _resolve_round(round: Optional[int] = None) -> Union[float, int]:
    return round if round is not None else float("Inf")


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


class DataStore(BaseDataStore, HyperparametersMixin):
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

    def show_batch(self, stage: RunningStage = RunningStage.TRAIN) -> Optional[Any]:
        loader = getattr(self, f"{stage}_loader")()
        if loader is not None:
            return next(iter(loader))


class PandasDataStore(DataStore):
    data: pd.DataFrame = None
    test_data: Optional[DATASET] = None
    validation_data: Optional[DATASET] = None

    _rng: RandomState
    input_names: List[str]
    target_name: str

    def from_dataset_dict(
        self, dataset_dict: DatasetDict, input_names: Union[str, List[str]], target_name: str
    ) -> None:
        self.target_name = target_name
        self.input_names = [self.input_names] if isinstance(input_names, str) else input_names

        data = dataset_dict["train"].to_pandas()
        if SpecialKeys.ID not in data.columns:
            data[SpecialKeys.ID] = data.index.copy()
        # check consistency of the index
        assert data[SpecialKeys.ID].nunique() == len(data)

        data = data.loc[:, self.input_names + [self.target_name, SpecialKeys.ID]].assign(
            **{
                SpecialKeys.IS_LABELLED: False,
                SpecialKeys.IS_VALIDATION: False,
                SpecialKeys.LABELLING_ROUND: -100,
            }
        )
        self.data = data
        for stage in (RunningStage.VALIDATION, RunningStage.TEST):
            setattr(self, f"{stage}_data", dataset_dict.get(stage, None))

    def _labelled_mask(self, round: Optional[int] = None) -> pd.Series:
        return (self.data[SpecialKeys.IS_LABELLED] == True) & (
            self.data[SpecialKeys.LABELLING_ROUND] <= _resolve_round(round)
        )

    def _train_mask(self, round: Optional[int] = None) -> pd.Series:
        return self._labelled_mask(round) & (self.data[SpecialKeys.IS_VALIDATION] == False)

    def _validation_mask(self, round: Optional[int] = None) -> pd.Series:
        return self._labelled_mask(round) & (self.data[SpecialKeys.IS_VALIDATION] == True)

    def _pool_mask(self, round: Optional[int] = None) -> pd.Series:
        return (self.data[SpecialKeys.IS_LABELLED] == False) | (
            self.data[SpecialKeys.LABELLING_ROUND] > _resolve_round(round)
        )

    def train_dataset(self, round: Optional[int] = None) -> Optional[DATASET]:
        round = _resolve_round(round)
        mask = self._train_mask(round)
        if mask.sum() > 0:
            return Dataset.from_pandas(self.data.loc[mask], preserve_index=False)

    def validation_dataset(self, round: Optional[int] = None) -> Optional[DATASET]:
        if self.validation_data:
            return self.validation_data

        round = _resolve_round(round)
        mask = self._validation_mask(round)
        if mask.sum() > 0:
            return Dataset.from_pandas(self.data.loc[mask], preserve_index=False)

    def test_dataset(self) -> DATASET:
        if self.test_data is not None:
            return self.test_data

    def pool_dataset(self, round: Optional[int] = None, subset_indices: Optional[List[int]] = None) -> DataLoader:
        round = _resolve_round(round)
        df = self.data.loc[self._pool_mask(round), [i for i in self.data.columns if i != InputKeys.TARGET]]
        if subset_indices is not None:
            df = df.loc[df[SpecialKeys.ID].isin(subset_indices)]
        return Dataset.from_pandas(df, preserve_index=False)

    def label(
        self,
        indices: List[int],
        round: int,
        validation_perc: Optional[float] = None,
        validation_sampling: Optional[str] = None,
    ) -> int:
        assert isinstance(indices, list), ValueError(f"`indices` must be of type `List[int]`, not {type(indices)}.")
        assert (isinstance(validation_perc, float) and validation_perc > 0.0) or validation_perc is None, ValueError(
            f"`validation_perc` must be of type `float` and > 0.0, not {type(validation_perc)}"
        )

        # label training data
        mask = self.data[SpecialKeys.ID].isin(indices)
        self.data.loc[mask, SpecialKeys.IS_LABELLED] = True
        self.data.loc[mask, SpecialKeys.LABELLING_ROUND] = round

        # train-validation split
        if validation_perc is not None:
            n_val = floor(validation_perc * len(indices)) or 1  # at least add one
            currentdata = self.data.loc[mask, [SpecialKeys.ID, InputKeys.TARGET]]
            val_indices = sample(
                indices=currentdata[SpecialKeys.ID].tolist(),
                size=n_val,
                labels=currentdata[InputKeys.TARGET],
                sampling=validation_sampling,
                random_state=self._rng,
            )
            self.data.loc[self.data[SpecialKeys.ID].isin(val_indices), SpecialKeys.IS_VALIDATION] = True

        return mask.sum()


class PandasHSNWDataStore(DataStore):
    """DataModule that defines dataloading and indexing logic."""

    index: hb.Index = None

    def load_index(self, path: Union[str, Path], embedding_dim: int) -> None:
        index = hb.Index(space="cosine", dim=embedding_dim)
        index.load_index(str(path))
        self.index = index

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


# class ActiveDataModule(DataModule):
#     data: pd.DataFrame = None


#     """
#     Get info
#     """

#     @property
#     def test_size(self) -> int:
#         if self.test_dataset is None:
#             return 0
#         return len(self.test_dataset)

#     @property
#     def has_test_data(self) -> bool:
#         return self.test_size > 0

#     @property
#     def last_labelling_round(self) -> int:
#         """Returns the number of the last active learning step."""
#         return int(self.datastore[SpecialKeys.LABELLING_ROUND].max())

#     @property
#     def initial_budget(self) -> int:
#         return (self.datastore[SpecialKeys.LABELLING_ROUND] == 0).sum()

#     def query_size(self, round: Optional[int] = None) -> int:
#         round = min(_resolve_round(round), self.last_labelling_round)
#         last = len(self.datastore.loc[self.datastore[SpecialKeys.LABELLING_ROUND] <= round])
#         prev = len(self.datastore.loc[self.datastore[SpecialKeys.LABELLING_ROUND] <= round - 1])
#         return last - prev

#     def train_size(self, round: Optional[int] = None) -> int:
#         return self._train_mask(round).sum()

#     def has_train_data(self, round: Optional[int] = None) -> bool:
#         return self.train_size(round) > 0

#     def validation_size(self, round: Optional[int] = None) -> int:
#         return self._validation_mask(round).sum()

#     def has_validation_data(self, round: Optional[int] = None) -> bool:
#         return self.validation_size(round) > 0

#     def total_labelled_size(self, round: Optional[int] = None) -> int:
#         return self._labelled_mask(round).sum()

#     def pool_size(self, round: Optional[int] = None) -> int:
#         return self._pool_mask(round).sum()

#     def has_unlabelled_data(self, round: Optional[int] = None) -> bool:
#         return self.pool_size(round) > 0

#     def train_indices(self, round: Optional[int] = None) -> np.ndarray:
#         return self.datastore.loc[self._train_mask(round), SpecialKeys.ID].values

#     def validation_indices(self, round: Optional[int] = None) -> np.ndarray:
#         return self.datastore.loc[self._validation_mask(round), SpecialKeys.ID].values

#     def pool_indices(self, round: Optional[int] = None) -> np.ndarray:
#         return self.datastore.loc[self._pool_mask(round), SpecialKeys.ID].values

#     def data_statistics(self, round: Optional[int] = None) -> Dict[str, int]:
#         return {
#             "train_size": self.train_size(round),
#             "validation_size": self.validation_size(round),
#             "test_size": self.test_size,
#             "pool_size": self.pool_size(round),
#             "total_labelled_size": self.total_labelled_size(round),
#         }

#     """
#     Labelling and Accounting
#     """

#     def _labelled_mask(self, round: Optional[int] = None) -> pd.Series:
#         return (self.datastore[SpecialKeys.IS_LABELLED] == True) & (
#             self.datastore[SpecialKeys.LABELLING_ROUND] <= _resolve_round(round)
#         )

#     def _train_mask(self, round: Optional[int] = None) -> pd.Series:
#         return self._labelled_mask(round) & (self.datastore[SpecialKeys.IS_VALIDATION] == False)

#     def _validation_mask(self, round: Optional[int] = None) -> pd.Series:
#         return self._labelled_mask(round) & (self.datastore[SpecialKeys.IS_VALIDATION] == True)

#     def _pool_mask(self, round: Optional[int] = None) -> pd.Series:
#         return (self.datastore[SpecialKeys.IS_LABELLED] == False) | (
#             self.datastore[SpecialKeys.LABELLING_ROUND] > _resolve_round(round)
#         )

#     """
#     Index
#     """

#     def _mask_train_from_index(self, round: Optional[int] = None) -> None:
#         if self.index is not None:
#             train_ids = self.train_indices(round)
#             for i in train_ids:
#                 self.index.mark_deleted(i)

#     def _unmask_train_from_index(self, round: Optional[int] = None) -> None:
#         if self.index is not None:
#             train_ids = self.train_indices(round)
#             for i in train_ids:
#                 self.index.unmark_deleted(i)

#     def get_train_embeddings(self, round: Optional[int] = None) -> np.ndarray:
#         self._unmask_train_from_index(round)
#         embeddings = self.index.get_items(self.train_indices(round))
#         self._mask_train_from_index(round)
#         return np.array(embeddings)

#     def get_embeddings(self, indices: Optional[List[int]] = None) -> np.ndarray:
#         indices = indices or np.array(self.index.get_ids_list())
#         self._unmask_train_from_index()
#         embeddings = self.index.get_items(indices)
#         self._mask_train_from_index()
#         return np.array(embeddings)

#     def get_pool_embeddings(self, round: Optional[int] = None) -> np.ndarray:
#         return np.array(self.index.get_items(self.pool_indices(round)))

#     """
#     Main methods
#     """

#     def label(
#         self,
#         indices: List[int],
#         round_idx: int,
#         validation_perc: Optional[float] = None,
#         validation_sampling: Optional[str] = None,
#         random_state: Optional[RandomState] = None,
#     ) -> int:
#         assert isinstance(indices, list), ValueError(f"`indices` must be of type `List[int]`, not {type(indices)}.")
#         assert isinstance(validation_perc, float) or validation_perc is None, ValueError(
#             f"`validation_perc` must be of type `float`, not {type(validation_perc)}"
#         )

#         mask = self.datastore[SpecialKeys.ID].isin(indices)
#         self.datastore.loc[mask, SpecialKeys.IS_LABELLED] = True
#         self.datastore.loc[mask, SpecialKeys.LABELLING_ROUND] = round_idx

#         if validation_perc is not None and validation_perc > 0.0:
#             n_val = round(validation_perc * len(indices)) or 1  # at least add one
#             currentdatastore = self.datastore.loc[mask, [SpecialKeys.ID, InputKeys.TARGET]]
#             val_indices = sample(
#                 indices=currentdatastore[SpecialKeys.ID].tolist(),
#                 size=n_val,
#                 labels=currentdatastore[InputKeys.TARGET],
#                 sampling=validation_sampling,
#                 random_state=random_state or self._rng,
#             )
#             self.datastore.loc[self.datastore[SpecialKeys.ID].isin(val_indices), SpecialKeys.IS_VALIDATION] = True

#         # remove instance from the index
#         if self._index is not None:
#             for idx in indices:
#                 self.index.mark_deleted(idx)

#         return mask.sum()

#     def set_initial_budget(
#         self,
#         budget: int,
#         validation_perc: Optional[float] = None,
#         sampling: Optional[str] = None,
#         seed: Optional[int] = None,
#     ) -> None:
#         df = self.datastore.loc[(self.datastore[SpecialKeys.IS_LABELLED] == False), [SpecialKeys.ID, InputKeys.TARGET]]
#         _rng = check_random_state(seed) if seed else self._rng

#         # sample from the pool
#         indices = sample(
#             indices=df[SpecialKeys.ID].tolist(),
#             size=budget,
#             labels=df[InputKeys.TARGET].tolist(),
#             sampling=sampling,
#             random_state=_rng,
#         )

#         # actually label
#         self.label(
#             indices=indices,
#             round_idx=0,
#             validation_perc=validation_perc,
#             validation_sampling=sampling,
#             random_state=_rng,
#         )

#     """
#     DataLoaders
#     """

#     def train_loader(self, round: Optional[int] = None) -> Optional[DataLoader]:
#         round = _resolve_round(round)
#         if self.train_size(round) > 0:
#             df = self.datastore.loc[self._train_mask(round)]
#             dataset = Dataset.from_pandas(df, preserve_index=False)

#             return self.get_loader(RunningStage.TRAIN, dataset)

#     def validation_loader(self, round: Optional[int] = None) -> Optional[DataLoader]:
#         round = _resolve_round(round)
#         if self.validation_size(round) > 0:
#             df = self.datastore.loc[self._validation_mask(round)]
#             dataset = Dataset.from_pandas(df, preserve_index=False)
#             return self.get_loader(RunningStage.VALIDATION, dataset)

#     def pool_loader(self, subset_indices: Optional[List[int]] = None) -> DataLoader:
#         df = self.datastore.loc[
#             (self.datastore[SpecialKeys.IS_LABELLED] == False),
#             [i for i in self.datastore.columns if i != InputKeys.TARGET],
#         ]

#         if subset_indices is not None:
#             df = df.loc[df[SpecialKeys.ID].isin(subset_indices)]

#         # for performance reasons
#         dataset = Dataset.from_pandas(
#             df=(
#                 df.assign(length=lambda df_: df_[InputKeys.INPUT_IDS].map(len))
#                 .sort_values("length")
#                 .drop(columns=["length"])
#             ),
#             preserve_index=False,
#         )

#         return self.get_loader(RunningStage.POOL, dataset)

#     def setup(self, stage: Optional[str] = None) -> None:
#         # with_format does not remove columns completely;
#         # when the Dataset is cast to pandas they remain, so remove
#         cols = list(self.train_dataset[0].keys())

#         self.datastore = (
#             self.train_dataset.to_pandas()
#             .loc[:, cols]
#             .assign(
#                 **{
#                     SpecialKeys.IS_LABELLED: False,
#                     SpecialKeys.IS_VALIDATION: False,
#                     SpecialKeys.LABELLING_ROUND: -100,
#                 }
#             )
#         )

#         # check consistency of the index
#         assert self.datastore[SpecialKeys.ID].nunique() == len(self.datastore)

#     def get_labelled_dataset(self) -> pd.DataFrame:
#         cols = [i for i in SpecialKeys] + [InputKeys.TARGET]
#         return self.datastore.loc[self.datastore[SpecialKeys.IS_LABELLED] == True, cols]

#     def set_labelled_dataset(self, df: pd.DataFrame) -> None:
#         assert df[SpecialKeys.ID].isin(self.datastore[SpecialKeys.ID]).all()

#         to_keep = self.datastore.loc[~self.datastore[SpecialKeys.ID].isin(df[SpecialKeys.ID])]

#         cols = [SpecialKeys.IS_LABELLED, SpecialKeys.IS_VALIDATION, SpecialKeys.LABELLING_ROUND]
#         to_update = pd.merge(df[cols + [SpecialKeys.ID]], self.datastore.drop(columns=cols), on=SpecialKeys.ID, how="inner")
#         assert not to_update[SpecialKeys.ID].isin(to_keep[SpecialKeys.ID]).all()

#         newdatastore = pd.concat([to_keep, to_update])
#         assert newdatastore.shape == self.datastore.shape
#         assert df[SpecialKeys.IS_LABELLED].sum() == newdatastore[SpecialKeys.IS_LABELLED].sum()
#         assert df[SpecialKeys.IS_VALIDATION].sum() == newdatastore[SpecialKeys.IS_VALIDATION].sum()

#         self.datastore = newdatastore.copy()

#     def save_labelled_dataset(self, save_dir: str) -> None:
#         save_dir = Path(save_dir)
#         save_dir.mkdir(parents=True, exist_ok=True)
#         self.get_labelled_dataset().to_parquet(save_dir / "labelled_dataset.parquet", index=False)

"""
Utils
"""


def sample(
    indices: List[int],
    size: int,
    random_state: RandomState,
    labels: Optional[List[int]],
    sampling: Optional[str] = None,
) -> List[int]:
    """Makes sure to seed everything consistently."""

    if sampling is None or sampling == "random":
        sample = random_state.choice(indices, size=size, replace=False)

    elif sampling == "stratified" and labels is not None:
        sample = resample(
            indices,
            replace=False,
            stratify=labels,
            n_samples=size,
            random_state=random_state,
        )

    else:
        raise ValueError("Only `random` and `stratified` are supported by default.")

    assert len(set(sample)) == size

    return sample


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
