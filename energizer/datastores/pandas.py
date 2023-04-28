from collections import Counter
from functools import partial
from math import floor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union, MutableMapping

import hnswlib as hb
import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from numpy.random import RandomState
from sklearn.utils import resample
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase

from energizer.datastores.base import DataStore
from energizer.enums import InputKeys, RunningStage, SpecialKeys
from energizer.types import DATASET
from energizer.utilities import _pad, ld_to_dl
from lightning.pytorch.utilities.parsing import AttributeDict

def _resolve_round(round: Optional[int] = None) -> Union[float, int]:
    return round if round is not None else float("Inf")


class PandasDataStore(DataStore):
    data: pd.DataFrame = None
    test_data: Optional[DATASET] = None
    validation_data: Optional[DATASET] = None

    input_names: List[str]
    target_name: str
    on_cpu: List[str]

    def from_dataset_dict(
        self,
        dataset_dict: DatasetDict,
        input_names: Union[str, List[str]],
        target_name: str,
        on_cpu: Optional[List[str]] = None,
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

        self.on_cpu = on_cpu if on_cpu is not None else [SpecialKeys.ID]
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

    def train_dataset(self, round: Optional[int] = None, passive: Optional[bool] = False) -> Optional[DATASET]:
        if passive: return Dataset.from_pandas(self.data, preserve_index=False)
        
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


class PandasDataStoreWithIndex(PandasDataStore):
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

    def label(
        self,
        indices: List[int],
        round: int,
        validation_perc: Optional[float] = None,
        validation_sampling: Optional[str] = None,
    ) -> int:
        out = super().label(indices, round, validation_perc, validation_sampling)
        # remove instance from the index
        if self._index is not None:
            for idx in indices:
                self.index.mark_deleted(idx)
        return out


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


class PandasDataStoreForSequenceClassification(PandasDataStore):
    _tokenizer: Optional[PreTrainedTokenizerBase]
    _labels: List[str]
    _label_distribution: Dict[str, int]

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
        max_length: Optional[int] = None,
    ) -> None:
        super().__init__(
            batch_size,
            eval_batch_size,
            num_workers,
            pin_memory,
            drop_last,
            persistent_workers,
            shuffle,
            seed,
            replacement,
        )
        self.max_length = max_length
    
    @property
    def hparams(self) -> Union[AttributeDict, MutableMapping]:
        hparams = super().hparams
        if hasattr(self, "_tokenizer"):
            hparams["name_or_path"] = self.tokenizer.name_or_path
        if hasattr(self, "max_length"):
            hparams["max_length"] = self.max_length
        return hparams

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @property
    def labels(self) -> List[str]:
        return self._labels

    @property
    def id2label(self) -> Dict[int, str]:
        return dict(enumerate(self.labels))

    @property
    def label2id(self) -> Dict[str, int]:
        return {v: k for k, v in self.id2label.items()}
    
    def label_distribution(self, normalized: bool = False) -> Dict[str, Union[float, int]]:
        if normalized:
            total = sum(self._label_distribution.values())
            return {k: self._label_distribution[k] / total for k in self._label_distribution}
        return dict(self._label_distribution)


    def from_dataset_dict(
        self,
        dataset_dict: DatasetDict,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        on_cpu: Optional[List[str]] = None,
    ) -> None:
        features = dataset_dict["train"].features
        assert (
            InputKeys.INPUT_IDS in features and InputKeys.ATT_MASK in features and InputKeys.TARGET in features
        ), ValueError(
            f"Columns `input_ids`, `attention_mask`, and `labels` must be in passed dataset_dict. Current columns {list(features.keys())}"
        )

        self._tokenizer = tokenizer
        self._labels = dataset_dict["train"].features[InputKeys.TARGET].names
        self._label_distribution = Counter(dataset_dict["train"][InputKeys.TARGET])

        super().from_dataset_dict(dataset_dict, [InputKeys.INPUT_IDS, InputKeys.ATT_MASK], InputKeys.TARGET, on_cpu)

    def get_collate_fn(self, stage: Optional[str] = None) -> Optional[Callable]:
        return partial(
            collate_fn,
            on_cpu=self.on_cpu,
            max_length=self.max_length,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else 0,
            pad_fn=_pad,
        )


def collate_fn(
    batch: List[Dict[str, Union[List[str], Tensor]]],
    on_cpu: List[str],
    max_length: int,
    pad_token_id: int,
    pad_fn: Callable,
) -> Dict[str, Union[List[str], Tensor]]:

    batch = ld_to_dl(batch)

    # remove string columns that cannot be transfered on gpu
    on_cpu = {col: batch.pop(col) for col in on_cpu if col in batch}

    labels = batch.pop(InputKeys.TARGET, None)

    # input_ids and attention_mask to tensor: truncate -> convert to tensor -> pad
    batch = {
        k: pad_fn(
            inputs=batch[k],
            padding_value=pad_token_id,
            max_length=max_length,
        )
        for k in (InputKeys.INPUT_IDS, InputKeys.ATT_MASK)
    }

    if labels is not None:
        batch[InputKeys.TARGET] = torch.tensor(labels, dtype=torch.long)

    # add things that need to remain on cpu
    if len(on_cpu) > 0:
        batch[InputKeys.ON_CPU] = on_cpu

    return batch
