from collections import Counter
from functools import partial
from math import floor
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import hnswlib as hb
import numpy as np
import pandas as pd
import srsly
import torch
from click import Option
from datasets import Dataset, DatasetDict, Features, Value, load_from_disk
from numpy.random import RandomState
from sklearn.utils import resample
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from energizer.datastores.base import Datastore
from energizer.enums import InputKeys, RunningStage, SpecialKeys
from energizer.utilities import _pad, ld_to_dl


class PandasDataStore(Datastore):
    data: pd.DataFrame
    test_data: Optional[Dataset] = None
    validation_data: Optional[Dataset] = None

    input_names: List[str]
    target_name: str
    on_cpu: List[str]
    _features: Features

    def train_dataset(self, round: Optional[int] = None, passive: Optional[bool] = False) -> Optional[Dataset]:
        if passive:
            return Dataset.from_pandas(self.data, preserve_index=False)

        mask = self._train_mask(round)
        if mask.sum() > 0:
            return Dataset.from_pandas(self.data.loc[mask], preserve_index=False)

    def validation_dataset(self, round: Optional[int] = None) -> Optional[Dataset]:
        if self.validation_data:
            return self.validation_data

        mask = self._validation_mask(round)
        if mask.sum() > 0:
            return Dataset.from_pandas(self.data.loc[mask], preserve_index=False)

    def test_dataset(self) -> Optional[Dataset]:
        if self.test_data is not None:
            return self.test_data

    def pool_dataset(
        self, round: Optional[int] = None, subset_indices: Optional[List[int]] = None
    ) -> Optional[Dataset]:
        df = self.data.loc[self._pool_mask(round), [i for i in self.data.columns if i != self.target_name]]
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
            currentdata = self.data.loc[mask, [SpecialKeys.ID, self.target_name]]
            val_indices = sample(
                indices=currentdata[SpecialKeys.ID].tolist(),
                size=n_val,
                labels=currentdata[self.target_name].tolist(),
                sampling=validation_sampling,
                random_state=self._rng,
            )
            self.data.loc[self.data[SpecialKeys.ID].isin(val_indices), SpecialKeys.IS_VALIDATION] = True

        return mask.sum()

    def sample_from_pool(
        self, size: int, mode: Optional[str], round: Optional[int] = None, random_state: Optional[RandomState] = None
    ) -> List[int]:
        data = self.data.loc[self._pool_mask(round), [SpecialKeys.ID, self.target_name]]
        return sample(
            indices=data[SpecialKeys.ID].tolist(),
            size=size,
            random_state=random_state or self._rng,
            labels=data[self.target_name].tolist(),
            sampling=mode,
        )

    def pool_size(self, round: Optional[int] = None) -> int:
        return self._pool_mask(round).sum()

    def train_size(self, round: Optional[int] = None) -> int:
        return self._train_mask(round).sum()

    def validation_size(self, round: Optional[int] = None) -> int:
        if self.validation_data is not None:
            return len(self.validation_data)
        return self._validation_mask(round).sum()

    def labelled_size(self, round: Optional[int] = None) -> int:
        return self._labelled_mask(round).sum()

    def test_size(self) -> int:
        return len(self.test_data) if self.test_data is not None else 0

    def get_num_labelled_at_round(self, round: Optional[int] = None) -> int:
        last_round = round or self.data[SpecialKeys.LABELLING_ROUND].max()
        if last_round < 0:
            return self.labelled_size(last_round)
        return self.labelled_size(last_round) - self.labelled_size(last_round - 1)

    def total_rounds(self) -> int:
        return self.data[SpecialKeys.LABELLING_ROUND].max()

    def from_datasets(
        self,
        input_names: Union[str, List[str]],
        target_name: str,
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        on_cpu: Optional[Union[str, List[str]]] = None,
    ) -> None:

        # set attributes
        self.target_name = target_name
        self.input_names = [input_names] if isinstance(input_names, str) else input_names
        on_cpu = on_cpu or []
        self.on_cpu = [on_cpu] if isinstance(on_cpu, str) else on_cpu
        # cols = self.input_names + [self.target_name] + list(SpecialKeys) + self.on_cpu

        # create training data
        data: pd.DataFrame = train_dataset.to_pandas()  # type: ignore
        features = {k: v for k, v in train_dataset.features.items()}

        # add special columns if not present
        for k, v, f in [
            (SpecialKeys.ID, None, Value(dtype="int64", id=None)),
            (SpecialKeys.IS_LABELLED, False, Value(dtype="bool", id=None)),
            (SpecialKeys.IS_VALIDATION, False, Value(dtype="bool", id=None)),
            (SpecialKeys.LABELLING_ROUND, -100, Value(dtype="int64", id=None)),
        ]:
            if k in data.columns:
                continue
            elif k == SpecialKeys.ID:
                data[SpecialKeys.ID.value] = data.index.copy()
                assert data[SpecialKeys.ID].nunique() == len(data)  # check consistency of the index
                features[SpecialKeys.ID.value] = f
            else:
                data[k.value] = v
                features[k.value] = f

        self.data = data  # .loc[:, cols]
        self._features = Features(**features)
        if validation_dataset is not None:
            self.validation_data = validation_dataset
        if test_dataset is not None:
            self.test_data = test_dataset

    def from_dataset_dict(
        self,
        dataset_dict: DatasetDict,
        input_names: Union[str, List[str]],
        target_name: str,
        on_cpu: Optional[List[str]] = None,
    ) -> None:
        self.from_datasets(
            train_dataset=dataset_dict[RunningStage.TRAIN],
            validation_dataset=dataset_dict.get(RunningStage.VALIDATION, None),
            test_dataset=dataset_dict.get(RunningStage.TEST, None),
            input_names=input_names,
            target_name=target_name,
            on_cpu=on_cpu,
        )

    def get_by_ids(self, ids: List[int]) -> pd.DataFrame:
        return self.data.loc[self.data[SpecialKeys.ID].isin(ids)]

    def _labelled_mask(self, round: Optional[int] = None) -> pd.Series:
        mask = self.data[SpecialKeys.IS_LABELLED] == True
        if round is not None:
            mask = mask & (self.data[SpecialKeys.LABELLING_ROUND] <= round)
        return mask

    def _train_mask(self, round: Optional[int] = None) -> pd.Series:
        return self._labelled_mask(round) & (self.data[SpecialKeys.IS_VALIDATION] == False)

    def _validation_mask(self, round: Optional[int] = None) -> pd.Series:
        return self._labelled_mask(round) & (self.data[SpecialKeys.IS_VALIDATION] == True)

    def _pool_mask(self, round: Optional[int] = None) -> pd.Series:
        mask = self.data[SpecialKeys.IS_LABELLED] == False
        if round is not None:
            mask = mask | (self.data[SpecialKeys.LABELLING_ROUND] > round)
        return mask


class PandasDataStoreWithIndex(PandasDataStore):
    """DataModule that defines dataloading and indexing logic."""

    index: hb.Index = None

    def label(
        self,
        indices: List[int],
        round: int,
        validation_perc: Optional[float] = None,
        validation_sampling: Optional[str] = None,
    ) -> int:
        out = super().label(indices, round, validation_perc, validation_sampling)
        # remove instance from the index
        if self.index is not None:
            for idx in indices:
                self.index.mark_deleted(idx)
        return out

    def search(
        self, query: np.ndarray, query_size: int, query_in_set: bool = True
    ) -> Tuple[List[List[int]], np.ndarray]:
        # retrieve one additional element if the query is in the set we are looking in
        # because the query itself is returned as the most similar element and we need to remove it
        query_size = query_size + 1 if query_in_set else query_size
        indices, distances = self.index.knn_query(data=query, k=query_size)
        if query_in_set:
            # remove the first element retrieved if the query is in the set since it's the element itself
            indices, distances = indices[:, 1:], distances[:, 1:]
        return indices.tolist(), distances

    def add_index(
        self,
        embedding_name: str,
        metric: str = "cosine",
        ef_construction: int = 200,
        ef: int = 200,
        M: int = 64,
        num_threads: int = 5,
    ) -> None:
        embeddings = np.stack(self.data[embedding_name].tolist())
        max_elements, dim = embeddings.shape

        # create hnsw index
        self.index = hb.Index(space=metric, dim=dim)
        self.index.set_ef(ef)
        self.index.init_index(max_elements=max_elements, M=M, ef_construction=ef_construction, random_seed=42)
        self.index.add_items(embeddings, self.data[SpecialKeys.ID].values, num_threads=num_threads)

    def save_index(self, dir: Union[str, Path]) -> None:
        if self.index is not None:
            self.index.save_index(str(Path(dir) / "hnswlib_index.bin"))
            meta = {"dim": self.index.dim, "metric": self.index.space}
            srsly.write_json(Path(dir) / "hnswlib_index_config.json", meta)

    def load_index(self, dir: Union[str, Path]) -> None:
        dir = Path(dir)
        if (dir / "hnswlib_index_config.json").exists():
            meta: Dict = srsly.read_json(dir / "hnswlib_index_config.json")  # type: ignore
            index = hb.Index(space=meta["metric"], dim=meta["dim"])
            index.load_index(str(dir / "hnswlib_index.bin"))
            self.index = index


class PandasDataStoreForSequenceClassification(PandasDataStoreWithIndex):
    _tokenizer: Optional[PreTrainedTokenizerBase]
    _labels: List[str]
    _label_distribution: Dict[str, int]

    def prepare_for_loading(
        self,
        batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        persistent_workers: bool = False,
        shuffle: bool = True,
        seed: int = 42,
        replacement: bool = False,
        max_length: int = 512,
    ) -> None:
        super().prepare_for_loading(
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
    def tokenizer(self) -> Optional[PreTrainedTokenizerBase]:
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

    def from_datasets(
        self,
        input_names: Union[str, List[str]],
        target_name: str,
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        on_cpu: Optional[List[str]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        self._tokenizer = tokenizer
        self._labels = train_dataset.features[target_name].names
        self._label_distribution = Counter(train_dataset[target_name])
        return super().from_datasets(
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            input_names=input_names,
            target_name=target_name,
            on_cpu=on_cpu,
        )

    def from_dataset_dict(
        self,
        dataset_dict: DatasetDict,
        input_names: Union[str, List[str]],
        target_name: str,
        on_cpu: Optional[List[str]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        self.from_datasets(
            train_dataset=dataset_dict[RunningStage.TRAIN],
            validation_dataset=dataset_dict.get(RunningStage.VALIDATION, None),
            test_dataset=dataset_dict.get(RunningStage.TEST, None),
            input_names=input_names,
            target_name=target_name,
            on_cpu=on_cpu,
            tokenizer=tokenizer,
        )

    def get_collate_fn(self, stage: Optional[RunningStage] = None) -> Optional[Callable]:
        on_cpu = self.on_cpu if self.on_cpu is not None else []
        return partial(
            collate_fn,
            input_names=self.input_names,
            target_name=self.target_name,
            on_cpu=on_cpu + [SpecialKeys.ID],
            max_length=self.max_length,
            pad_token_id=self.tokenizer.pad_token_id if self.tokenizer else 0,
            pad_fn=_pad,
        )

    def save(self, dir: Union[str, Path]) -> None:
        dir = Path(dir)

        datasets = {RunningStage.TRAIN: Dataset.from_pandas(self.data, preserve_index=False, features=self._features)}
        if self.validation_data:
            datasets[RunningStage.VALIDATION] = self.validation_data
        if self.test_data:
            datasets[RunningStage.TEST] = self.test_data

        for split, dataset in datasets.items():
            dataset.save_to_disk(dir / split)

        meta = {
            "input_names": self.input_names,
            "target_name": self.target_name,
            "on_cpu": self.on_cpu,
            "name_or_path": self.tokenizer.name_or_path if self.tokenizer else None,
            "seed": self.seed,
        }
        srsly.write_json(dir / "metadata.json", meta)
        self.save_index(dir)

    @classmethod
    def load(cls, dir: Union[str, Path]) -> "PandasDataStoreForSequenceClassification":
        dir = Path(dir)
        datasets = {split: load_from_disk(dir / split) for split in RunningStage if (dir / split).exists()}
        meta: Dict = srsly.read_json(dir / "metadata.json")  # type: ignore
        tokenizer = None
        if meta["name_or_path"] is not None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(meta["name_or_path"])

        out = cls(meta["seed"])
        out.from_datasets(
            train_dataset=datasets.get(RunningStage.TRAIN, None),  # type: ignore
            validation_dataset=datasets.get(RunningStage.VALIDATION, None),  # type: ignore
            test_dataset=datasets.get(RunningStage.TEST, None),  # type: ignore
            input_names=meta["input_names"],
            target_name=meta["target_name"],
            on_cpu=meta["on_cpu"],
            tokenizer=tokenizer,
        )
        out.load_index(dir)

        return out


def collate_fn(
    batch: List[Dict[str, Union[List[str], Tensor]]],
    input_names: List[str],
    target_name: str,
    on_cpu: List[str],
    max_length: Optional[int],
    pad_token_id: Optional[int],
    pad_fn: Callable,
) -> Dict[str, Union[List[str], Tensor]]:

    new_batch = ld_to_dl(batch)

    # remove string columns that cannot be transfered on gpu
    values_on_cpu = {col: new_batch.pop(col, None) for col in on_cpu if col in new_batch}

    labels = new_batch.pop(target_name, None)

    # input_ids and attention_mask to tensor: truncate -> convert to tensor -> pad
    new_batch = {
        k: pad_fn(
            inputs=new_batch[k],
            padding_value=pad_token_id,
            max_length=max_length,
        )
        for k in input_names
    }

    if labels is not None:
        new_batch[target_name] = torch.tensor(labels, dtype=torch.long)

    # add things that need to remain on cpu
    if len(on_cpu) > 0:
        new_batch[InputKeys.ON_CPU] = values_on_cpu

    return new_batch


def sample(
    indices: List[int],
    size: int,
    random_state: RandomState,
    labels: Optional[List[int]] = None,
    sampling: Optional[str] = None,
) -> List[int]:
    """Makes sure to seed everything consistently."""

    if sampling is None or sampling == "uniform":
        sample = random_state.choice(indices, size=size, replace=False).tolist()

    elif sampling == "stratified" and labels is not None:
        sample = resample(
            indices,
            replace=False,
            stratify=labels,
            n_samples=size,
            random_state=random_state,
        )
        sample = [i for arr in sample for i in arr.flatten()]

    else:
        raise ValueError("Only `uniform` and `stratified` are supported by default.")

    assert len(set(sample)) == size

    return sample