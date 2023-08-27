from energizer.datastores.base import Datastore
from typing import Optional, List, Union, Literal
from torch.utils.data import DataLoader
from energizer.enums import RunningStage, SpecialKeys, InputKeys
from numpy.random import RandomState
from energizer.datastores.base import PandasDataStore, IndexMixin
from datasets import Dataset
from pathlib import Path
import pandas as pd
from math import floor
from energizer.utilities import sample
import numpy as np


class ActiveLearningMixin:
    def pool_size(self, round: Optional[int] = None) -> int:
        raise NotImplementedError

    def labelled_size(self, round: Optional[int] = None) -> int:
        raise NotImplementedError

    def query_size(self, round: Optional[int] = None) -> int:
        raise NotImplementedError

    def total_rounds(self) -> int:
        raise NotImplementedError

    def label(
        self,
        indices: List[int],
        round: int,
        validation_perc: Optional[float] = None,
        validation_sampling: Literal["uniform", "stratified"] = "uniform",
    ) -> int:
        raise NotImplementedError

    def sample_from_pool(
        self,
        size: int,
        round: Optional[int] = None,
        random_state: Optional[RandomState] = None,
        with_indices: Optional[List[int]] = None,
        **kwargs,
    ) -> List[int]:
        raise NotImplementedError

    def save_labelled_dataset(self, save_dir: Union[str, Path]) -> None:
        raise NotImplementedError

    def pool_loader(self, *args, **kwargs) -> Optional[DataLoader]:
        return self.get_loader(RunningStage.POOL, *args, **kwargs)  # type: ignore

    def reset(self) -> None:
        raise NotImplementedError

    def get_train_ids(self, round: Optional[int] = None) -> List[int]:
        raise NotImplementedError

    def get_validation_ids(self, round: Optional[int] = None) -> List[int]:
        raise NotImplementedError

    def get_pool_ids(self, round: Optional[int] = None) -> List[int]:
        raise NotImplementedError


class ActiveDataStore(ActiveLearningMixin, Datastore):
    ...


class ActivePandasDataStore(ActiveLearningMixin, PandasDataStore):
    _train_data: pd.DataFrame
    _test_data: Optional[Dataset]

    def train_size(self, round: Optional[int] = None) -> int:
        return self._train_mask(round).sum()

    def pool_size(self, round: Optional[int] = None) -> int:
        return self._pool_mask(round).sum()

    def validation_size(self, round: Optional[int] = None) -> int:
        if self._validation_data is not None:
            return len(self._validation_data)
        return self._validation_mask(round).sum()

    def labelled_size(self, round: Optional[int] = None) -> int:
        return self._labelled_mask(round).sum()

    def query_size(self, round: Optional[int] = None) -> int:
        last_round = round or self._train_data[SpecialKeys.LABELLING_ROUND].max()
        if last_round < 0:
            return self.labelled_size(last_round)
        return self.labelled_size(last_round) - self.labelled_size(last_round - 1)

    def total_rounds(self) -> int:
        return self._train_data[SpecialKeys.LABELLING_ROUND].max()

    def train_dataset(
        self, round: Optional[int] = None, passive: Optional[bool] = False, with_indices: Optional[List[int]] = None
    ) -> Optional[Dataset]:
        if passive:
            return super().train_dataset()

        mask = self._train_mask(round)
        if mask.sum() > 0:
            if with_indices is not None:
                mask = mask & self._train_data[SpecialKeys.ID].isin(with_indices)
            return Dataset.from_pandas(self._train_data.loc[mask], preserve_index=False)

    def validation_dataset(self, round: Optional[int] = None) -> Optional[Dataset]:
        if self._validation_data is not None:
            return self._validation_data

        mask = self._validation_mask(round)
        if mask.sum() > 0:
            return Dataset.from_pandas(self._train_data.loc[mask], preserve_index=False)

    def pool_dataset(self, round: Optional[int] = None, with_indices: Optional[List[int]] = None) -> Optional[Dataset]:
        mask = self._pool_mask(round)
        if with_indices is not None:
            mask = mask & self._train_data[SpecialKeys.ID].isin(with_indices)
        return Dataset.from_pandas(
            self._train_data.loc[mask, [i for i in self._train_data.columns if i != InputKeys.TARGET]]
        )

    def label(
        self,
        indices: List[int],
        round: int,
        validation_perc: Optional[float] = None,
        validation_sampling: Literal["uniform", "stratified"] = "uniform",
    ) -> int:
        assert isinstance(indices, list), ValueError(f"`indices` must be of type `List[int]`, not {type(indices)}.")
        assert (isinstance(validation_perc, float) and validation_perc > 0.0) or validation_perc is None, ValueError(
            f"`validation_perc` must be of type `float` and > 0.0, not {type(validation_perc)}"
        )

        # label training data
        mask = self._train_data[SpecialKeys.ID].isin(indices)
        self._train_data.loc[mask, SpecialKeys.IS_LABELLED] = True
        self._train_data.loc[mask, SpecialKeys.LABELLING_ROUND] = round

        # train-validation split
        if validation_perc is not None:
            n_val = floor(validation_perc * len(indices)) or 1  # at least add one
            currentdata = self._train_data.loc[mask, [SpecialKeys.ID, InputKeys.TARGET]]
            val_indices = sample(
                indices=currentdata[SpecialKeys.ID].tolist(),
                size=n_val,
                labels=currentdata[InputKeys.TARGET].tolist(),
                mode=validation_sampling,
                random_state=self._rng,
            )
            self._train_data.loc[self._train_data[SpecialKeys.ID].isin(val_indices), SpecialKeys.IS_VALIDATION] = True

        return mask.sum()

    def sample_from_pool(
        self,
        size: int,
        round: Optional[int] = None,
        random_state: Optional[RandomState] = None,
        with_indices: Optional[List[int]] = None,
        **kwargs,
    ) -> List[int]:
        """Performs `uniform` or `stratified` sampling from the pool."""

        mask = self._pool_mask(round)
        if with_indices:
            mask = mask & self._train_data[SpecialKeys.ID].isin(with_indices)
        data = self._train_data.loc[mask, [SpecialKeys.ID, InputKeys.TARGET]]

        return sample(
            indices=data[SpecialKeys.ID].tolist(),
            size=size,
            random_state=random_state or self._rng,
            labels=data[InputKeys.TARGET].tolist(),
            **kwargs,
        )

    def save_labelled_dataset(self, save_dir: Union[str, Path]) -> None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        self._train_data.loc[self._labelled_mask()].to_parquet(path / "labelled_dataset.parquet", index=False)

    """
    Utilities
    """

    def _labelled_mask(self, round: Optional[int] = None) -> pd.Series:
        mask = self._train_data[SpecialKeys.IS_LABELLED] == True
        if round is not None:
            mask = mask & (self._train_data[SpecialKeys.LABELLING_ROUND] <= round)
        return mask

    def _train_mask(self, round: Optional[int] = None) -> pd.Series:
        return self._labelled_mask(round) & (self._train_data[SpecialKeys.IS_VALIDATION] == False)

    def _validation_mask(self, round: Optional[int] = None) -> pd.Series:
        return self._labelled_mask(round) & (self._train_data[SpecialKeys.IS_VALIDATION] == True)

    def _pool_mask(self, round: Optional[int] = None) -> pd.Series:
        mask = self._train_data[SpecialKeys.IS_LABELLED] == False
        if round is not None:
            mask = mask | (self._train_data[SpecialKeys.LABELLING_ROUND] > round)
        return mask

    def get_train_ids(self, round: Optional[int] = None) -> List[int]:
        return self._train_data.loc[self._train_mask(round), SpecialKeys.ID].tolist()

    def get_validation_ids(self, round: Optional[int] = None) -> List[int]:
        return self._train_data.loc[self._validation_mask(round), SpecialKeys.ID].tolist()

    def get_pool_ids(self, round: Optional[int] = None) -> List[int]:
        return self._train_data.loc[self._pool_mask(round), SpecialKeys.ID].tolist()


class ActiveIndexMixin(IndexMixin):
    def get_pool_embeddings(self, ids: List[int]) -> np.ndarray:
        return np.stack(self.index.get_items(ids))

    def get_train_embeddings(self, ids: List[int]) -> np.ndarray:
        # check all the ids are training ids
        assert len(set(self.get_train_ids()).intersection(set(ids))) == len(ids)  # type: ignore

        # now that we are sure, let's unmask them and get the items
        self.unmask_ids_from_index(ids)
        emb = np.stack(self.index.get_items(ids))

        # remask the training ids
        self.mask_ids_from_index(ids)

        return emb

    def get_embeddings(self, ids: List[int]) -> None:
        raise ValueError("Use `get_{train/pool}_embeddings` methods instead.")


class ActiveDataStoreWithIndex(ActiveIndexMixin, ActiveDataStore):
    ...


class ActivePandasDataStoreWithIndex(ActiveIndexMixin, ActivePandasDataStore):
    def label(
        self,
        indices: List[int],
        round: int,
        validation_perc: Optional[float] = None,
        validation_sampling: Literal["uniform", "stratified"] = "uniform",
    ) -> int:

        n_labelled = super().label(indices, round, validation_perc, validation_sampling)

        if self.index is not None:
            # remove instance from the index
            self.mask_ids_from_index(indices)

        return n_labelled