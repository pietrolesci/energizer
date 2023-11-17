from abc import ABC, abstractmethod
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import hnswlib as hb
import numpy as np
import srsly
from datasets import Dataset, DatasetDict  # type: ignore
from lightning_utilities.core.rank_zero import rank_zero_warn
from transformers import PreTrainedTokenizerBase
from typing_extensions import Self

from energizer.enums import InputKeys, RunningStage, SpecialKeys
from energizer.utilities import sequential_numbers


class TextMixin(ABC):
    MANDATORY_INPUT_NAMES: List[str] = [InputKeys.INPUT_IDS, InputKeys.ATT_MASK]
    OPTIONAL_INPUT_NAMES: List[str] = [InputKeys.TOKEN_TYPE_IDS]
    MANDATORY_TARGET_NAME: Optional[str] = None

    _tokenizer: PreTrainedTokenizerBase
    input_names: List[str]
    on_cpu: List[str]

    @abstractmethod
    def get_collate_fn(self, stage: Optional[RunningStage] = None, show_batch: bool = False) -> Optional[Callable]:
        ...

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return self._tokenizer

    @classmethod
    def from_datasets(
        cls,
        tokenizer: PreTrainedTokenizerBase,
        uid_name: Optional[str] = None,
        on_cpu: Optional[List[str]] = None,
        seed: Optional[int] = 42,
        train_dataset: Optional[Dataset] = None,
        validation_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
    ) -> Self:
        obj = cls(seed)  # type: ignore
        obj.load_datasets(
            tokenizer=tokenizer,
            uid_name=uid_name,
            on_cpu=on_cpu,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            mandatory_input_names=obj.MANDATORY_INPUT_NAMES,
            optional_input_names=obj.OPTIONAL_INPUT_NAMES,
            mandatory_target_name=obj.MANDATORY_TARGET_NAME,
        )
        return obj

    @classmethod
    def from_dataset_dict(
        cls,
        dataset_dict: DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
        uid_name: Optional[str] = None,
        on_cpu: Optional[List[str]] = None,
        seed: Optional[int] = 42,
    ) -> Self:
        obj = cls(seed)  # type: ignore
        obj.load_datasets(
            tokenizer=tokenizer,
            uid_name=uid_name,
            on_cpu=on_cpu,
            train_dataset=dataset_dict.get(RunningStage.TRAIN),
            validation_dataset=dataset_dict.get(RunningStage.VALIDATION),
            test_dataset=dataset_dict.get(RunningStage.TEST),
            mandatory_input_names=obj.MANDATORY_INPUT_NAMES,
            optional_input_names=obj.OPTIONAL_INPUT_NAMES,
            mandatory_target_name=obj.MANDATORY_TARGET_NAME,
        )
        return obj

    def load_datasets(
        self,
        mandatory_input_names: List[str],
        optional_input_names: List[str],
        mandatory_target_name: Optional[str],
        tokenizer: PreTrainedTokenizerBase,
        uid_name: Optional[str] = None,
        on_cpu: Optional[List[str]] = None,
        train_dataset: Optional[Dataset] = None,
        validation_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
    ) -> None:
        _datasets = {
            RunningStage.TRAIN: train_dataset,
            RunningStage.VALIDATION: validation_dataset,
            RunningStage.TEST: test_dataset,
        }
        datasets: Dict[RunningStage, Dataset] = {k: v for k, v in _datasets.items() if v is not None}
        if len(datasets) < 1:
            raise ValueError("You need to pass at least one dataset.")

        # === INPUT NAMES === #
        self._check_input_names(datasets, mandatory_input_names, optional_input_names)

        # === TARGET NAME === #
        if mandatory_target_name is not None:
            self._check_labels(datasets, mandatory_target_name)

        # === ON_CPU === #
        self._check_on_cpu(datasets, on_cpu)

        # === UID NAME === #
        datasets = self._check_uid(datasets, uid_name)
        self.on_cpu += [SpecialKeys.ID]

        # === FORMAT (KEEP ONLY USEFUL COLUMNS) === #
        datasets = self._format_datasets(datasets)

        # === SET ATTRIBUTES === #
        self._set_attributes(datasets, tokenizer)

    def _check_input_names(
        self,
        dataset_dict: Dict[RunningStage, Dataset],
        mandatory_input_names: List[str],
        optional_input_names: List[str],
    ) -> None:
        input_names = []
        for name in mandatory_input_names:
            for split, dataset in dataset_dict.items():
                if name in dataset.features:
                    input_names.append(name)
                else:
                    raise ValueError(f"Mandatory column {name} not in dataset[{split}].")

        for name in optional_input_names:
            for dataset in dataset_dict.values():
                if name in dataset.features:
                    input_names.append(name)

        self.input_names = list(set(input_names))

    def _check_on_cpu(self, dataset_dict: Dict[RunningStage, Dataset], on_cpu: Optional[List[str]]) -> None:
        _on_cpu = []
        if on_cpu is not None:
            for name in on_cpu:
                for split, dataset in dataset_dict.items():
                    assert name in dataset.features, f"{name=} not in dataset[{split}]={dataset.features.keys()}"
                _on_cpu.append(name)
        self.on_cpu = list(set(_on_cpu))

    def _check_uid(
        self, dataset_dict: Dict[RunningStage, Dataset], uid_name: Optional[str]
    ) -> Dict[RunningStage, Dataset]:
        uid_generator = sequential_numbers()
        new_datasets = {}
        for k, d in dataset_dict.items():
            if uid_name is None:
                uids = [next(uid_generator) for _ in range(len(d))]
                new_dataset = d.add_column(SpecialKeys.ID, uids)  # type: ignore
                print(f"UID column {SpecialKeys.ID} automatically created in dataset[{k}]")
            else:
                assert uid_name in d.features, f"{uid_name=} not in dataset[{k}]={d.features.keys()}"
                ids = d[uid_name]
                assert len(set(ids)) == len(ids), f"`uid_column` {uid_name} is not unique."

                new_dataset = d
                if uid_name != SpecialKeys.ID:
                    new_dataset = new_dataset.rename_columns({uid_name: SpecialKeys.ID})
                    print(f"UID column {uid_name} automatically renamed to {SpecialKeys.ID} in dataset[{k}]")

            new_datasets[k] = new_dataset

        return new_datasets

    def _check_labels(self, dataset_dict: Dict[RunningStage, Dataset], mandatory_target_name: str) -> None:
        for split, dataset in dataset_dict.items():
            assert (
                mandatory_target_name in dataset.features
            ), f"Mandatory column {mandatory_target_name} not in dataset[{split}]."

    def _format_datasets(self, dataset_dict: Dict[RunningStage, Dataset]) -> Dict[RunningStage, Dataset]:
        columns = self.input_names + self.on_cpu
        if self.MANDATORY_TARGET_NAME is not None:
            columns.append(self.MANDATORY_TARGET_NAME)
        return {k: v.with_format(columns=columns) for k, v in dataset_dict.items()}

    def _set_attributes(self, dataset_dict: Dict[RunningStage, Dataset], tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer
        self._train_data = dataset_dict.get(RunningStage.TRAIN)  # type: ignore
        self._validation_data = dataset_dict.get(RunningStage.VALIDATION)  # type: ignore
        self._test_data = dataset_dict.get(RunningStage.TEST)  # type: ignore


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
            rank_zero_warn(
                "Index has more ids than dataset. Masking the missing ids from the index. "
                "If this is expected (e.g., you downsampled your dataset), everything is fine."
            )
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
