from collections import Counter
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Union

import torch
from datasets import Dataset, DatasetDict  # type: ignore
from torch import Tensor
from transformers import PreTrainedTokenizerBase
from typing_extensions import Self

from energizer.datastores.base import PandasDataStoreWithIndex
from energizer.enums import InputKeys, RunningStage, SpecialKeys
from energizer.utilities import _pad, ld_to_dl, sequential_numbers


class SequenceClassificationMixin:
    MANDATORY_INPUT_NAMES: List[str] = [InputKeys.INPUT_IDS, InputKeys.ATT_MASK]
    OPTIONAL_INPUT_NAMES: List[str] = [InputKeys.TOKEN_TYPE_IDS]
    MANDATORY_TARGET_NAME: str = InputKeys.LABELS

    _tokenizer: PreTrainedTokenizerBase
    _labels: List[str]
    _label_distribution: Dict[str, int]
    input_names: List[str]
    on_cpu: List[str]

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
        super().prepare_for_loading(  # type: ignore
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
        self._loading_params["max_length"] = max_length  # type: ignore

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
        return _from_datasets(
            obj=obj,
            mandatory_input_names=cls.MANDATORY_INPUT_NAMES,
            optional_input_names=cls.OPTIONAL_INPUT_NAMES,
            mandatory_target_name=cls.MANDATORY_TARGET_NAME,
            tokenizer=tokenizer,
            uid_name=uid_name,
            on_cpu=on_cpu,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
        )

    @classmethod
    def from_dataset_dict(
        cls,
        dataset_dict: DatasetDict,
        tokenizer: PreTrainedTokenizerBase,
        uid_name: Optional[str] = None,
        on_cpu: Optional[List[str]] = None,
        seed: Optional[int] = 42,
    ) -> Self:
        return cls.from_datasets(
            tokenizer=tokenizer,
            uid_name=uid_name,
            on_cpu=on_cpu,
            seed=seed,
            train_dataset=dataset_dict.get(RunningStage.TRAIN),
            validation_dataset=dataset_dict.get(RunningStage.VALIDATION),
            test_dataset=dataset_dict.get(RunningStage.TEST),
        )

    def get_collate_fn(self, stage: Optional[RunningStage] = None, show_batch: bool = False) -> Optional[Callable]:
        return partial(
            collate_fn,
            input_names=self.input_names,
            on_cpu=self.on_cpu,
            max_length=None if show_batch else self.loading_params["max_length"],  # type: ignore
            pad_token_id=self.tokenizer.pad_token_id,
            pad_fn=_pad,
        )


class PandasDataStoreForSequenceClassification(SequenceClassificationMixin, PandasDataStoreWithIndex):
    ...


"""
Utilities
"""


def collate_fn(
    batch: List[Dict[str, Union[List[str], Tensor]]],
    input_names: List[str],
    on_cpu: List[str],
    max_length: Optional[int],
    pad_token_id: Optional[int],
    pad_fn: Callable,
) -> Dict[str, Union[List[str], Tensor]]:
    new_batch = ld_to_dl(batch)

    # remove string columns that cannot be transfered on gpu
    values_on_cpu = {col: new_batch.pop(col, None) for col in on_cpu if col in new_batch}

    labels = new_batch.pop(InputKeys.LABELS, None)

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
        new_batch[InputKeys.LABELS] = torch.tensor(labels, dtype=torch.long)

    # add things that need to remain on cpu
    if len(on_cpu) > 0:
        new_batch[InputKeys.ON_CPU] = values_on_cpu

    return new_batch


def _from_datasets(
    obj,
    mandatory_input_names: List[str],
    optional_input_names: List[str],
    mandatory_target_name: str,
    tokenizer: PreTrainedTokenizerBase,
    uid_name: Optional[str] = None,
    on_cpu: Optional[List[str]] = None,
    train_dataset: Optional[Dataset] = None,
    validation_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
) -> Any:

    _datasets = {
        RunningStage.TRAIN: train_dataset,
        RunningStage.VALIDATION: validation_dataset,
        RunningStage.TEST: test_dataset,
    }
    datasets: Dict[RunningStage, Dataset] = {k: v for k, v in _datasets.items() if v is not None}
    if len(datasets) < 1:
        raise ValueError("You need to pass at least one dataset.")

    # === INPUT NAMES === #
    input_names = []
    for name in mandatory_input_names:
        for split, dataset in datasets.items():
            if name in dataset.features:
                input_names.append(name)
            else:
                raise ValueError(f"Mandatory column {name} not in dataset[{split}].")

    for name in optional_input_names:
        for dataset in datasets.values():
            if name in dataset.features:
                input_names.append(name)

    # === TARGET NAME === #
    labels = []
    for split, dataset in datasets.items():
        assert (
            mandatory_target_name in dataset.features
        ), f"Mandatory column {mandatory_target_name} not in dataset[{split}]."
        labels.append(set(dataset.features[mandatory_target_name].names))

    # check labels are consistent
    assert all(s == labels[0] for s in labels), "Labels are inconsistent across splits"

    # === ON_CPU === #
    if on_cpu is not None:
        for name in on_cpu:
            for split, dataset in datasets.items():
                assert name in dataset.features, f"{name=} not in dataset[{split}]={dataset.features.keys()}"
    else:
        on_cpu = []

    # === UID NAME === #
    uid_generator = sequential_numbers()
    new_datasets = {}
    for k, d in datasets.items():
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

    on_cpu += [SpecialKeys.ID]

    # === FORMAT (KEEP ONLY USEFUL COLUMNS) === #
    columns = input_names + on_cpu + [mandatory_target_name]
    new_datasets = {k: v.with_format(columns=columns) for k, v in new_datasets.items()}

    # === SET ATTRIBUTES === #
    if RunningStage.TRAIN in new_datasets:
        obj._label_distribution = Counter(new_datasets[RunningStage.TRAIN][mandatory_target_name])

    obj._labels = next(iter(new_datasets.values())).features[mandatory_target_name].names
    obj.input_names = input_names
    obj.on_cpu = on_cpu
    obj._tokenizer = tokenizer
    obj._train_data = new_datasets[RunningStage.TRAIN].to_pandas()  # type: ignore
    obj._validation_data = new_datasets.get(RunningStage.VALIDATION)  # type: ignore
    obj._test_data = new_datasets.get(RunningStage.TEST)  # type: ignore

    return obj
