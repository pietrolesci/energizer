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
        self._loading_params["max_length"] = max_length

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
        input_names: Union[str, List[str]],
        target_name: str,
        tokenizer: PreTrainedTokenizerBase,
        uid_name: Optional[str] = None,
        on_cpu: Optional[List[str]] = None,
        seed: Optional[int] = 42,
        train_dataset: Optional[Dataset] = None,
        validation_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
    ) -> Self:
        obj = cls(seed)
        return _from_datasets(
            obj=obj,
            input_names=input_names,
            target_name=target_name,
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
        input_names: Union[str, List[str]],
        target_name: str,
        tokenizer: PreTrainedTokenizerBase,
        uid_name: Optional[str] = None,
        on_cpu: Optional[List[str]] = None,
        seed: Optional[int] = 42,
    ) -> Self:
        return cls.from_datasets(
            input_names=input_names,
            target_name=target_name,
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
            max_length=None if show_batch else self.loading_params["max_length"],
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

    labels = new_batch.pop(InputKeys.TARGET, None)

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
        new_batch[InputKeys.TARGET] = torch.tensor(labels, dtype=torch.long)

    # add things that need to remain on cpu
    if len(on_cpu) > 0:
        new_batch[InputKeys.ON_CPU] = values_on_cpu

    return new_batch


def _from_datasets(
    obj: Any,
    input_names: Union[str, List[str]],
    target_name: str,
    tokenizer: PreTrainedTokenizerBase,
    uid_name: Optional[str] = None,
    on_cpu: Optional[List[str]] = None,
    train_dataset: Optional[Dataset] = None,
    validation_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
) -> Any:
    obj._tokenizer = tokenizer

    datasets = {
        RunningStage.TRAIN: train_dataset,
        RunningStage.VALIDATION: validation_dataset,
        RunningStage.TEST: test_dataset,
    }
    datasets = DatasetDict({k: v for k, v in datasets.items() if v is not None})

    # label distribution
    dataset = train_dataset or validation_dataset or test_dataset
    if dataset is None:
        raise ValueError("You need to pass at least one dataset.")
    obj._labels = dataset.features[target_name].names
    obj._label_distribution = Counter(dataset[target_name])

    # column names
    obj.input_names = [input_names] if isinstance(input_names, str) else input_names
    assert all(
        i in d.features for i in obj.input_names + [target_name] for d in datasets.values()
    ), "Check input/target names passed."
    datasets = datasets.rename_columns({target_name: InputKeys.TARGET})

    obj.on_cpu = on_cpu or []
    for i in obj.on_cpu:
        for d in datasets.values():
            if i not in d.features:
                print(f"Some `on_cpu`={i} is not in dataset={d.features.keys()}")
    obj.on_cpu += [SpecialKeys.ID]

    if datasets.get(RunningStage.TRAIN) is not None:
        if uid_name is None:
            uid_generator = sequential_numbers()
            datasets = datasets.map(
                lambda ex: {SpecialKeys.ID: [next(uid_generator) for _ in range(len(ex[target_name]))]},
                batched=True,
            )
        else:
            # check
            col = list(datasets[RunningStage.TRAIN][SpecialKeys.ID])
            assert len(set(col)) == len(col), f"`uid_column` {uid_name} is not unique."
            datasets[RunningStage.TRAIN] = datasets[RunningStage.TRAIN].rename_columns({uid_name: SpecialKeys.ID})

    # set data sources
    datasets = datasets.select_columns(obj.input_names + obj.on_cpu + [InputKeys.TARGET])  # type: ignore
    obj._train_data = datasets[RunningStage.TRAIN].to_pandas()  # type: ignore
    obj._validation_data = datasets.get(RunningStage.VALIDATION)
    obj._test_data = datasets.get(RunningStage.TEST)

    return obj
