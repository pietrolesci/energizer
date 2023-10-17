from functools import partial
from itertools import chain
from typing import Any, Callable, Dict, List, Optional, Union

from datasets import Dataset, DatasetDict  # type: ignore
from torch import Tensor
from transformers import PreTrainedTokenizerBase
from typing_extensions import Self

from energizer.datastores.base import PandasDataStoreWithIndex
from energizer.enums import InputKeys, RunningStage, SpecialKeys
from energizer.utilities import _pad, ld_to_dl, sequential_numbers


class LanguageModellingMixin:
    _tokenizer: PreTrainedTokenizerBase

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
        obj = cls(seed)
        return _from_datasets(
            obj=obj,
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
            max_length=None if show_batch else self.loading_params["max_length"],
            pad_token_id=self.tokenizer.pad_token_id,
            pad_fn=_pad,
        )

    def group_texts(self, block_size: int) -> None:
        dataset = self.train_dataset()
        dataset = dataset.with_format(columns=self.input_names)
        dataset = dataset.map(lambda ex: group_texts(ex, block_size=block_size), batched=True)
        self._train_data = dataset.to_pandas()


class PandasDataStoreForLanguageModelling(LanguageModellingMixin, PandasDataStoreWithIndex):
    ...


"""
Utilities
"""


def collate_fn(
    batch: List[Dict[str, Union[List[str], Tensor]]],
    input_names: str,
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

    # labels substitute pad_token_id with -100
    labels = new_batch[InputKeys.INPUT_IDS].clone()
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    new_batch[InputKeys.LABELS] = labels

    # add things that need to remain on cpu
    if len(on_cpu) > 0:
        new_batch[InputKeys.ON_CPU] = values_on_cpu

    return new_batch


def _from_datasets(
    obj: Any,
    tokenizer: PreTrainedTokenizerBase,
    uid_name: Optional[str] = None,
    on_cpu: Optional[List[str]] = None,
    train_dataset: Optional[Dataset] = None,
    validation_dataset: Optional[Dataset] = None,
    test_dataset: Optional[Dataset] = None,
) -> Any:
    obj._tokenizer = tokenizer
    obj.input_names = [InputKeys.INPUT_IDS, InputKeys.ATT_MASK]
    obj.on_cpu = on_cpu or []

    _datasets = {
        RunningStage.TRAIN: train_dataset,
        RunningStage.VALIDATION: validation_dataset,
        RunningStage.TEST: test_dataset,
    }
    datasets: Dict[RunningStage, Dataset] = {k: v for k, v in _datasets.items() if v is not None}
    if len(datasets) < 1:
        raise ValueError("You need to pass at least one dataset.")

    uid_generator = sequential_numbers()
    columns = obj.input_names + obj.on_cpu
    if uid_name is not None:
        columns.append(uid_name)

    new_datasets = {}
    for k, d in datasets.items():
        # check all columns are present in all datasets
        for col in columns:
            assert col in d.features.keys(), f"{col=} not in dataset={d.features.keys()}"

        if uid_name is None:
            uids = [next(uid_generator) for _ in range(len(d))]
            new_dataset = d.add_column(SpecialKeys.ID, uids)
        else:
            # check
            col = list(datasets[RunningStage.TRAIN][SpecialKeys.ID])
            assert len(set(col)) == len(col), f"`uid_column` {uid_name} is not unique."
            new_dataset = d.rename_columns({uid_name: SpecialKeys.ID})

        new_datasets[k] = new_dataset

    obj.on_cpu += [SpecialKeys.ID]

    # set data sources
    new_datasets = {k: v.with_format(columns=obj.input_names + obj.on_cpu) for k, v in new_datasets.items()}

    obj._train_data = new_datasets[RunningStage.TRAIN].to_pandas()
    obj._validation_data = new_datasets.get(RunningStage.VALIDATION)
    obj._test_data = new_datasets.get(RunningStage.TEST)

    return obj


def group_texts(examples: Dict, block_size: int) -> Dict:
    """Concatenate all texts from our dataset and generate chunks of block_size.

    Refs: https://github.com/huggingface/transformers/blob/bfb1895e3346cb8a2bf2560c75d45e70edf46a47/examples/pytorch/language-modeling/run_clm_no_trainer.py#L456
    """

    # Concatenate all texts
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    # concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict
    # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
    total_length = (total_length // block_size) * block_size

    # Split by chunks of max_len
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()
    }

    return result
