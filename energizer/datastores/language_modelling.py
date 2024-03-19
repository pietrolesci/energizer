from collections.abc import Callable  # , Generator, Any
from dataclasses import dataclass
from functools import partial

from datasets import Dataset
from lightning_utilities.core.rank_zero import rank_zero_info
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from energizer.datastores.base import DataloaderArgs, Datastore, PandasDatastoreWithIndex
from energizer.datastores.mixins import TextMixin
from energizer.enums import InputKeys, RunningStage
from energizer.utilities import _pad, ld_to_dl

# from itertools import islice


def collate_fn_for_language_modelling(
    batch: list[dict[str, list[str] | Tensor]],
    input_names: list[str],
    on_cpu: list[str],
    max_length: int | None,
    pad_token_id: int | None,
    pad_fn: Callable,
    return_labels: bool,
) -> dict[str, list[str] | Tensor]:
    new_batch = ld_to_dl(batch)

    # remove string columns that cannot be transfered on gpu
    values_on_cpu = {col: new_batch.pop(col, None) for col in on_cpu if col in new_batch}

    labels = new_batch.pop(InputKeys.LABELS, None)

    # input_ids and attention_mask to tensor: truncate -> convert to tensor -> pad
    new_batch = {k: pad_fn(inputs=new_batch[k], padding_value=pad_token_id, max_length=max_length) for k in input_names}

    # labels substitute pad_token_id with -100
    if return_labels:
        labels = new_batch[InputKeys.INPUT_IDS].clone()
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100
        new_batch[InputKeys.LABELS] = labels

    # add things that need to remain on cpu
    if len(values_on_cpu) > 0:
        new_batch[InputKeys.ON_CPU] = values_on_cpu

    return new_batch


@dataclass
class LanguageModellingDataloaderArgs(DataloaderArgs):
    max_length: int


class LanguageModellingMixin(TextMixin):
    MANDATORY_INPUT_NAMES: list[str] = [InputKeys.INPUT_IDS]
    MANDATORY_TARGET_NAME: str | None = None
    OPTIONAL_INPUT_NAMES: list[str] = []

    _loading_params: LanguageModellingDataloaderArgs | None = None
    _return_labels: bool = True

    def set_return_labels(self, mode: bool = True) -> None:
        self._return_labels = mode
        msg = f"Return labels is {self._return_labels}."
        if not self._return_labels:
            msg += " Remeber that you now need to manually shift the `input_ids` to obtain the labels."
        rank_zero_info(msg)

    @property
    def return_labels(self) -> bool:
        return self._return_labels

    def get_collate_fn(self, stage: RunningStage | None = None, show_batch: bool = False) -> Callable | None:
        return partial(
            collate_fn_for_language_modelling,
            input_names=self.input_names,
            on_cpu=self.on_cpu,
            max_length=None if show_batch else self.loading_params.max_length,  # type: ignore
            pad_token_id=self.tokenizer.pad_token_id,
            pad_fn=_pad,
            return_labels=self.return_labels,
        )

    def prepare_for_loading(
        self,
        batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        drop_last: bool = False,
        persistent_workers: bool = False,
        shuffle: bool = True,
        replacement: bool = False,
        data_seed: int = 42,
        multiprocessing_context: str | None = None,
        max_length: int = 512,
    ) -> None:
        self._loading_params = LanguageModellingDataloaderArgs(
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            shuffle=shuffle,
            data_seed=data_seed,
            replacement=replacement,
            multiprocessing_context=multiprocessing_context,
            max_length=max_length,
        )

    # def get_packed_dataset(self, dataset: Dataset, block_size: int) -> Dataset:
    #     iterable_dataset = dataset.to_iterable_dataset()
    #     input_ids_iterator = iter(token for ex in iterable_dataset for token in ex[InputKeys.INPUT_IDS])
    #     packed_dataset = Dataset.from_generator(batched(input_ids_iterator, block_size))
    #     return packed_dataset

    # def train_dataset(self) -> Optional[Dataset]:
    #     if self._train_data is not None:
    #         return self._train_data

    # def validation_dataset(self) -> Optional[Dataset]:
    #     if self._validation_data is not None:
    #         return self._validation_data

    # def test_dataset(self) -> Optional[Dataset]:
    #     if self._test_data is not None:
    #         return self._test_data


class DatastoreForLanguageModelling(LanguageModellingMixin, Datastore): ...


class PandasDatastoreForLanguageModelling(LanguageModellingMixin, PandasDatastoreWithIndex):
    def _set_attributes(self, dataset_dict: dict[RunningStage, Dataset], tokenizer: PreTrainedTokenizerBase) -> None:
        super()._set_attributes(dataset_dict, tokenizer)
        self._train_data = self._train_data.to_pandas()  # type: ignore


# def batched(iterator, chunk_size) -> Generator[list[Any], Any, None]:
# 		while chunk := list(islice(iterator, chunk_size)):
# 			yield chunk

# def group_texts(examples: dict, block_size: int) -> dict:
# 	"""Concatenate all texts from our dataset and generate chunks of block_size.

# 	Refs: https://github.com/huggingface/transformers/blob/bfb1895e3346cb8a2bf2560c75d45e70edf46a47/examples/pytorch/language-modeling/run_clm_no_trainer.py#L456
# 	"""

# 	# Concatenate all texts
# 	concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
# 	# concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

# 	total_length = len(concatenated_examples[next(iter(examples.keys()))])

# 	# We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict
# 	# We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
# 	total_length = (total_length // block_size) * block_size

# 	# Split by chunks of max_len
# 	result = {
# 		k: [t[i : i + block_size] for i in range(0, total_length, block_size)] for k, t in concatenated_examples.items()
# 	}

# 	return result
