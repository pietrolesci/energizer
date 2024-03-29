from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

from datasets import Dataset
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from energizer.datastores.base import DataloaderArgs, Datastore, PandasDatastoreWithIndex
from energizer.datastores.mixins import TextMixin
from energizer.enums import InputKeys, RunningStage
from energizer.utilities import _pad, ld_to_dl


def collate_fn_for_seq2seq(
    batch: list[dict[str, list[str] | Tensor]],
    input_names: list[str],
    on_cpu: list[str],
    max_source_length: int | None,
    max_target_length: int | None,
    pad_token_id: int | None,
    pad_fn: Callable,
) -> dict[str, list[str] | Tensor]:
    new_batch = ld_to_dl(batch)

    # remove string columns that cannot be transfered on gpu
    values_on_cpu = {col: new_batch.pop(col, None) for col in on_cpu if col in new_batch}

    labels = new_batch.pop(InputKeys.LABELS, None)

    # input_ids and attention_mask to tensor: truncate -> convert to tensor -> pad
    new_batch = {
        k: pad_fn(inputs=new_batch[k], padding_value=pad_token_id, max_length=max_source_length) for k in input_names
    }

    # labels substitute pad_token_id with -100
    labels = pad_fn(inputs=labels, padding_value=-100, max_length=max_target_length)
    new_batch[InputKeys.LABELS] = labels

    # add things that need to remain on cpu
    if len(values_on_cpu) > 0:
        new_batch[InputKeys.ON_CPU] = values_on_cpu

    return new_batch


@dataclass
class Seq2SeqDataloaderArgs(DataloaderArgs):
    max_source_length: int
    max_target_length: int


class Seq2SeqMixin(TextMixin):
    MANDATORY_TARGET_NAME: str | None = InputKeys.LABELS
    OPTIONAL_INPUT_NAMES: list[str] = []
    BLOCK_SIZE: int = 1_000
    _loading_params: Seq2SeqDataloaderArgs | None = None

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
        max_source_length: int = 512,
        max_target_length: int = 512,
    ) -> None:
        self._loading_params = Seq2SeqDataloaderArgs(
            batch_size=batch_size,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            persistent_workers=persistent_workers,
            shuffle=shuffle,
            replacement=replacement,
            data_seed=data_seed,
            multiprocessing_context=multiprocessing_context,
            max_source_length=max_source_length,
            max_target_length=max_target_length,
        )

    def get_collate_fn(self, stage: RunningStage | None = None, show_batch: bool = False) -> Callable | None:
        return partial(
            collate_fn_for_seq2seq,
            input_names=self.input_names,
            on_cpu=self.on_cpu,
            max_source_length=None if show_batch else self.loading_params.max_source_length,  # type: ignore
            max_target_length=None if show_batch else self.loading_params.max_target_length,  # type: ignore
            pad_token_id=self.tokenizer.pad_token_id,
            pad_fn=_pad,
        )


class DatastoreForSeq2Seq(Seq2SeqMixin, Datastore):
    ...


class PandasDatastoreForSeq2Seq(Seq2SeqMixin, PandasDatastoreWithIndex):
    def _set_attributes(self, dataset_dict: dict[RunningStage, Dataset], tokenizer: PreTrainedTokenizerBase) -> None:
        super()._set_attributes(dataset_dict, tokenizer)
        self._train_data = self._train_data.to_pandas()  # type: ignore
