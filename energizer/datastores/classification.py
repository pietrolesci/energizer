from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial

import torch
from datasets import Dataset
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from energizer.datastores.base import DataloaderArgs, Datastore, PandasDatastoreWithIndex
from energizer.datastores.mixins import TextMixin
from energizer.enums import InputKeys, RunningStage
from energizer.utilities import _pad, ld_to_dl


def collate_fn_for_sequence_classification(
    batch: list[dict[str, list[str] | Tensor]],
    input_names: list[str],
    on_cpu: list[str],
    max_length: int | None,
    pad_token_id: int,
    pad_fn: Callable,
    build_attention_mask: bool,
) -> dict[str, list[str] | Tensor]:
    new_batch = ld_to_dl(batch)

    # remove string columns that cannot be transfered on gpu
    values_on_cpu = {col: new_batch.pop(col, None) for col in on_cpu if col in new_batch}

    labels = new_batch.pop(InputKeys.LABELS, None)

    # input_ids and attention_mask to tensor: truncate -> convert to tensor -> pad
    new_batch = {
        k: pad_fn(inputs=new_batch[k], padding_value=pad_token_id, max_length=max_length).contiguous()
        for k in input_names
    }

    if build_attention_mask:
        inp_ids = new_batch[InputKeys.INPUT_IDS]
        att_mask = torch.ones_like(inp_ids, device=inp_ids.device, dtype=torch.long).contiguous()
        new_batch[InputKeys.ATT_MASK] = att_mask.masked_fill(inp_ids == pad_token_id, pad_token_id)

    if labels is not None:
        new_batch[InputKeys.LABELS] = torch.tensor(labels, dtype=torch.long).contiguous()

    # add things that need to remain on cpu
    if len(values_on_cpu) > 0:
        new_batch[InputKeys.ON_CPU] = values_on_cpu

    return new_batch


@dataclass
class SequenceClassificationDataloaderArgs(DataloaderArgs):
    max_length: int


class SequenceClassificationMixin(TextMixin):
    MANDATORY_TARGET_NAME: str | None = InputKeys.LABELS
    _loading_params: SequenceClassificationDataloaderArgs | None = None

    _labels: list[str]
    _label_distribution: dict[str, dict[str, int]]

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
        assert self.tokenizer is not None, "Before being able to load data you need to attach a tokenizer"
        self._loading_params = SequenceClassificationDataloaderArgs(
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
            max_length=max_length,
        )

    @property
    def labels(self) -> list[str]:
        return self._labels

    @property
    def id2label(self) -> dict[int, str]:
        return dict(enumerate(self.labels))

    @property
    def label2id(self) -> dict[str, int]:
        return {v: k for k, v in self.id2label.items()}

    def label_distribution(self, normalized: bool = False) -> dict[str, dict]:
        if normalized:
            norm_label_distribution = {}
            for split, label_dist in self._label_distribution.items():
                total = sum(label_dist.values())
                norm_label_distribution[split] = {k: round(label_dist[k] / total, ndigits=3) for k in label_dist}
            lab_dist = norm_label_distribution

        else:
            lab_dist = self._label_distribution

        # sort by label_idx and add label in key
        out = {}
        for split, dist in lab_dist.items():
            sorted_dist = dict(sorted(dist.items()))
            out[split] = {f"{k}-({self.id2label[k]})": v for k, v in sorted_dist.items()}  # type: ignore
        return out

    def _set_attributes(self, dataset_dict: dict[RunningStage, Dataset], tokenizer: PreTrainedTokenizerBase) -> None:
        super()._set_attributes(dataset_dict, tokenizer)

        # === SET ATTRIBUTES === #
        _label_distribution = {}
        for k in dataset_dict:
            _label_distribution[k] = dict(Counter(dataset_dict[k][self.MANDATORY_TARGET_NAME]))  # type: ignore
        self._label_distribution = _label_distribution
        self._labels = next(iter(dataset_dict.values())).features[self.MANDATORY_TARGET_NAME].names

    def _check_labels(self, dataset_dict: dict[RunningStage, Dataset], mandatory_target_name: str) -> None:
        labels = []
        for split, dataset in dataset_dict.items():
            assert (
                mandatory_target_name in dataset.features
            ), f"Mandatory column {mandatory_target_name} not in dataset[{split}]."
            labels.append(set(dataset.features[mandatory_target_name].names))

        # check labels are consistent
        assert all(s == labels[0] for s in labels), "Labels are inconsistent across splits"

    def get_collate_fn(self, stage: RunningStage | None = None, show_batch: bool = False) -> Callable | None:
        assert self.tokenizer.pad_token_id is not None, "You need to set the `pad_token_id` in the tokenizer"
        return partial(
            collate_fn_for_sequence_classification,
            input_names=self.input_names,
            on_cpu=self.on_cpu,
            max_length=None if show_batch else self.loading_params.max_length,  # type: ignore
            pad_token_id=self.tokenizer.pad_token_id,
            pad_fn=_pad,
            build_attention_mask=InputKeys.ATT_MASK not in self.input_names,
        )


class DatastoreForSequenceClassification(SequenceClassificationMixin, Datastore):
    ...


class PandasDatastoreForSequenceClassification(SequenceClassificationMixin, PandasDatastoreWithIndex):
    def _set_attributes(self, dataset_dict: dict[RunningStage, Dataset], tokenizer: PreTrainedTokenizerBase) -> None:
        super()._set_attributes(dataset_dict, tokenizer)
        self._train_data = self._train_data.to_pandas()  # type: ignore
