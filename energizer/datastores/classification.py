from collections import Counter
from typing import Callable, Dict, List, Optional, Union
from functools import partial
from energizer.utilities import _pad

import torch
from datasets import Dataset
from torch import Tensor
from transformers import PreTrainedTokenizerBase

from energizer.datastores.base import PandasDatastoreWithIndex, Datastore
from energizer.enums import InputKeys, RunningStage
from energizer.utilities import ld_to_dl
from energizer.datastores.mixins import TextMixin


def collate_fn_for_sequence_classification(
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


class SequenceClassificationMixin(TextMixin):
    MANDATORY_TARGET_NAME: Optional[str] = InputKeys.LABELS

    _labels: List[str]
    _label_distribution: Dict[str, Dict[str, int]]

    @property
    def labels(self) -> List[str]:
        return self._labels

    @property
    def id2label(self) -> Dict[int, str]:
        return dict(enumerate(self.labels))

    @property
    def label2id(self) -> Dict[str, int]:
        return {v: k for k, v in self.id2label.items()}

    def label_distribution(self, normalized: bool = False) -> Dict[str, Dict]:
        if normalized:
            norm_label_distribution = {}
            for split, label_dist in self._label_distribution.items():
                total = sum(label_dist.values())
                norm_label_distribution[split] = {k: label_dist[k] / total for k in label_dist}
            return norm_label_distribution

        return self._label_distribution

    def _set_attributes(self, dataset_dict: Dict[RunningStage, Dataset], tokenizer: PreTrainedTokenizerBase) -> None:
        super()._set_attributes(dataset_dict, tokenizer)

        # === SET ATTRIBUTES === #
        _label_distribution = {}
        for k in dataset_dict:
            _label_distribution[k] = dict(Counter(dataset_dict[k][self.MANDATORY_TARGET_NAME]))  # type: ignore
        self._label_distribution = _label_distribution
        self._labels = next(iter(dataset_dict.values())).features[self.MANDATORY_TARGET_NAME].names

    def _check_labels(self, dataset_dict: Dict[RunningStage, Dataset], mandatory_target_name: str) -> None:
        labels = []
        for split, dataset in dataset_dict.items():
            assert (
                mandatory_target_name in dataset.features
            ), f"Mandatory column {mandatory_target_name} not in dataset[{split}]."
            labels.append(set(dataset.features[mandatory_target_name].names))

        # check labels are consistent
        assert all(s == labels[0] for s in labels), "Labels are inconsistent across splits"

    def get_collate_fn(self, stage: Optional[RunningStage] = None, show_batch: bool = False) -> Optional[Callable]:
        return partial(
            collate_fn_for_sequence_classification,
            input_names=self.input_names,
            on_cpu=self.on_cpu,
            max_length=None if show_batch else self.loading_params["max_length"],  # type: ignore
            pad_token_id=self.tokenizer.pad_token_id,
            pad_fn=_pad,
        )


class DatastoreForSequenceClassification(SequenceClassificationMixin, Datastore):
    ...


class PandasDatastoreForSequenceClassification(SequenceClassificationMixin, PandasDatastoreWithIndex):
    def _set_attributes(self, dataset_dict: Dict[RunningStage, Dataset], tokenizer: PreTrainedTokenizerBase) -> None:
        super()._set_attributes(dataset_dict, tokenizer)
        self._train_data = self._train_data.to_pandas()  # type: ignore
