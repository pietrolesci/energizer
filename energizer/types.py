from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Union

from torch import Tensor
from torchmetrics import Metric

METRIC = Union[Metric, Any]
DATASET = Iterable[Mapping]

BATCH_OUTPUT = Union[Tensor, Dict]

EPOCH_OUTPUT = List[Union[BATCH_OUTPUT, Any]]

ROUND_OUTPUT = Union[Mapping[str, Any], Any]
