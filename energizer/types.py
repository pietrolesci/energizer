from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Mapping, Tuple, Union

from torch import Tensor
from torchmetrics import Metric

METRIC = Union[Metric, Any]
DATASET = Iterable[Mapping]

BATCH_OUTPUT = Union[Tensor, Dict]

EPOCH_OUTPUT = List[Union[BATCH_OUTPUT, Any]]
FIT_OUTPUT = Tuple[EPOCH_OUTPUT, EPOCH_OUTPUT]

ROUND_OUTPUT = Union[Mapping[str, Any], Any]
