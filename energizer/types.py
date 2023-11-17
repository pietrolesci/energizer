from collections.abc import Mapping
from typing import Any, Union

from torch import Tensor
from torchmetrics import Metric

METRIC = Union[Metric, Any]
DATASET = list[Mapping]
DATA_SOURCE = Any

BATCH_OUTPUT = Union[Tensor, dict]

EPOCH_OUTPUT = Union[list[BATCH_OUTPUT], Any]
FIT_OUTPUT = tuple[EPOCH_OUTPUT, EPOCH_OUTPUT]

ROUND_OUTPUT = Union[Mapping[str, Any], Any]
