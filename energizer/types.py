from collections.abc import Mapping
from typing import Any, TypeAlias

from torch import Tensor
from torchmetrics import Metric

METRIC: TypeAlias = Metric | Any
DATASET: TypeAlias = list[Mapping]
DATA_SOURCE: TypeAlias = Any

BATCH_OUTPUT: TypeAlias = Tensor | dict

EPOCH_OUTPUT: TypeAlias = list[BATCH_OUTPUT] | Any
FIT_OUTPUT: TypeAlias = tuple[EPOCH_OUTPUT, EPOCH_OUTPUT]

ROUND_OUTPUT: TypeAlias = Mapping[str, Any] | Any
