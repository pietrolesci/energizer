from collections.abc import Generator
from enum import Enum, EnumMeta
from typing import Any


class ValueOnlyEnumMeta(EnumMeta):
    def __iter__(cls) -> Generator[Any, None, None]:
        return (cls._member_map_[name].value for name in cls._member_names_)


class StrEnum(str, Enum, metaclass=ValueOnlyEnumMeta):
    def __str__(self) -> str:
        # behaves like string when used in interpolation
        return str(self.value)

    def __eq__(self, other: object) -> bool:
        """Compare two instances."""
        if isinstance(other, Enum):
            other = other.value
        return self.value.lower() == str(other).lower()

    def __hash__(self) -> int:
        """Return unique hash."""
        # re-enable hashtable, so it can be used as a dict key or in a set
        return hash(self.value.lower())

    def __getattribute__(self, __name: str) -> Any:
        return super().__getattribute__(__name)


# pyright: reportGeneralTypeIssues=false
# use this otherwise complains with the literal string assigned to the enum
# https://github.com/microsoft/pyright/issues/1521
# https://stackoverflow.com/questions/70310588/python-type-hint-enum-member-value
class RunningStage(StrEnum):
    TRAIN: str = "train"
    VALIDATION: str = "validation"
    TEST: str = "test"
    POOL: str = "pool"


class Interval(StrEnum):
    BATCH: str = "batch"
    STEP: str = "step"
    EPOCH: str = "epoch"
    ROUND: str = "round"


class SpecialKeys(StrEnum):
    ID: str = "uid"
    IS_LABELLED: str = "is_labelled"
    IS_VALIDATION: str = "is_validation"
    LABELLING_ROUND: str = "labelling_round"


class InputKeys(StrEnum):
    LABELS: str = "labels"
    INPUT_IDS: str = "input_ids"
    ATT_MASK: str = "attention_mask"
    TOKEN_TYPE_IDS: str = "token_type_ids"
    ON_CPU: str = "on_cpu"
    TEXT: str = "text"


class OutputKeys(StrEnum):
    PREDS: str = "y_hat"
    LABELS: str = "y"
    LOSS: str = "loss"
    LOGS: str = "logs"
    LOGITS: str = "logits"
    BATCH_SIZE: str = "batch_size"
    METRICS: str = "metrics"
    SCORES: str = "scores"
    GRADS: str = "gradients"
    EMBEDDINGS: str = "embeddings"
