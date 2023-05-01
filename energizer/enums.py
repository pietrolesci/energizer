from lightning_utilities.core.enums import StrEnum as _StrEnum
from enum import EnumMeta


class ValueOnlyEnumMeta(EnumMeta):

    # def __members__(cls):
    #     return {member.value: member for member in cls}

    def __iter__(cls):
        """
        Returns members in definition order.
        """
        return (cls._member_map_[name].value for name in cls._member_names_)


class StrEnum(_StrEnum, metaclass=ValueOnlyEnumMeta):
    def __str__(self):
        # behaves like string when used in interpolation
        return str(self.value)
    


class RunningStage(StrEnum):
    TRAIN: str = "train"
    VALIDATION: str = "validation"
    TEST: str = "test"
    POOL: str = "pool"


class Interval(StrEnum):
    BATCH: str = "batch"
    EPOCH: str = "epoch"
    ROUND: str = "round"


class SpecialKeys(StrEnum):
    ID: str = "unique_id"
    IS_LABELLED: str = "is_labelled"
    IS_VALIDATION: str = "is_validation"
    LABELLING_ROUND: str = "labelling_round"


class InputKeys(StrEnum):
    TARGET: str = "labels"
    INPUT_IDS: str = "input_ids"
    ATT_MASK: str = "attention_mask"
    TOKEN_TYPE_IDS: str = "token_type_ids"
    ON_CPU: str = "on_cpu"
    TEXT: str = "text"


class OutputKeys(StrEnum):
    PRED: str = "y_hat"
    TARGET: str = "y"
    LOSS: str = "loss"
    LOGS: str = "logs"
    LOGITS: str = "logits"
    BATCH_SIZE: str = "batch_size"
    METRICS: str = "metrics"
    SCORES: str = "scores"
