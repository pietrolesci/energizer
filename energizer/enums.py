from lightning.pytorch.utilities.enums import LightningEnum


class RunningStage(LightningEnum):
    TRAIN: str = "train"
    VALIDATION: str = "validation"
    TEST: str = "test"
    POOL: str = "pool"


class Interval(LightningEnum):
    BATCH: str = "batch"
    EPOCH: str = "epoch"
    ROUND: str = "round"


class SpecialKeys(LightningEnum):
    ID: str = "unique_id"
    IS_LABELLED: str = "is_labelled"
    IS_VALIDATION: str = "is_validation"
    LABELLING_ROUND: str = "labelling_round"


class InputKeys(LightningEnum):
    TARGET: str = "labels"
    INPUT_IDS: str = "input_ids"
    ATT_MASK: str = "attention_mask"
    TOKEN_TYPE_IDS: str = "token_type_ids"
    ON_CPU: str = "on_cpu"
    TEXT: str = "text"


class OutputKeys(LightningEnum):
    PRED: str = "y_hat"
    TARGET: str = "y"
    LOSS: str = "loss"
    LOGS: str = "logs"
    LOGITS: str = "logits"
    BATCH_SIZE: str = "batch_size"
    METRICS: str = "metrics"
    SCORES: str = "scores"
