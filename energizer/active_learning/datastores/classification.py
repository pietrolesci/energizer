from typing import List, Optional

from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from typing_extensions import Self

from energizer.active_learning.datastores.base import ActivePandasDataStoreWithIndex
from energizer.datastores.classification import SequenceClassificationMixin, _from_datasets
from energizer.enums import SpecialKeys


class ActivePandasDataStoreForSequenceClassification(SequenceClassificationMixin, ActivePandasDataStoreWithIndex):
    @classmethod
    def from_datasets(
        cls,
        tokenizer: PreTrainedTokenizerBase,
        uid_name: Optional[str] = None,
        on_cpu: Optional[List[str]] = None,
        seed: Optional[int] = 42,
        train_dataset: Optional[Dataset] = None,
        validation_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
    ) -> Self:
        obj = cls(seed)  # type: ignore
        obj = _from_datasets(
            obj=obj,
            mandatory_input_names=cls.MANDATORY_INPUT_NAMES,
            optional_input_names=cls.OPTIONAL_INPUT_NAMES,
            mandatory_target_name=cls.MANDATORY_TARGET_NAME,
            tokenizer=tokenizer,
            uid_name=uid_name,
            on_cpu=on_cpu,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
        )

        # add special columns if not present
        for k, v in [
            (SpecialKeys.IS_LABELLED, False),
            (SpecialKeys.IS_VALIDATION, False),
            (SpecialKeys.LABELLING_ROUND, -100),
        ]:
            obj._train_data[k.value] = v  # type: ignore

        return obj

    def reset(self) -> None:
        for k, v in [
            (SpecialKeys.IS_LABELLED, False),
            (SpecialKeys.IS_VALIDATION, False),
            (SpecialKeys.LABELLING_ROUND, -100),
        ]:
            self._train_data[k.value] = v  # type: ignore
