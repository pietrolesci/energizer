from typing import List, Optional, Union

from datasets import Dataset
from transformers import PreTrainedTokenizerBase
from typing_extensions import Self

from energizer.enums import SpecialKeys


from energizer.active_learning.datastores.base import ActivePandasDataStoreWithIndex
from energizer.datastores.classification import SequenceClassificationMixin, _from_datasets


class ActivePandasDataStoreForSequenceClassification(SequenceClassificationMixin, ActivePandasDataStoreWithIndex):
    @classmethod
    def from_datasets(
        cls,
        input_names: Union[str, List[str]],
        target_name: str,
        tokenizer: PreTrainedTokenizerBase,
        train_dataset: Dataset,
        validation_dataset: Optional[Dataset] = None,
        test_dataset: Optional[Dataset] = None,
        uid_name: Optional[str] = None,
        on_cpu: Optional[List[str]] = None,
        seed: Optional[int] = 42,
    ) -> Self:
        obj = cls(seed)
        obj = _from_datasets(
            obj=obj,
            input_names=input_names,
            target_name=target_name,
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
