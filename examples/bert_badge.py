from typing import Any, Dict, List

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from lightning.fabric.wrappers import _FabricModule
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from energizer import seed_everything
from energizer.active_learning.datastores.classification import ActivePandasDataStoreForSequenceClassification
from energizer.active_learning.strategies.diversity import BADGE
from energizer.enums import InputKeys, OutputKeys, RunningStage
from energizer.utilities import move_to_cpu

MODEL_NAME = "google/bert_uncased_L-2_H-128_A-2"
SEED = 42


class BADGEForSequenceClassification(BADGE):
    def get_penultimate_layer_out(self, model: _FabricModule, batch: Any) -> torch.Tensor:
        inp = {k: v for k, v in batch.items() if k in (InputKeys.INPUT_IDS, InputKeys.ATT_MASK)}
        return model.bert(**inp).pooler_output

    def get_logits_from_penultimate_layer_out(
        self, model: _FabricModule, penultimate_layer_out: torch.Tensor
    ) -> torch.Tensor:
        return model.classifier(penultimate_layer_out)

    def step(
        self,
        stage: RunningStage,
        model,
        batch: Dict,
        batch_idx: int,
        loss_fn,
        metrics: MetricCollection,
    ) -> torch.Tensor:

        _ = batch.pop(InputKeys.ON_CPU, None)
        out = model(**batch)

        out_metrics = metrics(out.logits, batch[InputKeys.TARGET])
        if stage == RunningStage.TRAIN:
            logs = {OutputKeys.LOSS: out.loss, **out_metrics}
            self.log_dict({f"{stage}/{k}": v for k, v in logs.items()}, step=self.tracker.global_batch)

        return out.loss

    def epoch_end(self, stage: RunningStage, output: List[np.ndarray], metrics: MetricCollection) -> float:
        # aggregate and log
        aggregated_metrics = move_to_cpu(metrics.compute())  # NOTE: metrics are still on device
        aggregated_loss = round(np.mean(output).item(), 6)
        logs = {OutputKeys.LOSS: aggregated_loss, **aggregated_metrics}
        self.log_dict({f"{stage}_end/{k}": v for k, v in logs.items()}, step=self.tracker.safe_global_epoch)
        return aggregated_loss

    def configure_metrics(self, *_) -> MetricCollection:
        num_classes: int = self.model.num_labels  # type: ignore
        task = "multiclass"
        metrics = MetricCollection(
            {
                "accuracy": Accuracy(task, num_classes=num_classes),
                "f1_macro": F1Score(task, num_classes=num_classes, average="macro"),
                "precision_macro": Precision(task, num_classes=num_classes, average="macro"),
                "recall_macro": Recall(task, num_classes=num_classes, average="macro"),
                "f1_micro": F1Score(task, num_classes=num_classes, average="micro"),
                "precision_micro": Precision(task, num_classes=num_classes, average="micro"),
                "recall_micro": Recall(task, num_classes=num_classes, average="micro"),
            }
        )

        # NOTE: you are in charge of moving it to the correct device
        return metrics.to(self.device)


if __name__ == "__main__":

    # load data
    dataset_dict: DatasetDict = load_dataset("pietrolesci/agnews")  # type: ignore

    # subset for speed
    dataset_dict["train"] = dataset_dict["train"].select(range(1000))
    dataset_dict["test"] = dataset_dict["test"].select(range(1000))

    # tokenize
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    dataset_dict = dataset_dict.map(lambda ex: tokenizer(ex["text"]), batched=True)

    # create datastore
    datastore = ActivePandasDataStoreForSequenceClassification.from_dataset_dict(
        dataset_dict=dataset_dict,
        input_names=["input_ids", "attention_mask"],
        target_name="labels",
        tokenizer=tokenizer,
    )
    datastore.prepare_for_loading(batch_size=32, eval_batch_size=64, seed=SEED)

    # seed
    seed_everything(SEED)

    # model
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        id2label=datastore.id2label,
        label2id=datastore.label2id,
        num_labels=len(datastore.labels),
    )

    # active learning loop
    badge = BADGEForSequenceClassification(model, accelerator="gpu", seed=SEED)
    print(badge.model_summary)

    results = badge.active_fit(
        datastore=datastore,  # type: ignore
        max_rounds=5,
        query_size=15,
        max_epochs=3,
        learning_rate=0.001,
        optimizer="adamw",
        scheduler="cosine_schedule_with_warmup",
        scheduler_kwargs={"num_warmup_steps": 0.1},
    )
    print(results)
