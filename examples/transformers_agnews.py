import os
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from pytorch_lightning import LightningModule, seed_everything
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall
from transformers import (
    AdamW,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    get_constant_schedule_with_warmup,
)

from energizer import ActiveDataModule, Trainer
from energizer.query_strategies import (
    BALDStrategy,
    EntropyStrategy,
    LeastConfidenceStrategy,
    MarginStrategy,
    RandomStrategy,
)

MODEL_NAME_OR_PATH = "bert-base-uncased"
# MODEL_NAME_OR_PATH = "google/bert_uncased_L-2_H-128_A-2"
BATCH_SIZE = 32
EVAL_BATCH_SIZE = 128 # 512
LEARNING_RATE = 0.0001
NUM_WARMUP_STEPS = 50
DATASET_NAME = "pietrolesci/ag_news"
DATASET_SPLIT = "concat"
QUERY_SIZE = 10
MAX_EPOCHS = 3
MAX_LABELLING_EPOCHS = 100


def get_dataloaders(
    model_name_or_path: str, dataset_name: str, dataset_split: str, batch_size: int, eval_batch_size: int
):
    """Load and preprocess data, and prepare dataloaders."""
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # renames "label" to "labels"
    collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

    # load dataset
    dataset = load_dataset(dataset_name, dataset_split)

    # tokenize
    dataset = dataset.map(lambda ex: tokenizer(ex["text"]), batched=True)
    columns_to_keep = ["label", "input_ids", "token_type_ids", "attention_mask"]

    # train-val split and record datasets
    train_set, test_set = dataset["train"], dataset["test"]
    _split = train_set.train_test_split(0.3)
    _, val_set = _split["train"], _split["test"]

    labels = train_set.features["label"].names

    # create dataloaders
    train_dl = DataLoader(
        train_set.with_format(columns=columns_to_keep),
        batch_size=batch_size,
        collate_fn=collator,
        num_workers=os.cpu_count(),
    )
    val_dl = DataLoader(
        val_set.with_format(columns=columns_to_keep),
        batch_size=eval_batch_size,
        collate_fn=collator,
        num_workers=os.cpu_count(),
    )
    test_dl = DataLoader(
        test_set.with_format(columns=columns_to_keep),
        batch_size=eval_batch_size,
        collate_fn=collator,
        num_workers=os.cpu_count(),
    )

    return train_dl, val_dl, test_dl, labels


def get_lightning_model(model_name_or_path: str, num_classes: int, learning_rate: float, num_warmup_steps):
    class TransformerModel(LightningModule):
        def __init__(
            self,
            model_name_or_path: str,
            num_classes: int,
            learning_rate: float,
            num_warmup_steps: int,
        ) -> None:
            super().__init__()
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name_or_path,
                num_labels=num_classes,
            )
            self.learning_rate = learning_rate
            self.num_warmup_steps = num_warmup_steps
            for stage in ("train", "val", "test"):
                metrics = MetricCollection(
                    {
                        "accuracy": Accuracy(),
                        "precision_macro": Precision(num_classes=num_classes, average="macro"),
                        "recall_macro": Recall(num_classes=num_classes, average="macro"),
                        "f1_macro": F1Score(num_classes=num_classes, average="macro"),
                        "f1_micro": F1Score(num_classes=num_classes, average="micro"),
                    }
                )
                setattr(self, f"{stage}_metrics", metrics)

        def common_step(self, batch: Dict[str, Tensor], stage: str):
            """Outputs loss and logits, logs loss and metrics."""
            targets = batch.pop("labels")
            logits = self(batch)
            loss = F.cross_entropy(logits, targets)
            self.log(f"{stage}/loss", loss)

            metrics = getattr(self, f"{stage}_metrics")(logits, targets)
            self.log_dict(metrics)

            return loss

        def forward(self, batch) -> torch.Tensor:
            return self.model(**batch).logits

        def training_step(self, batch: Dict[str, Tensor], batch_idx: int = 0) -> Tensor:
            return self.common_step(batch, "train")

        def validation_step(self, batch: Any, batch_idx: int = 0) -> Tensor:
            return self.common_step(batch, "val")

        def test_step(self, batch: Any, batch_idx: int = 0) -> Tensor:
            return self.common_step(batch, "test")

        def configure_optimizers(self) -> Dict[str, Any]:
            optimizer = AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.learning_rate)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": get_constant_schedule_with_warmup(
                        optimizer=optimizer, num_warmup_steps=self.num_warmup_steps
                    ),
                    "monitor": "val/loss",
                    "frequency": 1,
                    "interval": "step",
                },
            }

    return TransformerModel(model_name_or_path, num_classes, learning_rate, num_warmup_steps)


def main():

    seed_everything(42)

    # data
    train_dl, val_dl, test_dl, labels = get_dataloaders(
        MODEL_NAME_OR_PATH, DATASET_NAME, DATASET_SPLIT, BATCH_SIZE, EVAL_BATCH_SIZE
    )
    initial_set_indices = np.random.randint(0, len(train_dl.dataset), size=20).tolist()

    # instantiate model
    model = get_lightning_model(MODEL_NAME_OR_PATH, len(labels), LEARNING_RATE, NUM_WARMUP_STEPS)

    # For clarity let's pack the trainer kwargs in a dictionary
    trainer_kwargs = {
        "query_size": QUERY_SIZE,  # new instances will be queried at each iteration
        "max_epochs": MAX_EPOCHS,  # the underlying model will be fit for 3 epochs
        "max_labelling_epochs": MAX_LABELLING_EPOCHS,  # how many times to run the active learning loop
        "accelerator": "gpu",  # use the gpu
        "test_after_labelling": True,  # since we have a test set, we test after each labelling iteration
        "reset_weights": True,
        "limit_val_batches": 0,  # do not validate
        "log_every_n_steps": 1,  # we will have a few batches while training, so log on each
    }

    # define strategies
    strategies = {
        "random": RandomStrategy(model),
        "entropy": EntropyStrategy(model),
        "leastconfidence": LeastConfidenceStrategy(model),
        "margin": MarginStrategy(model),
        "bald": BALDStrategy(model),
    }

    results_dict = {}
    for name, strategy in strategies.items():
        seed_everything(42)  # for reproducibility (e.g., dropout)

        # initial labelling
        datamodule = ActiveDataModule(train_dl, val_dl, test_dl)
        datamodule.label(initial_set_indices)

        # active learning
        trainer = Trainer(**trainer_kwargs)
        results = trainer.active_fit(model=strategy, datamodule=datamodule)

        # storeresults
        df = results.to_pandas()
        results_dict[name] = df

        df.to_csv(f"./results/bert_{name}.csv", index=False)


if __name__ == "__main__":
    main()
