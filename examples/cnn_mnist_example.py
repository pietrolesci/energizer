import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, seed_everything
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split
from torchmetrics import Accuracy, F1Score, MetricCollection, Precision, Recall
from torchvision import transforms
from torchvision.datasets import MNIST

from energizer import ActiveDataModule, Trainer
from energizer.query_strategies import (
    BALDStrategy,
    EntropyStrategy,
    LeastConfidenceStrategy,
    MarginStrategy,
    RandomStrategy,
)


def get_dataloaders(data_dir="./data"):
    """Load and preprocess data, and prepare dataloaders."""
    data_dir = "./data"
    preprocessing_pipe = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    train_set = MNIST(data_dir, train=True, download=True, transform=preprocessing_pipe)
    test_set = MNIST(data_dir, train=False, download=True, transform=preprocessing_pipe)
    train_set, val_set = random_split(train_set, [55000, 5000])

    # create dataloaders
    batch_size = 32
    eval_batch_size = 128  # this is use when evaluating on the pool too
    train_dl = DataLoader(train_set, batch_size=batch_size, num_workers=os.cpu_count())
    val_dl = DataLoader(val_set, batch_size=eval_batch_size, num_workers=os.cpu_count())
    test_dl = DataLoader(test_set, batch_size=eval_batch_size, num_workers=os.cpu_count())

    return train_dl, val_dl, test_dl


def get_lightning_model():
    class MNISTModel(LightningModule):
        def __init__(self, num_classes: int = 10) -> None:
            super().__init__()
            self.num_classes = num_classes
            self.model = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout2d(),
                nn.Conv2d(32, 64, kernel_size=5),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2),
                nn.Dropout2d(),
                nn.Flatten(),
                nn.Linear(1024, 128),
                nn.Dropout(),
                nn.Linear(128, num_classes),
            )
            for stage in ("train", "val", "test"):
                metrics = MetricCollection(
                    {
                        "accuracy": Accuracy(),
                        "precision_macro": Precision(num_classes=num_classes, average="macro"),
                        "precision_micro": Precision(num_classes=num_classes, average="micro"),
                        "recall_macro": Recall(num_classes=num_classes, average="macro"),
                        "recall_micro": Recall(num_classes=num_classes, average="micro"),
                        "f1_macro": F1Score(num_classes=num_classes, average="macro"),
                        "f1_micro": F1Score(num_classes=num_classes, average="micro"),
                    }
                )
                setattr(self, f"{stage}_metrics", metrics)

        def forward(self, batch: Tuple[Tensor, Tensor]) -> Tensor:
            # NOTE: here I am unpacking the batch in the forward pass
            x, _ = batch
            return self.model(x)

        def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
            return F.cross_entropy(logits, targets)

        def common_step(self, batch: Tuple[Tensor, Tensor], stage: str) -> Dict[str, Tensor]:
            _, y = batch
            logits = self(batch)  # feed the entire batch since forward know how to handle

            loss = self.loss(logits, y)
            self.log(f"{stage}/loss", loss, on_epoch=True, on_step=False, prog_bar=False)

            metrics = getattr(self, f"{stage}_metrics")(logits, y)
            self.log_dict(metrics, on_epoch=True, on_step=False, prog_bar=False)

            return loss

        def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
            return self.common_step(batch, "train")

        def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
            return self.common_step(batch, "val")

        def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
            return self.common_step(batch, "test")

        def configure_optimizers(self) -> None:
            return torch.optim.SGD(self.parameters(), lr=0.001)

    return MNISTModel()


def main():

    seed_everything(42)

    # data
    train_dl, val_dl, test_dl = get_dataloaders()
    initial_set_indices = np.random.randint(0, len(train_dl.dataset), size=20).tolist()

    # instantiate model
    model = get_lightning_model()

    # For clarity let's pack the trainer kwargs in a dictionary
    trainer_kwargs = {
        "query_size": 10,  # new instances will be queried at each iteration
        "max_epochs": 50,  # the underlying model will be fit for 3 epochs
        "max_labelling_epochs": 100,  # how many times to run the active learning loop
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
        
        df.to_csv(f"./results/{name}.csv", index=False)


if __name__ == "__main__":
    main()
