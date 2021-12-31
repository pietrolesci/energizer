from typing import Dict, Tuple, Union

import torch
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.utils.data import Dataset

NUM_CLASSES = 2
NUM_FEATURES = 100


class RandomSupervisedDataset(Dataset):
    """Generate a dummy dataset with inputs and labels."""

    def __init__(
        self, num_classes: int = NUM_CLASSES, num_features: int = NUM_FEATURES, num_samples: int = 250
    ) -> None:
        """
        It generates a dataset where each instance is a Tuple[Tensor, Tensor]
        where the first element is a tensor of inputs of size `(num_samples, num_features)`
        and the second element is a tensor of labels of size `(num_samples,)` and `num_labels`
        unique elements.

        Args:
            num_classes (int): The number of classes.
            num_features (int): The number of features.
            num_samples (int): The number of instances.
        """
        self.len = num_samples
        self.x = torch.randn(size=(num_samples, num_features))
        self.y = torch.randint(low=0, high=num_classes, size=(num_samples,))

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        return self.x[idx], self.y

    def __len__(self):
        return self.len


class BoringModel(LightningModule):
    """A simple LightningModule with a linear layer."""

    def __init__(
        self, backbone: torch.nn.Module, num_classes: int = NUM_CLASSES, num_features: int = NUM_FEATURES
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Linear(num_features, num_classes)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.backbone(x))

    def step(self, batch: Union[Tuple[Tensor, Tensor], Dict[str, Tensor]]) -> Tensor:
        if isinstance(batch, dict):
            batch = (batch["inputs"], batch["labels"])
        inputs, targets = batch
        preds = self(inputs)
        return self.loss(preds, targets)

    def training_step(self, batch, *args, **kwargs) -> Dict[str, Tensor]:
        return {"loss": self.step(batch)}

    def validation_step(self, batch, *args, **kwargs) -> Dict[str, Tensor]:
        return {"loss": self.step(batch)}

    def test_step(self, batch, *args, **kwargs) -> Dict[str, Tensor]:
        return {"loss": self.step(batch)}

    def configure_optimizers(self):
        return torch.optim.SGD(self.layer.parameters(), lr=0.01)
