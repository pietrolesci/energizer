# Pytorch-Energizer


[![pypi](https://img.shields.io/pypi/v/pytorch-energizer.svg)](https://pypi.org/project/pytorch-energizer/)
[![python](https://img.shields.io/pypi/pyversions/pytorch-energizer.svg)](https://pypi.org/project/pytorch-energizer/)
[![Build Status](https://github.com/pietrolesci/pytorch-energizer/actions/workflows/dev.yml/badge.svg)](https://github.com/pietrolesci/pytorch-energizer/actions/workflows/dev.yml)
[![codecov](https://codecov.io/gh/pietrolesci/pytorch-energizer/branch/main/graphs/badge.svg)](https://codecov.io/github/pietrolesci/pytorch-energizer)



An active learning library for PyTorch


* Documentation: <https://pietrolesci.github.io/pytorch-energizer>
* GitHub: <https://github.com/pietrolesci/pytorch-energizer>
* PyPI: <https://pypi.org/project/pytorch-energizer/>
* Free software: MIT


## Features

`Energizer` allows training PyTorch models using active learning. Being based on Pytorch-Lightning, it can easily scale to multi-node/multi-gpu settings. Also, importantly, abiding to the light-weight Pytorch-Lightning API allows the library to have a tidy interface and reduce boilerplate code.


## MNIST example

Import the required modules

```python
from typing import Dict, Tuple

import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from energizer.learners.acquisition_functions import entropy
from energizer.learners.base import Deterministic
from energizer.trainer import Trainer
```

Load and process the MNIST data

```python
data_dir = "./data"
preprocessing_pipe = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ]
)
train_set = MNIST(data_dir, train=True, download=False, transform=preprocessing_pipe)
test_set = MNIST(data_dir, train=False, download=False, transform=preprocessing_pipe)
train_set, val_set = random_split(train_set, [55000, 5000])

# create dataloaders
batch_size = 32
eval_batch_size = 32  # this is use when evaluating on the pool too
train_dl = DataLoader(train_set, batch_size=batch_size)
val_dl = DataLoader(val_set, batch_size=eval_batch_size)
test_dl = DataLoader(test_set, batch_size=eval_batch_size)
```

Define the `LightningModule`

```diff
- class MNISTModel(LightningModule):
+ class MNISTModel(Deterministic):
    def __init__(self) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5),
            nn.Dropout2d(),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1024, 128),
            nn.Dropout(),
            nn.Linear(128, 10),
        )
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def step(self, batch: Tuple[Tensor, Tensor], stage: str) -> Dict[str, Tensor]:
        x, y = batch
        logits = self(x)
        loss = self.loss(logits, y)
        self.log(f"{stage}/loss", loss)
        return {"loss": loss, "logits": logits}

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        return self.step(batch, "train")

    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        return self.step(batch, "val")

    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Dict[str, Tensor]:
        return self.step(batch, "test")

+    def pool_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
+        """A implememntation of the `Entropy` active learning strategy."""
+
+        # define how to perform the forward pass
+        x, _ = batch
+        logits = self(x)
+
+        # use an acquisition/scoring function
+        scores = entropy(logits)
+
+        return scores

    def configure_optimizers(self) -> None:
        return torch.optim.SGD(self.parameters(), lr=0.01)
```

Run active learning loop

```python

model = MNISTModel()

trainer = Trainer(
    query_size=2,
    max_epochs=3,
    max_labelling_epochs=4,
    total_budget=5,
    log_every_n_steps=1,
    test_after_labelling=True,
    
    # for testing purposes
    limit_train_batches=10,
    limit_val_batches=10,
    limit_test_batches=10,
    limit_pool_batches=10,
)

trainer.active_fit(
    model=model,
    train_dataloaders=train_dl,
    val_dataloaders=val_dl,
    test_dataloaders=test_dl,
)

print(trainer.datamodule.stats)
# {'total_data_size': 55000,
#  'train_size': 6,
#  'pool_size': 54994,
#  'num_train_batches': 1,
#  'num_pool_batches': 1719}
```





## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [waynerv/cookiecutter-pypackage](https://github.com/waynerv/cookiecutter-pypackage) project template.
