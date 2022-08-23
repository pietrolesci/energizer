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

`Energizer` allows training PyTorch models using active learning. Being based on Pytorch-Lightning, it can easily scale to multi-node/multi-gpu settings. Also, importantly, abiding to the light-weight Pytorch-Lightning API allows `energizer` to have a tidy interface and reduce boilerplate code.

A core principle of `energizer` is full compatibility with any `LightningModule`. By simply implementing the required hooks, it should be possible to actively train any `LightningModule`.

At the moment `energizer` supports only pool-based active learning and is focused on research settings: the labels are already available but masked, thus mimicking a true active learning loop. However, the plan for `energizer` is to make it fully compatible with open-source annotation tools such as [`Label-Studio` ](https://labelstud.io/) and [`Rubrix`](https://www.rubrix.ml/).

Furthermore, `energizer` aims to be an extension to Pytorch-Lightning providing the users with non-standard training loop. For example, in the future we plan to add support for self-supervised learning.


## MNIST example

In this example we will train a vision model on the famous MNIST dataset.
First let's import the required modules

```python
from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from energizer import DeterministicMixin, MCDropoutMixin, Trainer
from energizer.acquisition_functions import entropy, expected_entropy
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

Now, let's define a normal `LightningModule`, as usual

```python
class MNISTModel(LightningModule):
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

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        return F.cross_entropy(logits, targets)

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

    def configure_optimizers(self) -> None:
        return torch.optim.SGD(self.parameters(), lr=0.01)
```

As a side note, notice that, for convenience, in this example we have defined the loss in an isolated method. This is useful for the sake of this example bacause, below, to use MCDropout we need to redefine how the loss is computed. In this way, we can simply override the `loss` method.

In order to run pool-based active learning, we need to define how the model behaves when predicting on the unlabelled pool. This is achieved, by overriding the new "pool" hooks provided by `energizer`. To have access to those, just inherit from one of the available mixins. Currently,`energizer` provides two mixins: 

- `DeterministicMixin`: it is a (very) thin wrapper around a normal `LightningModule` and simply adds the "pool"-related hooks needed to run the active learning loop.

- `MCDropoutMixin`: similar to `DeterministicMixin`, but modifies the behaviour of the model. In particular, it patches all the dropout layers so that they will remain active even during evaluation: it also modifies the behaviour of the forward pass of the model making it run `num_inference_iters` returning a `torch.Tensor` with dimensions `(batch_size, num_classes, num_inference_iter)`.

```python
class DeterministicMNISTModel(MNISTModel, DeterministicMixin):
    """A implememntation of the `Entropy` active learning strategy."""
    def pool_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        # define how to perform the forward pass
        x, _ = batch
        logits = self(x)
        # use an acquisition/scoring function
        scores = entropy(logits)
        return scores


class StochasticMNISTModel(MNISTModel, MCDropoutMixin):
    """A implememntation of the `Entropy` active learning strategy.

    In this case we use the MCDropout technique to compute model
    uncertainty. Accordigly, we need to use `expected_entropy` as
    the acquisition function.
    """
    def loss(logits: Tensor, targets: Tensor) -> Tensor:
        # logits has now shape:
        # [num_samples, num_classes, num_iterations]
        # average over num_iterations
        logits = logits.mean(-1)
        return F.cross_entropy(logits, targets)
    
    def pool_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        """A implememntation of the `Entropy` active learning strategy."""
        # define how to perform the forward pass
        x, _ = batch
        logits = self(x)
        # use an acquisition/scoring function
        # NOTE: since we are using MCDropout we need to use the
        # `expected_entropy` acquisition function
        scores = expected_entropy(logits)
        return scores
```

To check how the two new methods behave, let's look at the shapes of their predictions

```python
model = MNISTModel()
deterministic_model = DeterministicMNISTModel()
stochastic_model = StochasticMNISTModel()

x, _ = next(iter(train_dl))
model(x).shape, deterministic_model(x).shape, stochastic_model(x).shape
# (torch.Size([32, 10]), torch.Size([32, 10]), torch.Size([32, 10, 10]))
```

As expected, the stochastic model produces an output with a third dimension which corresponds to the 10 MC iterations.

Once we have defined a `LightningModule` that also inherits from the mixins and overridden at least the `pool_step` method, then we can run the active learning loop.

To do that, import the `Trainer` from `energizer`. This trainer is a thin wrapper around the usual Pytorch-Lightning `Trainer` that implements the `active_fit` method.


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


## Contributing

Install `energizer` locally

```bash
conda create -n energizer-dev python=3.9 -y
conda install poetry -y
poetry install -E dev -E test -E doc -E text -E vision
```



## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [waynerv/cookiecutter-pypackage](https://github.com/waynerv/cookiecutter-pypackage) project template.
