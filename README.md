[![pypi](https://img.shields.io/pypi/v/energizer.svg)](https://pypi.org/project/energizer/)
[![python](https://img.shields.io/pypi/pyversions/energizer.svg)](https://pypi.org/project/energizer/)
[![Build Status](https://github.com/pietrolesci/energizer/actions/workflows/dev.yml/badge.svg)](https://github.com/pietrolesci/energizer/actions/workflows/dev.yml)

[![codecov](https://codecov.io/gh/pietrolesci/energizer/branch/main/graph/badge.svg?token=782XT9AQFZ)](https://codecov.io/gh/pietrolesci/energizer)


`Energizer` is an Active-Learning framework for PyTorch based on PyTorch-Lightning

* Documentation: <https://pietrolesci.github.io/energizer>
* GitHub: <https://github.com/pietrolesci/energizer>
* PyPI: <https://pypi.org/project/energizer/>
* Free software: MIT


## Features

`Energizer`

* allows training any PyTorch-Lightning model using Active-Learning with no code changes, requiring minimal information from the user

* is modular and easily extensible by using the energizer primitives, in case you need the extra flexibility

* provides a unified and tidy interfaces for Active-Learning so that you can easily mix and match query strategies, acquisition functions, etc with no boilerplate code

* can easily scale to multi-node/multi-gpu settings thanks to the Pytorch-Lighting backend


### Gotchas and future plans

At the moment `energizer` is focused on research settings. In other words, it works with datasets in which the labels are already available. Internally, it will mask the labels and mimick a true active learning setting. In the future, `energizer` will fully compatible with open-source annotation tools such as [`Label-Studio` ](https://labelstud.io/) and [`Rubrix`](https://www.rubrix.ml/). 

Currently `energizer` has been extensively tested on cpu and single-node/single-gpu settings due to availability issues. Support for multi-node/multi-gpu settings should work out of the box thanks to Pytorch-Lightning but has not been tested at this stage.

`energizer` supports pool-based active learning. We plan to add support for stream-based settings and for self-supervised training.


## Design

A core principle of `energizer` is full compatibility with Pytorch-Lightning, that is with any `LightningModule`. By simply implementing the required hooks in the query strategy, it should be possible to actively train any `LightningModule`.

The core objects in `energizer` are:

* The active learning loop: it is Pytorch-Lightning `Loop` that in essence implements the following steps at each labelling iteration
```python
for _ in range(max_labelling_epochs):
    if labelled_data_available:
        # fit the model
        fit_loop.run()

    if can_run_testing:  
        # if test_dataloader is provided
        test_loop.run()

    if unlabelled_data_available:
        indices = pool_loop.run()
        label_data(indices)
```

* Query strategy: it is a `LightningModule` itself that implements additional hooks and methods and is linked to a specific pool loop. The fundamental method of a query strategy is the `query` method which is in charge of returning the indices of the instances that needs to be labelled. To initialize a strategy, it is simply required to pass a `LightningModule` to the constructor
```python
from energizer.query_strategies import RandomQueryStrategy

model = MyGreatLightningModel()
query_strategy = RandomQueryStrategy(model)
```

* Trainer: it provides a simple extension to the Pytorch-Lightning trainer by implementing the `active_fit` method (and other state-tracking properties). The trainer knows that when the active learning loop is either testing or fitting it will use the underlying `LightningModule` passed to the strategy, i.e. `query_strategy.model` in the example above. When it needs to run on the pool it will use the `query_strategy` directly.


* Pool loops: a pool loop is implemented using the `Loop` abstraction in Pytorch-Lightning. Pool loops control how the query strategies behave on the pool. For example, the `RandomQueryStrategy` that queries random instances from the pool does not need to run any computation on the pool. Other query strategies might need to run model inference on the pool and thus need to be treated separately. For this reason, a pool loop is tightly linked to a query strategy


## Usage

The most basic usage of `energizer` requires minimal inputs from the user. 

Say we have the following `LightningModule`

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

    def forward(self, x: Union[Tensor, Tuple[Tensor, Tensor]]) -> Tensor:
        """NOTE: notice how we unpack the batch in forward method.
        
        More on this later.
        """
        if isinstance(x, tuple):
            x, y = batch
            return self.model(x), y
        return self.model(x)

    def common_step(self, batch: Tuple[Tensor, Tensor], stage: str) -> Tensor:
        """For convenience define a common step."""
        logits, y = self(batch)
        loss = F.cross_entropy(logits, y)
        self.log(f"{stage}/loss", loss)
        return loss

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self.common_step(batch, "train")
    
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self.common_step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.SGD(self.parameters(), lr=0.001)

model = MNISTModel()
```

We need to select a query strategy, say the `EntropyStrategy` that queries instances from the pool for which the model is most uncertain. Uncertainty is defined as the entropy of the model predictive label distribution

```python
from energizer.query_strategies import EntropyStrategy

entropy_strategy = EntropyStrategy(model)
```

> __NOTE__: When a strategy is instantiated, internally it creates a deep copy of the model passed. This is to avoid sharing states if you want to try out different query strategies on the same model. This might change in the future!

Now, to train the model with active learning we just need to apply the following changes

```diff
- from pytorch_lightning import Trainer
+ from energizer import Trainer
```

And then call the `active_fit` method

```python
trainer = Trainer(
    max_labelling_epochs=4,     # run the active learning loop 4 times
    query_size=10,              # at each loop query 10 instances
    max_epochs=3,               # fit the model on the labelled data for 3 epochs 
    test_after_labelling=True,  # test after each labelling
    # ... you can pass any other pl.Trainer arguments
)

results = trainer.active_fit(
    model=entropy_strategy,
    # ... dataloaders or datamodule
)
```

The `active_fit` method will take care of masking the train dataloader and create a pool dataloader.

And that's it! You will have a resulting model trained with active learning. 


### The anatomy of a query strategy

In the example abote, we used the `EntropyStrategy`. It needs to run model inference on the pool, get the logits, transform them into probabilities, and compute the entropy. So, contrarely to a `RandomStrategy`, we also need to implement how the model should behave when fed with a batch coming from the pool.

In `energizer` we implement a base class called `AccumulatorStrategy`. The name comes from the fact that it accumulates the results of each batch and the returns the indices corresponding to the Top-K instances. Do not worry if you have a huge pool, it performs a running Top-K operation and keeps in memory only `2 * K` instance at every time. 

In order to run pool-based active learning, we need to define how the model behaves when predicting on the unlabelled pool. This is achieved, by overriding the new "pool" hooks. An `AccumulatorStrategy` requires us to implement the `pool_step` method (similar to a `training_step` or `test_step` in Pytorch-Lightning) that runs inference on the batch and returns a 1-dimensional `Tensor` of scores (that are then Top-K-ed).

So, if we were to implement the `EntropyStrategy` ourselves, we would simply do

```python
from energizer.acquisition_functions import entropy

class EntropyStrategy(AccumulatorStrategy):
    def pool_step(self, batch: MODEL_INPUT, batch_idx: int, *args, **kwargs) -> Tensor:
        logits = self(batch)
        return entropy(logits)
```

As simple as this. We do not need to implement the `query` method in this case because for `AccumulatorStrategy`s, the output of `pool_step` is continually aggregated and we simply need to perform an argmax operation to obtain the indices. This is handled directly by `energizer`.


### Note on research settings

Finally, note that in our implementation of the `EntropyStrategy` the type of the batch input is `MODEL_INPUT`. This is to highlight that `pool_step` (by defaut, of course you can override it as you like) does not unpack a batch: it expectes that a batch can directly be passed to the `forward` of the underlying `LightningModule`. This is the case in real-world scenarios where you actually do not have a label.

However, for research settings, we do have a label in each batch. Since `energizer` cannot know how to unpack a batch (it can be a `dict`, a `tuple`, your own custom data structure, etc) it also implements an additional hook that can be used for this purpose `get_inputs_from_batch`. 

So if you are in a research setting (your batch contains the labels and needs to be unpacked in order to extract the inputs), you have the following 3 options:

1. Unpack a batch in the `forward` method of your `LightningModule` (as we did in the MNIST example)

1. Define a `forward` method that expects only the inputs in your `LightningModule` (as it is usually done), subclass a query strategy (e.g, `EntropyStrategy`), and implement the `get_inputs_from_batch`

1. Define a `forward` method that expects only the inputs in your `LightningModule` (as it is usually done), subclass a query strategy (e.g, `EntropyStrategy`), and implement the `pool_step` from scratch including the batch unpacking logic



## Contributing

Install `energizer` locally

```bash
conda create -n energizer-dev python=3.9 -y
# conda install poetry -y
curl -sSL https://install.python-poetry.org | python3 -
poetry install --all-extras --sync
```


## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [waynerv/cookiecutter-pypackage](https://github.com/waynerv/cookiecutter-pypackage) project template.
