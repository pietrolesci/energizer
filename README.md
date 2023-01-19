[![pypi](https://img.shields.io/pypi/v/energizer.svg)](https://pypi.org/project/energizer/)
[![python](https://img.shields.io/pypi/pyversions/energizer.svg)](https://pypi.org/project/energizer/)
[![Build Status](https://github.com/pietrolesci/energizer/actions/workflows/dev.yml/badge.svg)](https://github.com/pietrolesci/energizer/actions/workflows/dev.yml)
[![codecov](https://codecov.io/gh/pietrolesci/energizer/branch/main/graph/badge.svg?token=782XT9AQFZ)](https://codecov.io/gh/pietrolesci/energizer)


`Energizer` is an Active-Learning framework for PyTorch based on PyTorch-Lightning

* Documentation: <https://pietrolesci.github.io/energizer>
* GitHub: <https://github.com/pietrolesci/energizer>
* PyPI: <https://pypi.org/project/energizer/>
* Free software: Apache-2.0


## Installation

```bash
pip install energizer
```

To contribute, clone the `energizer` repo and use poetry to initialize the environment (if you don't have poetry run `curl -sSL https://install.python-poetry.org | python3 -`)

```bash
conda create -n energizer-dev python=3.9 -y
poetry install --all-extras --sync
```

## Features

`Energizer` come with the following features, it

* allows training any PyTorch-Lightning model using Active-Learning with no code changes, requiring minimal information from the user (see [Getting started](#getting-started))

* is modular and easily extensible by using the energizer primitives, in case you need the extra flexibility

* provides a unified and tidy interfaces for Active-Learning with a consistent and predictable API so that you can easily mix and match query strategies, acquisition functions, etc, with no boilerplate code

* can easily scale to multi-node/multi-gpu settings thanks to the Pytorch-Lighting backend


## Energizer in 30 seconds

The most basic usage of `energizer` requires minimal inputs from the user:

1. Define your `LightningModule`

    ```python
    model = MyLightningModel(*args, **kwargs)
    ```

1. Import a query strategy from `energizer` and instantiate it
    ```python
    from energizer.query_strategies import LeastConfidenceStrategy
    
    query_strategy = LeastConfidenceStrategy(model)
    ```

1. Import the `energizer` trainer
    ```diff
    - from pytorch_lightning import Trainer
    + from energizer import Trainer
    ```

1. Instantiate the trainer passing the `energizer`-specific arguments and the
usual `pl.Trainer` arguments

    ```python
    trainer = Trainer(
        max_labelling_epochs=4,     #(1) 
        query_size=10,              #(2)
        max_epochs=3,               #(3)
        test_after_labelling=True,  #(4)
        #(5)
    )
    ```


## Energizer in 15 minutes: fine-tuning BERT on AGNews

An active learning loop in `energizer` looks something like this

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

Currently `energizer` is more geared towards research settings. Therefore, we currently do not support interactive annotation and, thus, assume that your dataset already has annotated data. Internally, annotations will be masked to mimick a real active learning process with the only difference that when you "label" instances (calling `label_data` in the snippet above) you are simply unmasking their labels. However, support for interactive labelling is coming very soon (see #coming-next).

Without further ado, let's get into it. In the snippet below, we will ("actively") train a `bert-base-uncased` model on the AGNews dataset. First, let's load the dataset from the [HuggingFace Hub](https://huggingface.co/datasets/pietrolesci/ag_news) and tokenize it

```python
from datasets import load_dataset
from transformers import AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
dataset = load_dataset("pietrolesci/ag_news", "concat")
dataset = dataset.map(lambda ex: tokenizer(ex["text"]), batched=True)
```

Now, let's take 30% of the training set to construct a validation set, and create the respective dataloaders

```python
from transformers import DataCollatorWithPadding
from torch.utils.data import DataLoader


train_set, test_set = dataset["train"], dataset["test"]

# get the number of classes for later use
num_classes = len(train_set.features["label"].names)

# train-val split
train_val_splits = train_set.train_test_split(0.3)
train_set, val_set = train_val_splits["train"], train_val_splits["test"]

# define collator function that dynamically pads batches
collator_fn = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")

# select columns and create dataloaders
columns_to_keep = ["label", "input_ids", "token_type_ids", "attention_mask"]
train_dl = DataLoader(
    train_set.with_format(columns=columns_to_keep),
    batch_size=32,  # might need to adjust this based on your hardware
    collate_fn=collator_fn,
)
val_dl = DataLoader(
    val_set.with_format(columns=columns_to_keep),
    batch_size=128,  # might need to adjust this based on your hardware
    collate_fn=collator_fn,
)
test_dl = DataLoader(
    test_set.with_format(columns=columns_to_keep),
    batch_size=128,  # might need to adjust this based on your hardware
    collate_fn=collator_fn,
)
```

Great! We are done with the dataloading part. Now, let's focus on modelling. We define a normal `LightningModule` backed by the `bert-base-uncased` model

```python
# type annotations
from typing import Any, Dict
from torch.optim import Optimizer
from torch import Tensor

from pytorch_lightning import LightningModule
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
import torch.nn.functional as F


class TransformerClassifier(LightningModule):
    def __init__(
        self, name_or_path: str, num_classes: int, learning_rate: float = 1e-4
    ) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name_or_path,
            num_labels=num_classes,
        )

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        return self.model(**batch).logits

    def common_step(self, batch: Any, stage: str) -> Tensor:
        """Outputs loss and logits, logs loss and metrics."""
        targets = batch.pop("labels")
        logits = self(batch)
        loss = F.cross_entropy(logits, targets)
        self.log(f"{stage}_loss", loss)
        return loss

    def training_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        return self.common_step(batch, "train")

    def validation_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        return self.common_step(batch, "val")

    def test_step(self, batch: Any, batch_idx: int) -> Dict[str, Any]:
        return self.common_step(batch, "test")

    def configure_optimizers(self) -> Optimizer:
        return AdamW(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.learning_rate,
        )

# instantiate model
model = TransformerClassifier("bert-base-uncased", num_classes=num_classes)
```

> __NOTE__: many of the query strategies expect the model `forward` to output the logits. By default the HuggingFace transformers return a [`ModelOutput`](https://huggingface.co/docs/transformers/main_classes/output) dataclass, so we extracted the logits explicitly.

Now we need to select a query strategy. A good starting point is the entropy strategy that selects instances that maximize the predictive entropy. To use it in the active learning loop, simply import the `EntropyStrategy` and instantiate it passing the model instance

```python
from energizer.query_strategies import EntropyStrategy

entropy_strategy = EntropyStrategy(model=model)
```

Ok, now we have the dataloaders, the model, and the query strategy. We are ready to start. In order to use the active learning loop, instead of importing the trainer from Lightning, you need to import it from energizer. This is the same old trainer with the added bonus that it implements the `active_fit` method

```diff
- from pytorch_lightning import Trainer
+ from energizer import Trainer
```

Finally, instantiate the trainer. Since it is the same old trainer, you can pass any Lightning flag. In addition, you can pass additional arguments to customize your active learning loop. In this case, we tell energizer that we want to run 4 labelling iterations (`max_labelling_epochs=4`); at each iteration we query 10 datapoints (`query_size=10`); after labelling new instances we want the model to train for 3 epochs (`max_epochs=3`) and that after it is trained on the new labelled data, we want it to be tested on the test set (`test_after_labelling=True`)

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
    train_dataloaders=train_dl,
    val_dataloaders=val_dl,
    test_dataloaders=test_dl,
)
```

And that's it! Now, `entropy_strategy.model` is a model trained with active learning. You can explore the `results` object and you can get a pandas dataframe out of it by simply calling `results.to_pandas()`.

You can find more information about how `energizer` works in the [Design](#design) section.


## The anatomy of a query strategy

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


## Coming next

At the moment `energizer` is focused on research settings. In other words, it works with datasets in which the labels are already available. Internally, it will mask the labels and mimick a true active learning setting. In the future, `energizer` will fully compatible with open-source annotation tools such as [`Label-Studio`](https://labelstud.io/) and [`Rubrix`](https://www.rubrix.ml/).

Currently `energizer` has been extensively tested on cpu and single-node/single-gpu settings due to availability issues. Support for multi-node/multi-gpu settings should work out of the box thanks to Pytorch-Lightning but has not been tested at this stage.

`energizer` supports pool-based active learning. We plan to add support for stream-based settings and for self-supervised training.


## Credits

This package was created with [Cookiecutter](https://github.com/audreyr/cookiecutter) and the [waynerv/cookiecutter-pypackage](https://github.com/waynerv/cookiecutter-pypackage) project template.
