# `Energizer` at a glance


## Features

`Energizer`

* allows training any PyTorch-Lightning model using Active-Learning with no code changes, requiring minimal information from the user

* is modular and easily extensible by using the energizer primitives, in case you need the extra flexibility

* provides a unified and tidy interfaces for Active-Learning so that you can easily mix and match query strategies, acquisition functions, etc with no boilerplate code

* can easily scale to multi-node/multi-gpu settings thanks to the Pytorch-Lighting backend


## Gotchas and future plans

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


## A note on batch unpacking in research settings

Batch unpacking is problematic only in research settings where we actually have a labelled dataset and to mimick the annotation scenario we "mask" instances from the training set. However, `energizer` will not mask the labels: the dataloader will still return a batch "as is", i.e. with the labels. The `energizer.ActiveDataModule` simply divides instances in train/pool datasets via a mask, but does not actually "remove" the labels. This design choice was made because your batch can be any data structure: a tuple. a dict, a dataclass, etc. 

With only a few exceptions, the `pool_step` method of the available query strategies are all implemented assuming that the batch is formed by only the inputs and no batch unpacking is performed. The inputs to `pool_step` are directly passed to the `forward` method of the underlying `LightningModule`. 

```python
def pool_step(self, batch: MODEL_INPUT, batch_idx: int, *args, **kwargs) -> Tensor:
    logits = self.underlying_lightning_module(batch)  #(1)
    scores = some_acquisition_function(logits)        #(2)
    return scores
```

1. This is the name of the attribute that contains the `LightningModule` that we pass in the constructor of a query strategy. In practice, in the codebase, we do not use this nomenclature, but it is just for the sake of the example. 

2. This is simply a placeholder for any scoring function available in `energizer`.

This is problematic because now the burder of batch unpacking is shifted to the `forward` method of the underlying model and the [suggested best practice](https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#starter-example) says that the `forward` method of `LightningModule`s should simply perfom inference.

Therefore, to fix this there are 3 possible solutions:




Roughly, a default `pool_step` is defined as follows

To stress this, in `energizer` the type annotation of the batch argument in any `pool_step` is `MODEL_INPUT`. This is to highlight that `pool_step` by defaut expectes that a batch can directly be passed to the `forward` of the underlying `LightningModule`. 


Therefore, it's up to the user to tell `energizer` how to extract the inputs from the batch!


This is the case in real-world scenarios where you actually do not have a label.

, of course you can override it as you like) does not unpack a batch: it 


 Each query strategy implements a hook that can be used for this purpose: `get_inputs_from_batch`. It roughly works as follows:

```python
for batch in pool_dataloader:
    inputs = query_strategy.get_inputs_from_batch(batch)
    query_strategy.pool_step(inputs)
```

By default `get_inputs_from_batch` it simply returns its inputs

:::energizer.query_strategies.base.BaseQueryStrategy.get_inputs_from_batch

    

Each query strategy Under the hood, any query strategy in `energizer` simply calls the `forward` 
    method of the underlying `LightningModule` in the `pool_step`.

  
  
  Note that in the `forward` method implementation of `MNISTCNNModel` we accounted
    for the batch unpacking. Usually, the `forward` method is supposed to simply
    expect the inputs to perform inference and batch unpacking is done in the 
    `{training, validation, test, predict}_step`s. 
    
    This is done for convenience here!

    
    
    







    
    of the `EntropyStrategy` the type of the batch input is `MODEL_INPUT`. This is to highlight that `pool_step` (by defaut, of course you can override it as you like) does not unpack a batch: it expectes that a batch can directly be passed to the `forward` of the underlying `LightningModule`. This is the case in real-world scenarios where you actually do not have a label.

    However, for research settings, we do have a label in each batch. Since `energizer` cannot know how to unpack a batch (it can be a `dict`, a `tuple`, your own custom data structure, etc) it also implements an additional hook that can be used for this purpose `get_inputs_from_batch`.

    So if you are in a research setting (your batch contains the labels and needs to be unpacked in order to extract the inputs), you have the following 3 options:

    1. Unpack a batch in the `forward` method of your `LightningModule` (as we did in the MNIST example)

    1. Define a `forward` method that expects only the inputs in your `LightningModule` (as it is usually done), subclass a query strategy (e.g, `EntropyStrategy`), and implement the `get_inputs_from_batch`

    1. Define a `forward` method that expects only the inputs in your `LightningModule` (as it is usually done), subclass a query strategy (e.g, `EntropyStrategy`), and implement the `pool_step` from scratch including the batch unpacking logic