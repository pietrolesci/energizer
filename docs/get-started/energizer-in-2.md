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

    1. Run the active learning loop 4 times.
    2. At each loop query 10 instances.
    3. Fit the model on the labelled data for 3 epochs at each labelling iteration.
    4. Test after each labelling iteration.
    5.  ...and you can pass any other `pl.Trainer` arguments.


1. Finally, call the `active_fit` method and collect the results

    ```python
    results = trainer.active_fit(
        model=entropy_strategy,
        #(1)
    )
    ```

    1. You can pass the individual dataloaders, a `pl.DataModule`, 
    or the `energizer.ActiveDataModule` directly.
