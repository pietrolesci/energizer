# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from copy import deepcopy
from typing import Any, Dict, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loops import Loop
from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.loops.dataloader.prediction_loop import PredictionLoop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.progress import Progress
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from energizer.data import ActiveDataModule
from energizer.strategies.base import EnergizerStrategy


class ActiveLearningLoop(Loop):
    """Loop for active learning.

    Active learning loop mental model

    ```
    while has_unlabelled_data or total_labelled_size < total_budget or current_epoch < max_epoch:

        if datamodule.has_labelled_data:

            # FitLoop
            for epoch in range(label_epoch_frequency):

                # TrainingEpochLoop
                for batch_idx, batch in enumerate(train_dataloader):
                    # fit model

                # ValidationEpochLoop
                for batch_idx, batch in enumerate(val_dataloader):
                    # validate model

        # PoolLoop [EvaluationLoop]
        for batch_idx, batch in enumerate(pool_dataloader):
            # run heuristic and return indices of instances to label

        # LabellingLoop
        for idx in indices_to_label:
            # label pool_dataset[idx]
            datamodule.label()

        if reset_weights:
            # reset model weights
    ```
    """

    def __init__(
        self,
        strategy: EnergizerStrategy,
        query_size: int = 2,
        label_epoch_frequency: int = 1,
        reset_weights: bool = True,
        total_budget: int = -1,
        min_steps_per_epoch: int = None,
    ) -> None:
        """
        Args:
            strategy (EnergizerStrategy): An active learning strategy.
            label_epoch_frequency (int): Number of epoch to run before requesting labellization.
            reset_weights (bool): Whether to reset the weights to their initial state at
                the end of each labelling iteration.
            total_budget (int): Number of instances to acquire before stopping the training loop.
                Set to -1 to stop when the unlabelled pool is exhausted.
            min_steps_per_epoch (Optional[int]): Minumum number of steps per epoch. Especially in the beginning
                of active learning when there are a few labelled instances, this arguments allows the user to
                train for a minimum number of iterations by resampling the available data.

        Note that some of the required attributes will be set when the loop is actually run or connected.
        """
        super().__init__()
        self.strategy = strategy
        self.query_size = query_size
        self.label_epoch_frequency = label_epoch_frequency
        self.reset_weights = reset_weights
        self.total_budget = total_budget
        self.min_steps_per_epoch = min_steps_per_epoch

        self.progress = Progress()
        self.fit_loop: Optional[FitLoop] = None
        self.test_loop: Optional[EvaluationLoop] = None
        self.pool_loop: Optional[EvaluationLoop] = None
        self.lightning_module: Optional[LightningModule] = None
        self.lightning_module_state_dict: Optional[Dict[str, torch.Tensor]] = None

    @property
    def done(self) -> bool:
        """Define when the ActiveLearningLoop terminates.

        The loop terminates when one of these conditions is satisfied:

        - The pool dataloader has no more unlabelled data

        - The total labelling budget has been reached

        - The maximum number of epochs have been run
        """
        return (
            not self.trainer.datamodule.has_unlabelled_data
            or (self.total_budget > 0 and self.trainer.datamodule.total_labelled_size >= self.total_budget)
            or self.progress.current.completed >= self.max_epochs
        )

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def connect(self, trainer: Trainer) -> None:
        """Connect to the default fit_loop of the trainer.

        Specifically this method

        - Extracts the `fit_loop` from the trainer and uses it to train and validate the model

        - Extracts the `max_epochs` and uses it as a stopping condition for the `ActiveLearningLoop`

        - Patches the `max_epochs` argument of the `fit_loop` with `label_epoch_frequency` so that
        during each training phase it trains the model `label_epoch_frequency` times

        - Extracts the `test_loop` from the trainer and uses it to test the model

        - Extracts the `test_loop` from the trainer and uses it to run the model on the pool dataset;
        the `test_loop` is an `evaluation_loop` and rather convenient, with respect to the `prediction_loop`
        because it easily allows for multi-gpu training
        """
        # attach the loop for training and validation
        self.fit_loop = trainer.fit_loop

        # patch arguments
        self.max_epochs = self.fit_loop.max_epochs
        self.fit_loop.max_epochs = self.label_epoch_frequency

        # attach the loop for testing
        self.test_loop = trainer.test_loop

        # attach the loop for evaluation on the pool
        self.pool_loop = trainer.test_loop

    def __getattr__(self, key):
        """Connect self attributes to `fit_loop` attributes."""
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Validate inputs and gather essential attributes.

        In particular:

        - Records the model we wish to train as `self.lightning_module` for easy access

        - If `reset_weights` is True, it records `lightning_module_state_dict` in order to
        reset the weights at each labelling iteration

        - Initializes the `strategy` based on the arguments passes in this class constructor

        - Initializes the `datamodule` based on the arguments passes in this class constructor
        """
        if not isinstance(self.trainer.datamodule, ActiveDataModule):
            raise MisconfigurationException(
                f"ActiveLearningLoop requires the ActiveDataModule, not {type(self.trainer.datamodule)}."
            )
        self.lightning_module = self.trainer.lightning_module

        if self.reset_weights:
            self.lightning_module_state_dict = deepcopy(self.lightning_module.state_dict())

        self.strategy.connect(self.lightning_module, self.query_size)

        self.trainer.datamodule._min_steps_per_epoch = self.min_steps_per_epoch

        if not self.fit_loop or not self.test_loop or not self.pool_loop:
            raise MisconfigurationException("`ActiveLearningLoop` must be connected to a `Trainer`.")

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Reset the strategy internal states and increment progress."""
        self.strategy.reset()
        self.progress.increment_ready()

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Run one iteration of the active learning loop.

        At each iteration of the active learning loop the following steps are performed

        - If there are labelled data (no cold-starting),

            1. Reset dataloaders, states, and attach original model for training and validation

            1. Run training for `label_epoch_frequency` epochs and each epoch has a least `min_steps_per_epoch` steps

            1. Run validation

            1. Reset dataloaders, states, and attach original model for testing

            1. Run testing (useful to produce active learning outputs) and log metrics

        - If there are unlabelled data

            1. Reset dataloaders, states, and attach patched model for pool (alike testing)

            1. Run pool loop and collect the indices of the instance to label

            1. Label instances
        """
        self.progress.increment_started()
        print("\nEPOCH", flush=True)
        print("Active learning dataset:", self.trainer.datamodule._active_dataset)

        if self.trainer.datamodule.has_labelled_data:
            # training and validation
            self._reset_fitting()
            self.fit_loop.run()  # type: ignore

            # testing
            self._reset_testing()
            metrics = self.test_loop.run()  # type: ignore
            if metrics:
                self.trainer.logger.log_metrics(metrics[0], step=self.trainer.global_step)

        # pool evaluation
        if self.trainer.datamodule.has_unlabelled_data:
            self._reset_pool()
            self.pool_loop.run()  # type: ignore
            indices = self.strategy.indices.tolist()  # type: ignore

            # print("Indices Pool:", indices, flush=True)
            # print("Pool Indices:", self.trainer.datamodule.pool_dataset.indices, flush=True)
            # print("Indices Oracle:", self.trainer.datamodule._active_dataset._pool_to_oracle(indices), flush=True)
            # print()

            # labelling
            self.trainer.datamodule.label(indices)
        self.progress.increment_processed()

    def on_advance_end(self) -> None:
        """Reset the LightningModule, optionally reset its weights, and increment progress."""
        self._reset_fitting()  # reset the model to load the correct state_dict
        if self.reset_weights:
            self.trainer.lightning_module.load_state_dict(self.lightning_module_state_dict)
        self.progress.increment_completed()

    def _reset_fitting(self) -> None:
        """Reset training and validation dataloaders and states and attach the original LightningModule."""
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True
        self._attach_model(self.lightning_module)

    def _reset_testing(self) -> None:
        """Reset testing dataloaders and states and attach the original LightningModule."""
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True
        self._attach_model(self.lightning_module)

    def _reset_pool(self):
        """Reset pool dataloaders and states and attach the patched LightningModule.

        Since the `pool_loop` is a `test_loop`, the states are the same as those requried by
        the `test_loop`. The difference here is that the pool dataloader is loaded and a patched
        version of the model, called `strategy`, is used.
        """
        self._reset_pool_dataloader()
        # self.trainer.state.fn = TrainerFn.PREDICTING
        # self.trainer.predicting = True
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True
        self._attach_model(self.strategy)

    def _reset_pool_dataloader(self) -> None:
        # Hack
        dataloader = self.trainer.datamodule.pool_dataloader()
        self.trainer.num_test_batches = [len(dataloader)]
        self.trainer.test_dataloaders = [dataloader]

    def _attach_model(self, model: LightningModule):
        """Attach a LightningModule to use during a loop."""
        self.trainer.training_type_plugin.connect(model)
