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
from typing import Any, Callable, Dict, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.loops import Loop
from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.loops.dataloader.prediction_loop import PredictionLoop
from pytorch_lightning.loops.epoch.prediction_epoch_loop import PredictionEpochLoop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.connectors.data_connector import _DataLoaderSource
from pytorch_lightning.trainer.progress import Progress
from pytorch_lightning.trainer.states import RunningStage, TrainerFn, TrainerStatus
from pytorch_lightning.trainer.trainer import Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden

from energizer import data
from energizer.data import ActiveDataModule
from energizer.strategies import EnergizerStrategy


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

        # PoolLoop
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
        self.pool_loop: Optional[PredictionLoop] = None
        # self.labelling_loop = None
        self.lightning_module: Optional[LightningModule] = None
        self.strategy_state_dict: Optional[Dict[str, torch.Tensor]] = None

    @property
    def done(self) -> bool:
        return (
            not self.trainer.datamodule.has_unlabelled_data
            or (self.total_budget > 0 and self.trainer.datamodule.total_labelled_size >= self.total_budget)
            or self.progress.current.completed >= self.max_epochs
        )

    def reset(self) -> None:
        """Nothing to reset in this loop."""

    def connect(self, trainer: Trainer) -> None:
        """Connects to the default fit_loop of the trainer."""
        # attach the loop for training and validation
        self.fit_loop = trainer.fit_loop
        # attach the loop for testing
        self.test_loop = trainer.test_loop
        # attach the loop for evaluation on the pool
        self.pool_loop = trainer.predict_loop
        # patch arguments
        self.max_epochs = self.fit_loop.max_epochs
        self.fit_loop.max_epochs = self.label_epoch_frequency

    def __getattr__(self, key):
        """Connect self attributes to fit_loop attributes."""
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Validate inputs and gather essential attributes."""
        if not isinstance(self.trainer.datamodule, ActiveDataModule):
            raise MisconfigurationException(
                f"ActiveLearningLoop requires the ActiveDataModule, not {type(self.trainer.datamodule)}."
            )
        self.lightning_module = self.trainer.lightning_module
        self.strategy_state_dict = deepcopy(self.lightning_module.state_dict())
        self.strategy.connect(self.lightning_module, self.query_size)
        self.trainer.datamodule._min_steps_per_epoch = self.min_steps_per_epoch

        if not self.fit_loop or not self.test_loop or not self.pool_loop:
            raise MisconfigurationException("`ActiveLearningLoop` must be connected to a `Trainer`.")

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        self.strategy._reset()
        self.progress.increment_ready()

    def advance(self, *args: Any, **kwargs: Any) -> None:
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
            self._reset_predicting_pool()
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
        """Reset the LightningModule."""
        self._reset_fitting()  # reset the model to load the correct state_dict
        if self.reset_weights:
            self.trainer.lightning_module.load_state_dict(self.strategy_state_dict)
        self.progress.increment_completed()

    def _reset_fitting(self) -> None:
        """Reset training and validation dataloaders, and states."""
        self.trainer.reset_train_dataloader()
        self.trainer.reset_val_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True
        self._connect_model(self.lightning_module)

    def _reset_testing(self) -> None:
        """Reset testing dataloader, and states."""
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True
        self._connect_model(self.lightning_module)

    def _reset_predicting_pool(self):
        """Reset pool dataloader, and states."""
        self._reset_pool_dataloader()
        self.trainer.state.fn = TrainerFn.PREDICTING
        self.trainer.predicting = True
        self._connect_model(self.strategy)

    def _connect_model(self, model: LightningModule):
        self.trainer.training_type_plugin.connect(model)

    def _reset_pool_dataloader(self) -> None:
        # Hack
        dataloader = self.trainer.datamodule.pool_dataloader()
        self.trainer.num_predict_batches = [len(dataloader)]
        self.trainer.predict_dataloaders = [dataloader]
