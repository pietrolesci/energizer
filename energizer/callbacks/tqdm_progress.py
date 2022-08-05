import importlib
from typing import Any, Optional, Union

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm, TQDMProgressBar, _update_n, convert_inf

if importlib.util.find_spec("ipywidgets") is not None:
    from tqdm.auto import tqdm as _tqdm
else:
    from tqdm import tqdm as _tqdm

import sys


class TQDMProgressBarActiveLearning(TQDMProgressBar):
    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__(refresh_rate, process_position)
        self._pool_progress_bar: Optional[_tqdm] = None

    """
    Add pool progress bar
    """

    @property
    def pool_progress_bar(self) -> _tqdm:
        if self._pool_progress_bar is None:
            raise TypeError(f"The `{self.__class__.__name__}._pool_progress_bar` reference has not been set yet.")
        return self._pool_progress_bar

    @pool_progress_bar.setter
    def pool_progress_bar(self, bar: _tqdm) -> None:
        self._pool_progress_bar = bar

    def init_pool_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for testing."""
        bar = Tqdm(
            desc=self.pool_description,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    """
    Pool porperties
    """

    @property
    def pool_description(self) -> str:
        return "Pool"

    @property
    def pool_batch_idx(self) -> int:
        """The number of batches processed during testing.

        Use this to update your progress bar.
        """
        return self.trainer.active_learning_loop.pool_loop.epoch_loop.batch_progress.current.processed

    @property
    def total_pool_batches_current_dataloader(self) -> Union[int, float]:
        """The total number of testing batches, which may change from epoch to epoch for current dataloader.

        Use this to set the total number of iterations in the progress bar. Can return ``inf`` if the test dataloader is
        of infinite size.
        """
        assert self._current_eval_dataloader_idx is not None
        return self.trainer.num_pool_batches[self._current_eval_dataloader_idx]

    """
    Implement pool hooks
    """

    def on_pool_start(self, *_: Any) -> None:
        self.pool_progress_bar = self.init_pool_tqdm()

    def on_pool_batch_start(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if not self.has_dataloader_changed(dataloader_idx):
            return

        self.pool_progress_bar.reset(convert_inf(self.total_pool_batches_current_dataloader))
        self.pool_progress_bar.initial = 0
        self.pool_progress_bar.set_description(f"{self.pool_description} DataLoader {dataloader_idx}")

    def on_pool_batch_end(self, *_: Any) -> None:
        if self._should_update(self.pool_batch_idx, self.pool_progress_bar.total):
            _update_n(self.pool_progress_bar, self.pool_batch_idx)

    def on_pool_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.pool_progress_bar.close()
        self.reset_dataloader_idx_tracker()

    """
    Patch properties
    """

    ### NOTE: patch `fit_loop` -- take from pytorch_lightning.callbacks.progress.base ###
    @property
    def _val_processed(self) -> int:
        return self.trainer.active_learning_loop.fit_loop.epoch_loop.val_loop.epoch_loop.batch_progress.total.processed

    @property
    def train_batch_idx(self) -> int:
        """The number of batches processed during training.

        Use this to update your progress bar.
        """
        return self.trainer.active_learning_loop.fit_loop.epoch_loop.batch_progress.current.processed

    @property
    def val_batch_idx(self) -> int:
        """The number of batches processed during validation.

        Use this to update your progress bar.
        """
        loop = self.trainer.active_learning_loop.fit_loop.epoch_loop.val_loop
        current_batch_idx = loop.epoch_loop.batch_progress.current.processed
        return current_batch_idx

    @property
    def total_val_batches(self) -> Union[int, float]:
        """The total number of validation batches, which may change from epoch to epoch for all val dataloaders.

        Use this to set the total number of iterations in the progress bar. Can return ``inf`` if the predict dataloader
        is of infinite size.
        """
        return (
            sum(self.trainer.num_val_batches)
            if self.trainer.active_learning_loop.fit_loop.epoch_loop._should_check_val_epoch()
            else 0
        )

    @property
    def total_batches_current_epoch(self) -> Union[int, float]:
        total_train_batches = self.total_train_batches
        total_val_batches = self.total_val_batches
        assert self._trainer is not None

        if total_train_batches != float("inf") and total_val_batches != float("inf"):
            # val can be checked multiple times per epoch
            val_check_batch = self.trainer.val_check_batch
            if self.trainer.check_val_every_n_epoch is None:
                train_batches_processed = self.trainer.active_learning_loop.fit_loop.total_batch_idx + 1
                val_checks_per_epoch = ((train_batches_processed + total_train_batches) // val_check_batch) - (
                    train_batches_processed // val_check_batch
                )
            else:
                val_checks_per_epoch = total_train_batches // val_check_batch

            total_val_batches = total_val_batches * val_checks_per_epoch

        return total_train_batches + total_val_batches

    ### NOTE: patch `test_loop` -- take from pytorch_lightning.callbacks.progress.base ###
    @property
    def test_batch_idx(self) -> int:
        """The number of batches processed during testing.

        Use this to update your progress bar.
        """
        return self.trainer.active_learning_loop.test_loop.epoch_loop.batch_progress.current.processed

    ### NOTE: patch `print` method ###
    def print(self, *args: Any, sep: str = " ", **kwargs: Any) -> None:
        active_progress_bar = None

        if self._main_progress_bar is not None and not self.main_progress_bar.disable:
            active_progress_bar = self.main_progress_bar
        elif self._val_progress_bar is not None and not self.val_progress_bar.disable:
            active_progress_bar = self.val_progress_bar
        elif self._test_progress_bar is not None and not self.test_progress_bar.disable:
            active_progress_bar = self.test_progress_bar
        elif self._predict_progress_bar is not None and not self.predict_progress_bar.disable:
            active_progress_bar = self.predict_progress_bar
        elif self._pool_progress_bar is not None and not self.pool_progress_bar.disable:
            active_progress_bar = self.pool_progress_bar

        if active_progress_bar is not None:
            s = sep.join(map(str, args))
            active_progress_bar.write(s, **kwargs)

    def on_train_epoch_start(self, trainer: "pl.Trainer", *_: Any) -> None:
        total_batches = self.total_batches_current_epoch
        self.main_progress_bar.reset(convert_inf(total_batches))
        self.main_progress_bar.initial = 0
        self.main_progress_bar.set_description(f"Epoch {trainer.current_epoch}")
