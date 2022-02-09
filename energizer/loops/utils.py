import io
import os
import sys
from typing import Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm, TQDMProgressBar, _update_n, convert_inf
from pytorch_lightning.utilities.distributed import rank_zero_debug

from energizer.strategies.base import EnergizerStrategy


class EnergizerTQDMProgressBar(TQDMProgressBar):
    def __init__(self, refresh_rate: int = 1, process_position: int = 0):
        super().__init__(refresh_rate, process_position)
        self.al_progress_bar = None

    def __getstate__(self):
        # can't pickle the tqdm objects
        state = super().__getstate__()
        state["al_progress_bar"] = None
        return state

    def init_sanity_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for the validation sanity run."""
        bar = Tqdm(
            desc="Validation sanity check",
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    def init_train_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for training."""
        bar = Tqdm(
            desc="\tTraining",
            initial=self.train_batch_idx,
            # position=(2 * self.process_position),
            position=1,
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_predict_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for predicting."""
        bar = Tqdm(
            desc="Predicting",
            initial=self.train_batch_idx,
            position=(2 * self.process_position),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
            smoothing=0,
        )
        return bar

    def init_validation_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for validation."""
        # The main progress bar doesn't exist in `trainer.validate()`
        has_main_bar = self.main_progress_bar is not None
        bar = Tqdm(
            desc="\tValidating",
            position=(2 * self.process_position + has_main_bar),
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    def init_test_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for testing."""
        bar = Tqdm(
            desc="\tTesting",
            # position=(2 * self.process_position),
            position=2,
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    def init_pool_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for testing."""
        bar = Tqdm(
            desc="\tPool",
            # position=(2 * self.process_position),
            position=3,
            disable=self.is_disabled,
            leave=False,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    def init_al_tqdm(self) -> Tqdm:
        """Override this to customize the tqdm bar for testing."""
        bar = Tqdm(
            desc="Labelling Epoch",
            # position=(2 * self.process_position),
            position=0,
            disable=self.is_disabled,
            leave=True,
            dynamic_ncols=True,
            file=sys.stdout,
        )
        return bar

    def on_sanity_check_start(self, trainer, pl_module):
        self.val_progress_bar = self.init_sanity_tqdm()
        self.main_progress_bar = Tqdm(disable=True)  # dummy progress bar
        self.al_progress_bar = Tqdm(disable=True)  # dummy progress bar

    def on_sanity_check_end(self, trainer, pl_module):
        self.main_progress_bar.close()
        self.val_progress_bar.close()

    def on_train_start(self, trainer, pl_module):
        self.al_progress_bar = self.init_al_tqdm()
        self.main_progress_bar = self.init_train_tqdm()

    def on_train_epoch_start(self, trainer, pl_module):
        total_train_batches = self.total_train_batches
        total_val_batches = self.total_val_batches
        if total_train_batches != float("inf") and total_val_batches != float("inf"):
            # val can be checked multiple times per epoch
            val_checks_per_epoch = total_train_batches // trainer.val_check_batch
            total_val_batches = total_val_batches * val_checks_per_epoch

        total_batches = total_train_batches + total_val_batches
        self.main_progress_bar.total = convert_inf(total_batches)
        self.main_progress_bar.set_description(f"\tEpoch {trainer.current_epoch}")

        totat_test_batches = len(trainer.datamodule.test_dataloader())
        total_pool_batches = len(trainer.datamodule.pool_dataloader())
        total_al_batches = total_train_batches + total_val_batches + totat_test_batches + total_pool_batches
        self.al_progress_bar.total = convert_inf(total_al_batches)
        self.al_progress_bar.set_description(f"Labelling Epoch {trainer.datamodule.last_labelling_step}")

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._should_update(self.train_batch_idx):
            _update_n(self.al_progress_bar, self.train_batch_idx + self._val_processed)
            _update_n(self.main_progress_bar, self.train_batch_idx + self._val_processed)
            self.main_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        _update_n(self.al_progress_bar, self.train_batch_idx + self._val_processed)
        _update_n(self.main_progress_bar, self.train_batch_idx + self._val_processed)
        if not self.main_progress_bar.disable:
            self.main_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))

    def on_train_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.main_progress_bar.close()

    def on_validation_start(self, trainer, pl_module):
        if trainer.sanity_checking:
            self.val_progress_bar.total = sum(trainer.num_sanity_val_batches)
        else:
            self.val_progress_bar = self.init_validation_tqdm()
            self.val_progress_bar.total = convert_inf(self.total_val_batches)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self._should_update(self.val_batch_idx):
            _update_n(self.al_progress_bar, self.val_batch_idx)
            _update_n(self.val_progress_bar, self.val_batch_idx)
            if trainer.state.fn == "fit":
                _update_n(self.al_progress_bar, self.train_batch_idx + self._val_processed)
                _update_n(self.main_progress_bar, self.train_batch_idx + self._val_processed)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        _update_n(self.al_progress_bar, self._val_processed)
        _update_n(self.val_progress_bar, self._val_processed)

    def on_validation_end(self, trainer, pl_module):
        if self.main_progress_bar is not None and trainer.state.fn == "fit":
            self.main_progress_bar.set_postfix(self.get_metrics(trainer, pl_module))
        self.val_progress_bar.close()

    def on_test_start(self, trainer, pl_module):
        if isinstance(pl_module, EnergizerStrategy):
            self.test_progress_bar = self.init_pool_tqdm()
        else:
            self.test_progress_bar = self.init_test_tqdm()
        self.test_progress_bar.total = convert_inf(self.total_test_batches)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self._should_update(self.test_batch_idx):
            _update_n(self.al_progress_bar, self.test_batch_idx)
            _update_n(self.test_progress_bar, self.test_batch_idx)

    def on_test_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        _update_n(self.al_progress_bar, self.test_batch_idx)
        _update_n(self.test_progress_bar, self.test_batch_idx)

    def on_test_end(self, trainer, pl_module):
        self.test_progress_bar.close()
        if isinstance(pl_module, EnergizerStrategy):
            self.al_progress_bar.close()

    def on_predict_epoch_start(self, trainer, pl_module):
        self.predict_progress_bar = self.init_predict_tqdm()
        self.predict_progress_bar.total = convert_inf(self.total_predict_batches)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if self._should_update(self.predict_batch_idx):
            _update_n(self.predict_progress_bar, self.predict_batch_idx)

    def on_predict_end(self, trainer, pl_module):
        self.predict_progress_bar.close()

    def print(
        self, *args, sep: str = " ", end: str = os.linesep, file: Optional[io.TextIOBase] = None, nolock: bool = False
    ):
        active_progress_bar = None

        if self.main_progress_bar is not None and not self.main_progress_bar.disable:
            active_progress_bar = self.main_progress_bar
        elif self.val_progress_bar is not None and not self.val_progress_bar.disable:
            active_progress_bar = self.val_progress_bar
        elif self.test_progress_bar is not None and not self.test_progress_bar.disable:
            active_progress_bar = self.test_progress_bar
        elif self.predict_progress_bar is not None and not self.predict_progress_bar.disable:
            active_progress_bar = self.predict_progress_bar

        if active_progress_bar is not None:
            s = sep.join(map(str, args))
            active_progress_bar.write(s, end=end, file=file, nolock=nolock)

    def _should_update(self, idx: int) -> bool:
        return self.refresh_rate and idx % self.refresh_rate == 0

    @staticmethod
    def _resolve_refresh_rate(refresh_rate: int) -> int:
        if os.getenv("COLAB_GPU") and refresh_rate == 1:
            # smaller refresh rate on colab causes crashes, choose a higher value
            rank_zero_debug("Using a higher refresh rate on Colab. Setting it to `20`")
            refresh_rate = 20
        return refresh_rate
