from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import torch
from lightning.fabric import Fabric
from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.connector import _PRECISION_INPUT
from lightning.fabric.loggers import Logger
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule, _FabricOptimizer
from lightning.pytorch.core.mixins.hparams_mixin import HyperparametersMixin
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from src.energizer.enums import OutputKeys, RunningStage
from src.energizer.progress_trackers import ProgressTracker
from src.energizer.registries import OPTIMIZER_REGISTRY
from src.energizer.types import BATCH_OUTPUT, EPOCH_OUTPUT, METRIC
from src.energizer.utilities import init_deterministic, move_to_cpu
from src.energizer.utilities.model_summary import summarize


@dataclass
class FitEpochOutput:
    """Simple container for train and validation outputs for each epoch of fitting.

    This deserves a separate container bacause during fit we obtain EpochOutput's
    from both train and, possibly, validation.
    """

    train: EPOCH_OUTPUT = None
    validation: EPOCH_OUTPUT = None


class Estimator(HyperparametersMixin):
    _hparams_ignore: List[str] = ["model", "loggers", "callbacks"]
    _progress_tracker: ProgressTracker = None

    def __init__(
        self,
        model: torch.nn.Module,
        accelerator: Optional[Union[str, Accelerator]] = None,
        precision: _PRECISION_INPUT = 32,
        callbacks: Optional[Union[List[Any], Any]] = None,
        loggers: Optional[Union[Logger, List[Logger]]] = None,
        deterministic: bool = True,
    ) -> None:
        super().__init__()
        self.fabric = Fabric(
            accelerator=accelerator,
            precision=precision,
            callbacks=callbacks,
            loggers=loggers,
        )
        self.model = model
        init_deterministic(deterministic)
        self.save_hyperparameters(ignore=self._hparams_ignore)

    @property
    def device(self) -> torch.device:
        return self.fabric.device

    @property
    def progress_tracker(self) -> ProgressTracker:
        if self._progress_tracker is None:
            self._progress_tracker = ProgressTracker()
        return self._progress_tracker

    @property
    def model_summary(self) -> str:
        return summarize(self)

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: Optional[DataLoader] = None,
        max_epochs: Optional[int] = 3,
        min_steps: Optional[int] = None,
        learning_rate: float = 0.001,
        optimizer: str = "adamw",
        optimizer_kwargs: Optional[Dict] = None,
        scheduler: Optional[str] = None,
        scheduler_kwargs: Optional[Dict] = None,
        **kwargs,
    ) -> List[FitEpochOutput]:

        # start progress tracking
        self.progress_tracker.setup(
            RunningStage.TRAIN,
            max_epochs=max_epochs,
            min_steps=min_steps,
            num_train_batches=len(train_loader),
            num_validation_batches=len(validation_loader or []),
            **kwargs,
        )

        # configuration
        train_loader = self.configure_dataloader(train_loader)
        validation_loader = self.configure_dataloader(validation_loader)
        optimizer = self.configure_optimizer(optimizer, learning_rate, optimizer_kwargs)
        scheduler = self.configure_scheduler(scheduler, optimizer, scheduler_kwargs)
        model, optimizer = self.fabric.setup(self.model, optimizer)

        # run epochs
        output = self.run_fit(model, train_loader, validation_loader, optimizer, scheduler)

        return output

    def run_fit(
        self,
        model: _FabricModule,
        train_loader: _FabricDataLoader,
        validation_loader: Optional[_FabricDataLoader],
        optimizer: _FabricOptimizer,
        scheduler: Optional[str],
    ) -> FitEpochOutput:

        self.progress_tracker.start_fit()

        # call hook
        self.fabric.call("on_fit_start", estimator=self, model=model)

        output = []
        while not self.progress_tracker.is_fit_done():

            out = self.run_epoch(
                model=model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                optimizer=optimizer,
                scheduler=scheduler,
            )

            output.append(out)

            # update progress
            self.progress_tracker.increment_epoch()

        # call hook
        self.fabric.call("on_fit_end", estimator=self, model=model, output=output)

        self.progress_tracker.end_fit()

        return output

    def run_epoch(
        self,
        model: _FabricModule,
        train_loader: _FabricDataLoader,
        validation_loader: Optional[_FabricDataLoader],
        optimizer: _FabricOptimizer,
        scheduler: Optional[str],
    ) -> FitEpochOutput:
        """Runs a training epoch."""

        # configure progress tracking
        self.progress_tracker.start(RunningStage.TRAIN)

        # define metrics
        metrics = self.configure_metrics(RunningStage.TRAIN)
        loss_fn = self.configure_loss_fn(RunningStage.TRAIN)

        # train mode
        model.train()

        # call hook
        self.fabric.call("on_train_epoch_start", estimator=self, model=model, optimizer=optimizer)

        train_out, validation_out = [], []
        iterable = enumerate(train_loader)
        while not self.progress_tracker.is_done():

            batch_idx, batch = next(iterable)

            # put batch on correct device
            batch = self.transfer_to_device(batch)

            # call hook
            self.fabric.call(
                "on_train_batch_start",
                estimator=self,
                model=model,
                optimizer=optimizer,
                batch=batch,
                batch_idx=batch_idx,
            )

            # run model on batch
            batch_out = self.run_training_step(model, batch, batch_idx, optimizer, scheduler, loss_fn, metrics)

            # call hook
            self.fabric.call(
                "on_train_batch_end",
                estimator=self,
                model=model,
                output=batch_out,
                batch=batch,
                batch_idx=batch_idx,
            )

            # record output
            train_out.append(move_to_cpu(batch_out))

            # validation loop
            if self.progress_tracker.should_validate():
                out = self.run_evaluation(model, validation_loader, RunningStage.VALIDATION)
                if out is not None:
                    validation_out.append(out)

            # update progress tracker
            self.progress_tracker.increment()

        # method to possibly aggregate
        train_out = self.train_epoch_end(train_out, metrics)

        # call hook
        self.fabric.call(
            "on_train_epoch_end",
            estimator=self,
            model=model,
            output=train_out,
            metrics=metrics,
        )

        # validation loop
        if self.progress_tracker.should_validate():
            out = self.run_evaluation(model, validation_loader, RunningStage.VALIDATION)
            if out is not None:
                validation_out.append(out)

        self.progress_tracker.end()

        return FitEpochOutput(train=train_out, validation=validation_out)

    def run_training_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        optimizer: _FabricOptimizer,
        scheduler: _LRScheduler,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: Optional[METRIC],
    ) -> BATCH_OUTPUT:
        """Runs over a single batch of data."""

        # zero_grad
        optimizer.zero_grad()

        # compute loss
        output = self.train_step(model, batch, batch_idx, loss_fn, metrics)
        loss = output if isinstance(output, torch.Tensor) else output[OutputKeys.LOSS]

        # compute gradients
        self.fabric.backward(loss)  # instead of loss.backward()

        # update parameters
        self.fabric.call("on_before_optimizer_step", self, model, optimizer)
        optimizer.step()
        self.fabric.call("on_after_optimizer_step", self, model, optimizer)

        # update scheduler
        if scheduler is not None:
            scheduler.step()

        # update progress_tracker
        self.progress_tracker.increment_step()

        return output

    def test(self, test_loader: DataLoader, **kwargs) -> EPOCH_OUTPUT:
        """This method is useful because validation can run in fit when model is already setup."""
        self.progress_tracker.setup(RunningStage.TEST, num_batches=len(test_loader), **kwargs)

        # configuration
        loader = self.configure_dataloader(test_loader)
        model = self.fabric.setup(self.model)
        return self.run_evaluation(model, loader, RunningStage.TEST)

    def run_evaluation(self, model: _FabricModule, loader: _FabricDataLoader, stage: RunningStage) -> EPOCH_OUTPUT:
        """Runs over an entire evaluation dataloader."""

        # configure progress tracking
        self.progress_tracker.start(stage)

        # configure metrics
        metrics = self.configure_metrics(stage)
        loss_fn = self.configure_loss_fn(stage)

        # eval mode
        is_fitting = model.training
        model.eval()

        # call hook
        self.fabric.call(f"on_{stage}_epoch_start", estimator=self, model=model)

        output = []
        iterable = enumerate(loader)
        with torch.inference_mode():

            while not self.progress_tracker.is_done():

                batch_idx, batch = next(iterable)

                # put batch on correct device
                batch = self.transfer_to_device(batch)

                # call hook
                self.fabric.call(
                    f"on_{stage}_batch_start", estimator=self, model=model, batch=batch, batch_idx=batch_idx
                )

                # run model on batch
                batch_out = self.evaluation_step(model, batch, batch_idx, loss_fn, metrics, stage)

                # call hook
                self.fabric.call(
                    f"on_{stage}_batch_end",
                    estimator=self,
                    model=model,
                    output=batch_out,
                    batch=batch,
                    batch_idx=batch_idx,
                )

                # record output
                if batch_out is not None:
                    output.append(move_to_cpu(batch_out))

                # update progress tracker
                self.progress_tracker.increment()

        # method to possibly aggregate
        output = getattr(self, f"{stage}_epoch_end")(output, metrics)

        # call hook
        self.fabric.call(f"on_{stage}_epoch_end", estimator=self, model=model, output=output, metrics=metrics)

        # resets model training status
        model.train(is_fitting)

        self.progress_tracker.end()

        return output

    def evaluation_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: Optional[METRIC],
        stage: RunningStage,
    ) -> Optional[BATCH_OUTPUT]:
        """Runs over a single batch of data."""
        # this might seems redundant but it's useful for active learning to hook in
        return getattr(self, f"{stage}_step")(model, batch, batch_idx, loss_fn, metrics)

    """
    Methods
    """

    def transfer_to_device(self, batch: Any) -> Any:
        return self.fabric.to_device(batch)

    def configure_optimizer(
        self, optimizer: str, learning_rate: float, optimizer_kwargs: Optional[Dict] = None
    ) -> Optimizer:
        assert optimizer is not None, ValueError("You must provide an optimizer.")

        optimizer_fn = OPTIMIZER_REGISTRY[optimizer]
        optimizer_kwargs = optimizer_kwargs or {}

        # weight decay
        no_decay, weight_decay = optimizer_kwargs.get("no_decay", None), optimizer_kwargs.get("weight_decay", None)
        if no_decay is not None and (weight_decay is not None and weight_decay > 0.0):
            params = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            params = filter(lambda p: p.requires_grad, self.model.parameters())

        # instantiate optimizer
        optimizer = optimizer_fn(params, lr=learning_rate, **optimizer_kwargs)

        return optimizer

    def configure_scheduler(
        self,
        scheduler: str,
        optimizer: Optimizer,
        scheduler_kwargs: Optional[Dict] = None,
    ) -> Optional[_LRScheduler]:
        ...
        # if scheduler is None:
        #     return

        # scheduler_fn = SCHEDULER_REGISTRY[scheduler]

        # # collect scheduler kwargs
        # params = list(inspect.signature(scheduler_fn).parameters.keys())
        # scheduler_kwargs = scheduler_kwargs or {}
        # num_train_steps = self.progress_tracker.train_tracker.max
        # num_warmup_steps = scheduler_kwargs.get("num_warmup_steps", None)
        # if num_warmup_steps is not None and isinstance(num_warmup_steps, float):
        #     num_warmup_steps *= num_train_steps
        # if "num_train_steps" in params:
        #     scheduler_kwargs["num_train_steps"] = num_train_steps
        # if "num_warmup_steps" in params:
        #     scheduler_kwargs["num_warmup_steps"] = num_warmup_steps

        # scheduler = scheduler_fn(optimizer, **scheduler_kwargs)

        # return scheduler

    def configure_loss_fn(self, stage: RunningStage) -> torch.nn.Module:
        ...

    def configure_dataloader(self, loader: Optional[DataLoader]) -> Optional[_FabricDataLoader]:
        """Does not move the dataloader to the device."""
        if loader is None:
            return

        return self.fabric.setup_dataloaders(loader, replace_sampler=False, move_to_device=False)

    """
    Hooks
    """

    def configure_metrics(self, stage: Optional[RunningStage] = None) -> Optional[METRIC]:
        pass

    def train_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: Optional[METRIC] = None,
    ) -> BATCH_OUTPUT:
        raise NotImplementedError

    def validation_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: Optional[METRIC] = None,
    ) -> Optional[BATCH_OUTPUT]:
        ...

    def test_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: Optional[METRIC] = None,
    ) -> Optional[BATCH_OUTPUT]:
        ...

    def train_epoch_end(self, output: List[BATCH_OUTPUT], metrics: Optional[METRIC]) -> EPOCH_OUTPUT:
        return output

    def validation_epoch_end(self, output: List, metrics: Optional[METRIC]) -> EPOCH_OUTPUT:
        return output

    def test_epoch_end(self, output: List, metrics: Optional[METRIC]) -> EPOCH_OUTPUT:
        return output

    def log(self, name: str, value: Any, step: int) -> None:
        """Automatically moves to cpu and then logs value."""
        if self.progress_tracker.should_log():
            self.fabric.log(name, move_to_cpu(value), step)

    def log_dict(self, value_dict: Mapping[str, Any], step: int) -> None:
        """Automatically moves to cpu and then logs mapping of values."""
        if self.progress_tracker.should_log():
            self.fabric.log_dict(value_dict, step)

    def save_state_dict(self, cache_dir: Union[str, Path], name: str = "state_dict.pt") -> None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.fabric.save(self.model.state_dict(), cache_dir / name)

    def load_state_dict(self, cache_dir: Union[str, Path], name: str = "state_dict.pt") -> None:
        cache_dir = Path(cache_dir)
        self.model.load_state_dict(self.fabric.load(cache_dir / name))