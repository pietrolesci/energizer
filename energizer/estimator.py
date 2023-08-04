import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import numpy as np
import torch
from lightning.fabric import Fabric
from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.loggers.logger import Logger
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule, _FabricOptimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from energizer.enums import OutputKeys, RunningStage
from energizer.registries import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY
from energizer.trackers import ProgressTracker
from energizer.types import BATCH_OUTPUT, EPOCH_OUTPUT, FIT_OUTPUT, METRIC
from energizer.utilities import move_to_cpu, set_deterministic
from energizer.utilities.model_summary import summarize


@dataclass
class Args:
    def to_dict(self) -> Dict[str, Any]:
        out = copy.deepcopy(self.__dict__)
        return out


@dataclass
class OptimizerArgs(Args):
    set_to_none: bool = False
    no_decay: Optional[List[str]] = None
    weight_decay: Optional[float] = None
    clip_val: Optional[Union[float, int]] = None
    max_norm: Optional[Union[float, int]] = None
    norm_type: Union[float, int] = 2.0


@dataclass
class SchedulerArgs(Args):
    num_warmup_steps: Optional[int] = None


class Estimator:
    _model: Union[torch.nn.Module, Callable]
    _tracker: ProgressTracker
    _is_compiled: bool = False

    def __init__(
        self,
        model: Any,
        accelerator: Union[str, Accelerator] = "cpu",
        precision: _PRECISION_INPUT = 32,
        callbacks: Optional[Union[List[Any], Any]] = None,
        loggers: Optional[Union[Logger, List[Logger]]] = None,
        deterministic: bool = True,
        tf32_mode: str = "highest",
        **kwargs,
    ) -> None:
        super().__init__()
        self.fabric = Fabric(
            accelerator=accelerator,
            precision=precision,
            callbacks=callbacks,
            loggers=loggers,
            devices=1,  # only works with single-GPU
            num_nodes=1,  # only works with single-node
        )
        # sets deterministic convolutions too
        set_deterministic(deterministic)

        # equivalent to `torch.backends.cudnn.allow_tf32 = True`
        # convolutions are not changed, to do that you need
        # `torch.backends.cudnn.allow_tf32 = True`
        torch.set_float32_matmul_precision(tf32_mode)

        self.init_model(model, **kwargs)
        self.init_tracker()

    @property
    def model(self) -> Union[torch.nn.Module, Callable]:
        return self._model

    @property
    def tracker(self) -> ProgressTracker:
        return self._tracker

    @property
    def device(self) -> torch.device:
        return self.fabric.device

    @property
    def model_summary(self) -> str:
        return summarize(self)

    @property
    def is_compiled(self) -> bool:
        return self._is_compiled

    def init_tracker(self) -> None:
        self._tracker = ProgressTracker()

    def init_model(self, model: Any, **kwargs) -> None:
        self._model = model

    def compile(self, **kwargs) -> None:
        # model becomes a Callable
        self._model = torch.compile(self._model, **kwargs)
        self._is_compiled = True

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: Optional[DataLoader] = None,
        max_epochs: Optional[int] = 3,
        min_epochs: Optional[int] = None,
        max_steps: Optional[int] = None,
        min_steps: Optional[int] = None,
        validation_freq: Optional[str] = "1:epoch",
        gradient_accumulation_steps: Optional[int] = None,
        learning_rate: float = 0.001,
        optimizer: str = "adamw",
        optimizer_kwargs: Optional[Dict] = None,
        scheduler: Optional[str] = None,
        scheduler_kwargs: Optional[Dict] = None,
        log_interval: int = 1,
        enable_progress_bar: bool = True,
        limit_train_batches: Optional[int] = None,
        limit_validation_batches: Optional[int] = None,
    ) -> List[FIT_OUTPUT]:
        """Entry point for model training.

        Calls `fit -> run_fit -> run_epoch -> run_training_step`
        """

        # self.fabric.launch()  # NOTE: do not support distributed yet

        # start progress tracking
        self.tracker.setup(
            stage=RunningStage.TRAIN,
            log_interval=log_interval,
            enable_progress_bar=enable_progress_bar,
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_steps=max_steps,
            min_steps=min_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_batches=len(train_loader),
            num_validation_batches=len(validation_loader or []),
            validation_freq=validation_freq,
            limit_train_batches=limit_train_batches,
            limit_validation_batches=limit_validation_batches,
        )

        # configuration
        _train_loader = self.configure_dataloader(train_loader)
        _validation_loader = self.configure_dataloader(validation_loader)
        _optimizer = self.configure_optimizer(optimizer, learning_rate, optimizer_kwargs)
        _scheduler = self.configure_scheduler(scheduler, _optimizer, scheduler_kwargs)
        model, _optimizer = self.fabric.setup(self.model, _optimizer)  # type: ignore

        # run epochs
        return self.run_fit(model, _train_loader, _validation_loader, _optimizer, _scheduler)  # type: ignore

    def run_fit(
        self,
        model: _FabricModule,
        train_loader: _FabricDataLoader,
        validation_loader: Optional[_FabricDataLoader],
        optimizer: _FabricOptimizer,
        scheduler: Optional[_LRScheduler],
    ) -> List[FIT_OUTPUT]:
        self.tracker.start_fit()

        # call hook
        self.fabric.call("on_fit_start", estimator=self, model=model)

        output = []
        while not self.tracker.is_fit_done:
            out = self.run_epoch(
                model=model,
                train_loader=train_loader,
                validation_loader=validation_loader,
                optimizer=optimizer,
                scheduler=scheduler,
            )

            output.append(out)

            # update progress
            self.tracker.increment_epoch()

        # call hook
        self.fabric.call("on_fit_end", estimator=self, model=model, output=output)

        self.tracker.end_fit()

        return output

    def run_epoch(
        self,
        model: _FabricModule,
        train_loader: _FabricDataLoader,
        validation_loader: Optional[_FabricDataLoader],
        optimizer: _FabricOptimizer,
        scheduler: Optional[_LRScheduler],
    ) -> FIT_OUTPUT:
        """Runs a training epoch."""

        # configure progress tracking
        self.tracker.start(RunningStage.TRAIN)

        # define metrics
        metrics = self.configure_metrics(RunningStage.TRAIN)
        loss_fn = self.configure_loss_fn(RunningStage.TRAIN)

        # train mode
        model.train()

        # call hook
        self.fabric.call("on_train_epoch_start", estimator=self, model=model, optimizer=optimizer)

        train_out, validation_out = [], []
        iterable = enumerate(train_loader)
        while not self.tracker.is_done:
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

            # print("=======")

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
            # print("IN ->", self.tracker.should_validate, "Step:", self.tracker.global_step, "Batch:", self.tracker.global_batch)
            if self.tracker.should_validate:
                out = self.run_evaluation(model, validation_loader, RunningStage.VALIDATION)  # type: ignore
                if out is not None:
                    validation_out.append(out)

            # update progress tracker
            self.tracker.increment()

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
        # print("OUT ->", self.tracker.should_validate)
        if self.tracker.should_validate:
            out = self.run_evaluation(model, validation_loader, RunningStage.VALIDATION)  # type: ignore
            if out is not None:
                validation_out.append(out)

        self.tracker.end()

        return train_out, validation_out

    def run_training_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        optimizer: _FabricOptimizer,
        scheduler: Optional[_LRScheduler],
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: Optional[METRIC],
    ) -> BATCH_OUTPUT:
        """Runs over a single batch of data."""
        # print(self.tracker.is_accumulating, self.tracker.global_batch)

        with self.fabric.no_backward_sync(model, enabled=self.tracker.is_accumulating):
            # compute loss
            output = self.train_step(model, batch, batch_idx, loss_fn, metrics)
            loss = output if isinstance(output, torch.Tensor) else output[OutputKeys.LOSS]

            # compute gradients
            self.fabric.backward(loss / self.tracker.gradient_accumulation_steps)  # instead of loss.backward()

        # print("Accumulating?", self.tracker.is_accumulating)
        if not self.tracker.is_accumulating:
            # clip gradients (in configure_optimizer we set the attribute `_gradient_clipping_kwargs`)
            if optimizer._gradient_clipping_kwargs.get("clip_val") or optimizer._gradient_clipping_kwargs.get("max_norm"):  # type: ignore
                self.fabric.clip_gradients(model, optimizer, **optimizer._gradient_clipping_kwargs)  # type: ignore

            # update parameters
            self.fabric.call("on_before_optimizer", estimator=self, model=model, optimizer=optimizer)
            optimizer.step()
            self.fabric.call("on_after_optimizer", estimator=self, model=model, optimizer=optimizer)

            # zero_grad (in configure_optimizer we set the attribute `_set_to_none_flag`)
            optimizer.zero_grad(set_to_none=optimizer._set_to_none_flag)  # type: ignore

            # update scheduler
            if scheduler is not None:
                scheduler.step()

            # update tracker
            self.tracker.increment_step()

            # print("UPDATED")

        return output

    def test(
        self,
        test_loader: DataLoader,
        log_interval: int = 1,
        enable_progress_bar: bool = True,
        limit_batches: Optional[int] = None,
    ) -> EPOCH_OUTPUT:
        """This method is useful because validation can run in fit when model is already setup."""
        # self.fabric.launch()  # NOTE: do not support distributed yet

        self.tracker.setup(
            stage=RunningStage.TEST,
            log_interval=log_interval,
            enable_progress_bar=enable_progress_bar,
            num_batches=len(test_loader),
            limit_batches=limit_batches,
        )

        # configuration
        loader = self.configure_dataloader(test_loader)
        model = self.fabric.setup(self.model)  # type: ignore
        return self.run_evaluation(model, loader, RunningStage.TEST)  # type: ignore

    def run_evaluation(
        self, model: _FabricModule, loader: _FabricDataLoader, stage: Union[str, RunningStage]
    ) -> EPOCH_OUTPUT:
        """Runs over an entire evaluation dataloader."""

        # configure progress tracking
        self.tracker.start(stage)

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
        ctx = torch.no_grad() if self.is_compiled else torch.inference_mode()
        with ctx:  # to fix compilation which does not work with torch.compile
            while not self.tracker.is_done:
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
                self.tracker.increment()

        # method to possibly aggregate
        output = getattr(self, f"{stage}_epoch_end")(output, metrics)

        # call hook
        self.fabric.call(f"on_{stage}_epoch_end", estimator=self, model=model, output=output, metrics=metrics)

        # resets model training status
        model.train(is_fitting)

        self.tracker.end()

        return output

    def evaluation_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: Optional[METRIC],
        stage: Union[str, RunningStage],
    ) -> BATCH_OUTPUT:
        """Runs over a single batch of data."""
        # this might seems redundant but it's useful for active learning to hook in
        return getattr(self, f"{stage}_step")(model, batch, batch_idx, loss_fn, metrics)

    """
    Methods
    """

    def transfer_to_device(self, batch: Any) -> Any:
        # NOTE: fabric knows how to handle non-gpu stuff so the batch can have anything inside
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
        _optimizer = optimizer_fn(params, lr=learning_rate, **optimizer_kwargs)

        # attach an attribute that will tell the optimizer to set_to_none
        setattr(_optimizer, "_set_to_none_flag", optimizer_kwargs.get("set_to_none", False))

        # attach gradient clipping kwargs
        gradient_clipping_kwargs = dict(
            clip_val=optimizer_kwargs.get("clip_val"),
            max_norm=optimizer_kwargs.get("max_norm"),
            norm_type=optimizer_kwargs.get("norm_type", 2.0),
        )
        setattr(_optimizer, "_gradient_clipping_kwargs", gradient_clipping_kwargs)

        return _optimizer

    def configure_scheduler(
        self,
        scheduler: Optional[str],
        optimizer: Optimizer,
        scheduler_kwargs: Optional[Dict] = None,
    ) -> Optional[_LRScheduler]:
        ...
        if scheduler is None:
            return

        scheduler_fn = SCHEDULER_REGISTRY[scheduler]

        # collect scheduler kwargs
        scheduler_kwargs = scheduler_kwargs or {}

        num_train_steps = self.tracker.step_tracker.max

        num_warmup_steps = scheduler_kwargs.get("num_warmup_steps")
        if num_warmup_steps is not None and 0.0 < num_warmup_steps < 1.0:
            num_warmup_steps = int(np.ceil(num_warmup_steps * num_train_steps))

        scheduler_kwargs["num_training_steps"] = num_train_steps
        scheduler_kwargs["num_warmup_steps"] = num_warmup_steps

        return scheduler_fn(optimizer, **scheduler_kwargs)

    def configure_dataloader(self, loader: Optional[DataLoader]) -> Optional[_FabricDataLoader]:
        if loader is None:
            return
        return self.fabric.setup_dataloaders(loader, use_distributed_sampler=False, move_to_device=False)  # type: ignore

    def log(self, name: str, value: Any, step: int) -> None:
        """Automatically moves to cpu and then logs value."""
        if self.tracker.should_log:
            self.fabric.log(name, move_to_cpu(value), step)

    def log_dict(self, value_dict: Mapping[str, Any], step: int) -> None:
        """Automatically moves to cpu and then logs mapping of values."""
        if self.tracker.should_log:
            self.fabric.log_dict(value_dict, step)

    def save_state_dict(self, cache_dir: Union[str, Path], name: str = "state_dict.pt") -> None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.fabric.save(state=self.model.state_dict(), path=cache_dir / name)

    def load_state_dict(self, cache_dir: Union[str, Path], name: str = "state_dict.pt") -> None:
        cache_dir = Path(cache_dir)
        self.model.load_state_dict(self.fabric.load(cache_dir / name))

    """
    Hooks
    """

    def configure_loss_fn(self, stage: Union[str, RunningStage]) -> torch.nn.Module:
        ...

    def configure_metrics(self, stage: Optional[Union[str, RunningStage]] = None) -> Optional[METRIC]:
        ...

    def train_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: Optional[METRIC] = None,
    ) -> BATCH_OUTPUT:
        return self.step(RunningStage.TRAIN, model, batch, batch_idx, loss_fn, metrics)

    def validation_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: Optional[METRIC] = None,
    ) -> Optional[BATCH_OUTPUT]:
        return self.step(RunningStage.VALIDATION, model, batch, batch_idx, loss_fn, metrics)

    def test_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: Optional[METRIC] = None,
    ) -> Optional[BATCH_OUTPUT]:
        return self.step(RunningStage.TEST, model, batch, batch_idx, loss_fn, metrics)

    def step(
        self,
        stage: Union[str, RunningStage],
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: Optional[METRIC] = None,
    ) -> BATCH_OUTPUT:
        raise NotImplementedError

    def train_epoch_end(self, output: List[BATCH_OUTPUT], metrics: Optional[METRIC]) -> EPOCH_OUTPUT:
        return self.epoch_end(RunningStage.TRAIN, output, metrics)

    def validation_epoch_end(self, output: List[BATCH_OUTPUT], metrics: Optional[METRIC]) -> EPOCH_OUTPUT:
        return self.epoch_end(RunningStage.VALIDATION, output, metrics)

    def test_epoch_end(self, output: List[BATCH_OUTPUT], metrics: Optional[METRIC]) -> EPOCH_OUTPUT:
        return self.epoch_end(RunningStage.TEST, output, metrics)

    def epoch_end(
        self, stage: Union[str, RunningStage], output: List[BATCH_OUTPUT], metrics: Optional[METRIC]
    ) -> EPOCH_OUTPUT:
        return output
