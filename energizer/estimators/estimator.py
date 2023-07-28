from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union

import torch
from lightning_fabric import Fabric
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning_fabric.loggers.logger import Logger
from lightning_fabric.wrappers import _FabricDataLoader, _FabricModule, _FabricOptimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from src.enums import OutputKeys, RunningStage
from src.progress_trackers import ProgressTracker
from src.registries import OPTIMIZER_REGISTRY
from src.types import BATCH_OUTPUT, EPOCH_OUTPUT, FIT_OUTPUT, METRIC
from src.utilities import init_deterministic, move_to_cpu
from src.utilities.model_summary import summarize


class Estimator:

    _model: torch.nn.Module
    _tracker: ProgressTracker

    def __init__(
        self,
        model: torch.nn.Module,
        accelerator: Union[str, Accelerator] = "cpu",
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
            devices=1,  # only works with single-GPU
            num_nodes=1,  # only works with single-node
        )
        init_deterministic(deterministic)
        self.init_tracker()
        self.init_model(model)

    @property
    def model(self) -> torch.nn.Module:
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

    def init_tracker(self) -> None:
        self._tracker = ProgressTracker()
    
    def init_model(self, model: torch.nn.Module) -> None:
        self._model = model

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
        log_interval: int = 1,
        enable_progress_bar: bool = True,
        num_validation_per_epoch: int = 1,
        limit_train_batches: Optional[int] = None,
        limit_validation_batches: Optional[int] = None,
    ) -> List[FIT_OUTPUT]:
        """Entry point for model training.
        
        Calls `fit -> run_fit -> run_epoch -> run_training_step`
        """

        # self.fabric.launch()  # NOTE: do not support distributed yet
        assert max_epochs is not None or min_steps is not None, "`max_epochs` or `min_steps` must be passed."

        # start progress tracking
        self.tracker.setup(
            stage=RunningStage.TRAIN,
            log_interval=log_interval,
            enable_progress_bar=enable_progress_bar,
            max_epochs=max_epochs,
            min_steps=min_steps,
            num_train_batches=len(train_loader),
            num_validation_batches=len(validation_loader or []),
            num_validation_per_epoch=num_validation_per_epoch,
            limit_train_batches=limit_train_batches,
            limit_validation_batches=limit_validation_batches,
        )

        # configuration
        _train_loader = self.configure_dataloader(train_loader)
        _validation_loader = self.configure_dataloader(validation_loader)
        _optimizer = self.configure_optimizer(optimizer, learning_rate, optimizer_kwargs)
        _scheduler = self.configure_scheduler(scheduler, _optimizer, scheduler_kwargs)
        model, _optimizer = self.fabric.setup(self.model, _optimizer)

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
        while not self.tracker.is_fit_done():

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
        while not self.tracker.is_done():

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
            if self.tracker.should_validate() and validation_loader is not None:
                out = self.run_evaluation(model, validation_loader, RunningStage.VALIDATION)
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
        if self.tracker.should_validate() and validation_loader is not None:
            out = self.run_evaluation(model, validation_loader, RunningStage.VALIDATION)
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

        # zero_grad
        optimizer.zero_grad()  # type: ignore

        # compute loss
        output = self.train_step(model, batch, batch_idx, loss_fn, metrics)
        loss = output if isinstance(output, torch.Tensor) else output[OutputKeys.LOSS]

        # compute gradients
        self.fabric.backward(loss)  # instead of loss.backward()

        # update parameters
        optimizer.step()

        # update scheduler
        if scheduler is not None:
            scheduler.step()

        # update tracker
        self.tracker.increment_step()

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
        model = self.fabric.setup(self.model)
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
        with torch.inference_mode():

            while not self.tracker.is_done():

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
        return optimizer_fn(params, lr=learning_rate, **optimizer_kwargs)

    def configure_scheduler(
        self,
        scheduler: Optional[str],
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
        # num_train_steps = self.tracker.train_tracker.max
        # num_warmup_steps = scheduler_kwargs.get("num_warmup_steps", None)
        # if num_warmup_steps is not None and isinstance(num_warmup_steps, float):
        #     num_warmup_steps *= num_train_steps
        # if "num_train_steps" in params:
        #     scheduler_kwargs["num_train_steps"] = num_train_steps
        # if "num_warmup_steps" in params:
        #     scheduler_kwargs["num_warmup_steps"] = num_warmup_steps

        # scheduler = scheduler_fn(optimizer, **scheduler_kwargs)

        # return scheduler

    def configure_dataloader(self, loader: Optional[DataLoader]) -> Optional[_FabricDataLoader]:
        if loader is None:
            return
        return self.fabric.setup_dataloaders(loader, use_distributed_sampler=False, move_to_device=False)  # type: ignore

    def log(self, name: str, value: Any, step: int) -> None:
        """Automatically moves to cpu and then logs value."""
        if self.tracker.should_log():
            self.fabric.log(name, move_to_cpu(value), step)

    def log_dict(self, value_dict: Mapping[str, Any], step: int) -> None:
        """Automatically moves to cpu and then logs mapping of values."""
        if self.tracker.should_log():
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
