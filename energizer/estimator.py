import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Mapping, Optional, Union, Dict, Tuple
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
class SchedulerArgs(Args):
    name: Optional[str] = None
    num_warmup_steps: Optional[int] = None
    num_training_steps: Optional[int] = None
    init_kwargs: Dict = field(default_factory=dict)


@dataclass
class OptimizationArgs(Args):
    name: Optional[str] = None
    lr: Optional[float] = None
    weight_decay: Optional[float] = None
    no_decay: Optional[List[str]] = None
    set_to_none: bool = False
    clip_val: Optional[Union[float, int]] = None
    max_norm: Optional[Union[float, int]] = None
    norm_type: Union[float, int] = 2.0
    init_kwargs: Dict = field(default_factory=dict)
    scheduler_kwargs: SchedulerArgs = field(default_factory=lambda: SchedulerArgs())


class Estimator:
    _model: Union[torch.nn.Module, Callable]
    _tracker: ProgressTracker
    _optimization_args: OptimizationArgs
    _is_compiled: bool = False

    def __init__(
        self,
        model: Any,
        accelerator: Union[str, Accelerator] = "cpu",
        precision: _PRECISION_INPUT = 32,
        callbacks: Optional[Union[List[Any], Any]] = None,
        loggers: Optional[Union[Logger, List[Logger]]] = None,
        deterministic: Union[bool, Literal["warn_only"]] = "warn_only",
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

        self.set_deterministic(deterministic)
        self.set_torch_matmul_precision(tf32_mode)

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
    def precision(self) -> str:
        return self.fabric._precision.precision

    # @property
    # def prediction_dtype(self) -> torch.dtype:
    #     dtype = self.precision
    #     if "-" in dtype:
    #         dtype = dtype.split("-")[0]
    #         dtype = f"bfloat{dtype[2:]}" if dtype.startswith("b") else f"float{dtype}"
    #     return getattr(torch, dtype)

    @property
    def model_summary(self) -> str:
        return summarize(self)

    @property
    def is_compiled(self) -> bool:
        return self._is_compiled

    @property
    def optimization_args(self) -> OptimizationArgs:
        return self._optimization_args

    def init_tracker(self) -> None:
        self._tracker = ProgressTracker()

    def init_model(self, model: Any, **kwargs) -> None:
        self._model = model

    def compile(self, **kwargs) -> None:
        # model becomes a Callable
        self._model = torch.compile(self._model, **kwargs)
        self._is_compiled = True

    def set_torch_matmul_precision(self, tf32_mode: str = "highest") -> None:
        # equivalent to `torch.backends.cudnn.allow_tf32 = True`
        # convolutions are not changed, to do that you need
        # `torch.backends.cudnn.allow_tf32 = True`
        torch.set_float32_matmul_precision(tf32_mode)

    def set_deterministic(self, deterministic: Union[bool, Literal["warn_only"]]) -> None:
        # sets deterministic convolutions too
        set_deterministic(deterministic)

    def set_eval_mode(self) -> None:
        self.model.eval()

    def set_train_mode(self, mode: bool = True) -> None:
        self.model.train(mode)

    """
    Entry points
    """

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
        learning_rate: Optional[float] = None,
        optimizer: Optional[str] = None,
        optimizer_kwargs: Optional[Union[Dict, OptimizationArgs]] = None,
        scheduler: Optional[str] = None,
        scheduler_kwargs: Optional[Union[Dict, SchedulerArgs]] = None,
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
            num_train_batches=len(train_loader or []),
            num_validation_batches=len(validation_loader or []),
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_steps=max_steps,
            min_steps=min_steps,
            gradient_accumulation_steps=gradient_accumulation_steps,
            validation_freq=validation_freq,
            log_interval=log_interval,
            enable_progress_bar=enable_progress_bar,
            limit_train_batches=limit_train_batches,
            limit_validation_batches=limit_validation_batches,
        )

        model, _optimizer, _scheduler, _train_loader, _validation_loader = self._setup_fit(
            train_loader,
            validation_loader,
            learning_rate,
            optimizer,
            optimizer_kwargs,
            scheduler,
            scheduler_kwargs,
        )
        return self.run_fit(model, _optimizer, _scheduler, _train_loader, _validation_loader)  # type: ignore

    def test(
        self,
        loader: DataLoader,
        log_interval: int = 1,
        enable_progress_bar: bool = True,
        limit_batches: Optional[int] = None,
    ) -> EPOCH_OUTPUT:
        """This method is useful because validation can run in fit when model is already setup."""
        # self.fabric.launch()  # NOTE: do not support distributed yet

        # start progress tracking
        self.tracker.setup(
            stage=RunningStage.TEST,
            num_batches=len(loader or []),
            log_interval=log_interval,
            enable_progress_bar=enable_progress_bar,
            limit_batches=limit_batches,
        )

        # configuration
        _eval_loader = self.configure_dataloader(loader)
        model = self.fabric.setup(self.model)  # type: ignore

        return self.run_evaluation(model, _eval_loader, RunningStage.TEST)  # type: ignore

    """
    Training / Evaluation methods
    """

    def run_fit(
        self,
        model: _FabricModule,
        optimizer: _FabricOptimizer,
        scheduler: Optional[_LRScheduler],
        train_loader: _FabricDataLoader,
        validation_loader: Optional[_FabricDataLoader],
    ) -> List[FIT_OUTPUT]:

        self.tracker.start_fit()

        self.callback("on_fit_start", model=model)

        output = []
        while not self.tracker.is_fit_done:
            out = self.run_epoch(model, optimizer, scheduler, train_loader, validation_loader)
            output.append(out)

            self.tracker.increment_epoch()

        self.callback("on_fit_end", model=model, output=output)

        self.tracker.end_fit()

        return output

    def run_epoch(
        self,
        model: _FabricModule,
        optimizer: _FabricOptimizer,
        scheduler: Optional[_LRScheduler],
        train_loader: _FabricDataLoader,
        validation_loader: Optional[_FabricDataLoader],
    ) -> FIT_OUTPUT:
        """Runs a training epoch."""

        # configure progress tracking
        self.tracker.start(RunningStage.TRAIN)

        # define metrics
        metrics = self.configure_metrics(RunningStage.TRAIN)
        loss_fn = self.configure_loss_fn(RunningStage.TRAIN)

        # train mode
        model.train()

        self.callback("on_train_epoch_start", model=model, optimizer=optimizer)

        train_out, validation_out = [], []
        iterable = enumerate(train_loader)
        while not self.tracker.is_done:
            batch_idx, batch = next(iterable)

            # put batch on correct device
            batch = self.transfer_to_device(batch)

            self.callback("on_train_batch_start", model=model, optimizer=optimizer, batch=batch, batch_idx=batch_idx)

            # print("=======")

            # run model on batch
            batch_out = self.run_training_step(model, optimizer, scheduler, batch, batch_idx, loss_fn, metrics)

            self.callback("on_train_batch_end", model=model, output=batch_out, batch=batch, batch_idx=batch_idx)

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

        self.callback("on_train_epoch_end", model=model, output=train_out, metrics=metrics)

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
        optimizer: _FabricOptimizer,
        scheduler: Optional[_LRScheduler],
        batch: Any,
        batch_idx: int,
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

            # clip gradients
            if self.optimization_args.clip_val or self.optimization_args.max_norm:
                self.fabric.clip_gradients(
                    model,
                    optimizer,
                    clip_val=self.optimization_args.clip_val,
                    max_norm=self.optimization_args.max_norm,
                    norm_type=self.optimization_args.norm_type,
                )

            # update parameters
            self.callback("on_before_optimizer", model=model, optimizer=optimizer)
            optimizer.step()
            self.callback("on_after_optimizer", model=model, optimizer=optimizer)

            optimizer.zero_grad(set_to_none=self.optimization_args.set_to_none)  # type: ignore

            # update scheduler
            if scheduler is not None:
                scheduler.step()

            # update tracker
            self.tracker.increment_step()

            # print("UPDATED")

        return output

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

        self.callback(f"on_{stage}_epoch_start", model=model)

        output = []
        iterable = enumerate(loader)
        ctx = torch.no_grad() if self.is_compiled else torch.inference_mode()
        with ctx:  # to fix compilation which does not work with torch.compile
            while not self.tracker.is_done:
                batch_idx, batch = next(iterable)

                # put batch on correct device
                batch = self.transfer_to_device(batch)

                self.callback(f"on_{stage}_batch_start", model=model, batch=batch, batch_idx=batch_idx)

                # run model on batch
                batch_out = self.evaluation_step(model, batch, batch_idx, loss_fn, metrics, stage)

                self.callback(f"on_{stage}_batch_end", model=model, output=batch_out, batch=batch, batch_idx=batch_idx)

                # record output
                if batch_out is not None:
                    output.append(move_to_cpu(batch_out))

                # update progress tracker
                self.tracker.increment()

        # method to possibly aggregate
        output = getattr(self, f"{stage}_epoch_end")(output, metrics)

        self.callback(f"on_{stage}_epoch_end", model=model, output=output, metrics=metrics)

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

    def epoch_end(
        self, stage: Union[str, RunningStage], output: List[BATCH_OUTPUT], metrics: Optional[METRIC]
    ) -> EPOCH_OUTPUT:
        return output

    def train_epoch_end(self, output: List[BATCH_OUTPUT], metrics: Optional[METRIC]) -> EPOCH_OUTPUT:
        return self.epoch_end(RunningStage.TRAIN, output, metrics)

    def validation_epoch_end(self, output: List[BATCH_OUTPUT], metrics: Optional[METRIC]) -> EPOCH_OUTPUT:
        return self.epoch_end(RunningStage.VALIDATION, output, metrics)

    def test_epoch_end(self, output: List[BATCH_OUTPUT], metrics: Optional[METRIC]) -> EPOCH_OUTPUT:
        return self.epoch_end(RunningStage.TEST, output, metrics)

    """
    Configuration
    """

    def configure_optimization_args(
        self,
        learning_rate: Optional[float] = None,
        optimizer: Optional[str] = None,
        optimizer_kwargs: Optional[Union[Dict, OptimizationArgs]] = None,
        scheduler: Optional[str] = None,
        scheduler_kwargs: Optional[Union[Dict, SchedulerArgs]] = None,
    ) -> None:

        # parse optimizer args
        opt_kwargs = optimizer_kwargs or {}  # if None
        opt_kwargs = opt_kwargs.to_dict() if isinstance(opt_kwargs, OptimizationArgs) else opt_kwargs

        assert (
            optimizer is not None or opt_kwargs.get("name") is not None
        ), "optimizer or optimizer_kwargs['name'] must be set."
        opt_kwargs["name"] = optimizer or opt_kwargs.get("name")

        assert (
            learning_rate is not None or opt_kwargs.get("lr") is not None
        ), "learning_rate or optimizer_kwargs['lr'] must be set."
        opt_kwargs["lr"] = learning_rate or opt_kwargs.get("lr")

        # parse scheduler args
        sch_kwargs = scheduler_kwargs or {}  # if None
        sch_kwargs = sch_kwargs.to_dict() if isinstance(sch_kwargs, SchedulerArgs) else sch_kwargs

        sch_kwargs["name"] = scheduler or sch_kwargs.get("name")

        num_train_steps = self.tracker.step_tracker.max
        num_warmup_steps = sch_kwargs.get("num_warmup_steps")
        if num_warmup_steps is not None:
            num_warmup_steps = (
                num_warmup_steps if num_warmup_steps >= 1.0 else int(np.ceil(num_warmup_steps * num_train_steps))
            )
        sch_kwargs["num_training_steps"] = num_train_steps
        sch_kwargs["num_warmup_steps"] = num_warmup_steps

        optimization_args = {"scheduler_kwargs": SchedulerArgs(**sch_kwargs), **opt_kwargs}
        self._optimization_args = OptimizationArgs(**optimization_args)

    def configure_optimizer(self) -> Optimizer:
        opt_kw = self.optimization_args
        assert opt_kw is not None

        # weight decay
        if opt_kw.no_decay is not None and (opt_kw.weight_decay is not None and opt_kw.weight_decay > 0.0):
            params = [
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in opt_kw.no_decay) and p.requires_grad
                    ],
                    "weight_decay": opt_kw.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.named_parameters()
                        if any(nd in n for nd in opt_kw.no_decay) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            params = filter(lambda p: p.requires_grad, self.model.parameters())

        return OPTIMIZER_REGISTRY[opt_kw.name](params, lr=opt_kw.lr, **opt_kw.init_kwargs)  # type:ignore

    def configure_scheduler(self, optimizer: Optimizer) -> Optional[_LRScheduler]:
        assert self.optimization_args is not None
        sch_kw = self.optimization_args.scheduler_kwargs
        if sch_kw.name is not None:
            return SCHEDULER_REGISTRY[sch_kw.name](
                optimizer,
                num_training_steps=sch_kw.num_training_steps,
                num_warmup_steps=sch_kw.num_warmup_steps,
                **sch_kw.init_kwargs,
            )

    def configure_dataloader(self, loader: Optional[DataLoader]) -> Optional[_FabricDataLoader]:
        if loader is not None:
            return self.fabric.setup_dataloaders(loader, use_distributed_sampler=False, move_to_device=False)  # type: ignore

    def configure_loss_fn(self, stage: Union[str, RunningStage]) -> torch.nn.Module:
        ...

    def configure_metrics(self, stage: Optional[Union[str, RunningStage]] = None) -> Optional[METRIC]:
        ...

    """
    Utilities
    """

    def transfer_to_device(self, batch: Any) -> Any:
        # NOTE: fabric knows how to handle non-gpu stuff so the batch can have anything inside
        return self.fabric.to_device(batch)

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

    def callback(self, hook: str, *args, **kwargs) -> Optional[Any]:
        # passes self as first argument
        return self.fabric.call(hook, *args, **kwargs)

    def _setup_fit(
        self,
        train_loader: Optional[DataLoader],
        validation_loader: Optional[DataLoader],
        learning_rate: Optional[float],
        optimizer: Optional[str],
        optimizer_kwargs: Optional[Union[Dict, OptimizationArgs]],
        scheduler: Optional[str],
        scheduler_kwargs: Optional[Union[Dict, SchedulerArgs]],
    ) -> Tuple[_FabricModule, _FabricOptimizer, Optional[_LRScheduler], _FabricDataLoader, Optional[_FabricDataLoader]]:

        # configuration
        _train_loader = self.configure_dataloader(train_loader)
        _validation_loader = self.configure_dataloader(validation_loader)

        self.configure_optimization_args(learning_rate, optimizer, optimizer_kwargs, scheduler, scheduler_kwargs)
        _optimizer = self.configure_optimizer()
        _scheduler = self.configure_scheduler(_optimizer)
        model, _optimizer = self.fabric.setup(self.model, _optimizer)  # type: ignore

        return model, _optimizer, _scheduler, _train_loader, _validation_loader  # type: ignore
