import inspect
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import bitsandbytes as bnb
import numpy as np
import torch
from lightning.fabric import Fabric
from lightning.fabric.accelerators.accelerator import Accelerator
from lightning.fabric.connector import _PLUGIN_INPUT
from lightning.fabric.loggers.logger import Logger
from lightning.fabric.plugins.precision.bitsandbytes import BitsandbytesPrecision
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule, _FabricOptimizer
from lightning_utilities.core.rank_zero import rank_zero_info
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from energizer.enums import OutputKeys, RunningStage
from energizer.models import Model, TorchModel
from energizer.registries import OPTIMIZER_REGISTRY, SCHEDULER_REGISTRY
from energizer.trackers import ProgressTracker
from energizer.types import BATCH_OUTPUT, EPOCH_OUTPUT, FIT_OUTPUT, METRIC
from energizer.utilities import Args, move_to_cpu, set_deterministic


@dataclass
class SchedulerArgs(Args):
    name: str | None = None
    num_warmup_steps: int | None = None
    num_training_steps: int | None = None
    init_kwargs: dict = field(default_factory=dict)


@dataclass
class OptimizationArgs(Args):
    name: str | None = None
    lr: float | None = None
    weight_decay: float | None = None
    no_decay: list[str] | None = None
    set_to_none: bool = False
    clip_val: float | int | None = None
    max_norm: float | int | None = None
    norm_type: float | int = 2.0
    init_kwargs: dict = field(default_factory=dict)
    scheduler_kwargs: SchedulerArgs = field(default_factory=lambda: SchedulerArgs())
    backward_create_graph: bool = False
    backward_retain_graph: bool | None = None


class Estimator:
    _model: Model
    _tracker: ProgressTracker
    _optimization_args: OptimizationArgs
    _is_compiled: bool = False

    def __init__(
        self,
        model: torch.nn.Module | Model,
        accelerator: str | Accelerator = "cpu",
        precision: _PRECISION_INPUT = 32,
        callbacks: list[Any] | Any | None = None,
        loggers: Logger | list[Logger] | None = None,
        deterministic: bool | Literal["warn_only"] = "warn_only",
        tf32_mode: Literal["highest", "high", "medium"] = "highest",
        plugins: _PLUGIN_INPUT | list[_PLUGIN_INPUT] | None = None,
    ) -> None:
        super().__init__()

        if isinstance(model, torch.nn.Module):
            model = TorchModel(model)
        self._model = model

        self.fabric = Fabric(
            accelerator=accelerator,
            precision=precision,
            callbacks=callbacks,
            loggers=loggers,
            devices=1,  # only works with single-GPU
            num_nodes=1,  # only works with single-node
            plugins=plugins,
        )

        self.set_deterministic(deterministic)
        self.set_torch_matmul_precision(tf32_mode)
        self.configure_tracker()
        self.configure_model()

    def configure_tracker(self) -> None:
        self._tracker = ProgressTracker()

    """
    Properties
    """

    @property
    def model(self) -> Model:
        return self._model

    @property
    def tracker(self) -> ProgressTracker:
        return self._tracker

    @property
    def device(self) -> torch.device:
        return self.fabric.device

    @property
    def precision(self) -> str:
        # NOTE: doing this because model is not cast at __init__ but only when `self.fabric.setup` is called
        if self.is_quantized:
            rank_zero_info(
                "Model is loaded with the `BitsandbytesPrecision` plugin thus it currently is not cast "
                "to the correct dtype. It will only be cast during `fit` or `test`. Furthermore, the "
                f"linear layers will be cast to {self.quantization_mode}"
            )
            dtype = self.fabric._precision.dtype  # type: ignore
            return "bf16-true" if dtype == torch.bfloat16 else "16-true"
        return self.fabric._precision.precision

    @property
    def is_quantized(self) -> bool:
        # NOTE: hacky -- this is very specific to the BitsandbytesPrecision plugin
        return isinstance(self.fabric._precision, BitsandbytesPrecision)

    @property
    def quantization_mode(self) -> str | None:
        # NOTE: look at the BitsandbytesPrecision class
        cls_to_mode = {
            "_NF4Linear": "nf4",
            "_NF4DQLinear": "nf4-dq",
            "_FP4Linear": "fp4",
            "_FP4DQLinear": "fp4-dq",
            "_Linear8bitLt": "int8-training",
            "_Int8LinearInference": "int8",
        }
        if self.is_quantized:
            return cls_to_mode[self.fabric._precision._linear_cls.__name__]  # type: ignore
        rank_zero_info("Model is not quantized")

    @property
    def dtypes(self) -> set[str]:
        # need to setup for changes to take effect
        model = self.fabric.setup(self.model.model_instance.model_instance)
        return {str(p.dtype) for p in model.parameters()}

    @property
    def loggers(self) -> list[Logger]:
        """Returns all loggers passed to Fabric."""
        return self.fabric.loggers

    @property
    def logger(self) -> Logger:
        """Returns the first logger in the list passed to Fabric, which is considered the main logger."""
        return self.fabric.logger

    @property
    def model_summary(self) -> str:
        return self.model.summary

    @property
    def is_compiled(self) -> bool:
        return self._is_compiled

    @property
    def optimization_args(self) -> OptimizationArgs:
        return self._optimization_args

    @property
    def callbacks(self) -> list:
        return self.fabric._callbacks

    """
    Entry points
    """

    def fit(
        self,
        train_loader: DataLoader,
        validation_loader: DataLoader | None = None,
        max_epochs: int | None = None,
        min_epochs: int | None = None,
        max_steps: int | None = None,
        min_steps: int | None = None,
        validation_freq: str | None = "1:epoch",
        gradient_accumulation_steps: int | None = None,
        learning_rate: float | None = None,
        optimizer: str | None = None,
        optimizer_kwargs: dict | OptimizationArgs | None = None,
        scheduler: str | None = None,
        scheduler_kwargs: dict | SchedulerArgs | None = None,
        log_interval: int = 1,
        enable_progress_bar: bool = True,
        limit_train_batches: int | None = None,
        limit_validation_batches: int | None = None,
    ) -> list[FIT_OUTPUT]:
        """Entry point for model training.

        Calls `fit -> run_fit -> run_epoch -> run_training_step`.

        Data ordering: at each epoch the loader will have different order.
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
            train_loader, validation_loader, learning_rate, optimizer, optimizer_kwargs, scheduler, scheduler_kwargs
        )
        return self.run_fit(model, _optimizer, _scheduler, _train_loader, _validation_loader)  # type: ignore

    def test(
        self,
        loader: DataLoader,
        log_interval: int = 1,
        enable_progress_bar: bool = True,
        limit_batches: int | None = None,
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
        model = self.fabric.setup(self.model.model_instance)  # type: ignore

        return self.run_evaluation(model, _eval_loader, RunningStage.TEST)  # type: ignore

    """
    Training / Evaluation methods
    """

    def run_fit(
        self,
        model: _FabricModule,
        optimizer: _FabricOptimizer,
        scheduler: _LRScheduler | None,
        train_loader: _FabricDataLoader,
        validation_loader: _FabricDataLoader | None,
    ) -> list[FIT_OUTPUT]:
        self.tracker.start_fit()

        self.callback("on_fit_start", model=model)

        output = []
        while not self.tracker.is_fit_done:
            out = self.run_epoch(model, optimizer, scheduler, train_loader, validation_loader)
            output.append(out)

            self.tracker.increment_epoch_idx()

        self.callback("on_fit_end", model=model, output=output)

        self.tracker.end_fit()

        return output

    def run_epoch(
        self,
        model: _FabricModule,
        optimizer: _FabricOptimizer,
        scheduler: _LRScheduler | None,
        train_loader: _FabricDataLoader,
        validation_loader: _FabricDataLoader | None,
    ) -> FIT_OUTPUT:
        """Runs a training epoch."""

        # configure progress tracking
        self.tracker.start(RunningStage.TRAIN)

        # define metrics
        metrics = self.configure_metrics(RunningStage.TRAIN)

        # train mode
        model.train()

        self.callback("on_train_epoch_start", model=model, optimizer=optimizer)

        train_out, validation_out = [], []
        iterable = enumerate(train_loader)
        while not self.tracker.is_done:
            batch_idx, batch = next(iterable)

            # put batch on correct device
            batch = self.transfer_to_device(batch)

            # here global_step == batch_idx
            self.callback("on_train_batch_start", model=model, optimizer=optimizer, batch=batch, batch_idx=batch_idx)

            # run model on batch
            batch_out = self.run_training_step(model, optimizer, scheduler, batch, batch_idx, metrics)

            # here globa_step == batch_idx + 1
            self.callback("on_train_batch_end", model=model, output=batch_out, batch=batch, batch_idx=batch_idx)

            # record output
            if batch_out is not None:
                train_out.append(move_to_cpu(batch_out))

            # validation loop
            if self.tracker.should_validate:
                out = self.run_evaluation(model, validation_loader, RunningStage.VALIDATION)  # type: ignore
                if out is not None:
                    validation_out.append(out)

            # update progress tracker
            # here train_batch_counter.current == batch_idx + 1
            self.tracker.increment()

        # method to possibly aggregate
        train_out = self.train_epoch_end(train_out, metrics)

        self.callback("on_train_epoch_end", model=model, output=train_out, metrics=metrics)

        # validation loop
        if self.tracker.should_validate:
            out = self.run_evaluation(model, validation_loader, RunningStage.VALIDATION)  # type: ignore
            if out is not None:
                validation_out.append(out)

        self.tracker.end()

        # validation_out is already on cpu, but here we might need to move
        return move_to_cpu(train_out), validation_out

    def run_training_step(
        self,
        model: _FabricModule,
        optimizer: _FabricOptimizer,
        scheduler: _LRScheduler | None,
        batch: Any,
        batch_idx: int,
        metrics: METRIC | None,
    ) -> BATCH_OUTPUT:
        """Runs over a single batch of data."""

        with self.fabric.no_backward_sync(model, enabled=self.tracker.is_accumulating):
            # compute loss
            output = self.train_step(model, batch, batch_idx, metrics)
            loss = output if isinstance(output, torch.Tensor) else output[OutputKeys.LOSS]

            # compute gradients  (instead of loss.backward())
            self.fabric.backward(
                loss / self.tracker.gradient_accumulation_steps,
                create_graph=self.optimization_args.backward_create_graph,
                retain_graph=self.optimization_args.backward_retain_graph,
            )

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

            # reset the gradients
            optimizer.zero_grad(set_to_none=self.optimization_args.set_to_none)  # type: ignore

            # update scheduler
            if scheduler is not None:
                self.callback("on_before_scheduler", model=model, optimizer=optimizer, scheduler=scheduler)
                scheduler.step()
                self.callback("on_after_scheduler", model=model, optimizer=optimizer, scheduler=scheduler)

            # update tracker
            self.tracker.increment_step()

        return output

    def run_evaluation(
        self, model: _FabricModule, loader: _FabricDataLoader, stage: str | RunningStage
    ) -> EPOCH_OUTPUT:
        """Runs over an entire evaluation dataloader."""

        # configure progress tracking
        self.tracker.start(stage)

        # configure metrics
        metrics = self.configure_metrics(stage)

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
                batch_out = self.evaluation_step(model, batch, batch_idx, metrics, stage)

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

        return move_to_cpu(output)

    def evaluation_step(
        self, model: _FabricModule, batch: Any, batch_idx: int, metrics: METRIC | None, stage: str | RunningStage
    ) -> BATCH_OUTPUT:
        """Runs over a single batch of data."""
        # this might seems redundant but it's useful for active learning to hook in
        return getattr(self, f"{stage}_step")(model, batch, batch_idx, metrics)

    def step(
        self, stage: str | RunningStage, model: _FabricModule, batch: Any, batch_idx: int, metrics: METRIC | None = None
    ) -> BATCH_OUTPUT:
        raise NotImplementedError

    def train_step(
        self, model: _FabricModule, batch: Any, batch_idx: int, metrics: METRIC | None = None
    ) -> BATCH_OUTPUT:
        return self.step(RunningStage.TRAIN, model, batch, batch_idx, metrics)

    def validation_step(
        self, model: _FabricModule, batch: Any, batch_idx: int, metrics: METRIC | None = None
    ) -> BATCH_OUTPUT | None:
        return self.step(RunningStage.VALIDATION, model, batch, batch_idx, metrics)

    def test_step(
        self, model: _FabricModule, batch: Any, batch_idx: int, metrics: METRIC | None = None
    ) -> BATCH_OUTPUT | None:
        return self.step(RunningStage.TEST, model, batch, batch_idx, metrics)

    def epoch_end(self, stage: str | RunningStage, output: list[BATCH_OUTPUT], metrics: METRIC | None) -> EPOCH_OUTPUT:
        return output

    def train_epoch_end(self, output: list[BATCH_OUTPUT], metrics: METRIC | None) -> EPOCH_OUTPUT:
        return self.epoch_end(RunningStage.TRAIN, output, metrics)

    def validation_epoch_end(self, output: list[BATCH_OUTPUT], metrics: METRIC | None) -> EPOCH_OUTPUT:
        return self.epoch_end(RunningStage.VALIDATION, output, metrics)

    def test_epoch_end(self, output: list[BATCH_OUTPUT], metrics: METRIC | None) -> EPOCH_OUTPUT:
        return self.epoch_end(RunningStage.TEST, output, metrics)

    """
    Configuration
    """

    def configure_model(self) -> None:
        with self.fabric.init_module():
            self.model.configure_model()

    def configure_optimization_args(
        self,
        learning_rate: float | None = None,
        optimizer: str | None = None,
        optimizer_kwargs: dict | OptimizationArgs | None = None,
        scheduler: str | None = None,
        scheduler_kwargs: dict | SchedulerArgs | None = None,
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

        # defaults to constant schedule
        sch_kwargs["name"] = scheduler or sch_kwargs.get("name")

        num_train_steps = self.tracker.step_counter.max
        num_warmup_steps = sch_kwargs.get("num_warmup_steps")
        if num_warmup_steps is not None and num_train_steps is not None:
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
                        for n, p in self.model.model_instance.named_parameters()
                        if not any(nd in n for nd in opt_kw.no_decay) and p.requires_grad
                    ],
                    "weight_decay": opt_kw.weight_decay,
                },
                {
                    "params": [
                        p
                        for n, p in self.model.model_instance.named_parameters()
                        if any(nd in n for nd in opt_kw.no_decay) and p.requires_grad
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            params = filter(lambda p: p.requires_grad, self.model.model_instance.parameters())

        return OPTIMIZER_REGISTRY[opt_kw.name](params, lr=opt_kw.lr, **opt_kw.init_kwargs)  # type:ignore

    def configure_scheduler(self, optimizer: Optimizer) -> _LRScheduler | None:
        assert self.optimization_args is not None
        sch_kw = self.optimization_args.scheduler_kwargs
        if sch_kw.name is None:
            return

        sch_fn = SCHEDULER_REGISTRY[sch_kw.name]
        scheduler_fn_args = inspect.getfullargspec(sch_fn).args

        kwargs = {**sch_kw.init_kwargs}
        if "num_training_steps" in scheduler_fn_args:
            kwargs["num_training_steps"] = sch_kw.num_training_steps
        if "num_warmup_steps" in scheduler_fn_args:
            kwargs["num_warmup_steps"] = sch_kw.num_warmup_steps

        return sch_fn(optimizer, **kwargs, **sch_kw.init_kwargs)

    def configure_dataloader(self, loader: DataLoader | None) -> _FabricDataLoader | None:
        if loader is not None:
            return self.fabric.setup_dataloaders(loader, use_distributed_sampler=False, move_to_device=False)  # type: ignore

    def configure_metrics(self, stage: str | RunningStage | None = None) -> METRIC | None:
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
            self.fabric.log_dict(move_to_cpu(value_dict), step)

    def save_state_dict(self, cache_dir: str | Path, name: str = "state_dict.pt", **kwargs) -> None:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.fabric.save(state=self.model.model_instance.state_dict(), path=cache_dir / name)

    def load_state_dict(self, cache_dir: str | Path, name: str = "state_dict.pt", **kwargs) -> None:
        cache_dir = Path(cache_dir)
        self.model.model_instance.load_state_dict(self.fabric.load(cache_dir / name))

    def callback(self, hook: str, *args, **kwargs) -> Any | None:
        # if estimator has the method
        method = getattr(self, hook, None)
        if method is not None and callable(method):
            method(*args, **kwargs)

        # passes self as first argument
        return self.fabric.call(hook, self, *args, **kwargs)

    def compile(self, **kwargs) -> None:
        # model becomes a Callable
        self._model = torch.compile(self._model, **kwargs)  # type: ignore
        self._is_compiled = True

    def set_torch_matmul_precision(self, tf32_mode: Literal["highest", "high", "medium"] = "highest") -> None:
        # equivalent to `torch.backends.cudnn.allow_tf32 = True`
        # convolutions are not changed, to do that you need
        # `torch.backends.cudnn.allow_tf32 = True`
        torch.set_float32_matmul_precision(tf32_mode)

    def set_deterministic(self, deterministic: bool | Literal["warn_only"]) -> None:
        # sets deterministic convolutions too
        set_deterministic(deterministic)

    def set_eval_mode(self) -> None:
        self.model.model_instance.eval()

    def set_train_mode(self, mode: bool = True) -> None:
        self.model.model_instance.train(mode)

    def num_parameters(self, only_trainable: bool = False, exclude_embeddings: bool = False) -> int:
        """Get number of (optionally, trainable or non-embeddings) parameters in the module.

        Refs: https://github.com/huggingface/transformers/blob/ae093eef016533a3670561fa9e26addb42d446d1/src/transformers/modeling_utils.py#L976-L1021

        Args:
            only_trainable (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of trainable parameters

            exclude_embeddings (`bool`, *optional*, defaults to `False`):
                Whether or not to return only the number of non-embeddings parameters

        Returns:
            `int`: The number of parameters.
        """

        if exclude_embeddings:
            embedding_param_names = [
                f"{name}.weight"
                for name, module_type in self.model.model_instance.named_modules()
                if isinstance(module_type, torch.nn.Embedding)
            ]
            total_parameters = [
                parameter
                for name, parameter in self.model.model_instance.named_parameters()
                if name not in embedding_param_names
            ]
        else:
            total_parameters = list(self.model.model_instance.parameters())

        total_numel = []
        is_loaded_in_4bit = self.is_quantized and "4" in self.quantization_mode  # type: ignore
        for param in total_parameters:
            if param.requires_grad or not only_trainable:
                # For 4bit models, we need to multiply the number of parameters by 2 as half of the parameters are
                # used for the 4bit quantization (uint8 tensors are stored)
                if is_loaded_in_4bit and isinstance(param, bnb.nn.Params4bit):
                    total_numel.append(param.numel() * 2)
                else:
                    total_numel.append(param.numel())

        return sum(total_numel)

    """
    Private methods
    """

    def _setup_fit(
        self,
        train_loader: DataLoader | None,
        validation_loader: DataLoader | None,
        learning_rate: float | None,
        optimizer: str | None,
        optimizer_kwargs: dict | OptimizationArgs | None,
        scheduler: str | None,
        scheduler_kwargs: dict | SchedulerArgs | None,
    ) -> tuple[_FabricModule, _FabricOptimizer, _LRScheduler | None, _FabricDataLoader, _FabricDataLoader | None]:
        # configuration
        _train_loader = self.configure_dataloader(train_loader)
        _validation_loader = self.configure_dataloader(validation_loader)

        self.configure_optimization_args(learning_rate, optimizer, optimizer_kwargs, scheduler, scheduler_kwargs)
        _optimizer = self.configure_optimizer()
        _scheduler = self.configure_scheduler(_optimizer)
        model, _optimizer = self.fabric.setup(self.model.model_instance, _optimizer)  # type: ignore

        return model, _optimizer, _scheduler, _train_loader, _validation_loader  # type: ignore
