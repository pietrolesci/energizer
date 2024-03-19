from pathlib import Path
from typing import Any

import torch
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule, _FabricOptimizer

# https://scikit-learn.org/stable/developers/develop.html#random-numbers
from torch.optim.lr_scheduler import _LRScheduler

from energizer.coreset_selection.datastores.base import CoresetDatastore
from energizer.coreset_selection.strategies.base import CoresetSelectionStrategy
from energizer.enums import RunningStage
from energizer.estimator import DEFAULT_OPTIM, DEFAULT_SCHED, OptimizationArgs, OptimizerArgs, SchedulerArgs
from energizer.types import METRIC


class TrainingDynamicsStrategy(CoresetSelectionStrategy):
    """This strategy trains the model and evaluates on the training set after every epoch.
    
    Since the training sets might be very large, we write statistics to a jsonl file.
    """

    def fit(
        self,
        datastore: CoresetDatastore,
        max_epochs: int,
        gradient_accumulation_steps: int | None = None,
        optimizer_kwargs: OptimizerArgs | dict | None = DEFAULT_OPTIM,
        scheduler_kwargs: SchedulerArgs | dict | None = DEFAULT_SCHED,
        log_interval: int = 1,
        enable_progress_bar: bool = True,
        limit_train_batches: int | None = None,
        limit_pool_batches: int | None = None,
    ) -> None:
        
        # setup dataloaders
        _train_loader: _FabricDataLoader = self.configure_dataloader(datastore.train_loader())  # type: ignore
        _pool_loader: _FabricDataLoader = self.configure_dataloader(datastore.pool_loader())  # type: ignore

        # setup tracking
        self.tracker.setup(
            num_train_batches=len(_train_loader or []),
            num_pool_batches=len(_pool_loader or []),
            max_epochs=max_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            log_interval=log_interval,
            enable_progress_bar=enable_progress_bar,
            limit_train_batches=limit_train_batches,
            limit_pool_batches=limit_pool_batches,
        )

        # setup optimization arguments
        _optim_kwargs = (
            optimizer_kwargs
            if isinstance(optimizer_kwargs, OptimizerArgs | None)
            else OptimizerArgs(**optimizer_kwargs)
        )
        _sched_kwargs = (
            scheduler_kwargs
            if isinstance(scheduler_kwargs, SchedulerArgs | None)
            else SchedulerArgs(**scheduler_kwargs)
        )

        # setup optimization
        _optim = self.configure_optimizer(_optim_kwargs)
        _scheduler = self.configure_scheduler(_optim, _sched_kwargs)
        _model, _optim = self.fabric.setup(self.model_instance, _optim)  # type: ignore

        return self.run_coreset_selection(_model, _optim, _scheduler, _optim_kwargs, _train_loader, _pool_loader)

    def run_coreset_selection(
        self,
        model: _FabricModule,
        optimizer: _FabricOptimizer,
        scheduler: _LRScheduler | None,
        optimization_args: OptimizationArgs | None,
        train_loader: _FabricDataLoader,
        pool_loader: _FabricDataLoader,
        datastore: CoresetDatastore,
    ) -> None:
        _optim_kwargs = OptimizationArgs() if optimization_args is None else optimization_args

        self.tracker.start_fit()

        self.callback("on_fit_start", model=model)

        while not self.tracker.is_fit_done:
            
            # train for one epoch
            _ = self.run_epoch(model, optimizer, scheduler, _optim_kwargs, train_loader, None)
            
            # gather statistics on the training set
            pool_out = self.run_evaluation(model, pool_loader, RunningStage.POOL)
            
            datastore.record(pool_out)
            
            self.tracker.increment_epoch_idx()
            
            if len(datastore.get_cache_size()) == self.tracker.log_interval or self.tracker.is_last_epoch:
                datastore.write_to_cache()
        
        self.callback("on_fit_end", model=model, output=None)
        
        self.tracker.end_fit()

    def pool_step(
        self, model: _FabricModule, batch: Any, batch_idx: int, metrics: METRIC | None = None
    ) -> dict[str, Any]:
        raise NotImplementedError

    def pool_epoch_end(self, output: list[dict], metrics: METRIC | None) -> list[dict]:
        return output
