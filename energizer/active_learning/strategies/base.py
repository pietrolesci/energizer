from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union

import torch
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule, _FabricOptimizer
from torch.optim.lr_scheduler import _LRScheduler

from energizer.active_learning.datastores.base import ActiveDataStore
from energizer.active_learning.trackers import ActiveProgressTracker
from energizer.enums import RunningStage
from energizer.estimator import Estimator, OptimizationArgs, SchedulerArgs
from energizer.types import BATCH_OUTPUT, METRIC, ROUND_OUTPUT


class ActiveEstimator(Estimator):
    _tracker: ActiveProgressTracker

    def init_tracker(self) -> None:
        self._tracker = ActiveProgressTracker()

    @property
    def tracker(self) -> ActiveProgressTracker:
        return self._tracker

    """
    Active learning loop
    """

    def active_fit(
        self,
        datastore: ActiveDataStore,
        query_size: int,
        max_rounds: Optional[int] = None,
        max_budget: Optional[int] = None,
        validation_perc: Optional[float] = None,
        validation_sampling: Literal["uniform", "stratified"] = "uniform",
        reinit_model: bool = True,
        model_cache_dir: Union[str, Path] = ".model_cache",
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
        limit_test_batches: Optional[int] = None,
        limit_pool_batches: Optional[int] = None,
    ) -> Any:
        assert not reinit_model or (
            reinit_model and model_cache_dir
        ), "If `reinit_model` is True then you must specify `model_cache_dir`."

        # configure progress tracking
        self.tracker.setup_active(
            log_interval=log_interval,
            enable_progress_bar=enable_progress_bar,
            max_rounds=max_rounds,
            max_budget=max_budget,
            query_size=query_size,
            datastore=datastore,
            run_on_pool=getattr(self, "pool_step", None) is not None,
            validation_perc=validation_perc,
        )

        return self.run_active_fit(
            datastore,
            reinit_model,
            model_cache_dir,
            query_size=query_size,
            replay=False,
            validation_sampling=validation_sampling,
            validation_perc=validation_perc,
            limit_test_batches=limit_test_batches,
            limit_pool_batches=limit_pool_batches,
            fit_loop_kwargs=dict(
                max_epochs=max_epochs,
                min_epochs=min_epochs,
                max_steps=max_steps,
                min_steps=min_steps,
                validation_freq=validation_freq,
                gradient_accumulation_steps=gradient_accumulation_steps,
                limit_train_batches=limit_train_batches,
                limit_validation_batches=limit_validation_batches,
            ),
            fit_opt_kwargs=dict(
                learning_rate=learning_rate,
                optimizer=optimizer,
                optimizer_kwargs=optimizer_kwargs,
                scheduler=scheduler,
                scheduler_kwargs=scheduler_kwargs,
            ),
        )

    def run_active_fit(
        self,
        datastore: ActiveDataStore,
        reinit_model: bool,
        model_cache_dir: Union[str, Path],
        **kwargs,
    ) -> Any:
        if reinit_model:
            self.save_state_dict(model_cache_dir)

        self.callback("on_active_fit_start", datastore=datastore)

        output = []
        while not self.tracker.is_active_fit_done:
            if reinit_model:
                self.load_state_dict(model_cache_dir)

            self.callback("on_round_start", datastore=datastore)
            out = self.run_round(datastore, **kwargs)
            self.callback("on_round_end", datastore=datastore, output=out)

            output.append(out)

            # update progress
            self.tracker.increment_round()

            # check
            if not self.tracker.is_last_round:
                datastore_budget = datastore.labelled_size(self.tracker.global_round)
                tracker_budget = self.tracker.budget_tracker.current
                assert datastore_budget == tracker_budget, f"{datastore_budget=} and {tracker_budget=}"

            # print(f"END -- Round: {self.tracker.round_tracker}; Budget: {self.tracker.budget_tracker}")

        output = self.active_fit_end(output)

        self.callback("on_active_fit_end", datastore=datastore, output=output)

        self.tracker.end_active_fit()

        return output

    def run_round(
        self,
        datastore: ActiveDataStore,
        query_size: int,
        replay: bool,
        validation_perc: Optional[float],
        validation_sampling: Literal["uniform", "stratified"],
        limit_test_batches: Optional[int],
        limit_pool_batches: Optional[int],
        fit_loop_kwargs: Dict,
        fit_opt_kwargs: Dict,
    ) -> ROUND_OUTPUT:
        model, optimizer, scheduler, train_loader, validation_loader, test_loader, pool_loader = self._setup_round(
            datastore,
            replay,
            fit_loop_kwargs,
            fit_opt_kwargs,
            limit_test_batches,
            limit_pool_batches,
        )

        output = {}

        # fit
        if train_loader is not None:
            output[RunningStage.TRAIN] = self.run_fit(model, optimizer, scheduler, train_loader, validation_loader)

        # test
        if test_loader is not None:
            # print(f"TEST -- Round: {self.tracker.round_tracker}; Budget: {self.tracker.budget_tracker}")
            output[RunningStage.TEST] = self.run_evaluation(model, test_loader, RunningStage.TEST)

        # query and label
        n_labelled = None
        if (
            not replay  # do not annotate in replay
            and not self.tracker.is_last_round  # last round is used only to test
            and pool_loader is not None
            and len(pool_loader or []) > query_size  # enough instances
        ):
            n_labelled = self.run_annotation(
                model, pool_loader, datastore, query_size, validation_perc, validation_sampling
            )
        elif replay:
            n_labelled = datastore.query_size(self.tracker.global_round)

        if n_labelled:
            self.tracker.increment_budget(n_labelled)

        output = self.round_end(output, datastore)

        return output

    def run_annotation(
        self,
        model: _FabricModule,
        loader: _FabricDataLoader,
        datastore: ActiveDataStore,
        query_size: int,
        validation_perc: Optional[float],
        validation_sampling: Optional[Literal["uniform", "stratified"]],
    ) -> int:
        # query
        self.callback("on_query_start", model=model, datastore=datastore)

        indices = self.run_query(model, loader, datastore, query_size)

        # prevent to query more than available budget
        if self.tracker.global_budget + len(indices) >= self.tracker.budget_tracker.max:  # type: ignore
            remaining_budget = min(query_size, self.tracker.budget_tracker.get_remaining_budget())
            indices = indices[:remaining_budget]

        self.callback("on_query_end", model=model, datastore=datastore, indices=indices)

        # label
        self.callback("on_label_start", datastore=datastore)

        n_labelled = datastore.label(
            indices=indices,
            round=self.tracker.global_round + 1,  # because the data will be used in the following round
            validation_perc=validation_perc,
            validation_sampling=validation_sampling,
        )

        self.callback("on_label_end", datastore=datastore)

        return n_labelled

    def run_query(
        self,
        model: _FabricModule,
        loader: _FabricDataLoader,
        datastore: ActiveDataStore,
        query_size: int,
    ) -> List[int]:
        raise NotImplementedError

    def active_fit_end(self, output: List[ROUND_OUTPUT]) -> Any:
        return output

    def round_end(self, output: ROUND_OUTPUT, datastore: ActiveDataStore) -> ROUND_OUTPUT:
        return output

    def _setup_round(
        self,
        datastore: ActiveDataStore,
        replay: bool,
        fit_loop_kwargs: Dict,
        fit_opt_kwargs: Dict,
        limit_test_batches: Optional[int],
        limit_pool_batches: Optional[int],
    ) -> Tuple[
        _FabricModule,
        _FabricOptimizer,
        Optional[_LRScheduler],
        Optional[_FabricDataLoader],
        Optional[_FabricDataLoader],
        Optional[_FabricDataLoader],
        Optional[_FabricDataLoader],
    ]:
        # start progress tracking

        num_round = self.tracker.global_round if replay else None

        # configuration fit
        train_loader = datastore.train_loader(round=num_round)
        validation_loader = datastore.validation_loader(round=num_round)
        self.tracker.setup_fit(
            num_train_batches=len(train_loader or []),
            num_validation_batches=len(validation_loader or []),
            **fit_loop_kwargs,
        )
        model, optimizer, scheduler, train_loader, validation_loader = self._setup_fit(
            train_loader,
            validation_loader,
            **fit_opt_kwargs,
        )

        # configuration test
        test_loader = datastore.test_loader()
        self.tracker.setup_eval(RunningStage.TEST, num_batches=len(test_loader or []), limit_batches=limit_test_batches)
        test_loader = self.configure_dataloader(test_loader)

        # configuration pool
        pool_loader = None
        if not replay:
            pool_loader = datastore.pool_loader(round=num_round)
            self.tracker.setup_eval(
                RunningStage.POOL, num_batches=len(pool_loader or []), limit_batches=limit_pool_batches
            )
            pool_loader = self.configure_dataloader(pool_loader)

        return model, optimizer, scheduler, train_loader, validation_loader, test_loader, pool_loader


class PoolBasedStrategyMixin(ABC):
    @abstractmethod
    def pool_step(
        self,
        model: _FabricModule,
        batch: Any,
        batch_idx: int,
        loss_fn: Optional[Union[torch.nn.Module, Callable]],
        metrics: Optional[METRIC] = None,
    ) -> BATCH_OUTPUT:
        ...

    def pool_epoch_end(self, output: List[Dict], metrics: Optional[METRIC]) -> List[Dict]:
        return output
