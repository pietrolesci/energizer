from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from lightning.fabric.wrappers import _FabricDataLoader, _FabricModule, _FabricOptimizer
from torch.optim.lr_scheduler import _LRScheduler

from energizer.active_learning.datastores.base import ActiveDatastore
from energizer.active_learning.trackers import ActiveProgressTracker
from energizer.enums import InputKeys, OutputKeys, RunningStage, SpecialKeys
from energizer.estimator import Estimator, OptimizationArgs, SchedulerArgs
from energizer.types import BATCH_OUTPUT, METRIC, ROUND_OUTPUT
from energizer.utilities import ld_to_dl


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
        datastore: ActiveDatastore,
        query_size: int,
        max_rounds: int | None = None,
        max_budget: int | None = None,
        validation_perc: float | None = None,
        validation_sampling: Literal["uniform", "stratified"] = "uniform",
        reinit_model: bool = True,
        model_cache_dir: str | Path = ".model_cache",
        max_epochs: int | None = 3,
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
        limit_test_batches: int | None = None,
        limit_pool_batches: int | None = None,
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
            test_kwargs=dict(limit_test_batches=limit_test_batches),
            query_kwargs=dict(limit_pool_batches=limit_pool_batches),
            label_kwargs=dict(validation_perc=validation_perc, validation_sampling=validation_sampling),
        )

    def run_active_fit(
        self, datastore: ActiveDatastore, reinit_model: bool, model_cache_dir: str | Path, **kwargs
    ) -> Any:
        if reinit_model:
            self.save_state_dict(model_cache_dir)

        self.callback("on_active_fit_start", datastore=datastore)

        output = []
        while not self.tracker.is_active_fit_done:
            if reinit_model:
                self.load_state_dict(model_cache_dir)

            # === RUN ROUND === #
            self.callback("on_round_start", datastore=datastore)

            out = self.run_round(datastore, **kwargs)

            out = self.round_end(datastore, out)
            self.callback("on_round_end", datastore=datastore, output=out)

            output.append(out)

            self.tracker.increment_round()
            self.tracker.increment_budget()
            # ================= #

            # check
            if not self.tracker.is_last_round:
                datastore_budget = datastore.labelled_size(self.tracker.global_round)
                tracker_budget = self.tracker.budget_tracker.current
                assert datastore_budget == tracker_budget, f"{datastore_budget=} and {tracker_budget=}"

            # print(f"END -- Round: {self.tracker.round_tracker}; Budget: {self.tracker.budget_tracker}")

        output = self.active_fit_end(datastore, output)

        self.callback("on_active_fit_end", datastore=datastore, output=output)

        self.tracker.end_active_fit()

        return output

    def run_round(
        self,
        datastore: ActiveDatastore,
        query_size: int,
        replay: bool,
        fit_loop_kwargs: dict,
        fit_opt_kwargs: dict,
        test_kwargs: dict,
        query_kwargs: dict,
        label_kwargs: dict,
    ) -> ROUND_OUTPUT:
        model, optimizer, scheduler, train_loader, validation_loader, test_loader = self._setup_round(
            datastore, replay, fit_loop_kwargs, fit_opt_kwargs, test_kwargs
        )

        output = {}

        # ============== FIT AND TEST ============== #
        if train_loader is not None:
            output[RunningStage.TRAIN] = self.run_fit(model, optimizer, scheduler, train_loader, validation_loader)

        if test_loader is not None:
            # print(f"TEST -- Round: {self.tracker.round_tracker}; Budget: {self.tracker.budget_tracker}")
            output[RunningStage.TEST] = self.run_evaluation(model, test_loader, RunningStage.TEST)

        # ============== QUERY AND LABEL ==============#

        n_labelled = 0
        if (
            not replay  # do not annotate in replay
            and not self.tracker.is_last_round  # last round is used only to test
        ):
            n_labelled = self.run_annotation(model, datastore, query_size, query_kwargs, label_kwargs)
        elif replay:
            n_labelled = datastore.query_size(self.tracker.global_round)

        # update the query size for this round
        self.tracker.budget_tracker.query_size = n_labelled

        return output

    def run_annotation(
        self, model: _FabricModule, datastore: ActiveDatastore, query_size: int, query_kwargs: dict, label_kwargs: dict
    ) -> int:
        # === QUERY === #
        self.callback("on_query_start", model=model, datastore=datastore)

        # NOTE: run_query is in charge of defining the pool_loader and the relative tracker
        indices = self.run_query(model, datastore, query_size, **query_kwargs)

        # prevent to query more than available budget
        if self.tracker.global_budget + len(indices) >= self.tracker.budget_tracker.max:  # type: ignore
            remaining_budget = min(query_size, self.tracker.budget_tracker.get_remaining_budget())
            indices = indices[:remaining_budget]

        self.callback("on_query_end", model=model, datastore=datastore, indices=indices)

        # ============= #

        # if no indices are returned, no need to annotated
        if len(indices) == 0:
            return 0

        # === LABEL === #
        self.callback("on_label_start", datastore=datastore)

        n_labelled = datastore.label(
            indices=indices,
            round=self.tracker.global_round + 1,  # because the data will be used in the following round
            **label_kwargs,
        )

        self.callback("on_label_end", datastore=datastore)
        # ============= #

        return n_labelled

    def run_query(self, model: _FabricModule, datastore: ActiveDatastore, query_size: int, **kwargs) -> list[int]:
        raise NotImplementedError

    def active_fit_end(self, datastore: ActiveDatastore, output: list[ROUND_OUTPUT]) -> Any:
        return output

    def round_end(self, datastore: ActiveDatastore, output: ROUND_OUTPUT) -> ROUND_OUTPUT:
        return output

    def _setup_round(
        self, datastore: ActiveDatastore, replay: bool, fit_loop_kwargs: dict, fit_opt_kwargs: dict, test_kwargs: dict
    ) -> tuple[
        _FabricModule,
        _FabricOptimizer,
        _LRScheduler | None,
        _FabricDataLoader | None,
        _FabricDataLoader | None,
        _FabricDataLoader | None,
    ]:
        """Start progress tracking."""

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
            train_loader, validation_loader, **fit_opt_kwargs
        )

        # configuration test
        test_loader = datastore.test_loader()
        limit_test_batches = test_kwargs.get("limit_test_batches")
        self.tracker.setup_eval(RunningStage.TEST, num_batches=len(test_loader or []), limit_batches=limit_test_batches)
        test_loader = self.configure_dataloader(test_loader)

        return model, optimizer, scheduler, train_loader, validation_loader, test_loader


class PoolBasedMixin(ABC):
    """Allows strategy to use the pool and/or the training set during the query process."""

    POOL_OUTPUT_KEY: OutputKeys

    def run_pool_evaluation(self, model: _FabricModule, loader: _FabricDataLoader) -> dict[str, np.ndarray]:
        out: list[dict] = self.run_evaluation(model, loader, RunningStage.POOL)  # type: ignore
        _out = ld_to_dl(out)
        return {k: np.concatenate(v) for k, v in _out.items()}

    def evaluation_step(
        self, model: _FabricModule, batch: Any, batch_idx: int, metrics: METRIC | None, stage: str | RunningStage
    ) -> dict | BATCH_OUTPUT:
        if stage != RunningStage.POOL:
            return super().evaluation_step(model, batch, batch_idx, metrics, stage)  # type: ignore

        # keep IDs here in case user messes up in the function definition
        ids = batch[InputKeys.ON_CPU][SpecialKeys.ID]
        pool_out = self.pool_step(model, batch, batch_idx, metrics)

        assert isinstance(pool_out, torch.Tensor), f"`{stage}_step` must return a tensor`."

        # enforce that we always return a dict here
        return {self.POOL_OUTPUT_KEY: pool_out, SpecialKeys.ID: ids}

    @abstractmethod
    def pool_step(
        self, model: _FabricModule, batch: Any, batch_idx: int, metrics: METRIC | None = None
    ) -> torch.Tensor:
        ...

    def pool_epoch_end(self, output: list[dict], metrics: METRIC | None) -> list[dict]:
        return output

    def get_pool_loader(self, datastore: ActiveDatastore, **kwargs) -> _FabricDataLoader | None:
        subpool_ids = kwargs.get("subpool_ids", None)
        loader = datastore.pool_loader(with_indices=subpool_ids) if subpool_ids is not None else datastore.pool_loader()

        if loader is not None:
            if subpool_ids is not None:
                assert len(loader.dataset) == len(subpool_ids), "Problems subsetting pool"  # type: ignore
            pool_loader = self.configure_dataloader(loader)  # type: ignore
            self.tracker.setup_eval(  # type: ignore
                RunningStage.POOL, num_batches=len(pool_loader or []), limit_batches=kwargs.get("limit_pool_batches")
            )
            return pool_loader

    def get_train_loader(self, datastore: ActiveDatastore, **kwargs) -> _FabricDataLoader | None:
        # NOTE: hack -- load train dataloader with the evaluation batch size
        assert datastore._loading_params is not None
        batch_size = datastore.loading_params.batch_size
        datastore._loading_params.batch_size = datastore.loading_params.eval_batch_size
        loader = datastore.train_loader(**kwargs)
        datastore._loading_params.batch_size = batch_size

        if loader is not None:
            train_loader = self.configure_dataloader(loader)  # type: ignore
            self.tracker.setup_eval(  # type: ignore
                RunningStage.POOL, num_batches=len(train_loader or []), limit_batches=kwargs.get("limit_pool_batches")
            )
            return train_loader
