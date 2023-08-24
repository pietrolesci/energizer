from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from lightning.fabric.wrappers import _FabricModule, _FabricDataLoader

from energizer.active_learning.datastores.base import ActiveDataStore
from energizer.active_learning.trackers import ActiveProgressTracker
from energizer.enums import RunningStage
from energizer.estimator import Estimator
from energizer.types import ROUND_OUTPUT
from energizer.estimator import OptimizationArgs, SchedulerArgs


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
        validation_perc: Optional[float] = None,
        max_rounds: Optional[int] = None,
        max_budget: Optional[int] = None,
        validation_sampling: Optional[str] = None,
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
        self.tracker.setup(
            log_interval=log_interval,
            enable_progress_bar=enable_progress_bar,
            max_rounds=max_rounds,
            max_budget=max_budget,
            query_size=query_size,
            run_on_pool=getattr(self, "pool_step", None) is not None,
            datastore=datastore,
            validation_perc=validation_perc,
        )

        fit_kwargs = dict(
            max_epochs=max_epochs,
            min_epochs=min_epochs,
            max_steps=max_steps,
            min_steps=min_steps,
            validation_freq=validation_freq,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            log_interval=log_interval,
            enable_progress_bar=enable_progress_bar,
            limit_train_batches=limit_train_batches,
            limit_validation_batches=limit_validation_batches,
        )

        return self.run_active_fit(
            datastore=datastore,
            replay=False,
            reinit_model=reinit_model,
            model_cache_dir=model_cache_dir,
            query_size=query_size,
            validation_sampling=validation_sampling,
            validation_perc=validation_perc,
            limit_test_batches=limit_test_batches,
            limit_pool_batches=limit_pool_batches,
            fit_kwargs=fit_kwargs,
        )

    def run_active_fit(
        self,
        datastore: ActiveDataStore,
        replay: bool,
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

            out = self.run_round(datastore=datastore, replay=replay, **kwargs)

            self.callback("on_round_end", datastore=datastore, output=out)

            output.append(out)

            # update progress
            self.tracker.increment_round()

            # check
            if not self.tracker.is_last_round:
                total_budget = datastore.labelled_size(self.tracker.global_round)
                assert (
                    self.tracker.budget_tracker.current == total_budget
                ), f"{self.tracker.budget_tracker.current} == {total_budget}"

        output = self.active_fit_end(output)

        self.callback("on_active_fit_end", datastore=datastore, output=output)

        self.tracker.end_active_fit()

        return output

    def run_round(
        self,
        datastore: ActiveDataStore,
        replay: bool,
        fit_kwargs: Dict,
        query_size: int,
        validation_perc: Optional[float],
        validation_sampling: Optional[str],
        limit_test_batches: Optional[int],
        limit_pool_batches: Optional[int],
    ) -> ROUND_OUTPUT:

        # start progress tracking
        num_round = self.tracker.global_round if replay else None

        # configuration
        model, train_loader, validation_loader, _optimizer, _scheduler = self._setup_fit(
            train_loader=datastore.train_loader(round=num_round),  # type: ignore
            validation_loader=datastore.validation_loader(round=num_round),
            **fit_kwargs,
        )

        test_loader = self._setup_eval(
            stage=RunningStage.TEST,
            loader=datastore.test_loader(),
            limit_batches=limit_test_batches,
            setup_model=False,
        )

        pool_loader = None
        if not replay:
            pool_loader = self._setup_eval(
                stage=RunningStage.POOL,
                loader=datastore.pool_loader(round=num_round),
                limit_batches=limit_pool_batches,
                setup_model=False,
            )  # type: ignore

        output = {}

        # fit
        if len(train_loader or []) > 0:
            output["fit"] = self.run_fit(model, train_loader, validation_loader, _optimizer, _scheduler)  # type: ignore

        # test
        if len(test_loader or []) > 0:  # type: ignore
            output[RunningStage.TEST] = self.run_evaluation(model, test_loader, RunningStage.TEST)  # type: ignore

        # query and label
        n_labelled = None
        if (
            not replay  # do not annotate in replay
            and not self.tracker.is_last_round  # last round is used only to test
            and len(pool_loader or []) > query_size  # enough instances
        ):
            n_labelled = self.run_annotation(model, pool_loader, datastore, query_size, validation_perc, validation_sampling)  # type: ignore
        elif replay:
            n_labelled = datastore.query_size(num_round)

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
        validation_sampling: Optional[str],
    ) -> int:
        # query
        self.callback("on_query_start", model=model, datastore=datastore)

        indices = self.run_query(model, loader=loader, datastore=datastore, query_size=query_size)

        # prevent to query more than available budget
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
        self, model: _FabricModule, loader: _FabricDataLoader, datastore: ActiveDataStore, query_size: int
    ) -> List[int]:
        raise NotImplementedError

    def active_fit_end(self, output: List[ROUND_OUTPUT]) -> Any:
        return output

    def round_end(self, output: ROUND_OUTPUT, datastore: ActiveDataStore) -> ROUND_OUTPUT:
        return output
