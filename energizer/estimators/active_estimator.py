from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from lightning_fabric.wrappers import _FabricModule

from energizer.datastores.base import Datastore
from energizer.enums import RunningStage
from energizer.estimators.estimator import Estimator
from energizer.estimators.progress_trackers import ActiveProgressTracker
from energizer.types import ROUND_OUTPUT


class ActiveEstimator(Estimator):
    def setup_tracking(self) -> None:
        self._progress_tracker = ActiveProgressTracker()

    @property
    def progress_tracker(self) -> ActiveProgressTracker:
        return self._progress_tracker

    """
    Active learning loop
    """

    def active_fit(
        self,
        datastore: Datastore,
        query_size: int,
        validation_perc: Optional[float] = None,
        max_rounds: Optional[int] = None,
        max_budget: Optional[int] = None,
        validation_sampling: Optional[str] = None,
        reinit_model: bool = True,
        max_epochs: Optional[int] = 3,
        min_steps: Optional[int] = None,
        learning_rate: float = 0.001,
        optimizer: str = "adamw",
        optimizer_kwargs: Optional[Dict] = None,
        scheduler: Optional[str] = None,
        scheduler_kwargs: Optional[Dict] = None,
        model_cache_dir: Union[str, Path] = ".model_cache",
        log_interval: int = 1,
        enable_progress_bar: bool = True,
        limit_train_batches: Optional[int] = None,
        limit_validation_batches: Optional[int] = None,
        limit_test_batches: Optional[int] = None,
        limit_pool_batches: Optional[int] = None,
        num_validation_per_epoch: Optional[int] = None,
    ) -> Any:
        assert (
            max_budget is not None or max_rounds is not None
        ), "At least one of `max_rounds` or `max_budget` must be not None."
        assert not reinit_model or (
            reinit_model and model_cache_dir
        ), "If `reinit_model` is True then you must specify `model_cache_dir`."

        # configure progress tracking
        self.progress_tracker.setup_active(
            max_rounds=max_rounds,
            max_budget=int(min(datastore.pool_size(), max_budget or float("Inf"))),
            initial_budget=datastore.labelled_size(),
            query_size=query_size,
            has_test=datastore.test_size() > 0,
            run_on_pool=getattr(self, "pool_step", None) is not None,
            has_validation=datastore.validation_size() > 0 or validation_perc is not None,
            log_interval=log_interval,
            enable_progress_bar=enable_progress_bar,
        )

        return self.run_active_fit(
            datastore=datastore,
            replay=False,
            reinit_model=reinit_model,
            model_cache_dir=model_cache_dir,
            max_epochs=max_epochs,
            min_steps=min_steps,
            learning_rate=learning_rate,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            query_size=query_size,
            validation_sampling=validation_sampling,
            validation_perc=validation_perc,
            limit_train_batches=limit_train_batches,
            limit_validation_batches=limit_validation_batches,
            limit_test_batches=limit_test_batches,
            limit_pool_batches=limit_pool_batches,
            num_validation_per_epoch=num_validation_per_epoch,
        )

    def run_active_fit(
        self,
        datastore: Datastore,
        replay: bool,
        reinit_model: bool,
        model_cache_dir: Union[str, Path],
        **kwargs,
    ) -> Any:

        if reinit_model:
            self.save_state_dict(model_cache_dir)

        # call hook
        self.fabric.call("on_active_fit_start", estimator=self, datastore=datastore)

        output = []
        while not self.progress_tracker.is_active_fit_done():

            if reinit_model:
                self.load_state_dict(model_cache_dir)

            self.fabric.call("on_round_start", estimator=self, datastore=datastore)

            out = self.run_round(datastore=datastore, replay=replay, **kwargs)

            # method to possibly aggregate
            out = self.round_epoch_end(out, datastore)

            self.fabric.call("on_round_end", estimator=self, datastore=datastore, output=out)

            output.append(out)

            # update progress
            self.progress_tracker.increment_round()

            # check
            if not self.progress_tracker.is_last_round:
                # print(self.progress_tracker.round_tracker)
                total_budget = datastore.labelled_size(self.progress_tracker.global_round)
                assert (
                    self.progress_tracker.budget_tracker.current == total_budget
                ), f"{self.progress_tracker.budget_tracker.current} == {total_budget}"

        output = self.active_fit_end(output)

        # call hook
        self.fabric.call("on_active_fit_end", estimator=self, datastore=datastore, output=output)

        self.progress_tracker.end_active_fit()

        return output

    def run_round(
        self,
        datastore: Datastore,
        replay: bool,
        max_epochs: Optional[int],
        min_steps: Optional[int],
        learning_rate: float,
        optimizer: str,
        optimizer_kwargs: Optional[Dict],
        scheduler: Optional[str],
        scheduler_kwargs: Optional[Dict],
        query_size: int,
        validation_perc: Optional[float],
        validation_sampling: Optional[str],
        num_validation_per_epoch: int,
        limit_train_batches: Optional[int],
        limit_validation_batches: Optional[int],
        limit_test_batches: Optional[int],
        limit_pool_batches: Optional[int],
    ) -> ROUND_OUTPUT:

        num_round = self.progress_tracker.global_round if replay else None
        self.progress_tracker.setup_fit(
            max_epochs=max_epochs,
            min_steps=min_steps,
            num_train_batches=len(datastore.train_loader(round=num_round) or []),
            num_validation_batches=len(datastore.validation_loader(round=num_round) or []),
            num_validation_per_epoch=num_validation_per_epoch,
            limit_train_batches=limit_train_batches,
            limit_validation_batches=limit_validation_batches,
        )
        self.progress_tracker.setup_eval(
            RunningStage.TEST, num_batches=len(datastore.test_loader() or []), limit_batches=limit_test_batches
        )
        if not replay:
            self.progress_tracker.setup_eval(
                RunningStage.POOL,
                num_batches=len(datastore.pool_loader(round=num_round) or []),
                limit_batches=limit_pool_batches,
            )

        # loaders
        train_loader = self.configure_dataloader(datastore.train_loader(round=num_round))
        validation_loader = self.configure_dataloader(datastore.validation_loader(round=num_round))
        test_loader = self.configure_dataloader(datastore.test_loader())

        # optimization
        _optimizer = self.configure_optimizer(optimizer, learning_rate, optimizer_kwargs)
        _scheduler = self.configure_scheduler(scheduler, _optimizer, scheduler_kwargs)
        model, _optimizer = self.fabric.setup(self.model, _optimizer)

        output = {}

        # fit
        if datastore.train_size(num_round) > 0:
            output["fit"] = self.run_fit(model, train_loader, validation_loader, _optimizer, _scheduler)  # type: ignore

        # test
        if datastore.test_size() > 0:
            output[RunningStage.TEST] = self.run_evaluation(model, test_loader, RunningStage.TEST)  # type: ignore

        # query and label
        n_labelled = None
        if (
            not replay  # do not annotate in replay
            and not self.progress_tracker.is_last_round  # last round is used only to test
            and datastore.pool_size(num_round) > query_size  # enough instances
        ):
            n_labelled = self.run_annotation(model, datastore, query_size, validation_perc, validation_sampling)
        elif replay:
            n_labelled = datastore.query_size(num_round)

        if n_labelled:
            self.progress_tracker.increment_budget(n_labelled)

        return output

    def run_annotation(
        self,
        model: _FabricModule,
        datastore: Datastore,
        query_size: int,
        validation_perc: Optional[float],
        validation_sampling: Optional[str],
    ) -> int:

        # query
        self.fabric.call("on_query_start", estimator=self, model=model, datastore=datastore)

        indices = self.run_query(model, datastore=datastore, query_size=query_size)

        # prevent to query more than available budget
        remaining_budget = min(query_size, self.progress_tracker.budget_tracker.get_remaining_budget())
        indices = indices[:remaining_budget]

        self.fabric.call("on_query_end", estimator=self, model=model, datastore=datastore, indices=indices)

        # label
        self.fabric.call("on_label_start", estimator=self, datastore=datastore)

        n_labelled = datastore.label(
            indices=indices,
            round=self.progress_tracker.global_round + 1,  # because the data will be used in the following round
            validation_perc=validation_perc,
            validation_sampling=validation_sampling,
        )

        self.fabric.call("on_label_end", estimator=self, datastore=datastore)

        return n_labelled

    def run_query(self, model: _FabricModule, datastore: Datastore, query_size: int) -> List[int]:
        raise NotImplementedError

    def active_fit_end(self, output: List[ROUND_OUTPUT]) -> Any:
        return output

    def round_epoch_end(self, output: ROUND_OUTPUT, datastore: Datastore) -> ROUND_OUTPUT:
        return output

    def replay_active_fit(
        self,
        datastore: Datastore,
        reinit_model: bool = True,
        max_epochs: Optional[int] = 3,
        min_steps: Optional[int] = None,
        learning_rate: float = 0.001,
        optimizer: str = "adamw",
        optimizer_kwargs: Optional[Dict] = None,
        scheduler: Optional[str] = None,
        scheduler_kwargs: Optional[Dict] = None,
        model_cache_dir: Union[str, Path] = ".model_cache",
        log_interval: int = 1,
        enable_progress_bar: bool = True,
        limit_train_batches: Optional[int] = None,
        limit_validation_batches: Optional[int] = None,
        limit_test_batches: Optional[int] = None,
        limit_pool_batches: Optional[int] = None,
        num_validation_per_epoch: Optional[int] = None,
    ) -> Any:

        assert not reinit_model or (
            reinit_model and model_cache_dir
        ), "If `reinit_model` is True then you must specify `model_cache_dir`."

        # configure progress tracking
        self.progress_tracker.setup_active(
            max_rounds=datastore.total_rounds(),
            max_budget=datastore.labelled_size(),
            initial_budget=datastore.labelled_size(0),
            query_size=datastore.query_size(1),
            has_test=datastore.test_size() > 0,
            run_on_pool=False,
            has_validation=datastore.validation_size() > 0,
            log_interval=log_interval,
            enable_progress_bar=enable_progress_bar,
        )

        return self.run_active_fit(
            datastore=datastore,
            replay=True,
            reinit_model=reinit_model,
            model_cache_dir=model_cache_dir,
            max_epochs=max_epochs,
            min_steps=min_steps,
            learning_rate=learning_rate,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            query_size=None,
            validation_sampling=None,
            validation_perc=None,
            limit_train_batches=limit_train_batches,
            limit_validation_batches=limit_validation_batches,
            limit_test_batches=limit_test_batches,
            limit_pool_batches=limit_pool_batches,
            num_validation_per_epoch=num_validation_per_epoch,
        )
