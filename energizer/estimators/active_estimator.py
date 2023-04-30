from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from lightning.fabric.wrappers import _FabricModule
from torch.utils.data import DataLoader

from energizer.datastores.base import Datastore
from energizer.enums import RunningStage
from energizer.estimators.estimator import Estimator
from energizer.estimators.progress_trackers import ActiveProgressTracker
from energizer.types import EPOCH_OUTPUT, FIT_OUTPUT, ROUND_OUTPUT


@dataclass
class RoundOutput:
    fit: List[FIT_OUTPUT] = None
    test: EPOCH_OUTPUT = None
    indices: List[int] = None


class ActiveEstimator(Estimator):
    _progress_tracker: ActiveProgressTracker = None

    @property
    def progress_tracker(self) -> ActiveProgressTracker:
        if self._progress_tracker is None:
            self._progress_tracker = ActiveProgressTracker()
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
        model_cache_dir: Optional[Union[str, Path]] = ".model_cache",
        log_interval: Optional[int] = 1,
        enable_progress_bar: Optional[bool] = True,
        limit_train_batches: Optional[int] = None,
        limit_validation_batches: Optional[int] = None,
        limit_test_batches: Optional[int] = None,
        limit_pool_batches: Optional[int] = None,
        validation_interval: Optional[int] = None,
    ) -> Any:

        # configure progress tracking
        assert max_budget is not None or max_rounds is not None, ValueError(
            "At least one of `max_rounds` or `max_budget` must be not None."
        )
        self.progress_tracker.setup(
            max_rounds=max_rounds or float("Inf"),
            max_budget=min(datastore.pool_size(), max_budget or float("Inf")),
            initial_budget=datastore.initial_budget,
            query_size=query_size,
            has_test=datastore.has_test_data,
            has_pool=getattr(self, "pool_step", None) is not None,
            has_validation=datastore.has_validation_data() or validation_perc,
            log_interval=log_interval,
            enable_progress_bar=enable_progress_bar,
        )

        return self.run_active_fit(
            replay=False,
            datastore=datastore,
            max_epochs=max_epochs,
            min_steps=min_steps,
            learning_rate=learning_rate,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            reinit_model=reinit_model,
            query_size=query_size,
            validation_sampling=validation_sampling,
            validation_perc=validation_perc,
            model_cache_dir=model_cache_dir,
            limit_train_batches=limit_train_batches,
            limit_validation_batches=limit_validation_batches,
            limit_test_batches=limit_test_batches,
            limit_pool_batches=limit_pool_batches,
            validation_interval=validation_interval,
        )

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
        model_cache_dir: Optional[Union[str, Path]] = ".model_cache",
        log_interval: Optional[int] = 1,
        enable_progress_bar: Optional[bool] = True,
        limit_train_batches: Optional[int] = None,
        limit_validation_batches: Optional[int] = None,
        limit_test_batches: Optional[int] = None,
        limit_pool_batches: Optional[int] = None,
        validation_interval: Optional[int] = None,
    ) -> Any:

        # configure progress tracking
        self.progress_tracker.setup(
            max_rounds=datastore.last_labelling_round,
            max_budget=None,
            initial_budget=datastore.initial_budget,
            query_size=datastore.query_size(1),
            has_validation=datastore.has_validation_data(),
            has_test=datastore.has_test_data,
            has_pool=False,
            log_interval=log_interval,
            enable_progress_bar=enable_progress_bar,
        )

        return self.run_active_fit(
            replay=True,
            datastore=datastore,
            max_epochs=max_epochs,
            min_steps=min_steps,
            learning_rate=learning_rate,
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            scheduler=scheduler,
            scheduler_kwargs=scheduler_kwargs,
            reinit_model=reinit_model,
            model_cache_dir=model_cache_dir,
            limit_train_batches=limit_train_batches,
            limit_validation_batches=limit_validation_batches,
            limit_test_batches=limit_test_batches,
            limit_pool_batches=limit_pool_batches,
            validation_interval=validation_interval,
            query_size=None,
            validation_perc=None,
            validation_sampling=None,
        )

    def run_active_fit(
        self,
        replay: bool,
        datastore: Datastore,
        max_epochs: Optional[int],
        min_steps: Optional[int],
        learning_rate: float,
        optimizer: str,
        optimizer_kwargs: Optional[Dict],
        scheduler: Optional[str],
        scheduler_kwargs: Optional[Dict],
        reinit_model: bool,
        query_size: Optional[int],
        validation_sampling: Optional[str],
        validation_perc: Optional[float],
        model_cache_dir: Optional[Union[str, Path]],
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

            out = self.run_round(
                replay=replay,
                datastore=datastore,
                query_size=query_size,
                validation_perc=validation_perc,
                validation_sampling=validation_sampling,
                max_epochs=max_epochs,
                min_steps=min_steps,
                learning_rate=learning_rate,
                optimizer=optimizer,
                optimizer_kwargs=optimizer_kwargs,
                scheduler=scheduler,
                scheduler_kwargs=scheduler_kwargs,
                **kwargs,
            )

            # method to possibly aggregate
            output = self.round_epoch_end(output, datastore)

            self.fabric.call("on_round_end", estimator=self, datastore=datastore, output=out)

            output.append(out)

            # update progress
            self.progress_tracker.increment_round()

            # check
            if not self.progress_tracker.is_last_round:
                # print(self.progress_tracker.round_tracker)
                total_budget = datastore.total_labelled_size(self.progress_tracker.global_round)
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
        replay: bool,
        datastore: Datastore,
        max_epochs: Optional[int],
        min_steps: Optional[int],
        learning_rate: float,
        optimizer: str,
        optimizer_kwargs: Optional[Dict],
        scheduler: Optional[str],
        scheduler_kwargs: Optional[Dict],
        query_size: Optional[int],
        validation_perc: Optional[float],
        validation_sampling: Optional[str],
        limit_train_batches: Optional[int],
        limit_validation_batches: Optional[int],
        limit_test_batches: Optional[int],
        limit_pool_batches: Optional[int],
        validation_interval: Optional[int],
    ) -> ROUND_OUTPUT:

        num_round = self.progress_tracker.global_round if replay else None

        self.progress_tracker.setup_round_tracking(
            # fit
            max_epochs=max_epochs,
            min_steps=min_steps,
            num_train_batches=len(datastore.train_loader(num_round) or []),
            num_validation_batches=len(datastore.validation_loader(num_round) or []),
            limit_train_batches=limit_train_batches,
            limit_validation_batches=limit_validation_batches,
            validation_interval=validation_interval,
            # test
            num_test_batches=len(datastore.test_loader() or []),
            limit_test_batches=limit_test_batches,
            # pool
            num_pool_batches=len(datastore.pool_loader(num_round) or []) if not replay else None,
            limit_pool_batches=limit_pool_batches if not replay else None,
        )

        # loaders
        train_loader = self.configure_dataloader(datastore.train_loader(num_round))
        validation_loader = self.configure_dataloader(datastore.validation_loader(num_round))
        test_loader = self.configure_dataloader(datastore.test_loader())

        # optimization
        optimizer = self.configure_optimizer(optimizer, learning_rate, optimizer_kwargs)
        scheduler = self.configure_scheduler(scheduler, optimizer, scheduler_kwargs)
        model, optimizer = self.fabric.setup(self.model, optimizer)

        output = RoundOutput()

        # fit
        if datastore.has_train_data(num_round):
            output.fit = self.run_fit(model, train_loader, validation_loader, optimizer, scheduler)

        # test
        if datastore.has_test_data:
            output.test = self.run_evaluation(model, test_loader, RunningStage.TEST)

        # query and label
        if (
            not replay  # do not annotate in replay
            and not self.progress_tracker.is_last_round  # last round is used only to test
            and datastore.pool_size(num_round) > query_size  # enough instances
        ):
            n_labelled = self.run_annotation(model, datastore, query_size, validation_perc, validation_sampling)
        elif replay:
            n_labelled = datastore.query_size(num_round)

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
        self.fabric.call("on_query_start", estimator=self, model=model)

        indices = self.run_query(model, datastore=datastore, query_size=query_size)
        if self.progress_tracker.budget_tracker.remaining_budget < query_size:
            indices = indices[: self.progress_tracker.budget_tracker.remaining_budget]

        self.fabric.call("on_query_end", estimator=self, model=model, output=indices)

        # label
        self.fabric.call("on_label_start", estimator=self, datastore=datastore)

        n_labelled = datastore.label(
            indices=indices,
            round_idx=self.progress_tracker.global_round + 1,  # because the data will be used in the following round
            validation_perc=validation_perc,
            validation_sampling=validation_sampling,
        )

        self.fabric.call("on_label_end", estimator=self, datastore=datastore)

        return n_labelled

    def run_query(self, model: _FabricModule, datastore: Datastore, query_size: int) -> List[int]:
        raise NotImplementedError

    def active_fit_end(self, output: List[ROUND_OUTPUT]) -> Any:
        return output

    def round_epoch_end(self, output: RoundOutput, datastore: Datastore) -> ROUND_OUTPUT:
        return output

    """
    Methods
    """

    def get_pool_loader(self, datastore: Datastore) -> DataLoader:
        return datastore.pool_loader()
