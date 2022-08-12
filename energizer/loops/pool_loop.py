import os
from typing import Any, List, Sequence, Tuple, Union, Optional

from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.trainer.connectors.logger_connector.result import _OUT_DICT
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from torch.utils.data import DataLoader

from energizer.loops.pool_epoch_loop import PoolEvaluationEpochLoop
from pytorch_lightning.loops.loop import Loop


class PoolNoEvalLoop(Loop):
    """Calls the `query` method directly, without running on the pool."""

    def reset(self) -> None:
        pass

    def advance(self, *args: Any, **kwargs: Any) -> None:
        pass

    def done(self) -> bool:
        return True

    def on_run_end(self) -> Tuple[List[_OUT_DICT], List[int]]:
        output = super().on_run_end()
        indices = self.trainer.lightning_module.query()
        return output, indices


class PoolEvaluationLoop(EvaluationLoop):
    """Loops over all dataloaders for evaluation."""

    def __init__(self, query_size: Optional[int] = None, verbose: Optional[bool] = False) -> None:
        super().__init__(verbose)
        self.epoch_loop = PoolEvaluationEpochLoop(query_size)
        # self._results = _ResultCollection(training=False)

    @property
    def dataloaders(self) -> Sequence[DataLoader]:
        """Returns the validation or test dataloaders."""
        dataloaders = self.trainer.pool_dataloaders
        if dataloaders is None:
            return []
        return dataloaders

    @property
    def prefetch_batches(self) -> int:
        batches = self.trainer.num_pool_batches
        is_unsized = batches[self.current_dataloader_idx] == float("inf")
        inter_batch_parallelism = os.getenv("PL_INTER_BATCH_PARALLELISM", "0") == "1"
        return 1 if is_unsized or inter_batch_parallelism else 0

    def _get_max_batches(self) -> List[int]:
        """Returns the max number of batches for each dataloader."""
        return self.trainer.num_pool_batches

    def _reload_evaluation_dataloaders(self) -> None:
        """Reloads dataloaders if necessary."""
        if self.trainer.testing:
            self.trainer.reset_pool_dataloader()

    def _on_evaluation_start(self, *args: Any, **kwargs: Any) -> None:
        """Runs ``on_pool_start`` hooks."""
        assert self._results is not None
        self._results.to(device=self.trainer.lightning_module.device)

        self.trainer._call_callback_hooks("on_pool_start", *args, **kwargs)
        self.trainer._call_lightning_module_hook("on_pool_start", *args, **kwargs)

    def _on_evaluation_model_eval(self) -> None:
        """Sets model to eval mode."""
        self.trainer._call_lightning_module_hook("on_pool_model_eval")

    def _on_evaluation_model_train(self) -> None:
        """Sets model to train mode."""
        self.trainer._call_lightning_module_hook("on_pool_model_train")

    def _on_evaluation_end(self, *args: Any, **kwargs: Any) -> None:
        """Runs ``on_pool_end`` hook."""
        self.trainer._call_callback_hooks("on_pool_end", *args, **kwargs)
        self.trainer._call_lightning_module_hook("on_pool_end", *args, **kwargs)

        # reset the logger connector state
        self.trainer._logger_connector.reset_results()

    def _on_evaluation_epoch_start(self, *args: Any, **kwargs: Any) -> None:
        """Runs `on_pool_epoch_start` hook."""
        self.trainer._logger_connector.on_epoch_start()
        self.trainer._call_callback_hooks("on_pool_epoch_start", *args, **kwargs)
        self.trainer._call_lightning_module_hook("on_pool_epoch_start", *args, **kwargs)

        # manually reset accumulation metric
        # self.trainer.lightning_module.accumulator.reset()

    def _evaluation_epoch_end(self, outputs: List[EPOCH_OUTPUT]) -> None:
        """Runs ``pool_epoch_end``"""
        self.trainer._logger_connector._evaluation_epoch_end()

        # with a single dataloader don't pass a 2D list
        output_or_outputs: Union[EPOCH_OUTPUT, List[EPOCH_OUTPUT]] = (
            outputs[0] if len(outputs) > 0 and self.num_dataloaders == 1 else outputs
        )

        # call the model epoch end
        self.trainer._call_lightning_module_hook("pool_epoch_end", output_or_outputs)

    def _on_evaluation_epoch_end(self) -> None:
        """Runs ``on_pool_epoch_end`` hook."""
        self.trainer._call_callback_hooks("on_pool_epoch_end")
        self.trainer._call_lightning_module_hook("on_pool_epoch_end")
        self.trainer._logger_connector.on_epoch_end()

    def on_run_end(self) -> Tuple[List[_OUT_DICT], List[int]]:
        output = super().on_run_end()
        indices = self.trainer.lightning_module.query()
        return output, indices
