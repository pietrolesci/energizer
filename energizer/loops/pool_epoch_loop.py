from functools import lru_cache
from typing import Any, Optional

from pytorch_lightning.loops.epoch.evaluation_epoch_loop import EvaluationEpochLoop
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.types import STEP_OUTPUT

from energizer.learners.base import Learner
from energizer.metrics import AccumulateTopK


class PoolEvaluationEpochLoop(EvaluationEpochLoop):
    """This is the loop performing the evaluation.

    It mainly loops over the given dataloader and runs the validation or test step (depending on the trainer's current
    state).
    """

    def __init__(self, query_size: int) -> None:
        super().__init__()
        self.query_size = query_size
        self.accumulator = AccumulateTopK(k=query_size)  # need to move this to same device as model

    def _evaluation_step(self, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        """The evaluation step (pool_step).

        Args:
            batch: The current batch to run through the step.
            batch_idx: The index of the current batch
            dataloader_idx: the index of the dataloader producing the current batch

        Returns:
            the outputs of the step
        """
        # NOTE: this should be calling on the strategy to be consistent with PL
        # output = self.trainer._call_lightning_strategy_hook("pool_step", *kwargs.values())
        output = self.trainer._call_lightning_module_hook("pool_step", *kwargs.values())
        self.accumulator.update(output)
        return output

    def _evaluation_step_end(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        """Calls the `pool_step_end` hook."""
        model_output = self.trainer._call_lightning_module_hook("pool_step_end", *args, **kwargs)
        # NOTE: this should be calling on the strategy to be consistent with PL
        # strategy_output = self.trainer._call_strategy_hook("pool_step_end", *args, **kwargs)
        strategy_output = None
        output = strategy_output if model_output is None else model_output
        return output

    def _on_evaluation_batch_start(self, **kwargs: Any) -> None:
        """Calls the ``on_pool_batch_start`` hook.

        Args:
            batch: The current batch to run through the step
            batch_idx: The index of the current batch
            dataloader_idx: The index of the dataloader producing the current batch

        Raises:
            AssertionError: If the number of dataloaders is None (has not yet been set).
        """
        self.trainer._logger_connector.on_batch_start(**kwargs)
        kwargs.setdefault("dataloader_idx", 0)  # TODO: the argument should be keyword for these
        self.trainer._call_callback_hooks("on_pool_batch_start", *kwargs.values())
        self.trainer._call_lightning_module_hook("on_pool_batch_start", *kwargs.values())

    def _on_evaluation_batch_end(self, output: Optional[STEP_OUTPUT], **kwargs: Any) -> None:
        """The ``on_pool_batch_end`` hook.

        Args:
            output: The output of the performed step
            batch: The input batch for the step
            batch_idx: The index of the current batch
            dataloader_idx: Index of the dataloader producing the current batch
        """
        kwargs.setdefault("dataloader_idx", 0)  # TODO: the argument should be keyword for these
        self.trainer._call_callback_hooks("on_pool_batch_end", output, *kwargs.values())
        self.trainer._call_lightning_module_hook("on_pool_batch_end", output, *kwargs.values())
        self.trainer._logger_connector.on_batch_end()

    @lru_cache(1)
    def _should_track_batch_outputs_for_epoch_end(self) -> bool:
        """Whether the batch outputs should be stored for later usage."""
        model = self.trainer.lightning_module
        return is_overridden("pool_epoch_end", model, Learner)
