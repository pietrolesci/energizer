from functools import lru_cache
from typing import Any, Optional, Callable, Type

from pytorch_lightning.loops.epoch.evaluation_epoch_loop import EvaluationEpochLoop
from pytorch_lightning.utilities.types import STEP_OUTPUT
import pytorch_lightning as pl
from energizer.learners.base import Learner
import torch
from torch import Tensor
from torchmetrics import Metric
from functools import partial
from unittest.mock import Mock


class AccumulateTopK(Metric):
    def __init__(
        self,
        k: int,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
    ) -> None:
        super().__init__(compute_on_step, dist_sync_on_step, process_group, dist_sync_fn)
        self.k = k
        self.add_state("topk_scores", torch.tensor([float("-inf")] * self.k))
        self.add_state("indices", torch.ones(self.k, dtype=torch.int64).neg())
        self.add_state("size", torch.tensor(0, dtype=torch.int64))

    def update(self, scores: Tensor) -> None:
        batch_size = scores.numel()

        # indices with respect to the pool dataset and scores
        current_indices = torch.arange(self.size, self.size + batch_size, 1)

        # compute topk comparing current batch with the states
        all_scores = torch.cat([self.topk_scores, scores], dim=0)
        all_indices = torch.cat([self.indices, current_indices], dim=0)

        # aggregation
        topk_scores, idx = torch.topk(all_scores, k=min(self.k, batch_size))

        self.topk_scores.copy_(topk_scores)
        self.indices.copy_(all_indices[idx])
        self.size += batch_size

        print("current batch_size:", batch_size)
        print("current_indices:", current_indices)
        print("size:", self.size)
        print("all_scores:", all_scores)
        print("all_indices:", all_indices)
        print("top_scores:", topk_scores)
        print("top_indices:", all_indices[idx], "\n")

    def compute(self) -> Tensor:
        print("compute indices:", self.indices)
        return self.indices


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
        output = self.trainer._call_lightning_module_hook("pool_step", *kwargs.values())
        self.accumulator.update(output)
        return output

    def _evaluation_step_end(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        """Calls the `pool_step_end` hook."""
        # TODO: why the strategy - why it is useful?
        try:
            strategy_output = self.trainer._call_strategy_hook("pool_step_end", *args, **kwargs)
        except AttributeError:
            strategy_output = None
        model_output = self.trainer._call_lightning_module_hook("pool_step_end", *args, **kwargs)
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
        try:
            self.trainer._call_callback_hooks("on_pool_batch_start", *kwargs.values())
        except AttributeError:
            pass
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
        try:
            self.trainer._call_callback_hooks("on_pool_batch_end", output, *kwargs.values())
        except AttributeError:
            pass
        self.trainer._call_lightning_module_hook("on_pool_batch_end", output, *kwargs.values())
        self.trainer._logger_connector.on_batch_end()

    @lru_cache(1)
    def _should_track_batch_outputs_for_epoch_end(self) -> bool:
        """Whether the batch outputs should be stored for later usage."""
        model = self.trainer.lightning_module
        return is_overridden("pool_epoch_end", model)


def is_overridden(method_name: str, instance: Optional[object] = None, parent: Optional[Type[object]] = None) -> bool:
    if instance is None:
        # if `self.lightning_module` was passed as instance, it can be `None`
        return False

    if parent is None:
        if isinstance(instance, Learner):
            parent = Learner
        elif isinstance(instance, pl.LightningDataModule):
            parent = pl.LightningDataModule
        elif isinstance(instance, pl.Callback):
            parent = pl.Callback
        if parent is None:
            raise ValueError("Expected a parent")

    instance_attr = getattr(instance, method_name, None)
    if instance_attr is None:
        return False
    # `functools.wraps()` support
    if hasattr(instance_attr, "__wrapped__"):
        instance_attr = instance_attr.__wrapped__
    # `Mock(wraps=...)` support
    if isinstance(instance_attr, Mock):
        # access the wrapped function
        instance_attr = instance_attr._mock_wraps
    # `partial` support
    elif isinstance(instance_attr, partial):
        instance_attr = instance_attr.func
    if instance_attr is None:
        return False

    parent_attr = getattr(parent, method_name, None)
    if parent_attr is None:
        raise ValueError("The parent should define the method")

    return instance_attr.__code__ != parent_attr.__code__
