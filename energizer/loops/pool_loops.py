import os
from functools import lru_cache
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning.loops.dataloader.evaluation_loop import EvaluationLoop
from pytorch_lightning.loops.epoch.evaluation_epoch_loop import EvaluationEpochLoop
from pytorch_lightning.loops.loop import Loop
from pytorch_lightning.trainer.connectors.logger_connector.result import _OUT_DICT
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
from torch import Tensor
from torch.utils.data import DataLoader
from torchmetrics import Metric

"""
Loops that does not require evaluating on pool
"""


class PoolNoEvaluationLoop(Loop):
    """Calls the `query` method directly, without running on the pool.

    This is used for the `RandomStrategy`.
    """

    def reset(self) -> None:
        pass

    def advance(self, *args: Any, **kwargs: Any) -> None:
        pass

    def done(self) -> bool:
        return True

    def on_run_end(self) -> Tuple[List[_OUT_DICT], List[int]]:
        output = super().on_run_end()
        indices = self.trainer._call_lightning_module_hook("query")
        # indices = self.trainer.lightning_module.query()
        return output, indices


"""
Loops that evaluate on pool and accumulate batch results
"""


class AccumulateTopK(Metric):
    full_state_update = False

    def __init__(self, k: int) -> None:
        super().__init__()
        self.k = k
        self.add_state("topk_scores", torch.tensor([float("-inf")] * self.k))
        self.add_state("indices", torch.ones(self.k, dtype=torch.int64).neg())
        self.add_state("size", torch.tensor(0, dtype=torch.int64))

    def update(self, scores: Tensor) -> None:
        batch_size = scores.numel()

        # indices with respect to the pool dataset and scores
        current_indices = torch.arange(self.size, self.size + batch_size, 1).type_as(self.indices)

        # compute topk comparing current batch with the states
        all_scores = torch.cat([self.topk_scores, scores], dim=0)
        all_indices = torch.cat([self.indices, current_indices], dim=0)

        # aggregation
        topk_scores, idx = torch.topk(all_scores, k=self.k)

        self.topk_scores.copy_(topk_scores)
        self.indices.copy_(all_indices[idx])
        self.size += batch_size

        # print("current batch_size:", batch_size)
        # print("current_indices:", current_indices)
        # print("size:", self.size)
        # print("all_scores:", all_scores)
        # print("all_indices:", all_indices)
        # print("top_scores:", topk_scores)
        # print("top_indices:", all_indices[idx], "\n")

    def compute(self) -> Tensor:
        # print("compute indices:", self.indices)
        return self.indices


class PoolEvaluationEpochLoop(EvaluationEpochLoop):
    """This is the loop performing the evaluation.

    It mainly loops over the given dataloader and runs the `pool_step` method.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def accumulator(self) -> AccumulateTopK:
        """Returns the `AccumulateTopK` metric."""
        return self._accumulator

    @accumulator.setter
    def accumulator(self, accumulator: AccumulateTopK) -> None:
        # need to move this to same device as model
        self._accumulator = accumulator

    def _evaluation_step(self, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        """The evaluation step (`pool_step`).

        Args:
            batch: The current batch to run through the step.
            batch_idx: The index of the current batch
            dataloader_idx: the index of the dataloader producing the current batch

        Returns:
            the outputs of the step
        """
        # unpack batch with custom logic
        kwargs["batch"] = self.trainer._call_lightning_module_hook("get_inputs_from_batch", kwargs["batch"])

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
        """Calls the `on_pool_batch_start` hook.

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
        """The `on_pool_batch_end` hook.

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
        from energizer.query_strategies.base import AccumulatorStrategy

        model = self.trainer.lightning_module
        return is_overridden("pool_epoch_end", model, AccumulatorStrategy)


class PoolEvaluationLoop(EvaluationLoop):
    """Loops over the pool dataloaders for evaluation."""

    def __init__(self, verbose: Optional[bool] = False) -> None:
        super().__init__(verbose)
        self.epoch_loop = PoolEvaluationEpochLoop()
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
        # TODO: check if this is correct device
        assert self._results is not None
        self._results.to(device=self.trainer.lightning_module.device)
        self.epoch_loop.accumulator.to(device=self.trainer.lightning_module.device)

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
        self.epoch_loop.accumulator.reset()

    def _evaluation_epoch_end(self, outputs: List[EPOCH_OUTPUT]) -> None:
        """Runs ``pool_epoch_end`."""
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
        logged_outputs = super().on_run_end()
        indices = self.trainer.lightning_module.query()
        return logged_outputs, indices
