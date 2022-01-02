from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor

from energizer.inference.inference_modules import EnergizerInference
from energizer.strategies.hooks import PoolHooksMixin


class EnergizerStrategy(LightningModule, PoolHooksMixin):
    """Base class for a strategy that is a thin wrapper around `LightningModule`.

    It defines the `pool_*` methods and hooks that the user can redefine.
    Since the `pool_loop` is an `evaluation_loop` under the hood, this class calls
    the `pool_*` and `on_pool_*` methods in the respective `test_*` and `on_test_*` methods.
    This is necessary in order for the lightning `evaluation_loop` to pick up and run the
    methods. On the other hand, the user is able to deal with the `pool_*` and `on_pool_*`
    methods which makes the API cleaner.

    One important feature of this class is that the scores computed on the pool need not be
    kept in memory, but they are computed per each batch and a state is kept in the strategy
    class. On every batch, the state is updated and contains the maximum/minimum (depending on
    the strategy) scores seen so far. The states have dimension equal to the `query_size`.
    Therefore, the memory overhead is negligible in many cases, compared to other implementations
    that first compute the scores on the entire pool and then optimize them and extrarct the indices.
    """

    def __init__(self, inference_module: Optional[EnergizerInference]) -> None:
        """Initializes a strategy.

        Args:
            inference_module (EnergizerInference): An inference module that modifies the forward behaviour
                of the underlying module.
            requires_grad (bool): If True, it keeps track of the gradients while performing the `pool_step`.
                By default this is set to True and the operations are performed in the new
                `torch.inference_mode()` context.
        """
        super().__init__()
        self.inference_module = inference_module
        self.trainer: Optional[Trainer] = None
        self.query_size: Optional[int] = None

        self._counter: Optional[int] = None
        self.values: Optional[Tensor] = None
        self.indices: Optional[Tensor] = None

    def connect(self, module: LightningModule, query_size: int) -> None:
        """Deferred initialization of the strategy.

        Method to

        - Connect inference module to the underlying original module

        - Assign the trainer as an attribute

        - Assign the `query_size` parameters from the `ActiveLearningLoop`
        """
        if self.inference_module is not None:
            self.inference_module.connect(module)
        self.trainer = module.trainer
        self.query_size = query_size

    def reset(self) -> None:
        """Reset the states. This method must be called at the end of each active learning loop."""
        self._counter = 0
        self.values = torch.zeros(self.query_size, dtype=torch.float32, device=self.device, requires_grad=False)
        self.indices = -torch.ones(self.query_size, dtype=torch.int64, device=self.device, requires_grad=False)

    def forward(self, *args, **kwargs) -> Any:
        """Calls the forward method of the inference module."""
        return self.inference_module(*args, **kwargs)  # type: ignore

    def pool_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        """Main entry point to define a custom strategy.

        Since the `pool_loop` is a lightning `evalutation_loop`, in particular at `test_loop`, this method will be
        called inplace of the usual `test_step` method.

        This method should implement the logic to apply to each batch in the pool dataset and must return a tuple
        where the first element is a tensor with the actual scores and the second element is a tensor with the
        relative indices with respect to the batch.

        This method simply performs the following steps

        ```python
        # compute logits
        logits = self(batch)

        # compute scores
        logits = self.on_before_objective(logits)
        scores = self.objective(logits)
        scores = self.on_after_objective(scores)

        # optimize objective
        values, indices = self.optimize_objective(scores, query_size)

        # finally returns the tuple (values, indices)
        ```
        """
        # compute logits
        logits = self(batch)

        # compute scores
        logits = self.on_before_objective(logits)
        scores = self.objective(logits)
        scores = self.on_after_objective(scores)

        # optimize objective
        values, indices = self.optimize_objective(scores)

        return values, indices

    def pool_step_end(self, outputs: Tuple[Tensor, Tensor]) -> None:
        """Aggregate results across machines and update states."""
        # values = torch.cat([out[0] for out in outputs], dim=0)
        # indices = torch.cat([out[1] for out in outputs], dim=0)

        values, indices = outputs

        all_values = torch.cat([self.values, values], dim=0)
        all_indices = torch.cat([self.indices, indices], dim=0)

        new_values, idx = self.optimize_objective(all_values)
        self.values.copy_(new_values)  # type: ignore
        self.indices.copy_(all_indices[idx])  # type: ignore

    def pool_epoch_end(self, outputs: Any) -> None:
        pass

    def test_step(
        self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> Tuple[Tuple[Tensor, Tensor], int]:
        """Call the `pool_step` method and performs bookkeeping.

        This method allows to abstract away some of the underlying logic to transform indices relative to
        the batch in indices relative to the pool dataset

        It performs the `pool_step` and modifies the indices that are returned from the `pool_step` (which
        are relative to the batch) in indices relative to the pool dataset.
        """
        if isinstance(batch, Mapping):
            batch_size = batch[list(batch.keys())[0]].shape[0]
        elif isinstance(batch, Sequence):
            batch_size = batch[0].shape[0]
        else:
            raise MisconfigurationException(f"Batch of type {type(batch)} not supported, use Mapping or Sequence.")
        return self.pool_step(batch, batch_idx, dataloader_idx), batch_size

    def test_step_end(self, outputs: Tuple[Tuple[Tensor, Tensor], int]) -> None:
        (values, indices), batch_size = outputs

        # make indices relative to pool
        indices = self._batch_to_pool(indices, batch_size)

        self.pool_step_end((values, indices))

    def test_epoch_end(self, outputs: Any) -> None:
        self.pool_epoch_end(outputs)

    def on_before_objective(self, logits: Tensor) -> Tensor:
        """Run before the scores are computed. By default it simply returns its inputs."""
        return logits

    def objective(self, logits: Tensor) -> Tensor:
        """Must be redefined with the specific active learning objective, aka the acquisition function.

        Args:
            logits (Tensor): The output of the inference module forward pass.

        Returns:
            A tensor containing the scores that have to be optimized. By default, this tensor
            is then flatten by the `on_after_objective` method.
        """
        raise NotImplementedError

    def on_after_objective(self, scores: Tensor) -> Tensor:
        """Run after the scores are computed. By default if flattens the scores."""
        return scores.flatten()

    def optimize_objective(self, scores: Tensor) -> Tuple[Tensor, Tensor]:
        """Define how the active learning objective is optimized. By default it maximizes it.

        Returns:
            A tuple of two tensors. Both tensors have length `query_size`. The first element is the tensor
            of the topk scores and the second element is the tensor of the relative topk indices.
        """
        query_size = min(scores.shape[0], self.query_size)
        return torch.topk(scores, query_size, dim=0)

    def _batch_to_pool(self, indices: Tensor, batch_size: int) -> Tensor:
        """Transform index relative to the batch in indices relative to the pool."""
        indices += self._counter
        self._counter += batch_size  # type: ignore
        pool_size = self.trainer.datamodule.pool_size  # type: ignore
        if self._counter > pool_size:
            raise RuntimeError("Strategy states must be reset at the end of each labelling iteration.")
        return indices

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """Call the `poolest_batch_start` method and the relative method from each callback."""
        self.on_pool_batch_start(batch, batch_idx, dataloader_idx)
        self._call_callback_hook("on_pool_batch_start", batch, batch_idx, dataloader_idx)

    def on_test_batch_end(self, outputs: Optional[Any], batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        """Call the `poolest_batch_end` method and the relative method from each callback."""
        self.on_pool_batch_end(batch, batch_idx, dataloader_idx)
        self._call_callback_hook("on_pool_batch_end", batch, batch_idx, dataloader_idx)

    def on_test_epoch_start(self) -> None:
        """Call the `poolest_epoch_start` method and the relative method from each callback."""
        self.on_pool_epoch_start()
        self._call_callback_hook("on_pool_epoch_start")

    def on_test_epoch_end(self) -> None:
        """Call the `poolest_epoch_end` method and the relative method from each callback."""
        self.on_pool_epoch_end()
        self._call_callback_hook("on_pool_epoch_end")

    def on_test_start(self) -> None:
        """Call the `poolest_start` method and the relative method from each callback."""
        self.on_pool_start()
        self._call_callback_hook("on_pool_start")

    def on_test_end(self) -> None:
        """Call the `poolest_end` method and the relative method from each callback."""
        self.on_pool_end()
        self._call_callback_hook("on_pool_end")

    def on_test_model_eval(self) -> None:
        """Call the `poolest_model_eval` method and the relative method from each callback."""
        self.on_pool_model_eval()
        self._call_callback_hook("on_pool_model_eval")

    def on_test_model_train(self) -> None:
        """Call the `poolest_model_train` method and the relative method from each callback."""
        self.on_pool_model_train()
        self._call_callback_hook("on_pool_model_train")
