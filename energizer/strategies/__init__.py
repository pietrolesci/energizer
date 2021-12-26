from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor


class EnergizerStrategy(LightningModule):
    def __init__(self):
        super().__init__()
        self.parent_module: Optional[LightningModule] = None
        self.trainer: Optional[Trainer] = None
        self.query_size: Optional[int] = None

    def connect(self, module: LightningModule, query_size: int) -> None:
        self.parent_module = module
        self.trainer = module.trainer
        self.query_size = query_size

    def forward(self, batch):
        return self.parent_module(batch)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        # print("INF - PREDICTING")

        with torch.inference_mode():
            logits = self(batch)
            batch_size = logits.shape[0]

            scores = self.objective(logits)

            # print("\tBATCH STARTING:", flush=True)
            # print("\tScores:", scores, flush=True)
            # print("\tBatch_size:", batch_size, flush=True)
            # print("\tBatch_idx:", batch_idx, flush=True)

            values, indices = self.optimize_objective(scores, min(batch_size, self.query_size))

            # print("\tIndices Batch:", indices.tolist(), flush=True)
            # print("\tInstance Counter:", self._counter, flush=True)
            # print("\tInstance Progress:", self.batch_progress, flush=True)

            indices = self._batch_to_pool(indices, batch_size)
            # print("\tIndices Batch to Pool:", indices.tolist(), flush=True)
            # print("\tAll indeces:", self.indices.tolist(), flush=True)

            self._update(values, indices)
            # print("\tUpdated All indeces:", self.indices.tolist(), "\n", flush=True)
            # self.batch_progress.increment_completed()

    def _reset(self) -> None:
        self._counter = 0
        self.values = torch.zeros(self.query_size, dtype=torch.float32, device=self.device, requires_grad=False)
        self.indices = -torch.ones(self.query_size, dtype=torch.int64, device=self.device, requires_grad=False)

    def objective(self, logits: Tensor) -> Tensor:
        raise NotImplementedError

    def optimize_objective(self, scores: Tensor, query_size: int) -> Tuple[Tensor, Tensor]:
        return torch.topk(scores, query_size, dim=0)

    def _update(self, values, indices):
        all_values = torch.cat([self.values, values], dim=0)
        all_indices = torch.cat([self.indices, indices], dim=0)

        new_values, idx = self.optimize_objective(all_values, self.query_size)
        self.values.copy_(new_values)
        self.indices.copy_(all_indices[idx])

    def _batch_to_pool(self, indices, batch_size):
        indices += self._counter
        self._counter += batch_size
        return indices


class LeastConfidence(EnergizerStrategy):
    """Implements the least confidence strategy.

    References: https://towardsdatascience.com/active-learning-sampling-strategies-f8d8ac7037c8

    This strategy allows an active learner to select the unlabeled data
    samples for which the model is least confident (i.e., most uncertain)
    in prediction or class assignment.

    It selects an instance x_{LC} according to:

        $$x_{LC} = argmax_{x} 1 - P(y|x)$$

    where y is the most probable class for each instance

    The least confidence strategy considers only the most probable class
    for evaluation
    """

    def __init__(self):
        super().__init__()

    def objective(self, logits: Tensor) -> Tensor:
        probs = F.softmax(logits, dim=-1)  # N x C
        topone = probs.max(dim=-1).values  # N x 1
        least_confidence = 1.0 - topone  # N x 1

        return least_confidence


class MarginStrategy(EnergizerStrategy):
    """Implements the Margin strategy.

    Reference: https://towardsdatascience.com/active-learning-sampling-strategies-f8d8ac7037c8

    The least confidence strategy considers only the most probable class
    for evaluation. This strategy considers the two most probable classes
    i.e. the classes having the highest and second-highest probabilities.

    It selects an instance x_{M} according to:

        $$x_{M} = argmax_{x} 1 - P(y|x)$$

    where y is the most probable class for each instance

    The margin strategy considers the two most probable classes i.e.
    the classes having the highest and second-highest probabilities
    """

    def __init__(self):
        super().__init__()

    def objective(self, logits_N_C: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits_N_C, dim=-1)  # N x C
        toptwo = probs.topk(2, dim=-1).values  # N x 2
        margin = toptwo[:, 0] - toptwo[:, 1]  # N x 1

        return margin


class EntropyStrategy(EnergizerStrategy):
    """This class implements the least confidence strategy.

    https://towardsdatascience.com/active-learning-sampling-strategies-f8d8ac7037c8

    The two previous strategies consider one or two most probable classes
    for sampling. The information lying in the remaining classes' probability
    distribution is unused. One way to look at uncertainty in a set of
    predictions is by whether you expect to be surprised by the outcome. This
    is the concept behind entropy

    It selects an instance x_{E} according to:

        $$x_{E} = argmax_{x} - sum_{i} p_\theta(y_{i}|x) * log_{2}(p_\theta(y_{i}|x))$$

    where y_{i} are the normalized logits (i.e., the probabilities) per each
    instance

    The margin strategy considers the two most probable classes i.e.
    the classes having the highest and second-highest probabilities
    """

    def __init__(self):
        super().__init__()

    def compute_scores(self, logits_N_C: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits_N_C, dim=-1)  # N x C
        nats = probs * probs.log()  # N x C
        entropy = nats.sum(dim=1).neg()  # N x 1

        return entropy
