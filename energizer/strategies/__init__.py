from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from pytorch_lightning import LightningModule, Trainer
from torch import Tensor

import energizer.functional as EF
from energizer.inference import Deterministic, EnergizerInference


class EnergizerStrategy(LightningModule):
    def __init__(self, inference_module: EnergizerInference) -> None:
        super().__init__()
        self.inference_module = inference_module
        self.trainer: Optional[Trainer] = None
        self.query_size: Optional[int] = None

        self._counter: Optional[int] = None
        self.values: Optional[Tensor] = None
        self.indices: Optional[Tensor] = None

    def connect(self, module: LightningModule, query_size: int) -> None:
        """Deferred initialization of the strategy."""
        self.inference_module.connect(module)
        self.trainer = module.trainer
        self.query_size = query_size

    def forward(self, *args, **kwargs) -> Any:
        return self.inference_module(*args, **kwargs)

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Any:
        # print("INF - PREDICTING")

        with torch.inference_mode():
            logits = self(batch)
            batch_size = logits.shape[0]

            scores = self.objective(logits).flatten()

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

    def optimize_objective(self, scores: Tensor, query_size: Optional[int] = 1) -> Tuple[Tensor, Tensor]:
        return torch.topk(scores, query_size, dim=0)

    def _update(self, values: Tensor, indices: Tensor) -> None:
        all_values = torch.cat([self.values, values], dim=0)
        all_indices = torch.cat([self.indices, indices], dim=0)

        new_values, idx = self.optimize_objective(all_values, self.query_size)
        self.values.copy_(new_values)  # type: ignore
        self.indices.copy_(all_indices[idx])  # type: ignore

    def _batch_to_pool(self, indices: Tensor, batch_size: int) -> Tensor:
        indices += self._counter
        self._counter += batch_size  # type: ignore
        return indices


class LeastConfidence(EnergizerStrategy):
    r"""Implements the least confidence strategy.

    References: http://burrsettles.com/pub/settles.activelearning.pdf.

    This strategy allows an active learner to select the unlabeled data
    samples for which the model is least confident (i.e., most uncertain)
    in prediction or class assignment.

    It selects an instance $x$ such that

    $$\arg \max_{x} \; 1 - p(y_{max}|x, \theta)$$

    where $y_{max} = \arg\max_y p(y|x, \theta)$, i.e. the class label with the
    highest posterior probability under the model $\theta$. One way to interpret
    this uncertainty measure is the expected 0/1-loss, i.e., the model's belief
    that it will mislabel $x$.

    If samples from a posterior distributions are provided, it computes

    $$\arg \max_{x} \; 1 - \mathrm{E}_{p(\theta| D)} p(y_{max}|x, \theta)$$
    """

    def __init__(self, inference_module: EnergizerInference = Deterministic()) -> None:
        super().__init__(inference_module)

    def objective(self, logits: Tensor) -> Tensor:
        if logits.ndim == 3:
            confidence = EF.expected_confidence(logits, k=1)
            return 1.0 - confidence

        confidence = EF.confidence(logits, k=1)
        return 1.0 - confidence


class MarginStrategy(EnergizerStrategy):
    r"""Implements the margin strategy.

    Reference: http://burrsettles.com/pub/settles.activelearning.pdf.

    Margin sampling aims to correct for a shortcoming in least
    confident strategy, by incorporating the posterior of the second most likely label.
    Intuitively, instances with large margins are easy, since the classifier has little
    doubt in differentiating between the two most likely class labels. Instances with
    small margins are more ambiguous, thus knowing the true label would help the
    model discriminate more effectively between them.

    It selects an instance $x$ such that

    $$\arg\min_{x} P(y_1|x, \theta) - P(y_2|x, \theta)$$

    where $y_1$ and $y_2$ are the first and second most probable class labels under the
    model defined by $\theta$, respectively.

    If samples from a posterior distributions are provided, it computes

    $$\arg\min_{x} \mathrm{E}_{p(\theta| D)} P(y_1|x, \theta) - \mathrm{E}_{p(\theta| D)} P(y_2|x, \theta)$$
    """

    def __init__(self, inference_module: EnergizerInference = Deterministic()) -> None:
        super().__init__(inference_module)

    def objective(self, logits: Tensor) -> Tensor:
        if logits.ndim == 3:
            confidence_top2 = EF.expected_confidence(logits, k=2)

        confidence_top2 = EF.confidence(logits, k=2)

        # since it's a minimization
        return (confidence_top2[:, 0] - confidence_top2[:, 1]).neg()


class EntropyStrategy(EnergizerStrategy):
    r"""This class implements the entropy strategy.

    Reference: http://burrsettles.com/pub/settles.activelearning.pdf.

    Entropy is an information-theoretic measure that represents the amount of information
    needed to “encode” a distribution. As such, it is often thought of as a measure of
    uncertainty or impurity in machine learning. For binary classification, entropy-based
    sampling reduces to the margin and least confident strategies above; in fact all three
    are equivalent to querying the instance with a class posterior closest to 0.5. However,
    the entropybased approach generalizes easily to probabilistic multi-label classifiers and
    probabilistic models for more complex structured instances, such as sequences.

    The two previous strategies consider one or two most probable classes
    for sampling. The information lying in the remaining classes' probability
    distribution is unused. One way to look at uncertainty in a set of
    predictions is by whether you expect to be surprised by the outcome. This
    is the concept behind entropy

    It selects an instance $x$ such that

    $$\arg\max_x \mathrm{H}\left(\mathrm{p}(X)\right) = \arg\max_x - \sum_c p_c \; \log p_c$$

    where $\mathrm{p}$ is a probability mass function for a random variable $X$ such that
    $\mathrm{p}(X = c) = p_c$.

    If samples from a posterior distributions are provided, it computes

    $$\arg\max_x \mathrm{E}_{p(\theta| D)} \mathrm{H}\left(\mathrm{p}(X)\right)$$
    """

    def __init__(self, inference_module: EnergizerInference) -> None:
        super().__init__(inference_module)

    def compute_scores(self, logits: Tensor) -> Tensor:
        if logits.ndim == 3:
            return EF.expected_entropy(logits)

        return EF.entropy(logits)
