from typing import Any, Callable, Optional

import torch
from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import Tensor
from torchmetrics import Metric

import energizer.strategies.functional as F
from energizer.strategies.inference import Adapter


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


class EnergizerStrategy(Adapter):
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

    _is_on_pool: bool = False

    def __init__(self, adapter: Adapter, query_size: int) -> None:
        if not hasattr(adapter, "pool_step") and not hasattr(adapter.adapter, "pool_step"):
            raise MisconfigurationException("You must implement `pool_step` in your LightningModule.")
        super().__init__(adapter)
        self.query_size = query_size
        self.accumulation_metric = AccumulateTopK(self.query_size)

    @property
    def is_on_pool(self) -> bool:
        """Returns whether we are evaluating on the pool or on the test set."""
        return self._is_on_pool

    @is_on_pool.setter
    def is_on_pool(self, value) -> None:
        self._is_on_pool = value
        if (
            getattr(self, "trainer", None) is not None
            and getattr(self.trainer, "datamodule", None) is not None
            and getattr(self.trainer.datamodule, "_is_on_pool", None) is not None
        ):
            self.trainer.datamodule.is_on_pool = value

    def on_test_epoch_start(self) -> None:
        """Reset the accumulation metric manually.

        Since we are not logging it, torchmetrics does not have any way of figuring
        out when to reset it.
        """
        self.accumulation_metric.reset()

    def forward(self, *args, **kwargs) -> Any:
        """Dispaches the forward pass to the right object.

        Calls the forward step of the strategy or the underlying LightningModule
        based on whether we are on the pool or {train, val, test} set.
        """
        if self.is_on_pool:
            return self.adapter(*args, **kwargs)
        return self.adapter.adapter(*args, **kwargs)

    def test_step(self, batch, batch_idx, *args, **kwargs) -> None:
        print("\nbatch_idx:", batch_idx)
        if self.is_on_pool:
            print("len:", len(self.trainer.datamodule.pool_fold))
            
            # consider including score into pool_ste to be consistent with PL logic
            outputs = self.pool_step(batch, batch_idx, *args, **kwargs)
            scores = self.score(outputs)
            
            # TODO: why do I have gradients here?
            print("requires grad:", scores.requires_grad, self.trainer.testing)
            self.accumulation_metric.update(scores)
        return LightningModule.test_step(self.adapter, batch, batch_idx, *args, **kwargs)

    def pool_step(self, batch, batch_idx, *args, **kwargs) -> None:
        return self.adapter.adapter.pool_step(batch, batch_idx, *args, **kwargs)


class RandomStrategy(EnergizerStrategy):
    """Implements the RandomStrategy."""


class LeastConfidenceStrategy(EnergizerStrategy):
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

    def score(self, logits: Tensor) -> Tensor:
        r"""Compute the least confidence scores.

        $$1 - p(y_{max}|x, \theta)$$ or $$1 - \mathrm{E}_{p(\theta| D)} p(y_{max}|x, \theta)$$
        """
        if logits.ndim == 3:
            confidence = F.expected_confidence(logits, k=1)
            return 1.0 - confidence

        confidence = F.confidence(logits, k=1)
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

    def score(self, logits: Tensor) -> Tensor:
        r"""Compute the margin scores.

        $$P(y_1|x, \theta) - P(y_2|x, \theta)$$ or
        $$\mathrm{E}_{p(\theta| D)} P(y_1|x, \theta) - \mathrm{E}_{p(\theta| D)} P(y_2|x, \theta)$$
        """
        if logits.ndim == 3:
            confidence_top2 = F.expected_confidence(logits, k=2)

        confidence_top2 = F.confidence(logits, k=2)

        # since it's a minimization
        return -(confidence_top2[:, 0] - confidence_top2[:, 1])


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

    def score(self, logits: Tensor) -> Tensor:
        r"""Compute the entropy scores.

        $$\mathrm{H}\left(\mathrm{p}(X)\right) = \arg\max_x - \sum_c p_c \; \log p_c$$ or
        $$\mathrm{E}_{p(\theta| D)} \mathrm{H}\left(\mathrm{p}(X)\right)$$
        """
        if logits.ndim == 3:
            return F.expected_entropy(logits)

        return F.entropy(logits)
