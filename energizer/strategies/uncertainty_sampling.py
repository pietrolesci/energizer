from torch import Tensor

import energizer.functional as F
from energizer.inference.inference_modules import Deterministic, EnergizerInference
from energizer.strategies.base import EnergizerStrategy


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

    def __init__(self, inference_module: EnergizerInference = Deterministic()) -> None:
        super().__init__(inference_module)

    def objective(self, logits: Tensor) -> Tensor:
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

    def __init__(self, inference_module: EnergizerInference = Deterministic()) -> None:
        super().__init__(inference_module)

    def objective(self, logits: Tensor) -> Tensor:
        if logits.ndim == 3:
            confidence_top2 = F.expected_confidence(logits, k=2)

        confidence_top2 = F.confidence(logits, k=2)

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
            return F.expected_entropy(logits)

        return F.entropy(logits)
