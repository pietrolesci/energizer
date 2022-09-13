import torch
from torch import Tensor
from torch.special import entr  # https://pytorch.org/docs/stable/special.html


def entropy(logits: Tensor) -> Tensor:
    r"""Computes Shannon's entropy in nats.

    It expects a tensor of logits with the following dimensions: `(B: batch_size, C: num_classes)`.

    This function implements the following steps, for each element along the `B: batch_size` dimension:

    - Converts logits in probabilities along the `C: num_classes` dimension
    $$p_{bc} = e^{l_{bc}} / \sum_j e^{l_{bj}}$$


    - Computes Shannon's entropy along the `C: num_classes` dimension
    $$\mathrm{H}_b\left(\mathrm{p}(X)\right) = - \sum_c p_{bc} \log(p_{bc})$$

    where $l_{bc}$ is the logit for class $c$ for the $b$-th element in the batch, and $\mathrm{p}$ is a
    probability mass function for a random variable $X$ such that $\mathrm{p}(X = c) = p_c$.

    Args:
        logits (Tensor): A tensor of dimensions `(B: batch_size, C: num_classes)`.

    Returns:
        The Shannon's entropy, i.e. a vector of dimensions `(B: batch_size,)`.
    """
    probs = logits.softmax(dim=-1)
    return entr(probs).sum(dim=-1)  # remember: you need to sum across classes


def predictive_entropy(logits: Tensor) -> Tensor:
    r"""Computes the predictive Shannon's entropy in nats.

    It expects a tensor of logits with the following dimensions:
    `(B: batch_size, C: num_classes, S: num_inference_iterations)`.
    This function implements the following steps, for each element along the `B: batch_size` dimension:

    - Converts logits in probabilities along the `C: num_classes` dimension
    $$p_{bcs} = e^{l_{bcs}} / \sum_j e^{l_{bjs}}$$

    - Averages the output probabilities per class across samples
    $$p_{bc} = \frac{1}{S} \sum_s p_{bcs}

    - Computes Shannon's entropy along the `C: num_classes` dimension
    $$\mathrm{H}_{b}\left(\mathrm{p}(X) \right) = - \sum_c p_{bc} \log(p_{bc})$$

    where $l_{bcs}$ is the logit for class $c$ for the $b$-th element in the batch in the $s$-th sample,
    and $\mathrm{p}$ is a probability mass function for a random variable $X$ such that $\mathrm{p}(X = c) = p_c$.

    You can see the `entropy` function as a restriction of this in which we
    only have one sample.

    Args:
        logits (Tensor): A tensor of dimensions `(B: batch_size, C: num_classes, S: num_inference_iterations)`.

    Returns:
        The Shannon's entropy, i.e. a vector of dimensions `(B: batch_size,)`.
    """
    avg_probs = logits.softmax(dim=-2).mean(dim=-1)
    return entr(avg_probs).sum(dim=-1)


def expected_entropy(logits: Tensor) -> Tensor:
    r"""Computes the expected Shannon's entropy in nats.

    It expects a tensor of logits with the following dimensions:
    `(B: batch_size, C: num_classes, S: num_inference_iterations)`.
    This function implements the following steps, for each element along the `B: batch_size` dimension:

    - Converts logits in probabilities along the `C: num_classes` dimension
    $$p_{bcs} = e^{l_{bcs}} / \sum_j e^{l_{bjs}}$$

    - Computes Shannon's entropy along the `C: num_classes` dimension
    $$\mathrm{H}_{bs}\left(\mathrm{p}(X) \right) = - \sum_c p_{bcs} \log(p_{bcs})$$

    - Averages the Shannon's entropy along the `S: num_samples` dimension
    $$\frac{1}{S} \sum_s \mathrm{H}_{bs}\left(\mathrm{p}(X)\right)$$

    where $l_{bcs}$ is the logit for class $c$ for the $b$-th element in the batch in the $s$-th sample,
    and $\mathrm{p}$ is a probability mass function for a random variable $X$ such that $\mathrm{p}(X = c) = p_c$.

    Args:
        logits (Tensor): A tensor of dimensions `(B: batch_size, C: num_classes, S: num_inference_iterations)`.

    Returns:
        The Shannon's entropy, i.e. a vector of dimensions `(B: batch_size,)`.
    """
    probs = logits.softmax(dim=-2)
    entropies = entr(probs).sum(dim=-2)
    return entropies.mean(dim=-1)


def bald(logits: Tensor) -> Tensor:
    r"""Compute the BALD acquisition function.

    NOTE: this could have been simply implemented as

    ```python
    predictive_entropy(logits) - expected_entropy(logits)
    ```

    however, both functions would need to compute the softmax internally.
    To avoid doubling the computation uselessly, we implement this function
    so that it only computes the softmax ones.

    """
    # predictive_entropy(logits) - expected_entropy(logits)
    probs = logits.softmax(dim=-2)

    # To get the first term, we make many runs, average the output, and measure the entropy.
    predictive_entropy = entr(probs.mean(dim=-1)).sum(dim=-1)

    # To get the second term, we make many runs, measure the entropy of every run, and take the average.
    expected_entropy = entr(probs).sum(dim=-2).mean(dim=-1)

    return predictive_entropy - expected_entropy


def confidence(logits: Tensor, k: int = 1) -> Tensor:
    r"""Computes confidence based on logits.

    Computes the confidence defined as the highest probability the model assigns to a class, that is

    $$\max_c p_{bc}$$

    where $p_{bc}$ is the probability for class $c$ for instance $b$ in the batch.

    Args:
        logits (Tensor): A tensor of dimensions `(B: batch_size, C: num_classes)`.
        k (int): The "k" in "top-k".

    Returns:
        The confidence defined as the maximum probability assigned to a class, i.e. a vector of
        dimensions `(B: batch_size, k)`.
    """
    probs = logits.softmax(dim=-1)
    return torch.topk(probs, k=k, sorted=True, dim=-1).values


def expected_confidence(logits: Tensor, k: int = 1) -> Tensor:
    r"""Computes the expected confidence based on logits.

    Computes the expected confidence across samples, defined as the highest probability the model assigns
    to a class, that is

    $$\sum_s \max_c p_{bcs}$$

    where $p_{bcs}$ is the probability for class $c$ for instance $b$ in batch of sample $s$.

    Args:
        logits (Tensor): A tensor of dimensions `(B: batch_size, C: num_classes, S: num_samples)`.
        k (int): The "k" in "top-k".

    Returns:
        The confidence defined as the maximum probability assigned to a class, i.e. a vector of
        dimensions `(B: batch_size, k)`.
    """
    probs = logits.softmax(dim=-2)
    confidence = torch.topk(probs, k=k, sorted=True, dim=-2).values
    return confidence.mean(dim=-1)


def least_confidence(logits: Tensor) -> Tensor:
    r"""Implements the least confidence acquisition function.

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
    return 1.0 - confidence(logits, k=1).flatten()


def expected_least_confidence(logits: Tensor, k: int = 1) -> Tensor:
    r"""Implements the least confidence acquisition function.

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
    return 1.0 - expected_confidence(logits, k=k).flatten()


def margin_confidence(logits: Tensor) -> Tensor:
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
    confidence_top2 = confidence(logits, k=2)
    # we want the instances with the smallest gap, so we need to negate
    return -(confidence_top2[:, 0] - confidence_top2[:, 1]).flatten()


def expected_margin_confidence(logits: Tensor):
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
    confidence_top2 = expected_confidence(logits, k=2)
    return -(confidence_top2[:, 0] - confidence_top2[:, 1]).flatten()
