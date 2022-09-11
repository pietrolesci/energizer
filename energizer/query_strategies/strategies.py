from typing import List, Tuple

import numpy as np
from torch import Tensor

from energizer.acquisition_functions import (
    bald,
    entropy,
    expected_entropy,
    expected_least_confidence,
    expected_margin_confidence,
    least_confidence,
    margin_confidence,
    predictive_entropy,
)
from energizer.query_strategies.base import AccumulatorStrategy, MCAccumulatorStrategy, NoAccumulatorStrategy
from energizer.utilities.types import MODEL_INPUT

"""
NoAccumulatorStrategy's
"""


class RandomStrategy(NoAccumulatorStrategy):
    """Query random instances from the pool.
    
    !!! note "Naming conventions"
        
        In the literature it is sometimes referred to as "Uniform" strategy.
    """
    def query(self) -> List[int]:
        pool_size = self.trainer.datamodule.pool_size
        return np.random.randint(low=0, high=pool_size, size=self.query_size).tolist()


"""
AccumulatorStrategy's
"""


class EntropyStrategy(AccumulatorStrategy):
    r"""Query instances with the highest predictive entropy.

    This strategy selects instances that maximize the predictive entropy 
    $$ x_H = \underset{x \in D_{pool}}{\arg\max} \; − \sum_{k=1}^K \; p_\theta(y_k \mid x) \; \log p_\theta(y_k  \mid x) $$
    
    where $p_\theta(y_k \mid x)$ is the posterior probability (i.e., the $k$-th
    softmax-ed logit) of class $k$ according to the classifier parametrized by
    the parameters $\theta$.

    It implements the entropy calculation using the [`entropy`][energizer.acquisition_functions.entropy] 
    function. Look the for more details on how it is implemented.

    !!! note "Naming conventions"
        
        In the literature it is sometimes referred to as "Max-Entropy" strategy.
    """
    def pool_step(self, batch: MODEL_INPUT, batch_idx: int, *args, **kwargs) -> Tensor:
        logits = self(batch)
        return entropy(logits)


class LeastConfidenceStrategy(AccumulatorStrategy):
    def pool_step(self, batch: MODEL_INPUT, batch_idx: int, *args, **kwargs) -> Tensor:
        logits = self(batch)
        return least_confidence(logits)


class MarginStrategy(AccumulatorStrategy):
    r"""Queries instances whose two most likely labels have the smallest difference. 
    
    In particular, once we collect the posterior probabilities according to the model
    with parameters $\theta$ (i.e., the softmax-ed logit), we then compute the difference
    between the two highest, and use this difference as a measure of uncertainty. In math,

    $$ x_H = \underset{x \in D_{pool}}{\arg\max} \; p_\theta(\hat{y}_1) - p_\theta(\hat{y}_2) $$

    where $\hat{y}_1$ and $\hat{y}_2$ are the first and second most probable class
    according to the model, respectively.

    Intuitively, if the classifiers is certain about a prediction, it would assign the majority of the probability mass to a specific class.

    It implements the entropy calculation using the [`margin_confidence`][energizer.acquisition_functions.margin_confidence] 
    function. Look the for more details on how it is implemented.
    """

    def pool_step(self, batch: MODEL_INPUT, batch_idx: int, *args, **kwargs) -> Tensor:
        logits = self(batch)
        return margin_confidence(logits)


"""
MCAccumulatorStrategy's
"""


class ExpectedEntropyStrategy(MCAccumulatorStrategy):
    def pool_step(self, batch: MODEL_INPUT, batch_idx: int, *args, **kwargs) -> Tensor:
        logits = self(batch)
        return expected_entropy(logits)


class PredictiveEntropyStrategy(MCAccumulatorStrategy):
    def pool_step(self, batch: MODEL_INPUT, batch_idx: int, *args, **kwargs) -> Tensor:
        logits = self(batch)
        return predictive_entropy(logits)


class ExpectedLeastConfidenceStrategy(MCAccumulatorStrategy):
    def pool_step(self, batch: MODEL_INPUT, batch_idx: int, *args, **kwargs) -> Tensor:
        logits = self(batch)
        return expected_least_confidence(logits)


class ExpectedMarginStrategy(MCAccumulatorStrategy):
    def pool_step(self, batch: MODEL_INPUT, batch_idx: int, *args, **kwargs) -> Tensor:
        logits = self(batch)
        return expected_margin_confidence(logits)


class BALDStrategy(MCAccumulatorStrategy):
    def pool_step(self, batch: MODEL_INPUT, batch_idx: int, *args, **kwargs) -> Tensor:
        logits = self(batch)
        return bald(logits)
