"""
This module is based on the following implementations:

    - Baal library with [LICENSE](https://github.com/ElementAI/baal/blob/master/LICENSE):
        - https://github.com/ElementAI/baal/blob/master/baal/bayesian/dropout.py
        - https://github.com/ElementAI/baal/blob/master/baal/bayesian/consistent_dropout.py

    - BatchBALD_Redux with [LICENCE](https://github.com/BlackHC/batchbald_redux/blob/master/LICENSE):
        - https://github.com/BlackHC/batchbald_redux/blob/master/batchbald_redux/consistent_mc_dropout.py
"""

import copy
from typing import Optional

import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn
from torch.nn.modules.dropout import _DropoutNd


class Dropout(_DropoutNd):
    """Dropout layer that remains always on, even during evaluation."""

    def forward(self, input):
        return F.dropout(input, self.p, True, self.inplace)


class Dropout2d(_DropoutNd):
    """Dropout2d layer that remains always on, even during evaluation."""

    def forward(self, input):
        return F.dropout2d(input, self.p, True, self.inplace)


class ConsistentDropout(_DropoutNd):
    """Dropout that guarantees that masks are the same across batches.

    Dropout layer that uses consistent masks for inference. That means, that we draw K
    masks and then keep them fixed while drawing the K inference samples for each input
    in the evaluation set. During training, masks are redrawn for every sample.

    It is useful when doing research as it guarantees that while the masks are the
    same across batches during inference. The masks are different within the batch.

    This is slower than using regular Dropout, but it is useful when you want to use
    the same set of weights for each sample used in inference.

    From BatchBALD (Kirsch et al, 2019), this is necessary to use BatchBALD and remove noise
    from the prediction.

    Args:
        p (float): probability of an element to be zeroed. Default: 0.5

    Notes:
        For optimal results, you should use a batch size of one during inference time.
        Furthermore, to guarantee that each sample uses the same set of weights, you must use
        `replicate_in_memory=True` in ModelWrapper,
        which is the default.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__(p=p, inplace=inplace)
        self.reset()

    def forward(self, x):
        if self.training:
            return F.dropout(x, self.p, training=True, inplace=False)
        else:
            if self._mask is None or self._mask.shape != x.shape:
                self._mask = self._make_mask(x)
            return torch.mul(x, self._mask)

    def _make_mask(self, x):
        return F.dropout(torch.ones_like(x, device=x.device), self.p, training=True)

    def reset(self):
        self._mask = None

    def eval(self):
        self.reset()
        return super().eval()


class ConsistentDropout2d(_DropoutNd):
    """Dropout that guarantees that masks are the same across batches.

    It is useful when doing research as it guarantees that while the masks are the
    same across batches during inference. The masks are different within the batch.

    This is slower than using regular Dropout, but it is useful when you want to use
    the same set of weights for each sample used in inference.

    From BatchBALD (Kirsch et al, 2019), this is necessary to use BatchBALD and remove noise
    from the prediction.

    Args:
        p (float): probability of an element to be zeroed. Default: 0.5

    Notes:
        For optimal results, you should use a batch size of one during inference time.
        Furthermore, to guarantee that each sample uses the same set of weights, you must
        use `replicate_in_memory=True` in ModelWrapper,
        which is the default.
    """

    def __init__(self, p: float = 0.5, inplace: bool = False):
        super().__init__(p=p, inplace=inplace)
        self.reset()

    def forward(self, x):
        if self.training:
            return F.dropout2d(x, self.p, training=True, inplace=False)
        else:
            if self._mask is None or self._mask.shape != x.shape:
                self._mask = self._make_mask(x)
            return torch.mul(x, self._mask)

    def _make_mask(self, x):
        return F.dropout2d(torch.ones_like(x, device=x.device), self.p, training=True)

    def reset(self):
        self._mask = None

    def eval(self):
        self.reset()
        return super().eval()


def patch_module(module: nn.Module, inplace: bool = True) -> nn.Module:
    """Replace dropout layers in a model with MCDropout layers.

    Args:
        module (nn.Module):
            The module in which you would like to replace dropout layers.
        inplace (bool, optional):
            Whether to modify the module in place or return a copy of the module.

    Raises:
        UserWarning if no layer is modified.

    Returns:
        nn.Module
            The modified module, which is either the same object as you passed in
            (if inplace = True) or a copy of that object.
    """
    if not inplace:
        module = copy.deepcopy(module)
    changed = patch_dropout_layers(module)
    if not changed:
        raise MisconfigurationException("The model should contain at least 1 dropout layer.")
    return module


def patch_dropout_layers(module: nn.Module, consistent: bool = False, prob: Optional[float] = None) -> bool:
    """Recursively iterate over the children of a module and replace the dropout layers.

    Recursively replace dropout layers in a `torch.nn.Module` with the custom `Dropout` or `ConsistentDropout`
    modules needed to run MCDropout, i.e. to keep subsampling active even during evaluation.
    This function operates in-place.

    Args:
        module (nn.Module): The module to patch.
        consistent (bool): Whether to use a consistent version of dropout that keeps the mask consistent
            across batches (note that the mask is different across samples within a batch).
        prob (float): If specified, this changes the dropout probability of all layers.

    Returns:
        Flag indicating if a layer was modified.
    """
    if prob and (prob < 0.0 or prob > 1.0):
        raise ValueError(f"Dropout probability must be 0 < prob < 1, not {prob}.")

    changed = False
    for name, child in module.named_children():
        new_module: Optional[nn.Module] = None
        if isinstance(child, nn.Dropout):
            if consistent:
                new_module = ConsistentDropout(p=prob if prob else child.p, inplace=child.inplace)
            else:
                new_module = Dropout(p=prob if prob else child.p, inplace=child.inplace)

        elif isinstance(child, nn.Dropout2d):
            if consistent:
                new_module = ConsistentDropout2d(p=prob if prob else child.p, inplace=child.inplace)
            else:
                new_module = Dropout2d(p=prob if prob else child.p, inplace=child.inplace)

        if new_module is not None:
            changed = True
            module.add_module(name, new_module)

        # recursively apply to child
        changed |= patch_dropout_layers(child)
    return changed
