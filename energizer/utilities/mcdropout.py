import contextlib
import random
from itertools import cycle
from typing import Generator, Optional

import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.seed import _collect_rng_states, _set_rng_states
from torch import Tensor, nn
from torch.nn.modules.dropout import _DropoutNd


@contextlib.contextmanager
def local_seed(seed: int) -> Generator[None, None, None]:
    """A context manager that allows to locally change the seed.

    Upon exit from the context manager it resets the random number generator state
    so that the operations that happen in the context do not affect randomness outside
    of it.
    """
    # collect current states
    states = _collect_rng_states()

    # set seed in the context
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # run code in context
    yield

    # reset states when exiting the context
    _set_rng_states(states)


class EnergizerDropoutLayer(_DropoutNd):
    """Base class for dropout layers that remains active even during evaluation.

    This class is used as a base to redefine all dropout layers present in Pytorch such
    that they remain always on even during evaluation. The elements to zero are randomized
    on every forward call during training time.

    This is obtained by overloading the original forward method of the class such that

    ```python
    # this
    F.dropout(input=input, p=self.p, training=self.training, inplace=self.inplace)

    # is turned into
    F.dropout(input=input, p=self.p, training=True, inplace=self.inplace)
    ```

    that is, the `training` argument is always set to `True`.

    Check the original documentation here: https://pytorch.org/docs/stable/nn.html#dropout-layers.

    Args:
        p (float): probability of an element to be zeroed. Default: 0.5
        inplace (bool): If set to True, will do this operation in-place.
    """

    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError


class MCDropout(EnergizerDropoutLayer):
    """Patch of `torch.nn.Dropout`."""

    def forward(self, input: Tensor) -> Tensor:
        """Calls the respective functional version with the `training` argument set to `True`."""
        return F.dropout(input=input, p=self.p, training=True, inplace=self.inplace)


class MCDropout2d(EnergizerDropoutLayer):
    """Patch of `torch.nn.Dropout2d`."""

    def forward(self, input: Tensor) -> Tensor:
        """Calls the respective functional version with the `training` argument set to `True`."""
        return F.dropout2d(input=input, p=self.p, training=True, inplace=self.inplace)


class MCDropout3d(EnergizerDropoutLayer):
    """Patch of `torch.nn.Dropout3d`."""

    def forward(self, input: Tensor) -> Tensor:
        """Calls the respective functional version with the `training` argument set to `True`."""
        return F.dropout3d(input=input, p=self.p, training=True, inplace=self.inplace)


class MCAlphaDropout(EnergizerDropoutLayer):
    """Patch of `torch.nn.AlphaDropout`."""

    def forward(self, input: Tensor) -> Tensor:
        """Calls the respective functional version with the `training` argument set to `True`."""
        return F.alpha_dropout(input=input, p=self.p, training=True)


class MCFeatureAlphaDropout(EnergizerDropoutLayer):
    """Patch of `torch.nn.FeatureAlphaDropout`."""

    def forward(self, input: Tensor) -> Tensor:
        """Calls the respective functional version with the `training` argument set to `True`."""
        return F.feature_alpha_dropout(input=input, p=self.p, training=True)


def replace_dropout_layers(module: nn.Module, prob: Optional[float] = None) -> None:
    """Replace dropout layers in a model with MCDropout layers.

    Args:
        module (nn.Module): The module in which dropout layers should be replaced.
        prob (float): If specified, this changes the dropout probability of all layers to `prob`. If `None` the dropout
            probability is the same as the original layer. Must be 0 < prob < 1.
        consistent (bool): If True, it uses the consistent version of dropout that fixes the mask across batches.

    Raises:
        MisconfigurationException if no layer is modified.

    Returns:
        A patched nn.Module which is either the same object passed in (if inplace = True) or a copy of that object.
    """
    if (prob is not None) and (prob < 0 or prob > 1):
        raise ValueError(f"Dropout probability must be 0 <= prob <= 1, not {prob}.")

    changed = _patch_dropout(module=module, prob=prob)
    if not changed:
        raise MisconfigurationException("The model should contain at least one dropout layer.")


def _patch_dropout(module: nn.Module, prob: Optional[float] = None) -> bool:
    """Recursively iterate over the children of a module and replace the dropout layers.

    This function operates in-place.

    Args:
        module (nn.Module): The module to patch.
        prob (float): If specified, this changes the dropout probability of all layers.
        consistent (bool): If True, it uses the consistent version of dropout that fixes the mask across batches.

    Keyword Args:
        num_inference_iters (int): Number of masks to sample.

    Returns:
        Flag indicating if at least a layer is modified.
    """
    changed = False
    for name, child in module.named_children():
        new_module: Optional[nn.Module] = None

        for dropout_layer in (nn.Dropout, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout):
            if isinstance(child, dropout_layer):

                # NOTE: `eval(dropout_layer.__name__)` works because the __name__ of the original dropout
                # layers is the same as the __name__ of the patched layers defined above
                new_module = eval(f"MC{dropout_layer.__name__}")(p=prob if prob else child.p, inplace=child.inplace)

        if new_module is not None:
            changed = True
            module.add_module(name, new_module)

        # recursively apply to child
        changed |= _patch_dropout(module=child, prob=prob)
    return changed


def replace_energizer_dropout_layers(module: nn.Module, prob: Optional[float] = None) -> None:
    """Replace Energizer dropout layers in a model with normal torch Dropout layers.

    Args:
        module (nn.Module): The module in which dropout layers should be replaced.
        prob (float): If specified, this changes the dropout probability of all layers to `prob`. If `None` the dropout
            probability is the same as the original layer. Must be 0 < prob < 1.

    Raises:
        MisconfigurationException if no layer is modified.

    Returns:
        A patched nn.Module which is either the same object passed in (if inplace = True) or a copy of that object.
    """
    if (prob is not None) and (prob < 0 or prob > 1):
        raise ValueError(f"Dropout probability must be 0 <= prob <= 1, not {prob}.")

    changed = _patch_energizer_dropout(module=module, prob=prob)
    if not changed:
        raise MisconfigurationException("The model should contain at least one dropout layer.")


def _patch_energizer_dropout(module: nn.Module, prob: Optional[float] = None) -> bool:
    """Recursively iterate over the children of a module and replace the dropout layers.

    This function operates in-place.

    Args:
        module (nn.Module): The module to patch.
        prob (float): If specified, this changes the dropout probability of all layers.

    Returns:
        Flag indicating if at least a layer is modified.
    """
    changed = False
    for name, child in module.named_children():
        new_module: Optional[nn.Module] = None

        for dropout_layer in (MCDropout, MCDropout2d, MCDropout3d, MCAlphaDropout, MCFeatureAlphaDropout):
            if isinstance(child, dropout_layer):

                new_module = eval(f"nn.{dropout_layer.__name__.lstrip('MC')}")(p=prob if prob else child.p)

        if new_module is not None:
            changed = True
            module.add_module(name, new_module)

        # recursively apply to child
        changed |= _patch_energizer_dropout(module=child, prob=prob)
    return changed
