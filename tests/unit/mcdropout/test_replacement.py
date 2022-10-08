from typing import List, Tuple

import pytest
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch import nn

from energizer.utilities.mcdropout import replace_dropout_layers, replace_energizer_dropout_layers


def get_dropout_modules(module: nn.Module) -> List[Tuple[str, float]]:
    return [
        (m.__class__.__name__, m.p) for _, m in module.named_modules() if isinstance(m, nn.modules.dropout._DropoutNd)
    ]


def test_replace_dropout(dropout_module):
    """Checks that replacement of layers works.

    In particular, it checks that the layer type is change but the original
    probabilty for each layer is preserved.
    """

    original_target = get_dropout_modules(dropout_module)
    new_target = [(f"MC{m}", p) for (m, p) in original_target]

    # silly check to see whether it works
    assert get_dropout_modules(dropout_module) == original_target

    # now replace torch dropout with energizer dropout
    replace_dropout_layers(dropout_module)
    assert get_dropout_modules(dropout_module) == new_target

    # now replace energizer dropout with torch dropout
    replace_energizer_dropout_layers(dropout_module)
    assert get_dropout_modules(dropout_module) == original_target

    # change probability
    replace_dropout_layers(dropout_module, 0.9)
    assert get_dropout_modules(dropout_module) == [(m, 0.9) for (m, _) in new_target]

    # change probability
    replace_energizer_dropout_layers(dropout_module, 0.9)
    assert get_dropout_modules(dropout_module) == [(m, 0.9) for (m, _) in original_target]


def test_consistent_inputs(dropout_module):

    # checks inputs
    with pytest.raises(MisconfigurationException):
        replace_dropout_layers(dropout_module, consistent=True)

    with pytest.raises(AssertionError):
        replace_dropout_layers(dropout_module, consistent=True, num_inference_iters=10, seeds=[0])


def test_consistent_dropout(dropout_module):
    replace_dropout_layers(dropout_module, consistent=True, num_inference_iters=10)

    # outputs =
