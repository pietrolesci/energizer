import pytest
import torch
from pytorch_lightning.utilities.exceptions import MisconfigurationException

from energizer.inference.utils import (
    AlphaDropout,
    Dropout,
    Dropout2d,
    Dropout3d,
    FeatureAlphaDropout,
    local_seed,
    patch_dropout_layers,
)
from tests.utils import NUM_FEATURES


def test_patch_exceptions(boring_model):
    """Test if errors are raised on inputs."""
    model = boring_model(torch.nn.Linear(NUM_FEATURES, NUM_FEATURES))
    with pytest.raises(MisconfigurationException):
        patch_dropout_layers(model)

    for i in (-0.01, 1.01):
        with pytest.raises(ValueError):
            patch_dropout_layers(model, prob=i)


@pytest.mark.parametrize("inplace", (True, False))
def test_patch_inplace(boring_model, inplace):
    """Test if inplace works and patch replaces all dropout layers."""
    model = boring_model(torch.nn.Sequential(torch.nn.Linear(NUM_FEATURES, NUM_FEATURES), torch.nn.Dropout()))
    patched_module = patch_dropout_layers(model, inplace=inplace)
    if inplace:
        assert patched_module is model
    else:
        assert patched_module is not model
    assert not any(isinstance(module, torch.nn.Dropout) for module in patched_module.modules())
    assert any(isinstance(module, Dropout) for module in patched_module.modules())


@pytest.mark.parametrize(
    "dropout_cls",
    [
        AlphaDropout,
        Dropout,
        Dropout2d,
        Dropout3d,
        FeatureAlphaDropout,
    ],
)
def test_consistent_dropout(dropout_cls):
    """Test that the consistent mechanism works."""
    t = torch.ones((1, 10, 10))
    NUM_INFERENCE_ITERS = 10
    dropout_layer = dropout_cls(prob=0.5, consistent=True, num_inference_iters=3)
    dropout_layer.eval()  # put layer in eval so that it uses the consistent mechanism

    for NUM_INFERENCE_ITERS in (10, 20):
        dropout_layer.reset_mask(NUM_INFERENCE_ITERS)  # reset with different number of iters
        a_seeds = [next(dropout_layer.seeds) for _ in range(NUM_INFERENCE_ITERS)]
        a = torch.cat([dropout_layer(t) for _ in range(NUM_INFERENCE_ITERS)])
        b_seeds = [next(dropout_layer.seeds) for _ in range(NUM_INFERENCE_ITERS)]
        b = torch.cat([dropout_layer(t) for _ in range(NUM_INFERENCE_ITERS)])
        dropout_layer.reset_mask()  # reset with same number of iters
        c_seeds = [next(dropout_layer.seeds) for _ in range(NUM_INFERENCE_ITERS)]
        c = torch.cat([dropout_layer(t) for _ in range(NUM_INFERENCE_ITERS)])

        assert a_seeds == b_seeds
        assert a_seeds != c_seeds
        assert torch.all(a == b)
        assert torch.any(a != c)


@pytest.mark.parametrize(
    "dropout_cls",
    [
        AlphaDropout,
        Dropout,
        Dropout2d,
        Dropout3d,
        FeatureAlphaDropout,
    ],
)
def test_dropout_errors(dropout_cls):
    with pytest.raises(MisconfigurationException):
        dropout_cls(consistent=True)


def test_local_seed_context():
    seed = torch.initial_seed()
    state = torch.get_rng_state()
    with local_seed(56):
        [torch.rand((1,)) for _ in range(10)]

    new_seed = torch.initial_seed()
    new_state = torch.get_rng_state()

    assert seed == new_seed
    assert torch.all(state == new_state)
