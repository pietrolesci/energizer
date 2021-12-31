import pytest
import torch
from pytorch_lightning import seed_everything

from energizer.inference.inference_modules import Deterministic, MCDropout
from tests.utils import NUM_CLASSES, NUM_FEATURES


@pytest.mark.parametrize("dataloader_arg", ["mock_dataloader", "mock_hf_dataloader"], indirect=True)
def test_deterministic_forward(dataloader_arg, boring_model):
    """Test if connect works and output of patched model is the same as the original."""
    model = boring_model(torch.nn.Linear(NUM_FEATURES, NUM_FEATURES))
    module = Deterministic()

    # check connect
    assert module.module is None
    module.connect(model)
    assert module.module is not None

    # check same outputs of the original model
    for batch in dataloader_arg:
        if isinstance(batch, dict):
            batch = (batch["inputs"], batch["labels"])
        inputs, _ = batch
        assert torch.all(module(inputs) == model(inputs))


@pytest.mark.parametrize("dataloader_arg", ["mock_dataloader", "mock_hf_dataloader"], indirect=True)
def test_mcdropout_forward(dataloader_arg, boring_model):
    """Test if connect works, output has the correct shape, and output is always different."""
    model = boring_model(torch.nn.Sequential(torch.nn.Linear(NUM_FEATURES, NUM_FEATURES), torch.nn.Dropout()))
    module = MCDropout(num_inference_iters=10)

    # check connect
    assert module.module is None
    module.connect(model)
    assert module.module is not None

    # check same outputs of the original model
    for batch in dataloader_arg:
        if isinstance(batch, dict):
            batch = (batch["inputs"], batch["labels"])
        inputs, _ = batch
        assert module(inputs).shape == (inputs.shape[0], NUM_CLASSES, 10)

    # check output is stochastic even when manually set to eval
    seed_everything(42)
    batch = next(iter(dataloader_arg))
    if isinstance(batch, dict):
        batch = (batch["inputs"], batch["labels"])
    inputs, _ = batch

    module.eval()
    assert not all((module(inputs) == module(inputs)).all() for _ in range(10))


@pytest.mark.parametrize("module", (Deterministic, MCDropout))
def test_no_copy(module, boring_model):
    """Test that when model is connected no copy is performed."""
    model = boring_model(torch.nn.Sequential(torch.nn.Linear(NUM_FEATURES, NUM_FEATURES), torch.nn.Dropout()))

    inference_module = module()
    inference_module.connect(model)

    # check parameters are the same
    for (_, param1), (_, param2) in zip(inference_module.named_parameters(), model.named_parameters()):
        assert param1 is param2
