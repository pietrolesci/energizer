import datasets
import pytest
from torch.utils.data import DataLoader
from transformers import default_data_collator

from tests.utils import BoringModel, RandomSupervisedDataset


@pytest.fixture
def mock_dataset():
    return RandomSupervisedDataset()


@pytest.fixture
def mock_hf_dataset():
    dataset = RandomSupervisedDataset()
    return datasets.Dataset.from_dict({"inputs": dataset.x, "labels": dataset.y})


@pytest.fixture
def dataset_arg(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def mock_dataloader(mock_dataset):
    return DataLoader(mock_dataset, batch_size=10)


@pytest.fixture
def mock_hf_dataloader(mock_hf_dataset):
    return DataLoader(mock_hf_dataset, batch_size=10, collate_fn=default_data_collator)


@pytest.fixture
def dataloader_arg(request):
    return request.getfixturevalue(request.param)


@pytest.fixture
def boring_model():
    return BoringModel
