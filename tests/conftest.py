#!/usr/bin/env python
"""Tests for `energizer` package."""

import datasets
import numpy as np
import pytest
from torch.utils.data import Dataset


@pytest.fixture
def mock_dataset():
    class PytorchDataset(Dataset):
        def __init__(self):
            super().__init__()
            self.dataset = list(zip(range(10), np.random.randint(0, 1, 10)))

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            return self.dataset[index]

    return PytorchDataset()


@pytest.fixture
def mock_hf_dataset():
    data = list(zip(range(10), np.random.randint(0, 1, 10)))
    dataset = datasets.Dataset.from_dict({"input": [d[0] for d in data], "target": [d[1] for d in data]})

    return dataset


@pytest.fixture
def dataset_arg(request):
    return request.getfixturevalue(request.param)
