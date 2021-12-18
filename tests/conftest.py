#!/usr/bin/env python
"""Tests for `energizer` package."""

import datasets
import pytest
from torch.utils.data import Dataset


@pytest.fixture
def mock_dataset():
    class PytorchDataset(Dataset):
        def __init__(self):
            super().__init__()
            self.dataset = [(f"instance_{i}", f"label_{i}") for i in range(10)]

        def __len__(self):
            return len(self.dataset)

        def __getitem__(self, index):
            return self.dataset[index]

    return PytorchDataset()


@pytest.fixture
def mock_hf_dataset():
    data = [(f"instance_{i}", f"label_{i}") for i in range(10)]
    dataset = datasets.Dataset.from_dict({"input": [d[0] for d in data], "target": [d[1] for d in data]})

    return dataset
