import pytest

from energizer.datastores.datamodule import ActiveDataModule


def test_len(dataloader):
    """Test that measures of length are consistent."""
    # no instances
    dm = ActiveDataModule(train_dataloader=dataloader)
    assert len(dataloader.dataset) == dm.train_size + dm.pool_size
    assert len(dataloader.dataset) == dm.total_data_size
    pool_size = dm.pool_size

    dm.label(0)
    assert len(dataloader.dataset) == dm.train_size + dm.pool_size
    assert len(dataloader.dataset) == dm.total_data_size
    assert dm.train_size == 1
    assert dm.pool_size == pool_size - 1


def test_indexing(dataloader):
    """Test that ActiveDataModule is not indexable directly."""
    dm = ActiveDataModule(train_dataloader=dataloader)
    with pytest.raises(TypeError):
        assert dm[0]


def test_labelling(dataloader):
    """Test that labelling changes all the required states."""
    dm = ActiveDataModule(train_dataloader=dataloader)
    len_dataset = len(dataloader.dataset)

    with pytest.raises(AssertionError):
        dm.label("0")

    with pytest.raises(ValueError):
        dm.label(list(range(len_dataset + 100)))

    with pytest.raises(ValueError):
        dm.label(int(1e6))

    assert dm.last_labelling_step == 0
    assert dm.train_size == 0
    assert dm.pool_size == len_dataset
    assert dm.has_labelled_data is False
    assert dm.has_unlabelled_data is True
    assert dm.train_dataset.indices == []

    for i in range(1, len_dataset + 1):
        dm.label(0)  # always label the first instance in the pool

        assert dm.last_labelling_step == i
        assert dm.train_size == i
        assert dm.pool_size == len_dataset - dm.train_size
        assert dm.has_labelled_data is True
        if i < len_dataset:
            assert dm.has_unlabelled_data is True
        else:
            assert dm.has_unlabelled_data is False
        assert dm.train_dataset.indices == list(range(i))

    assert dm.last_labelling_step == len_dataset
    assert dm.train_size == len_dataset
    assert dm.pool_size == len_dataset - dm.train_size
    assert dm.has_labelled_data is True
    assert dm.has_unlabelled_data is False
    assert dm.train_dataset.indices == list(range(len_dataset))


def test_labelling_multiple_indices(dataloader):
    """Test labelling multiple instances at once."""
    dm = ActiveDataModule(train_dataloader=dataloader)
    pool_ids = [0, 8, 7]  # they are the first to be labelled so correspond to ids in oracle
    dm.label(pool_ids)

    assert dm.train_dataset.indices == sorted(pool_ids)


def test_labelling_duplicates(dataloader):
    """Test that labelling duplicate indices results in a single instance to be labelled."""

    # check behaviour when batch of indices contains
    dm = ActiveDataModule(train_dataloader=dataloader)
    pool_ids = [0, 0]  # they are the first to be labelled so correspond to ids in oracle
    dm.label(pool_ids)
    assert dm.train_size == 1


def test_reset_at_labelling_step(dataloader):
    """Test that resetting the labelling steps sets the correct states."""
    dm = ActiveDataModule(train_dataloader=dataloader)
    len_dataset = len(dataloader.dataset)

    dm.label(0)  # label first
    assert dm.last_labelling_step == 1
    assert dm.train_size == 1
    assert dm.pool_size == len_dataset - dm.train_size
    assert dm.has_labelled_data is True
    assert dm.has_unlabelled_data is True
    assert dm.train_dataset.indices == [0]

    dm.label(list(range(len_dataset - 1)))  # label the rest
    assert dm.train_size == len_dataset
    assert dm.pool_size == len_dataset - dm.train_size
    assert dm.has_labelled_data is True
    assert dm.has_unlabelled_data is False
    assert dm.train_dataset.indices == list(range(len_dataset))

    dm.reset_at_labelling_step(1)  # go back to when there was one instance
    assert dm.train_size == 1
    assert dm.pool_size == len_dataset - dm.train_size
    assert dm.has_labelled_data is True
    assert dm.has_unlabelled_data is True
    assert dm.train_dataset.indices == [0]

    dm.reset_at_labelling_step(0)  # go back to when there was nothing labelled
    assert dm.last_labelling_step == 2
    assert dm.train_size == 0
    assert dm.pool_size == len_dataset - dm.train_size
    assert dm.has_labelled_data is False
    assert dm.has_unlabelled_data is True
    assert dm.train_dataset.indices == []

    dm.reset_at_labelling_step(dm.last_labelling_step)  # reset to the last step
    assert dm.train_size == len_dataset
    assert dm.pool_size == len_dataset - dm.train_size
    assert dm.has_labelled_data is True
    assert dm.has_unlabelled_data is False
    assert dm.train_dataset.indices == list(range(len_dataset))

    with pytest.raises(ValueError):
        assert dm.reset_at_labelling_step(100)
