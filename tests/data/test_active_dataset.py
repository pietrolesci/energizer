import pytest

from energizer.data import ActiveDataset


def test_len(mock_dataset):
    ads = ActiveDataset(mock_dataset)
    assert len(ads) == ads.labelled_size
    assert len(mock_dataset) == ads.pool_size

    ads.label(0)
    assert len(ads) == ads.labelled_size
    assert len(mock_dataset) == ads.labelled_size + ads.pool_size


def test_reset_at_labelling_step(mock_dataset):
    ads = ActiveDataset(mock_dataset)
    assert ads.last_labelling_step == 0
    assert ads.labelled_size == 0
    assert ads.has_labelled_data is False
    assert ads.has_unlabelled_data is True

    ads.label(0)
    assert ads.last_labelling_step == 1
    assert ads.labelled_dataset.indices == [0]
    assert ads.has_labelled_data is True

    ads.label(0)
    assert ads.last_labelling_step == 2
    assert ads.labelled_dataset.indices == [0, 1]
    assert ads.has_labelled_data is True

    ads.reset_at_labelling_step(1)
    assert ads.last_labelling_step == 2
    assert ads.labelled_dataset.indices == [0]
    assert ads.has_labelled_data is True

    ads.reset_at_labelling_step(0)
    assert ads.last_labelling_step == 2
    assert ads.labelled_dataset.indices == []
    assert ads.has_labelled_data is False

    ads.reset_at_labelling_step(ads.last_labelling_step)
    assert ads.last_labelling_step == 2
    assert ads.labelled_dataset.indices == [0, 1]
    assert ads.has_labelled_data is True

    ads.label(list(range(ads.pool_size)))
    assert ads.last_labelling_step == 3
    assert ads.labelled_dataset.indices == list(range(len(mock_dataset)))
    assert ads.has_labelled_data is True
    assert ads.has_unlabelled_data is False

    with pytest.raises(ValueError):
        assert ads.reset_at_labelling_step(100)


def test_label(mock_dataset):
    ads = ActiveDataset(mock_dataset)

    assert ads._pool_to_oracle(0) == [0]
    ads.label(0)
    assert ads.labelled_dataset.indices == [0]

    assert ads._pool_to_oracle(1) == [2]
    ads.label(1)  # relative to the pool this is 2
    assert ads.labelled_dataset.indices == [0, 2]

    assert ads._pool_to_oracle([0, 0]) == [1]
    ads.label([0, 0])
    assert ads.labelled_dataset.indices == [0, 1, 2]

    with pytest.raises(ValueError):
        assert ads.label(list(range(ads.pool_size + 10)))

    with pytest.raises(ValueError):
        assert ads.label(ads.pool_size + 10)


def test_dataset_indexing(mock_dataset):
    ads = ActiveDataset(mock_dataset)
    ads.label([0, 1, 2])

    for i in [0, 1, 2]:
        assert ads[i] == mock_dataset[i]

    assert ads[[0, 1]] == [mock_dataset[i] for i in [0, 1]]


def test_hf_dataset_indexing(mock_hf_dataset):
    ads = ActiveDataset(mock_hf_dataset)
    ads.label([0, 1, 2])

    for i in [0, 1, 2]:
        assert ads[i] == mock_hf_dataset[i]

    assert ads[[0, 1]] == mock_hf_dataset[[0, 1]]


def test_sample(mock_dataset):
    ads = ActiveDataset(mock_dataset)

    with pytest.raises(ValueError):
        assert ads.sample_pool_idx(-1)

    with pytest.raises(ValueError):
        assert ads.sample_pool_idx(0)

    with pytest.raises(ValueError):
        assert ads.sample_pool_idx(ads.pool_size + 1)

    assert len(ads.sample_pool_idx(ads.pool_size)) == ads.pool_size
    assert len(ads.sample_pool_idx(1)) == 1
