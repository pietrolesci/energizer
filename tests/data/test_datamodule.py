import pytest
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data.sampler import SequentialSampler

from energizer.data import ActiveDataModule
from energizer.data.datamodule import FixedLengthSampler


@pytest.mark.parametrize("dataset_arg", ["mock_dataset", "mock_hf_dataset"], indirect=True)
def test_len(dataset_arg):
    """Test that measures of length are consistent."""

    # no instances
    ads = ActiveDataModule(num_classes=2, train_dataset=dataset_arg)
    ads.prepare_data()  # useless: just pass but for coverage
    ads.setup()  # useless: just pass but for coverage
    assert ads.total_labelled_size == ads.train_size + ads.val_size
    assert len(ads.train_dataset) == ads.train_size == ads.val_size == ads.total_labelled_size == 0
    assert len(dataset_arg) == len(ads.pool_dataset) == ads.pool_size
    assert len(dataset_arg) == ads.total_labelled_size + ads.pool_size

    # one instance in the train dataset
    ads.label(0)
    assert ads.total_labelled_size == ads.train_size + ads.val_size
    assert len(ads.train_dataset) == ads.train_size == ads.total_labelled_size == 1
    assert ads.val_dataset is None
    assert len(dataset_arg) - ads.total_labelled_size == len(ads.pool_dataset) == ads.pool_size
    assert len(dataset_arg) == ads.total_labelled_size + ads.pool_size

    # one instance in the train dataset and one in the val dataset
    ads.val_split = 0.5  # hack
    ads.label([0, 1])
    assert ads.total_labelled_size == ads.train_size + ads.val_size
    assert len(ads.train_dataset) == ads.train_size == 2
    assert len(ads.val_dataset) == ads.val_size == 1
    assert len(dataset_arg) - ads.total_labelled_size == len(ads.pool_dataset) == ads.pool_size
    assert len(dataset_arg) == ads.total_labelled_size + ads.pool_size


@pytest.mark.parametrize("dataset_arg", ["mock_dataset", "mock_hf_dataset"], indirect=True)
def test_indexing(dataset_arg):
    """Test that ActiveDataModule is not indexable directly."""
    ads = ActiveDataModule(num_classes=2, train_dataset=dataset_arg)
    with pytest.raises(TypeError):
        assert ads[0]


@pytest.mark.parametrize("dataset_arg", ["mock_dataset", "mock_hf_dataset"], indirect=True)
def test_labelling(dataset_arg):
    """Test that labelling changes all the required states."""
    ads = ActiveDataModule(num_classes=2, train_dataset=dataset_arg)
    len_dataset_arg = len(dataset_arg)

    assert ads.last_labelling_step == 0
    assert ads.train_size == 0
    assert ads.pool_size == len_dataset_arg
    assert ads.has_labelled_data is False
    assert ads.has_unlabelled_data is True
    assert ads.train_dataset.indices == []

    for i in range(1, len_dataset_arg + 1):
        ads.label(0)  # always label the first instance in the pool

        assert ads.last_labelling_step == i
        assert ads.train_size == i
        assert ads.pool_size == len_dataset_arg - ads.train_size
        assert ads.has_labelled_data is True
        if i < len_dataset_arg:
            assert ads.has_unlabelled_data is True
        else:
            assert ads.has_unlabelled_data is False
        assert ads.train_dataset.indices == list(range(i))

    assert ads.last_labelling_step == len_dataset_arg
    assert ads.train_size == len_dataset_arg
    assert ads.pool_size == len_dataset_arg - ads.train_size
    assert ads.has_labelled_data is True
    assert ads.has_unlabelled_data is False
    assert ads.train_dataset.indices == list(range(len_dataset_arg))


@pytest.mark.parametrize("dataset_arg", ["mock_dataset", "mock_hf_dataset"], indirect=True)
def test_labelling_multiple_indices(dataset_arg):
    """Test labelling multiple instances at once."""
    ads = ActiveDataModule(num_classes=2, train_dataset=dataset_arg)
    pool_ids = [0, 8, 7]  # they are the first to be labelled so correspond to ids in oracle
    ads.label(pool_ids)

    assert ads.train_dataset.indices == sorted(pool_ids)


@pytest.mark.parametrize("dataset_arg", ["mock_dataset", "mock_hf_dataset"], indirect=True)
def test_labelling_duplicates(dataset_arg):
    """Test that labelling duplicate indices results in a single instance to be labelled."""

    # check behaviour when batch of indices contains
    ads = ActiveDataModule(num_classes=2, train_dataset=dataset_arg)
    pool_ids = [0, 0]  # they are the first to be labelled so correspond to ids in oracle
    ads.label(pool_ids)
    assert ads.train_size == 1

    # check behaviour when batch of indices contains
    ads = ActiveDataModule(num_classes=2, train_dataset=dataset_arg, val_split=0.5)
    pool_ids = [0, 0, 1]  # they are the first to be labelled so correspond to ids in oracle
    ads.label(pool_ids)
    assert ads.train_size == ads.val_size == 1


@pytest.mark.parametrize("dataset_arg", ["mock_dataset", "mock_hf_dataset"], indirect=True)
def test_labelling_val_split(dataset_arg):
    """Test that labelling with val_split works."""

    # check split works
    ads = ActiveDataModule(num_classes=2, train_dataset=dataset_arg, val_split=0.5)
    pool_ids = [0, 1]  # they are the first to be labelled so correspond to ids in oracle
    ads.label(pool_ids)
    assert ads.train_size == ads.val_size == 1

    # check that val_split receives at least 1 instance when there are two labelled instances
    # and the probability is too small that it randomly would receive just one
    ads = ActiveDataModule(num_classes=2, train_dataset=dataset_arg, val_split=0.0001)
    pool_ids = [0, 1]  # they are the first to be labelled so correspond to ids in oracle
    ads.label(pool_ids)
    assert ads.train_size == ads.val_size == 1

    # check behaviour when there is only one instance (bonus: using a duplicate)
    ads = ActiveDataModule(num_classes=2, train_dataset=dataset_arg, val_split=0.99)
    pool_ids = [0, 0]  # they are the first to be labelled so correspond to ids in oracle
    ads.label(pool_ids)
    assert ads.train_size == 1


@pytest.mark.parametrize("dataset_arg", ["mock_dataset", "mock_hf_dataset"], indirect=True)
def test_reset_at_labelling_step(dataset_arg):
    """Test that resetting the labelling steps sets the correct states."""
    ads = ActiveDataModule(num_classes=2, train_dataset=dataset_arg)
    len_dataset_arg = len(dataset_arg)

    ads.label(0)  # label first
    assert ads.last_labelling_step == 1
    assert ads.train_size == 1
    assert ads.pool_size == len_dataset_arg - ads.train_size
    assert ads.has_labelled_data is True
    assert ads.has_unlabelled_data is True
    assert ads.train_dataset.indices == [0]

    ads.label(list(range(len_dataset_arg - 1)))  # label the rest
    assert ads.train_size == len_dataset_arg
    assert ads.pool_size == len_dataset_arg - ads.train_size
    assert ads.has_labelled_data is True
    assert ads.has_unlabelled_data is False
    assert ads.train_dataset.indices == list(range(len_dataset_arg))

    ads.reset_at_labelling_step(1)  # go back to when there was one instance
    assert ads.train_size == 1
    assert ads.pool_size == len_dataset_arg - ads.train_size
    assert ads.has_labelled_data is True
    assert ads.has_unlabelled_data is True
    assert ads.train_dataset.indices == [0]

    ads.reset_at_labelling_step(0)  # go back to when there was nothing labelled
    assert ads.last_labelling_step == 2
    assert ads.train_size == 0
    assert ads.pool_size == len_dataset_arg - ads.train_size
    assert ads.has_labelled_data is False
    assert ads.has_unlabelled_data is True
    assert ads.train_dataset.indices == []

    ads.reset_at_labelling_step(ads.last_labelling_step)  # reset to the last step
    assert ads.train_size == len_dataset_arg
    assert ads.pool_size == len_dataset_arg - ads.train_size
    assert ads.has_labelled_data is True
    assert ads.has_unlabelled_data is False
    assert ads.train_dataset.indices == list(range(len_dataset_arg))

    with pytest.raises(ValueError):
        assert ads.reset_at_labelling_step(100)


@pytest.mark.parametrize("dataset_arg", ["mock_dataset", "mock_hf_dataset"], indirect=True)
def test_sample_pool_indices(dataset_arg):
    ads = ActiveDataModule(num_classes=2, train_dataset=dataset_arg)

    with pytest.raises(ValueError):
        assert ads.sample_pool_idx(-1)

    with pytest.raises(ValueError):
        assert ads.sample_pool_idx(0)

    with pytest.raises(ValueError):
        assert ads.sample_pool_idx(ads.pool_size + 1)

    assert len(ads.sample_pool_idx(ads.pool_size)) == ads.pool_size
    assert len(ads.sample_pool_idx(1)) == 1


@pytest.mark.parametrize("dataset_arg", ["mock_dataset", "mock_hf_dataset"], indirect=True)
def test_curriculum(dataset_arg):
    ads = ActiveDataModule(num_classes=2, train_dataset=dataset_arg)

    for _ in range(5):
        ads.label(0)

    assert ads.curriculum_dataset().indices == list(range(5))


@pytest.mark.parametrize("dataset_arg", ["mock_dataset", "mock_hf_dataset"], indirect=True)
def test_initial_labelling(dataset_arg):
    ads = ActiveDataModule(num_classes=2, train_dataset=dataset_arg)
    assert ads.train_size == 0

    ads = ActiveDataModule(num_classes=2, train_dataset=dataset_arg, initial_labels=[0])
    assert ads.train_size == 1

    ads = ActiveDataModule(num_classes=2, train_dataset=dataset_arg, initial_labels=2)
    assert ads.train_size == 2

    ads = ActiveDataModule(num_classes=2, train_dataset=dataset_arg, initial_labels=2, val_split=0.5)
    assert ads.train_size == ads.val_size == 1


@pytest.mark.parametrize("dataset_arg", ["mock_dataset", "mock_hf_dataset"], indirect=True)
def test_dataloader_len(dataset_arg):
    for batch_size in range(1, len(dataset_arg) + 1):
        ads = ActiveDataModule(
            num_classes=2,
            train_dataset=dataset_arg,
            initial_labels=2,
            batch_size=batch_size,
        )
        assert ads.train_dataloader().batch_size is None
        assert ads.train_dataloader().batch_sampler.batch_size == batch_size
        assert len(ads.train_dataloader().batch_sampler) == len(ads.train_dataloader())

    # min_steps_per_epoch
    for shuffle in (True, False):
        ads = ActiveDataModule(num_classes=2, train_dataset=dataset_arg, initial_labels=2, shuffle=shuffle)
        ads._min_steps_per_epoch = 1
        assert len(ads.train_dataloader().batch_sampler) == len(ads.train_dataloader()) == 2
        for _ in range(2):
            assert next(iter(ads.train_dataloader()))

        ads = ActiveDataModule(num_classes=2, train_dataset=dataset_arg, initial_labels=2, shuffle=shuffle)
        ads._min_steps_per_epoch = 10
        assert len(ads.train_dataloader().batch_sampler) == len(ads.train_dataloader()) == 10
        for _ in range(10):
            assert next(iter(ads.train_dataloader()))


@pytest.mark.parametrize("dataset_arg", ["mock_dataset", "mock_hf_dataset"], indirect=True)
def test_sampler_type(dataset_arg):
    ads = ActiveDataModule(
        num_classes=2,
        train_dataset=dataset_arg,
        test_dataset=dataset_arg,
        predict_dataset=dataset_arg,
        val_dataset=dataset_arg,
        initial_labels=2,
        batch_size=1,
    )

    assert isinstance(ads.train_dataloader().batch_sampler.sampler, FixedLengthSampler)
    assert isinstance(ads.pool_dataloader().batch_sampler.sampler, SequentialSampler)
    assert isinstance(ads.val_dataloader().batch_sampler.sampler, SequentialSampler)
    assert isinstance(ads.test_dataloader().batch_sampler.sampler, SequentialSampler)
    assert isinstance(ads.predict_dataloader().batch_sampler.sampler, SequentialSampler)


@pytest.mark.parametrize("dataset_arg", ["mock_dataset", "mock_hf_dataset"], indirect=True)
def test_raise_errors(dataset_arg):
    for i in (-0.5, 1.0):
        with pytest.raises(MisconfigurationException):
            ActiveDataModule(num_classes=2, train_dataset=dataset_arg, val_split=i)

    with pytest.raises(MisconfigurationException):
        ActiveDataModule(num_classes=2, train_dataset=dataset_arg, val_split=i, val_dataset=dataset_arg)

    with pytest.raises(RuntimeError):
        ads = ActiveDataModule(num_classes=2, train_dataset=dataset_arg)
        next(iter(ads.train_dataloader()))
