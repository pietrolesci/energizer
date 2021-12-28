from functools import partial
from typing import Callable, List, Optional, Union

import datasets
import numpy as np
from numpy.random import default_rng
from pytorch_lightning import LightningDataModule
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler, SequentialSampler

from energizer.data.dataset import ActiveDataset, EnergizerSubset


class ActiveDataModule(LightningDataModule):
    """A `pytorch_lightning.LightningDataModule` that handles data manipulation for active learning."""

    def __init__(
        self,
        num_classes: int,
        train_dataset: Union[Dataset, datasets.Dataset],
        initial_labels: Optional[Union[int, List[int]]] = None,
        val_dataset: Optional[Union[Dataset, datasets.Dataset]] = None,
        val_split: Optional[float] = None,
        test_dataset: Optional[Union[Dataset, datasets.Dataset]] = None,
        predict_dataset: Optional[Union[Dataset, datasets.Dataset]] = None,
        batch_size: int = 1,
        shuffle: Optional[bool] = False,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        persistent_workers: bool = False,
        seed: Optional[int] = None,
    ):
        """Handles data manipulation for active learning and offers access to the underlying ActiveDataset.

        Args:
            num_classes (int): The number of classes in the dataset. Must be known.
            train_dataset (Union[Dataset, datasets.Dataset]): The dataset to use for training. This
                will be transformed into an `ActiveDataset`. If `val_split` is specified, validation set will
                be active too and will be created as a portion of the labelled instances in each iteration.
            initial_labels (Optional[Union[int, List[int]]]): The instances to label prior to starting the
                training loop.  If it is a `List[int]` it is interpreteed as the actual indices of the instances
                to label. If it is an `int` it will be the size of the indeces to randomly draw from the pool
                and label.
            val_dataset (Optional[Union[Dataset, datasets.Dataset]]): The validation dataset.
            val_split (Optional[float]): The proportion of labelled data to add to the validation set at each
                labelling iteration.
            test_dataset (Optional[Union[Dataset, datasets.Dataset]]): The test dataset.
            test_dataset (Optional[Union[Dataset, datasets.Dataset]]): The prediction dataset.
            batch_size (int): The batch size used to train the model. The batch size for evaluating the
                model on the pool is automatically set by the `ActiveTrainer` and exploits the maximum available
                bandwidth.
            shuffle (Optional[bool]): Whether to shuffle the data at each training iteration.
            num_workers (int): How many subprocesses to use for data loading. ``0`` means that the data
                will be loaded in the main process.
            collate_fn (Optional[Callable]): Merges a sequence of samples to form a mini-batch of Tensor(s).
            pin_memory (bool): TBD
            drop_last (bool): TBD
            persistent_workers (bool): TBD
            seed (Optional[int]): Seed used to shuffle the data if `shuffle` is True and to split data in the train
                and validation set if `val_split` is True.
        """
        super().__init__()
        self.num_classes = num_classes
        self.initial_labels = initial_labels
        self.val_split = val_split
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.collate_fn = collate_fn
        self.pin_memory = pin_memory
        self.drop_last = drop_last
        self.persistent_workers = persistent_workers
        self.seed = seed

        self._active_dataset = ActiveDataset(train_dataset, self.seed)
        self._val_dataset = val_dataset
        self._test_dataset = test_dataset
        self._predict_dataset = predict_dataset

        # this will be overwritten in the ActiveLearningLoop
        self._min_steps_per_epoch: Optional[int] = None

        # check val_dataset and val_split
        if self._val_dataset and self.val_split:
            raise MisconfigurationException("`val_dataset` and `val_split` are mutually exclusive.")
        elif self.val_split and (self.val_split < 0 or self.val_split >= 1):
            raise MisconfigurationException(f"`val_split` should 0 <= `val_split` < 1, not {self.val_split}.")
        elif self.val_split and (self.val_split > 0 or self.val_split < 1):
            print(
                f"Validation dataset is not specfied and `val_split == {self.val_split}`, therefore at each training "
                f"step {self.val_split} of the labelled data will be added to the validation dataset."
            )
        elif not self._val_dataset and self.val_split == 0:
            print("`val_dataset` is None and `val_split == 0` or is None, no validation will be performed.")

        # initial labelling
        if self.initial_labels is not None:
            if isinstance(self.initial_labels, int):
                self.initial_labels = self._active_dataset.sample_pool_idx(self.initial_labels)
            self.label(self.initial_labels)  # type: ignore

        if not self._active_dataset.has_labelled_data:
            print("Cold-starting: The training dataset does not contain any labelled instance.")

        # setup dataloaders
        self.train_dataloader = partial(self._make_dataloader, self.train_dataset, RunningStage.TRAINING)
        self.pool_dataloader = partial(self._make_dataloader, self.pool_dataset, RunningStage.PREDICTING)

        if self.val_dataset:
            self.val_dataloader = partial(self._make_dataloader, self.val_dataset, RunningStage.VALIDATING)

        if self.test_dataset:
            self.test_dataloader = partial(self._make_dataloader, self.test_dataset, RunningStage.TESTING)

        if self.predict_dataset:
            self.predict_dataloader = partial(self._make_dataloader, self.predict_dataset, RunningStage.PREDICTING)

        self.save_hyperparameters("num_classes", "batch_size", "shuffle", "val_split", "initial_labels", "seed")

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None) -> None:
        pass

    def _make_dataloader(
        self, dataset: Union[EnergizerSubset, Dataset, datasets.Dataset], running_stage: RunningStage
    ) -> DataLoader:
        if running_stage in (RunningStage.TRAINING, RunningStage.VALIDATING) and isinstance(dataset, EnergizerSubset):
            sampler = FixedLengthSampler(
                dataset, n_steps=self._min_steps_per_epoch, shuffle=self.shuffle, seed=self.seed
            )
        else:
            sampler = SequentialSampler(dataset)  # type: ignore

        return DataLoader(
            dataset,  # type: ignore
            batch_sampler=BatchSampler(
                sampler=sampler,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
            ),
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    @property
    def train_dataset(self) -> EnergizerSubset:
        """Returns the train dataset from the underlying active dataset."""
        return self._active_dataset.train_dataset

    @property
    def pool_dataset(self) -> EnergizerSubset:
        """Returns the pool dataset from the underlying active dataset."""
        return self._active_dataset.pool_dataset

    @property
    def val_dataset(self) -> Optional[Union[EnergizerSubset, Dataset, datasets.Dataset]]:
        """Returns the validation dataset.

        It returns the underlying active dataset if `val_split` is True or if `val_dataset` is provided.
        """
        if self._val_dataset:
            return self._val_dataset
        elif self.val_split and self.val_split > 0:
            return self._active_dataset.val_dataset
        else:
            return None

    @property
    def test_dataset(self) -> Optional[Union[Dataset, datasets.Dataset]]:
        """Returns the test dataset from the underlying active dataset."""
        return self._test_dataset

    @property
    def predict_dataset(self) -> Optional[Union[Dataset, datasets.Dataset]]:
        """Returns the predict dataset from the underlying active dataset."""
        return self._predict_dataset

    @property
    def train_size(self) -> int:
        """Returns the length of the `labelled_dataset` that has been assigned to training."""
        return self._active_dataset.train_size

    @property
    def val_size(self) -> int:
        """Returns the length of the `labelled_dataset` that has been assigned to validation."""
        return self._active_dataset.val_size

    @property
    def total_labelled_size(self) -> int:
        """Returns the number of all the labelled instances."""
        return self._active_dataset.total_labelled_size

    @property
    def pool_size(self) -> int:
        """Returns the length of the `pool_dataset`."""
        return self._active_dataset.pool_size

    @property
    def has_labelled_data(self) -> bool:
        """Checks whether there are labelled data available."""
        return self._active_dataset.has_labelled_data

    @property
    def has_unlabelled_data(self) -> bool:
        """Checks whether there are data to be labelled."""
        return self._active_dataset.has_unlabelled_data

    @property
    def last_labelling_step(self) -> int:
        """Returns the last active learning step."""
        return self._active_dataset.last_labelling_step

    def reset_at_labelling_step(self, labelling_step: int) -> None:
        """Resets the dataset at the state in which it was after the last `labelling_step`.

        To bring it back to the latest labelling step, simply run
        `obj.reset_at_labellling_step(obj.last_labelling_step)`.

        Args:
            labelling_step (int): The labelling step at which the dataset status should be brought.
        """
        self._active_dataset.reset_at_labelling_step(labelling_step)

    def sample_pool_idx(self, size: int) -> List[int]:
        """Samples indices from pool uniformly.

        Args:
            size (int): The number of indices to sample from the pool. Must be 0 < size <= pool_size

        Returns:
            The list of indices from the pool to label.
        """
        return self._active_dataset.sample_pool_idx(size)

    def label(self, pool_idx: Union[int, List[int]]) -> None:
        """Convenience method that calls the underlying self._active_dataset.label method.

        Adds instances to the `train_dataset` and, optionally, to the `val_dataset`.

        Note (1): indices are sorted. For example, assuming this is the first labelling iteration so
        that pool indices and oracle indices coincide, `label([0, 8, 7])` will add `[0, 7, 8]` to the
        `train_dataset.indices`.

        Note (2): when `val_split` is used, the validation dataset will always receive at least one instance
        if at least two indices are passed to be labelled. This is to avoid that the validation dataset
        remains empty.

        Note (3): when a list of indices is passed, only the unique values are counted. That is
        `pool_idx = [0, 0]` will only label instance 0. It is not recursively applied so that
        there are multiple calls to `label`.

        Note (4): if `val_split` is passed, but only one instance is labelled, the instance will always be
        to the train dataset and thus the validation set will always be empty.

        Args:
            pool_idx (List[int]): The index (relative to the pool_dataset, not the overall data) to label.
                If `len(pool_idx) == 1`, the `val_split` argument is ignore and nothing will be added to the
                validation dataset.
            val_split (Optional[float]): The proportion of the number of instances to add to the validation set.
                This is considered only when `len(pool_idx) > 1`.
        """
        self._active_dataset.label(pool_idx, self.val_split)


class FixedLengthSampler(Sampler):
    """A sampler that allows to iterate over a dataset an arbitraty number of times regardless the length of the dataset.

    Sometimes, you really want to do more with little data without increasing the number
    of epochs. This sampler takes a `dataset` and draws `size` shuffled indices from the
    range 0, size - 1 (with repetition).
    The indices produced by this sampler are then
    combined together by the batch_sampler that creates a batch
    of dimension batch_size (as specified in the DataLoader)

    NOTE: here we pass an instance of `EnergizerSubset` since it is mutable. This is
    crucial to allow the acquired indices to get picked up automatically
    without redefining the DataLoader after each acquisition
    """

    def __init__(
        self,
        dataset: EnergizerSubset,
        n_steps: Optional[int] = None,
        shuffle: Optional[bool] = False,
        seed: Optional[int] = None,
    ):
        """A sampler that resamples instances if needed.

        Args:
            dataset (EnergizerSubset): The dataset.
            n_steps (int): The minimum number of steps to iterate over the dataset. If this `n_steps > len(dataset)`
                instances will be resampled.
            shuffle (Optional[bool]): Whether to shuffle indices.
            seed (Optional[int]): Seed for the random shuffling procedure.
        """
        self.dataset = dataset
        self.n_steps = n_steps
        self.shuffle = shuffle
        self.seed = seed
        self._rng = default_rng(self.seed)

    def __iter__(self):
        # if dataset has no labelled data
        if len(self.dataset) < 1:
            raise RuntimeError("Dataset has no labelled instances yet, not possible to iterate.")

        # ensure that we don't lose data by accident when we have more
        # data than the required n_steps
        if self.n_steps is None or self.n_steps < len(self.dataset):
            if self.shuffle:
                return iter(self._rng.permutation(len(self.dataset)).tolist())
            else:
                return iter(range(len(self.dataset)))

        else:
            if self.shuffle:
                return iter((self._rng.permutation(self.n_steps) % len(self.dataset)).tolist())
            else:
                return iter((np.arange(self.n_steps) % len(self.dataset)).tolist())

    def __len__(self):
        return len(self.dataset) if not self.n_steps else max(self.n_steps, len(self.dataset))
