from typing import Callable, List, Optional, Union

import datasets
import numpy as np
from numpy.random import default_rng
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import BatchSampler, DataLoader, Dataset, Sampler, SequentialSampler

from energizer.data.active_dataset import ActiveDataset, EnergizerSubset


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
        min_steps_per_epoch: Optional[int] = None,
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
            initial_labels (Optional[Union[int, List[int]]]): The indices to randomly label prior to starting
                training.
            val_dataset (Optional[Union[Dataset, datasets.Dataset]]): The validation dataset.
            val_split (Optional[float]): The proportion of labelled data to add to the validation set at each
                labelling iteration.
            test_dataset (Optional[Union[Dataset, datasets.Dataset]]): The test dataset.
            min_steps_per_epoch (Optional[int]): Minumum number of steps per epoch. Especially in the beginning
                of active learning when there are a few labelled instances, this arguments allows the user to
                train for a minimum number of iterations by resampling the available data.
            batch_size (Optional[int]): The batch size used to train the model. The batch size for evaluating the
                model on the pool is automatically set by the `ActiveTrainer` and exploits the maximum available
                bandwidth.
            shuffle (Optional[bool]): Whether to shuffle the data at each training iteration.
            num_workers (Optional[int]): How many subprocesses to use for data loading. ``0`` means that the data
                will be loaded in the main process.
            collate_fn (Optional[Callable]): Merges a sequence of samples to form a mini-batch of Tensor(s).
            pin_memory (Optional[bool]): TBD
            drop_last (Optional[bool]): TBD
            persistent_workers (Optional[bool]): TBD
            seed (Optional[int]): Seed used to shuffle the data if `shuffle` is True and to split data in the train
                and validation set if `val_split` is True.
        """
        super().__init__()
        self.num_classes = num_classes
        self.initial_labels = initial_labels
        self.val_split = val_split
        self.min_steps_per_epoch = min_steps_per_epoch
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
            self.label(self.initial_labels, self.val_split)  # type: ignore

        if not self._active_dataset.has_labelled_data:
            print("Cold-starting: The training dataset does not contain any labelled instance.")

        self.save_hyperparameters("num_classes", "batch_size", "shuffle", "val_split", "initial_labels", "seed")

    def _make_dataloader(self, dataset: Union[EnergizerSubset, Dataset, datasets.Dataset]) -> DataLoader:
        if isinstance(dataset, EnergizerSubset):
            sampler = FixedLengthSampler(
                dataset, n_steps=self.min_steps_per_epoch, shuffle=self.shuffle, seed=self.seed
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
    def train_dataset(self):
        """Returns the train dataset from the underlying active dataset."""
        return self._active_dataset.train_dataset

    @property
    def pool_dataset(self):
        """Returns the pool dataset from the underlying active dataset."""
        return self._active_dataset.pool_dataset

    @property
    def val_dataset(self):
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
    def test_dataset(self):
        """Returns the test dataset from the underlying active dataset."""
        return self._test_dataset

    def train_dataloader(self) -> DataLoader:
        """Returns the train dataloader."""
        return self._make_dataloader(self.train_dataset)

    def val_dataloader(self) -> Optional[DataLoader]:  # type: ignore
        """Returns the validation dataloader."""
        if not self.val_dataset:
            return None
        return self._make_dataloader(self.val_dataset)

    def test_dataloader(self) -> Optional[DataLoader]:  # type: ignore
        """Returns the test dataloader."""
        if not self.test_dataset:
            return None
        return self._make_dataloader(self.test_dataset)

    def pool_dataloader(self) -> DataLoader:
        """Returns the pool dataloader."""
        return self._make_dataloader(self.pool_dataset)

    def label(self, pool_idx: Union[int, List[int]], val_split: Optional[float] = None) -> None:
        """Convenience method that calls the underlying self._active_dataset.label method.

        Adds instances to the `labelled_dataset`.

        Args:
            pool_idx (List[int]): The index (relative to the pool_dataset, not the overall data) to label.
                If `len(pool_idx) == 1`, the `val_split` argument is ignore and nothing will be added to the
                validation dataset.
            val_split (Optional[float]): The proportion of the number of instances to add to the validation set.
                This is considered only when `len(pool_idx) > 1`.
        """
        self._active_dataset.label(pool_idx, val_split)


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
