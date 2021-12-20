from typing import Any, Dict, List, Optional, Union

import datasets
import numpy as np
from numpy.random import default_rng
from torch.utils.data import Dataset


class EnergizerSubset:
    """This is a shallow reimplementation of `torch.data.utils.Subset`.

    It serves as base class for the source-specific implementation of Subset.
    """

    dataset: Any
    indices: List[int]

    def __repr__(self) -> str:
        return (
            f"Subset({{\n    "
            f"original_size: {len(self.dataset)},\n    "
            f"subset_size: {len(self.indices)},\n    "
            f"base_class: {type(self.dataset)},\n}})"
        )

    def __len__(self):
        return len(self.indices)


class TorchSubset(EnergizerSubset):
    """Defines a batch-indexable `Subset` for `torch.data.utils.Dataset`'s."""

    def __init__(self, dataset: Dataset, indices: List[int]) -> None:
        """Subset of a torch.utils.Dataset at specified indices.

        Args:
            dataset (Dataset): A Pytorch Dataset
            indices (List[int]): Indices in the whole set selected for subset
        """
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx: Union[int, List[int]]) -> Union[Any, List]:
        if isinstance(idx, list):
            return [self.dataset[self.indices[i]] for i in idx]
        return self.dataset[self.indices[idx]]


class HFSubset(EnergizerSubset):
    """Defines a batch-indexable `Subset` for `datasets.Dataset`'s."""

    def __init__(self, dataset: datasets.Dataset, indices: List[int]) -> None:
        """Subset of a datasets.Dataset at specified indices.

        Args:
            dataset (datasets.Dataset): An HuggingFace Dataset
            indices (List[int]): Indices in the whole set selected for subset
        """
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx: Union[int, List[int]]) -> Union[Dict, List]:
        if isinstance(idx, list):
            # HuggingFace datasets are indeed indexable by List[int] in their
            # implementation they report Union[int, slice, str]
            return self.dataset[[self.indices[i] for i in idx]]  # type: ignore
        return self.dataset[self.indices[idx]]


class ActiveDataset:
    """Makes a dataset ready for active learning.

    Splits an already labelled `torch.utils.data.Dataset` or `datasets.Dataset` into a
    `train_dataset`, a `pool_dataset`, and optionally a `val_dataset`. At its core it uses
    an implementation of the `Subset` class available in Pytorch that allows to arbitrarily
    make instances available. For this reason, it is suited for research cases in which
    the underlying dataset contains the true label and needs to be masked for the purpose of
    active learning. Therefore, it does not allow for interactive labelling, e.g.,
    human-in-the-loop annotations.

    This class is inspired and combines the best features of
    [baal](https://github.com/ElementAI/baal) and [batchbald_redux](https://github.com/BlackHC/batchbald_redux).
    Additionally, it allows to have an active validation set. This is useful to mimick real-world
    scenarios in which there is not a data-abundant validation set to tune hyper-parameters.
    This allows to perform experiments in which part of the labelling budget is assigned to validation.

    How does it work? At a high level, this is a thin wrapper around the `torch.data.utils.Subset`.
    As such, this implementation does not create a copy of the `dataset`. Instead, it uses indices to
    create two views, i.e. `train_dataset` and `pool_dataset`, of the same object in memory.
    This is achieved by having a unique reference to the indices of `dataset`, called `_mask`.
    `_mask` is a integer-valued array.
    Roughly it is implemented as follows:

    ```python
    class Subset:
        def __init__(self, dataset: Dataset, indices: List[int]) -> None:
            self.dataset = dataset
            self.indices = indices

        def __getitem__(self, idx: int) -> Union[Any, List]:
            return self.dataset[self.indices[idx]]

    # at the beginning of active learning all data are in the pool, i.e. masked with a 0
    _mask = np.zeros((len(dataset),), dtype=int)
    train_dataset = Subset(dataset, np.where(_mask > 0)[0])
    pool_dataset = Subset(dataset, np.where(_mask == 0)[0])
    ```

    Therefore, when indexing in a `Subset`, the indices are first indexed and then the dataset. This
    allows to have a different number of instances in a `Subset` without changing the underlying data.
    Note that this also means that during training, we do not need to redefine the `DataLoader` after
    each iteration of active learning.

    At each step of the labelling process, this class keeps track of what instances are labelled and when.
    It does so by keeping track of how many times the `label` method has been called and registers its value
    in the `_mask` attribute. The `_mask` attribute is a numpy array that has the same length of the original
    dataset. Entries with value 0 are associated with the pool dataset by default. After the first labelling step
    is perfomed and an entry `m` at index `n` indicates that instance `n` has been labelled in labelling step `m`.
    In this way, it easy to go back to a specific "snapshot" of the train dataset.

    In addition, as said above, this class allows to split labelled instance between train and validation sets.
    This allows to also have a growing validation set and mimicks real-world scenarios. The active validation set
    is implemented using a secondary boolean mask, `_val_mask`. This is a growing numpy array and has, at each
    iteration, the same length of the indices in the train dataset. Roughly, assignment to train and validation
    is implemented as follows

    ```python
    indices_of_labelled = np.where(_mask > 0)[0]
    train_dataset.indices = indices_of_labelled[~_val_mask]
    val_dataset.indices = indices_of_labelled[_val_mask]
    ```

    In more detail,

    - There is a unique underlying datastore is called `dataset`
    - The `_mask` is integer `numpy.ndarray` that has the same length of `dataset`: an entry `m` at index `n` indicates
    that instance `n` has been labelled in labelling step `m`
    - Indexing of the `train_dataset` is performed by checking where `_mask > 0` amd `_val_mask is False`
    - Indexing of the `val_dataset` is performed by checking where `_mask > 0` amd `_val_mask is True`
    - Indexing of the `pool_dataset` is performed by checking when the `_mask == 0`
    - The `_mask` contains information about when each instance has been labelled, therefore it is possible to
    go back to any "snapshot" of the dataset

    NOTE: the underlying `Subset`'s are implemented to support batch-indexing.
    """

    def __init__(self, dataset: Union[Dataset, datasets.Dataset], seed: Optional[int] = None) -> None:
        """Makes a `torch.utils.data.Dataset` or a `dataset.Dataset` active.

        Args:
            dataset (Union[Dataset, datasets.Dataset]): The original dataset containing both the inputs and the targets.
                This usually is the concatenation of the training and validation data. This is done to mimick real-world
                scenarios where the labelled instances are split between training and validation sets instead of having
                a growing training set and a fixed, possibly big, validation set.
            seed (Optional[int]): Seed used to randomly assign newly labelled instances to train or validation datasets.

        Attributes:
            train_dataset (EnergizerSubset): The train dataset.
            val_dataset (EnergizerSubset): The validation dataset.
            pool_dataset (EnergizerSubset): The pool dataset.
        """
        self.dataset = dataset
        self.seed = seed
        subset_cls = TorchSubset if isinstance(self.dataset, Dataset) else HFSubset

        self._mask = np.zeros((len(dataset),), dtype=int)
        self._val_mask = np.array([], dtype=np.bool)
        self._rng = default_rng(self.seed)

        self.train_dataset = subset_cls(self.dataset, [])
        self.val_dataset = subset_cls(self.dataset, [])
        self.pool_dataset = subset_cls(self.dataset, [])
        self._update_index_map()

    def __repr__(self) -> str:
        return (
            f"ActiveDataset({{\n    "
            f"original_dataset_size: {len(self.dataset)},\n    "
            f"train_size: {self.train_size},\n    "
            f"val_size: {self.val_size},\n    "
            f"pool_size: {self.pool_size},\n    "
            f"base_class: {type(self.dataset)},\n}})"
        )

    @property
    def train_size(self) -> int:
        """Returns the length of the `labelled_dataset` that has been assigned to training."""
        return len(self.train_dataset)

    @property
    def val_size(self) -> int:
        """Returns the length of the `labelled_dataset` that has been assigned to validation."""
        return len(self.val_dataset)

    @property
    def total_labelled_size(self) -> int:
        """Returns the number of all the labelled instances."""
        return self.train_size + self.val_size

    @property
    def pool_size(self) -> int:
        """Returns the length of the `pool_dataset`."""
        return len(self.pool_dataset)

    @property
    def has_labelled_data(self) -> bool:
        """Checks whether there are labelled data available."""
        return self.total_labelled_size > 0

    @property
    def has_unlabelled_data(self) -> bool:
        """Checks whether there are data to be labelled."""
        return self.pool_size > 0

    @property
    def last_labelling_step(self) -> int:
        """Returns the last active learning step."""
        return int(self._mask.max())

    def _update_index_map(self) -> None:
        """Updates indices in each `torch.utils.data.Subset`.

        It stores the indices as `List[int]` and updates the relative `{labelled, pool}_size` attributes.
        """
        self.pool_dataset.indices = np.where(self._mask == 0)[0].tolist()

        indices = np.where(self._mask > 0)[0]
        self.train_dataset.indices = indices[~self._val_mask].tolist()
        self.val_dataset.indices = indices[self._val_mask].tolist()

    def _pool_to_oracle(self, pool_idx: Union[int, List[int]]) -> List[int]:
        """Transform indices in `pool_dataset` to indices in the original `dataset`."""
        if isinstance(pool_idx, list):
            # NOTE: remove repetitions - this can be handled by numpy indexing but it is
            # better to make it explicit using `np.unique`
            return [self.pool_dataset.indices[i] for i in np.unique(pool_idx)]
        return [self.pool_dataset.indices[pool_idx]]

    def label(self, pool_idx: Union[int, List[int]], val_split: Optional[float] = None) -> None:
        """Adds instances to the `labelled_dataset`.

        Args:
            pool_idx (List[int]): The index (relative to the pool_dataset, not the overall data) to label.
                If `len(pool_idx) == 1`, the `val_split` argument is ignore and nothing will be added to the
                validation dataset.
            val_split (Optional[float]): The proportion of the number of instances to add to the validation set.
                This is considered only when `len(pool_idx) > 1`.
        """
        if not isinstance(pool_idx, int) and not isinstance(pool_idx, list):
            raise ValueError(f"`pool_idx` must be of type `int` or `List[int]`, not {type(pool_idx)}.")
        if isinstance(pool_idx, list):
            if len(np.unique(pool_idx)) > self.pool_size:
                raise ValueError(
                    f"Pool has {self.pool_size} instances, cannot label {len(np.unique(pool_idx))} instances."
                )

            elif max(pool_idx) > self.pool_size:
                raise ValueError(f"Cannot label instance {max(pool_idx)} for pool dataset of size {self.pool_size}.")

        else:
            if pool_idx > self.pool_size:
                raise ValueError(f"Cannot label instance {pool_idx} for pool dataset of size {self.pool_size}.")

        if val_split and (val_split < 0 or val_split >= 1):
            raise ValueError(f"`val_split` should 0 <= `val_split` < 1, not {val_split}.")

        labelling_step = self.last_labelling_step
        indices = self._pool_to_oracle(pool_idx)
        self._mask[indices] = labelling_step + 1

        if val_split and len(indices) > 1:
            how_many = int(np.floor(len(indices) * val_split).item())
            if how_many == 0:
                how_many += 1
            val_mask = ([True] * how_many) + ([False] * (len(indices) - how_many))
            val_mask = self._rng.permutation(val_mask)
            self._val_mask = np.append(self._val_mask, val_mask)
        else:
            self._val_mask = np.append(self._val_mask, [False] * len(indices))

        self._update_index_map()

    def reset_at_labelling_step(self, labelling_step: int) -> None:
        """Resets the dataset at the state in which it was after the last `labelling_step`.

        To bring it back to the latest labelling step, simply run
        `obj.reset_at_labellling_step(obj.last_labelling_step)`.

        Args:
            labelling_step (int): The labelling step at which the dataset status should be brought.
        """
        if labelling_step > self.last_labelling_step:
            raise ValueError(
                f"`labelling_step` ({labelling_step}) must be less than the total number "
                f"of labelling steps performed ({self.last_labelling_step})."
            )

        self.pool_dataset.indices = np.where((self._mask == 0) | (self._mask > labelling_step))[0].tolist()

        indices = np.where((self._mask > 0) & (self._mask <= labelling_step))[0]
        val_mask = self._val_mask[: len(indices)]
        self.train_dataset.indices = indices[~val_mask].tolist()
        self.val_dataset.indices = indices[val_mask].tolist()

    def sample_pool_idx(self, size: int) -> List[int]:
        """Samples indices from pool uniformly.

        Args:
            size (int): The number of indices to sample from the pool. Must be 0 < size <= pool_size
        """
        if size <= 0 or size > self.pool_size:
            raise ValueError(f"`size` must be 0 < size <= {self.pool_size} not {size}.")

        return np.random.permutation(self.pool_size)[:size].tolist()
