from typing import Any, Dict, List, Union

import datasets
import numpy as np
from torch.utils.data import Dataset


class Subset:
    """This is a shallow reimplementation of `torch.data.utils.Subset`."""

    dataset: Any
    indices: List[int]

    def __repr__(self) -> str:
        return f"<Subset of {type(self.dataset)}>"

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx: Union[int, List[int]]):
        raise NotImplementedError


class DatasetSubset(Subset):
    """Defines a batch-indexable `Subset` for `torch.data.utils.Dataset`'s."""

    def __init__(self, dataset: Dataset, indices: List[int]) -> None:
        """Subset of a dataset at specified indices.

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


class HFSubset(Subset):
    """Defines a batch-indexable `Subset` for `datasets.Dataset`'s."""

    def __init__(self, dataset: datasets.Dataset, indices: List[int]) -> None:
        """Subset of a dataset at specified indices.

        Args:
            dataset (datasets.Dataset): An HuggingFace Dataset
            indices (List[int]): Indices in the whole set selected for subset
        """
        self.dataset = dataset
        self.indices = indices

    def __getitem__(self, idx: Union[int, List[int]]) -> Union[Dict, List]:
        if isinstance(idx, list):
            # HggingFace datasets are indeed indexable by List[int] in their
            # implementation they report Union[int, slice, str]
            return self.dataset[[self.indices[i] for i in idx]]  # type: ignore
        return self.dataset[self.indices[idx]]


class ActiveDataset:
    """Makes a dataset ready for active learning.

    Splits an already labelled `torch.utils.data.Dataset` or `datasets.Dataset` into a
    `labelled_dataset` and a pool dataset. This class combines the best features of
    [baal](https://github.com/ElementAI/baal) and [batchbald_redux](https://github.com/BlackHC/batchbald_redux).
    This class is suited for research purposes in which the underlying dataset contains the true label.
    Therefore, note, that it is not possible to set a label.

    How does it work? At a high level, this is a thin wrapper around the `torch.data.utils.Subset`.
    As such it does not create a copy of the `dataset`. Instead, it uses indices to create two views,
    i.e. `labelled_dataset` and `pool_dataset`, of the same object in memory, i.e. `dataset`.

    At each step of the labelling process, this class keeps track of what instances are labelled and when.
    It does so by keeping track of how many times the `label` method has been called and registers its value
    in the `_mask` attribute. The `_mask` attribute is a numpy array that has the same length of the original
    dataset. Entries with value 0 are associated with the pool dataset by default. After the first labelling step
    is perfomed and an entry `m` at index `n` indicates that instance `n` has been labelled in labelling step `m`.
    In this way, it easy to go back to a specific "snapshot" of the `labelled_dataset`.
    The method `reset_at_labelling_step` allows the user to do that.

    In more detail,

    - There is a unique underlying datastore is called `dataset`
    - The `_mask` is integer `numpy.ndarray` that has the same length of `dataset`: an entry `m` at index `n` indicates
    that instance `n` has been labelled in labelling step `m`
    - The `labelled_dataset` is a `Subset`, i.e. a view, of the `dataset` containing instances for which
    the label is made available
    - Indexing of the labelled dataset is performed by checking when the `_mask > 0`
    - Accordingly, `pool_dataset` is a `Subset`, i.e. a view, of the `dataset` containing instance for which
    the label has been masked
    - Similarly, indexing of the pool dataset is performed by checking when the `_mask == 0`
    - To label an instance means changing the boolean flag of the `_mask`
    - `labelled_dataset` and `pool_dataset` are `torch.utils.data.Subset`'s where
    `{labelled, pool}_dataset.dataset` is either a `torch.utils.data.Dataset` or a `dataset.Dataset`
    and `{labelled, pool}_dataset.indices` is a `List[int]`

    Note that being a thin wrapper around the underlying dataset, it preserves its indexing capability.
    """

    def __init__(self, dataset: Union[Dataset, datasets.Dataset]) -> None:
        """Makes a `torch.utils.data.Dataset` or a `dataset.Dataset` active.

        Args:
            dataset (Union[Dataset, datasets.Dataset]): The original dataset containing both the inputs and the targets.
                This usually is the concatenation of the training and validation data. This is done to mimick real-world
                scenarios where the labelled instances are split between training and validation sets instead of having
                a growing training set and a fixed, possibly big, validation set.

        Attributes:
            labelled_dataset (Subset): The labelled dataset.
            pool_dataset (Subset): The pool dataset.
        """
        self.dataset = dataset
        subset_cls = DatasetSubset if isinstance(self.dataset, Dataset) else HFSubset

        self._mask = np.full((len(dataset),), 0)
        self.labelled_dataset = subset_cls(self.dataset, [])
        self.pool_dataset = subset_cls(self.dataset, [])
        self._update_index_map()

    def __len__(self):
        """Returns the length of the `labelled_dataset`."""
        return self.labelled_size

    def __getitem__(self, idx: Any) -> Any:
        """Returns element from the `labelled_dataset`.

        It allows for batch-indexing since it indexes an instance of the class `Subset`.
        So, even `torch.utils.data.Dataset`'s can be indexed with `List[int]`.

        Args:
            idx (Any): The index to retrieve. Its data type depends on the data type that the
                underlying dataset accepts in its `__getitem__` method.
        """
        return self.labelled_dataset[idx]

    def __repr__(self) -> str:
        return (
            f"ActiveDataset({{\n    "
            f"labelled_size: {self.labelled_size},\n    "
            f"pool_size: {self.pool_size},\n    "
            f"base_class: {type(self.dataset)},\n}})"
        )

    @property
    def labelled_size(self) -> int:
        """Returns the length of the `labelled_dataset`."""
        return len(self.labelled_dataset)

    @property
    def pool_size(self) -> int:
        """Returns the length of the `pool_dataset`."""
        return len(self.pool_dataset)

    @property
    def has_labelled_data(self) -> bool:
        """Checks whether there are labelled data available."""
        return self.labelled_size > 0

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
        # NOTE: for compatibility with HuggingFace, indices must be lists not numpy.ndarray
        self.labelled_dataset.indices = np.where(self._mask > 0)[0].tolist()
        self.pool_dataset.indices = np.where(self._mask == 0)[0].tolist()

    def _pool_to_oracle(self, pool_idx: Union[int, List[int]]) -> List[int]:
        """Transform indices in `pool_dataset` to indices in the original `dataset`."""
        if isinstance(pool_idx, list):
            # NOTE: remove repetitions - this can be handled by numpy indexing but it is
            # better to make it explicit using `np.unique`
            return [self.pool_dataset.indices[i] for i in np.unique(pool_idx)]
        return [self.pool_dataset.indices[pool_idx]]

    def label(self, pool_idx: Union[int, List[int]]) -> None:
        """Adds instances to the `labelled_dataset`.

        Args:
            pool_idx (List[int]): the index (relative to the pool_dataset, not the overall data) to label.
        """
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

        labelling_step = self.last_labelling_step
        indices = self._pool_to_oracle(pool_idx)
        self._mask[indices] = labelling_step + 1
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
        self.labelled_dataset.indices = np.where((self._mask > 0) & (self._mask <= labelling_step))[0].tolist()
        self.pool_dataset.indices = np.where(self._mask == 0)[0].tolist()

    def sample_pool_idx(self, size: int) -> np.ndarray:
        """Samples indices from pool uniformly.

        Args:
            size (int): The number of indices to sample from the pool. Must be 0 < size <= pool_size
        """
        if size <= 0 or size > self.pool_size:
            raise ValueError(f"`size` must be 0 < size <= {self.pool_size} not {size}.")

        return np.random.permutation(self.pool_size)[:size]
