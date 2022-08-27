from typing import Any, Dict, List, Optional, Union

import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from torch.utils.data import DataLoader, Dataset, Subset
import faiss
from energizer.utilities.type_converters import list_to_numpyint64, array_to_numpyfloat32
from energizer.utilities.logger import logger


class DataloaderToDataModule(LightningDataModule):
    """Given individual dataloaders, create a datamodule."""

    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloaders: Union[DataLoader, List[DataLoader]],
        test_dataloaders: Union[DataLoader, List[DataLoader]],
    ) -> None:
        super().__init__()
        self._train_dataloader = train_dataloader
        self._val_dataloaders = val_dataloaders
        self._test_dataloaders = test_dataloaders

    def train_dataloader(self) -> DataLoader:
        return self._train_dataloader

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self._val_dataloaders

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self._test_dataloaders


class ActiveDataModule(LightningDataModule):
    train_fold: Optional[Dataset] = None
    pool_fold: Optional[Dataset] = None

    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        test_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
    ):
        if train_dataloader is None and datamodule is None:
            raise MisconfigurationException("Either `train_dataloader` or `datamodule` argument should be provided")
        if train_dataloader is not None and datamodule is not None:
            raise MisconfigurationException(
                "Only one of `train_dataloader` and `datamodule` argument should be provided"
            )
        if train_dataloader is not None:
            self.datamodule = DataloaderToDataModule(train_dataloader, val_dataloaders, test_dataloaders)
        if datamodule is not None:
            self.datamodule = datamodule

            # set up datamodule by default
            self.datamodule.setup()

        super().__init__()
        self._train_dataloader_args = None
        self._eval_dataloader_args = None
        self.setup_folds()

    """
    Properties
    """

    @property
    def total_data_size(self) -> int:
        return self.train_size + self.pool_size

    @property
    def train_size(self) -> int:
        return len(self.train_fold)

    @property
    def pool_size(self) -> int:
        return len(self.pool_fold)

    @property
    def train_dataloader_args(self) -> Dict[str, Any]:
        return self._train_dataloader_args

    @property
    def eval_dataloader_args(self) -> Dict[str, Any]:
        return self._eval_dataloader_args

    @property
    def has_labelled_data(self) -> bool:
        """Checks whether there are labelled data available."""
        return len(self.train_fold) > 0

    @property
    def has_unlabelled_data(self) -> bool:
        """Checks whether there are data to be labelled."""
        return len(self.pool_fold) > 0

    @property
    def last_labelling_step(self) -> int:
        """Returns the number of the last active learning step."""
        return int(self.train_mask.max())

    @property
    def train_dataset(self) -> Any:
        return self.train_dataloader().dataset

    @property
    def val_dataset(self) -> Any:
        return self.val_dataloader().dataset

    @property
    def test_dataset(self) -> Any:
        return self.test_dataloader().dataset

    """
    Data preparations
    """

    def _get_dataloader_args(self, dataloader: DataLoader) -> Optional[Dict[str, Any]]:
        # TODO: look at how lightning does this inspection of the parameters, might work better
        if dataloader is not None:
            dataloader = dataloader[0] if isinstance(dataloader, List) else dataloader
            return {
                "batch_size": dataloader.batch_size,
                "num_workers": dataloader.num_workers,
                "collate_fn": dataloader.collate_fn,
                "pin_memory": dataloader.pin_memory,
                "drop_last": dataloader.drop_last,
                "timeout": dataloader.timeout,
                "worker_init_fn": dataloader.worker_init_fn,
                "prefetch_factor": dataloader.prefetch_factor,
                "persistent_workers": dataloader.persistent_workers,
                "shuffle": False,  # be explicit about it
            }

    def prepare_data(self) -> None:
        self.datamodule.prepare_data()

    def setup(self, stage: Optional[str] = None) -> None:
        self.datamodule.setup(stage)

    """
    Helper methods
    """

    def setup_folds(self) -> None:
        train_dl = self.datamodule.train_dataloader()
        train_dataset = train_dl.dataset
        self._train_dataloader_args = self._get_dataloader_args(train_dl)
        self.train_fold = Subset(train_dataset, [])
        self.pool_fold = Subset(train_dataset, [])

        eval_dl = self.datamodule.test_dataloader() or self.datamodule.val_dataloader()
        self._eval_dataloader_args = self._get_dataloader_args(eval_dl)
        self._eval_dataloader_args = self._eval_dataloader_args or self._train_dataloader_args

        # states
        self.train_mask = np.zeros((len(train_dataset),), dtype=int)
        self.setup_fold_index()

    def pool_to_original(self, pool_idx: List[int]) -> List[int]:
        """Transform indices in `pool_fold` to indices in the original `dataset`."""
        return [self.pool_fold.indices[i] for i in np.unique(pool_idx)]

    def train_to_original(self, train_idx: List[int]) -> List[int]:
        """Transform indices in `train_fold` to indices in the original `dataset`."""
        return [self.train_fold.indices[i] for i in np.unique(train_idx)]

    def original_to_pool(self, indices: List[int]) -> List[int]:
        """Transform indices wrt the original `dataset` to indices wrt `pool_fold`."""
        return [self.pool_fold.indices.index(i) for i in np.unique(indices)]

    def original_to_train(self, indices: List[int]) -> List[int]:
        """Transform indices wrt the original `dataset` to indices wrt `train_fold`."""
        return [self.train_fold.indices.index(i) for i in np.unique(indices)]

    def setup_fold_index(self) -> None:
        """Updates indices in each `torch.utils.data.Subset`.

        It stores the indices as `List[int]` and updates the relative `{labelled, pool}_size` attributes.
        """
        self.pool_fold.indices = np.where(self.train_mask == 0)[0].tolist()
        self.train_fold.indices = np.where(self.train_mask > 0)[0].tolist()

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
        self.pool_pool.indices = np.where((self.train_mask == 0) | (self.train_mask > labelling_step))[0].tolist()
        self.train_pool.indices = np.where((self.train_mask > 0) & (self.train_mask <= labelling_step))[0].tolist()

    """
    Main methods
    """

    def get_statistics(self) -> Dict[str, int]:
        return {
            "total_data_size": self.total_data_size,
            "train_size": self.train_size,
            "pool_size": self.pool_size,
            "num_train_batches": len(self.train_dataloader()),
            "num_pool_batches": len(self.pool_dataloader()),
        }

    def label(self, pool_idx: Union[int, List[int]]) -> None:
        """Moves instances at index `pool_idx` from the `pool_fold` to the `train_fold`.

        Args:
            pool_idx (List[int]): The index (relative to the pool_fold, not the overall data) to label.
        """
        assert isinstance(pool_idx, int) or isinstance(pool_idx, list), ValueError(
            f"`pool_idx` must be of type `int` or `List[int]`, not {type(pool_idx)}."
        )

        pool_idx = [pool_idx] if isinstance(pool_idx, int) else pool_idx

        if len(np.unique(pool_idx)) > len(self.pool_fold):
            raise ValueError(
                f"Pool has {len(self.pool_fold)} instances, cannot label {len(np.unique(pool_idx))} instances."
            )
        if max(pool_idx) > len(self.pool_fold):
            raise ValueError(f"Cannot label instance {max(pool_idx)} for pool dataset of size {len(self.pool_fold)}.")

        # acquire labels
        indices = self.pool_to_original(pool_idx)
        self.train_mask[indices] = self.last_labelling_step + 1  # current labelling step
        self.setup_fold_index()

    """
    DataLoaders
    """

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_fold, **self.train_dataloader_args)

    def val_dataloader(self) -> Optional[DataLoader]:
        return self.datamodule.val_dataloader()

    def test_dataloader(self) -> Optional[DataLoader]:
        return self.datamodule.test_dataloader()

    def pool_dataloader(self) -> DataLoader:
        return DataLoader(self.pool_fold, **self.eval_dataloader_args)


class ActiveDataModuleWithIndex(ActiveDataModule):

    def __init__(
        self,
        train_dataloader: DataLoader,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        test_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
        faiss_index_path: Optional[str] = None,
    ):
        super().__init__(train_dataloader, val_dataloaders, test_dataloaders, datamodule)

        self.faiss_index_path = faiss_index_path
        self._faiss_index = faiss.read_index(faiss_index_path)
        assert len(train_dataloader.dataset) == self._faiss_index.ntotal

    @property
    def faiss_index(self) -> "faiss.IndexIDMap":
        return self._faiss_index

    @faiss_index.setter
    def faiss_index(self, faiss_index_path: str) -> None:
        self._faiss_index = faiss.read_index(faiss_index_path)

    @property
    def faiss_index_size(self) -> int:
        return self.faiss_index.ntotal

    @property
    def is_synced(self) -> bool:
        return self.pool_size == self.faiss_index_size

    def get_array_at_ids(self, indices: List[int]) -> np.ndarray:
        # indices wrt `train_fold`` to original
        indices = self.train_to_original(indices)
        ids = [self.faiss_index.id_map.at(idx) for idx in indices]
        search_query =  np.vstack([self.faiss_index.index.reconstruct(idx) for idx in ids])
        return array_to_numpyfloat32(search_query)

    def search(self, search_query: np.ndarray, k: int) -> List[int]:
        logger.info("Searching `faiss_index`")
        retrieved_indices = self.faiss_index.assign(search_query, k)
        indices = np.unique(retrieved_indices.flatten()).tolist()
        return self.original_to_pool(indices)

    def remove_ids_from_faiss_index(self, indices: List[int]) -> None:
        logger.info("Updating `faiss_index`")
        indices = self.pool_to_original(indices)
        n_removed = self.faiss_index.remove_ids(list_to_numpyint64(indices))
        assert n_removed == len(indices)

    def label(self, pool_idx: Union[int, List[int]]) -> None:
        self.remove_ids_from_faiss_index(pool_idx)
        super().label(pool_idx)




"""
To have the `pool_dataloader` return batches without the label
"""


class ActiveDatamodule2(ActiveDataModule):
    def __init__(
        self,
        input_idx_or_keys: Union[int, str, List[str]],
        train_dataloader: DataLoader,
        val_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        test_dataloaders: Optional[Union[DataLoader, List[DataLoader]]] = None,
        datamodule: Optional[LightningDataModule] = None,
    ):
        super().__init__(train_dataloader, val_dataloaders, test_dataloaders, datamodule)
        self.input_idx_or_keys = input_idx_or_keys

    def setup_folds(self) -> None:
        train_dl = self.datamodule.train_dataloader()
        train_dataset = train_dl.dataset
        self._train_dataloader_args = self._get_dataloader_args(train_dl)
        self.train_fold = Subset(train_dataset, [])
        self.pool_fold = (
            TupleSubset(train_dataset, [], self.input_idx_or_keys)
            if isinstance(self.input_idx_or_keys, int)
            else DictSubset(train_dataset, [], self.input_idx_or_keys)
        )

        eval_dl = self.datamodule.test_dataloader() or self.datamodule.val_dataloader()
        self._eval_dataloader_args = self._get_dataloader_args(eval_dl)
        self._eval_dataloader_args = self._eval_dataloader_args or self._train_dataloader_args

        # states
        self.train_mask = np.zeros((len(train_dataset),), dtype=int)
        self.setup_fold_index()


class EnergizerSubset:
    """This is a shallow reimplementation of `torch.data.utils.Subset`.
    It serves as base class for the source-specific implementation of Subset.
    """

    def __init__(self, dataset: Any, indices: List[int]) -> None:
        """Initializes a subset, akin the `torch.data.utils.Subset`."""
        if not isinstance(indices, list):
            raise MisconfigurationException(f"Indices must be of type `List[int], not {type(indices)}")

        self.dataset = dataset
        self.indices = indices

    def __repr__(self) -> str:
        return (
            f"Subset({{\n    "
            f"original_size: {len(self.dataset)},\n    "
            f"subset_size: {len(self.indices)},\n    "
            f"base_class: {type(self.dataset)},\n}})"
        )

    def __len__(self):
        return len(self.indices)


class TupleSubset(EnergizerSubset):
    def __init__(self, dataset: Any, indices: List[int], input_idx: int) -> None:
        super().__init__(dataset, indices)
        self.input_idx = input_idx

    def __getitem__(self, idx: int) -> Any:
        return self.dataset[self.indices[idx]][self.input_idx]


class DictSubset(EnergizerSubset):
    def __init__(self, dataset: Any, indices: List[int], keys_list: List[str]) -> None:
        super().__init__(dataset, indices)
        self.keys_list = [keys_list] if isinstance(keys_list, str) else keys_list

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return {key: self.dataset[self.indices[idx]][key] for key in self.keys_list}
