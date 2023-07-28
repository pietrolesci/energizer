# import inspect
import contextlib
import os
import random
from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch
from lightning_fabric.utilities.seed import _collect_rng_states, _set_rng_states
from lightning_utilities.core.apply_func import apply_to_collection
from numpy import generic, ndarray
from numpy.random import RandomState
from sklearn.utils import resample
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

# from torch.utils.data import BatchSampler, SequentialSampler
# from energizer.enums import RunningStage


def tensor_to_python(t: Tensor, *_) -> Union[ndarray, float, int]:
    """Converts `torch.Tensor` to a `numpy.ndarray` or python scalar type."""
    # if t.numel() > 1:
    return t.detach().cpu().numpy()
    # return round(t.detach().cpu().item(), 6)


def make_dict_json_serializable(d: Dict) -> Dict:
    return {k: round(v.item(), 6) if isinstance(v, (ndarray, generic)) else v for k, v in d.items()}


def move_to_cpu(output: Any) -> Any:
    args = (Tensor, tensor_to_python, "cpu")
    return apply_to_collection(output, *args)


def ld_to_dl(ld: List[Dict]) -> Dict[str, List]:
    return {k: [dic[k] for dic in ld] for k in ld[0]}


@contextlib.contextmanager
def local_seed(seed: int) -> Generator[None, None, None]:
    """A context manager that allows to locally change the seed.

    Upon exit from the context manager it resets the random number generator state
    so that the operations that happen in the context do not affect randomness outside
    of it.
    """
    # collect current states
    states = _collect_rng_states()

    # set seed in the context
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # run code in context
    yield

    # reset states when exiting the context
    _set_rng_states(states)


def init_deterministic(deterministic: bool) -> None:
    # NOTE: taken from the lightning Trainer
    torch.use_deterministic_algorithms(deterministic)
    if deterministic:
        # fixing non-deterministic part of horovod
        # https://github.com/Lightning-AI/lightning/pull/1572/files#r420279383
        os.environ["HOROVOD_FUSION_THRESHOLD"] = "0"

        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def _pad(inputs: List[int], padding_value: float, max_length: int) -> Tensor:
    # truncate -> convert to tensor -> pad
    return pad_sequence(
        [torch.tensor(t[:max_length]) for t in inputs],
        batch_first=True,
        padding_value=padding_value,
    )


def sample(
    indices: List[int],
    size: int,
    random_state: RandomState,
    labels: Optional[List[int]] = None,
    sampling: Optional[str] = None,
) -> List[int]:
    """Makes sure to seed everything consistently."""

    if sampling is None or sampling == "uniform":
        sample = random_state.choice(indices, size=size, replace=False).tolist()

    elif sampling == "stratified" and labels is not None:
        sample = resample(
            indices,
            replace=False,
            stratify=labels,
            n_samples=size,
            random_state=random_state,
        )

    else:
        raise ValueError("Only `uniform` and `stratified` are supported by default.")

    assert len(set(sample)) == size

    return sample  # type: ignore


def sequential_numbers() -> Generator[int, Any, None]:
    n = 0
    while True:
        yield n
        n += 1
