# import inspect
import contextlib
import copy
import os
import random
import re
from collections.abc import Generator, Iterator
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
from lightning.fabric.utilities.seed import _collect_rng_states, _set_rng_states
from lightning_utilities.core.apply_func import apply_to_collection
from numpy import generic, ndarray
from numpy.random import RandomState
from sklearn.utils import resample
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


@dataclass
class Args:
    """Dataclass which is subscriptable like a dict"""

    def to_dict(self) -> dict[str, Any]:
        out = copy.deepcopy(self.__dict__)
        return out

    def __getitem__(self, k: str) -> Any:
        return self.__dict__[k]

    def __iter__(self) -> Iterator[str]:
        return iter(self.__dict__)

    def __len__(self) -> int:
        return len(self.__dict__)


def parse_locals(vars) -> dict:
    return {k: v for k, v in vars.items() if k not in ("self", "__class__")}


def camel_to_snake(name: str) -> str:
    name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    return re.sub("([a-z0-9])([A-Z])", r"\1_\2", name).lower()


def tensor_to_python(t: Tensor, *_) -> ndarray | float | int:
    """Converts `torch.Tensor` to a `numpy.ndarray` or python scalar type."""
    # if t.numel() > 1:
    cpu_t = t.detach().cpu()
    if cpu_t.dtype == torch.bfloat16:
        cpu_t = cpu_t.to(torch.float16)
    return cpu_t.numpy()
    # return round(t.detach().cpu().item(), 6)


def make_dict_json_serializable(d: dict) -> dict:
    return {k: round(v.item(), 6) if isinstance(v, ndarray | generic) else v for k, v in d.items()}


def move_to_cpu(output: Any) -> Any:
    args = (Tensor, tensor_to_python, "cpu")
    return apply_to_collection(output, *args)


def ld_to_dl(ld: list[dict]) -> dict[str, list]:
    return {k: [dic[k] for dic in ld] for k in ld[0]}


def dl_to_ld(dl: dict[str, list]) -> list[dict]:
    return [dict(zip(dl, t, strict=False)) for t in zip(*dl.values(), strict=False)]


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


def set_deterministic(deterministic: bool | Literal["warn_only"]) -> None:
    kwargs = {}
    if isinstance(deterministic, str):
        assert deterministic == "warn_only", "deterministic can be a bool or `warn_only`"
        deterministic, kwargs = True, {"warn_only": True}

    # NOTE: taken from the lightning Trainer
    torch.use_deterministic_algorithms(deterministic, **kwargs)

    if deterministic:
        # fixing non-deterministic part of horovod
        # https://github.com/Lightning-AI/lightning/pull/1572/files#r420279383
        os.environ["HOROVOD_FUSION_THRESHOLD"] = "0"

        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        # Enable CUDNN deterministic mode
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore


def _pad(inputs: list[list[int | float]], padding_value: int | float, max_length: int) -> Tensor:
    # truncate -> convert to tensor -> pad
    return pad_sequence([torch.tensor(t[:max_length]) for t in inputs], batch_first=True, padding_value=padding_value)


def sample(
    indices: list[int],
    size: int,
    random_state: RandomState,
    mode: Literal["uniform", "stratified"] = "uniform",
    **kwargs,
) -> list[int]:
    """Makes sure to seed everything consistently."""

    if mode == "uniform":
        sample = random_state.choice(indices, size=size, replace=False).tolist()

    elif mode == "stratified":
        assert "labels" in kwargs, ValueError("Must pass `labels` for stratified sampling.")
        sample = resample(
            indices, replace=False, stratify=kwargs.get("labels"), n_samples=size, random_state=random_state
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
