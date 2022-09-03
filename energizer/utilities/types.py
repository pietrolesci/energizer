from typing import Union, List, Any

from numpy import ndarray, array, int64, float32
from torch import Tensor


def tensor_to_python(t: Tensor, *_) -> Union[ndarray, float, int]:
    if t.numel() > 1:
        return t.numpy()
    return round(t.item(), 6)


def list_to_numpyint64(l: List[int]) -> ndarray:
    return array(l, dtype=int64)


def array_to_numpyfloat32(a: ndarray) -> ndarray:
    return a.astype(float32)

def check_type(expected_type, x: Any) -> None:
    assert isinstance(x, expected_type), TypeError(f"input must be of type {expected_type}, not {type(x)}")