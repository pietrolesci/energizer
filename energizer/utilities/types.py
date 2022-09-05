from typing import Any, List, Union

from numpy import array, float32, int64, ndarray
from torch import Tensor


def tensor_to_python(t: Tensor, *_) -> Union[ndarray, float, int]:
    """Converts `torch.Tensor` to a `numpy.ndarray` or python scalar type."""
    if t.numel() > 1:
        return t.numpy()
    return round(t.item(), 6)


def list_to_numpyint64(list_of_ints: List[int]) -> ndarray:
    """Converts python list to `numpy.ndarray` of type int64."""
    return array(list_of_ints, dtype=int64)


def array_to_numpyfloat32(numpy_array: ndarray) -> ndarray:
    """Casts `numpy.ndarray` to type float32."""
    return numpy_array.astype(float32)


def check_type(expected_type, obj: Any) -> None:
    """Asserts whether obj is of the specified type."""
    assert isinstance(obj, expected_type), TypeError(f"input must be of type {expected_type}, not {type(obj)}")
