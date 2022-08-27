from typing import Union

from numpy import ndarray
from torch import Tensor


def convert_to_numpy(t: Tensor, *_) -> Union[ndarray, float, int]:
    if t.numel() > 1:
        return t.numpy()
    return round(t.item(), 6)
