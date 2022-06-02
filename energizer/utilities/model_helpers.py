from functools import partial
from typing import Optional, Type
from unittest.mock import Mock

import pytorch_lightning as pl

from energizer.learners.base import Learner


def is_overridden(method_name: str, instance: Optional[object] = None, parent: Optional[Type[object]] = None) -> bool:
    if instance is None:
        # if `self.lightning_module` was passed as instance, it can be `None`
        return False

    if parent is None:
        if isinstance(instance, Learner):
            parent = Learner
        elif isinstance(instance, pl.LightningDataModule):
            parent = pl.LightningDataModule
        elif isinstance(instance, pl.Callback):
            parent = pl.Callback
        if parent is None:
            raise ValueError("Expected a parent")

    instance_attr = getattr(instance, method_name, None)
    if instance_attr is None:
        return False
    # `functools.wraps()` support
    if hasattr(instance_attr, "__wrapped__"):
        instance_attr = instance_attr.__wrapped__
    # `Mock(wraps=...)` support
    if isinstance(instance_attr, Mock):
        # access the wrapped function
        instance_attr = instance_attr._mock_wraps
    # `partial` support
    elif isinstance(instance_attr, partial):
        instance_attr = instance_attr.func
    if instance_attr is None:
        return False

    parent_attr = getattr(parent, method_name, None)
    if parent_attr is None:
        raise ValueError("The parent should define the method")

    return instance_attr.__code__ != parent_attr.__code__
