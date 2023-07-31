import inspect
import re
from types import ModuleType
from typing import Any, Callable, Generator, List, Optional, Tuple, Type

import torch
import torch_optimizer
import transformers
from transformers.optimization import TYPE_TO_SCHEDULER_FUNCTION

from energizer.utilities import camel_to_snake


class Registry(dict):
    def register_functions(self, module: ModuleType, filter_fn: Optional[Callable] = None) -> None:
        filter_fn = filter_fn if filter_fn is not None else lambda k, v: True
        for k, v in inspect.getmembers(module, inspect.isfunction):
            if filter_fn(k, v):
                self[k.lower()] = v

    def register_classes(
        self,
        module: ModuleType,
        base_cls: Type,
        override: bool = False,
        to_snake_case: bool = False,
    ) -> None:
        """This function is an utility to register all classes from a module."""
        for cls in self.get_members(module, base_cls):
            key = camel_to_snake(cls.__name__) if to_snake_case else cls.__name__
            if key not in self or override:
                self[key.lower()] = cls

    @staticmethod
    def get_members(module: ModuleType, base_cls: Type) -> Generator[Type, None, None]:
        return (
            cls
            for _, cls in inspect.getmembers(module, predicate=inspect.isclass)
            if issubclass(cls, base_cls) and cls != base_cls
        )

    @property
    def names(self) -> List[str]:
        """Returns the registered names."""
        return list(self.keys())

    @property
    def classes(self) -> Tuple[Type, ...]:
        """Returns the registered classes."""
        return tuple(self.values())

    def __str__(self) -> str:
        return f"Registered objects: {self.names}"

    def __getitem__(self, __key: str) -> Any:
        return super().__getitem__(__key.lower())


# redefine this to have torch and transformers overwrite torch_optimizer
OPTIMIZER_REGISTRY = Registry()
OPTIMIZER_REGISTRY.register_classes(torch_optimizer, torch.optim.Optimizer)
OPTIMIZER_REGISTRY.register_classes(transformers.optimization, torch.optim.Optimizer, override=True)
OPTIMIZER_REGISTRY.register_classes(torch.optim, torch.optim.Optimizer, override=True)


# add trasformers convenience functions
SCHEDULER_REGISTRY = Registry()
SCHEDULER_REGISTRY.register_classes(torch.optim.lr_scheduler, torch.optim.lr_scheduler._LRScheduler)
SCHEDULER_REGISTRY.update({v.__name__[4:]: v for v in TYPE_TO_SCHEDULER_FUNCTION.values()})


def filter_fn(k: str, v: Any) -> bool:
    signature = inspect.signature(v).parameters
    return not k.startswith("_") and all(i in signature for i in ["input", "target"])


LOSS_FUNCTIONS_REGISTRY = Registry()
LOSS_FUNCTIONS_REGISTRY.register_functions(torch.nn.functional, filter_fn)
LOSS_FUNCTIONS_REGISTRY.register_classes(torch.nn.modules.loss, torch.nn.modules.loss._Loss, to_snake_case=True)
