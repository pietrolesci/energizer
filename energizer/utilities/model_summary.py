from collections import OrderedDict
from typing import Dict, List, Tuple, cast

import numpy as np
import torch
from lightning.pytorch.utilities.model_summary.model_summary import (
    _format_summary_table,
    _is_lazy_weight_tensor,
    get_human_readable_count,
)


class LayerSummary:
    def __init__(self, module: torch.nn.Module) -> None:
        super().__init__()
        self._module = module

    @property
    def layer_type(self) -> str:
        """Returns the class name of the module."""
        return str(self._module.__class__.__name__)

    @property
    def num_parameters(self) -> int:
        """Returns the number of parameters in this module."""
        return sum(
            cast(int, np.prod(p.shape)) if not _is_lazy_weight_tensor(p) else 0 for p in self._module.parameters()
        )


class Summary:
    def __init__(self, estimator, max_depth: int = 1) -> None:
        self._estimator = estimator

        if not isinstance(max_depth, int) or max_depth < -1:
            raise ValueError(f"`max_depth` can be -1, 0 or > 0, got {max_depth}.")

        self._max_depth = max_depth
        self._layer_summary = self.summarize()
        # 1 byte -> 8 bits
        # TODO: how do we compute precision_megabytes in case of mixed precision?
        precision = self._estimator.fabric._precision if isinstance(self._estimator.fabric._precision, int) else 32
        self._precision_megabytes = (precision / 8.0) * 1e-6

    @property
    def named_modules(self) -> List[Tuple[str, torch.nn.Module]]:
        mods: List[Tuple[str, torch.nn.Module]]
        if self._max_depth == 0:
            mods = []
        elif self._max_depth == 1:
            # the children are the top-level modules
            mods = list(self._estimator.model.named_children())
        else:
            mods = self._estimator.model.named_modules()
            mods = list(mods)[1:]  # do not include root module (LightningModule)
        return mods

    @property
    def layer_names(self) -> List[str]:
        return list(self._layer_summary.keys())

    @property
    def layer_types(self) -> List[str]:
        return [layer.layer_type for layer in self._layer_summary.values()]

    @property
    def in_sizes(self) -> List:
        return [layer.in_size for layer in self._layer_summary.values()]  # type: ignore

    @property
    def out_sizes(self) -> List:
        return [layer.out_size for layer in self._layer_summary.values()]  # type: ignore

    @property
    def param_nums(self) -> List[int]:
        return [layer.num_parameters for layer in self._layer_summary.values()]

    @property
    def total_parameters(self) -> int:
        return sum(p.numel() if not _is_lazy_weight_tensor(p) else 0 for p in self._estimator.model.parameters())

    @property
    def trainable_parameters(self) -> int:
        return sum(
            p.numel() if not _is_lazy_weight_tensor(p) else 0
            for p in self._estimator.model.parameters()
            if p.requires_grad
        )

    @property
    def model_size(self) -> float:
        # todo: seems it does not work with quantized models - it returns 0.0
        return self.total_parameters * self._precision_megabytes

    def summarize(self) -> Dict[str, LayerSummary]:
        summary = OrderedDict((name, LayerSummary(module)) for name, module in self.named_modules)

        if self._max_depth >= 1:
            # remove summary entries with depth > max_depth
            for k in [k for k in summary if k.count(".") >= self._max_depth]:
                del summary[k]

        return summary

    def _get_summary_data(self) -> List[Tuple[str, List[str]]]:
        """Makes a summary listing with:

        Layer Name, Layer Type, Number of Parameters, Input Sizes, Output Sizes, Model Size
        """
        arrays = [
            (" ", list(map(str, range(len(self._layer_summary))))),
            ("Name", self.layer_names),
            ("Type", self.layer_types),
            ("Params", list(map(get_human_readable_count, self.param_nums))),
        ]

        return arrays

    def __str__(self) -> str:
        arrays = self._get_summary_data()

        total_parameters = self.total_parameters
        trainable_parameters = self.trainable_parameters
        model_size = self.model_size

        return _format_summary_table(total_parameters, trainable_parameters, model_size, *arrays)

    def __repr__(self) -> str:
        return str(self)


def summarize(estimator, max_depth: int = 1) -> str:
    model_summary = Summary(estimator, max_depth)
    summary_data = model_summary._get_summary_data()
    total_parameters = model_summary.total_parameters
    trainable_parameters = model_summary.trainable_parameters
    model_size = model_summary.model_size

    summary = _format_summary_table(total_parameters, trainable_parameters, model_size, *summary_data)
    if estimator.fabric.device.type == "cuda":
        s = "{:<{}}"
        summary += "\n" + s.format(f"{torch.cuda.max_memory_allocated() / 1e9:.02f} GB", 10)
        summary += "CUDA Memory used"

    return summary
