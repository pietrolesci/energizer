from typing import Dict, Union
from energizer.utilities import move_to_cpu

from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer
from energizer.callbacks import Callback
from energizer.estimator import Estimator

import torch
from torch.nn import Module


def grad_norm(module: Module, norm_type: Union[float, int, str], group_separator: str = "/") -> Dict[str, float]:
    """Compute each parameter's gradient's norm and their overall norm.

    The overall norm is computed over all gradients together, as if they
    were concatenated into a single vector.

    Args:
        module: :class:`torch.nn.Module` to inspect.
        norm_type: The type of the used p-norm, cast to float if necessary.
            Can be ``'inf'`` for infinity norm.
        group_separator: The separator string used by the logger to group
            the gradients norms in their own subfolder instead of the logs one.

    Return:
        norms: The dictionary of p-norms of each parameter's gradient and
            a special entry for the total p-norm of the gradients viewed
            as a single vector.

    """
    norm_type = float(norm_type)
    if norm_type <= 0:
        raise ValueError(f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {norm_type}")

    # compute on device
    norms = {
        f"grad_{norm_type}_norm{group_separator}{name}": torch.linalg.vector_norm(
            p.grad.detach().data.flatten(), ord=norm_type
        )
        for name, p in module.named_parameters()
        if p.grad is not None and p.requires_grad
    }
    if norms:
        total_norm = torch.linalg.vector_norm(torch.tensor(list(norms.values())).flatten(), ord=norm_type)
        norms[f"grad_{norm_type}_norm_total"] = total_norm

    return norms


class GradNorm(Callback):
    def __init__(self, norm_type: Union[float, int, str], group_separator: str = "/") -> None:
        """Compute each parameter's gradient's norm and their overall norm.

        The overall norm is computed over all gradients together, as if they
        were concatenated into a single vector.

        Args:
            norm_type: The type of the used p-norm, cast to float if necessary.
                Can be ``'inf'`` for infinity norm.
            group_separator: The separator string used by the logger to group
                the gradients norms in their own subfolder instead of the logs one.

        """
        self.group_separator = group_separator
        self.norm_type = float(norm_type)
        if self.norm_type <= 0:
            raise ValueError(f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {self.norm_type}")

    def on_before_optimizer(self, estimator: Estimator, model: _FabricModule, optimizer: _FabricOptimizer) -> None:
        # compute on device
        norms = grad_norm(model, norm_type=self.norm_type, group_separator=self.group_separator)

        # then move to cpu
        norms = move_to_cpu(norms)

        estimator.log_dict(norms, step=estimator.tracker.global_step)
