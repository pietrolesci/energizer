from collections.abc import Iterable
from typing import Union

import torch
from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer

from energizer.callbacks import Callback
from energizer.estimator import Estimator


def grad_norm(module: torch.nn.Module, norm_types: list[float], group_separator: str = "/") -> dict[str, float]:
    # compute on device
    return {
        f"grad_{norm_type}_norm{group_separator}{n}": torch.linalg.vector_norm(
            p.grad.detach().data.flatten(), ord=norm_type
        ).item()
        for norm_type in norm_types
        for n, p in module.named_parameters()
        if p.grad is not None and p.requires_grad
    }


def empirical_fim_norm(
    module: torch.nn.Module, norm_types: list[float], group_separator: str = "/"
) -> dict[str, float]:
    # compute on device
    return {
        f"efim_{norm_type}_norm{group_separator}{n}": torch.linalg.vector_norm(
            p.grad.detach().data.flatten() ** 2, ord=norm_type
        ).item()
        for norm_type in norm_types
        for n, p in module.named_parameters()
        if p.grad is not None and p.requires_grad
    }


def empirical_fim_trace(module: torch.nn.Module, group_separator: str = "/") -> dict[str, float]:
    # compute on device
    return {
        f"efim_trace{group_separator}{n}": torch.sum(p.grad.detach().data.flatten() ** 2).item()
        for n, p in module.named_parameters()
        if p.grad is not None and p.requires_grad
    }


def update_size_norm(
    differences: list[tuple[str, torch.Tensor]], norm_types: list[float], group_separator: str = "/"
) -> dict[str, float]:
    """Compute each parameter's gradient update (difference between after and before update) norm.

    Args:
        current_params: the new parameters after the update
        previous_params: the paramenters before the update
        norm_type: The type of the used p-norm, cast to float if necessary.
            Can be ``'inf'`` for infinity norm.
        group_separator: The separator string used by the logger to group
            the gradients norms in their own subfolder instead of the logs one.

    Return:
        norms: The dictionary of p-norms of each parameter's gradient and
            a special entry for the total p-norm of the gradients viewed
            as a single vector.

    """

    # compute on device
    return {
        f"update_{norm_type}_norm{group_separator}{n}": torch.linalg.vector_norm(diff.flatten(), ord=norm_type).item()
        for norm_type in norm_types
        for n, diff in differences
    }


def relative_stdlog10_update_size(
    differences: list[tuple[str, torch.Tensor]], previous_params: list[torch.Tensor], group_separator: str = "/"
) -> dict[str, float]:
    # compute on device
    return {
        f"relative_stdlog10_update{group_separator}{n}": (diff.std() / p_before.std()).log10().item()
        for (n, diff), p_before in zip(differences, previous_params)
    }


def relative_norm_update_size(
    differences: list[tuple[str, torch.Tensor]],
    previous_params: list[torch.Tensor],
    norm_types: list[float],
    group_separator: str = "/",
) -> dict[str, float]:
    # compute on device
    return {
        f"relative_update_{norm_type}_norm{group_separator}{n}": torch.linalg.vector_norm(
            diff.flatten(), ord=norm_type
        ).item()
        / torch.linalg.vector_norm(p_before.flatten(), ord=norm_type).item()
        for norm_type in norm_types
        for (n, diff), p_before in zip(differences, previous_params)
    }


class GradNorm(Callback):
    def __init__(
        self, norm_types: Union[float, int, str, list[Union[float, int, str]]], group_separator: str = "/"
    ) -> None:
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

        if not isinstance(norm_types, Iterable) or isinstance(norm_types, str):
            norm_types = [norm_types]

        self.norm_types = [float(i) for i in norm_types]

        assert all(i > 0 for i in self.norm_types), ValueError(
            f"`norm_type` must be a positive number or 'inf' (infinity norm). Got {self.norm_types}"
        )

    def on_before_optimizer(self, estimator: Estimator, model: _FabricModule, optimizer: _FabricOptimizer) -> None:
        norms = grad_norm(model, norm_types=self.norm_types, group_separator=self.group_separator)
        estimator.log_dict(norms, step=estimator.tracker.global_step)


class EmpiricalFIMNorm(GradNorm):
    def on_before_optimizer(self, estimator: Estimator, model: _FabricModule, optimizer: _FabricOptimizer) -> None:
        norms = empirical_fim_norm(model, norm_types=self.norm_types, group_separator=self.group_separator)
        estimator.log_dict(norms, step=estimator.tracker.global_step)


class EmpiricalFIMTrace(Callback):
    def __init__(self, group_separator: str = "/") -> None:
        self.group_separator = group_separator

    def on_before_optimizer(self, estimator: Estimator, model: _FabricModule, optimizer: _FabricOptimizer) -> None:
        norms = empirical_fim_trace(model, group_separator=self.group_separator)
        estimator.log_dict(norms, step=estimator.tracker.global_step)


class ParameterUpdateStats(GradNorm):
    _previous_params: list[torch.Tensor] = []

    def __init__(
        self,
        norm_types: Union[float, int, str, list[Union[float, int, str]]],
        group_separator: str = "/",
        return_update_size_norm: bool = True,
        return_relative_std_update: bool = True,
        return_relative_norm_update: bool = True,
    ) -> None:
        """Compute each parameter's gradient's norm and their overall norm.

        The overall norm is computed over all gradients together, as if they
        were concatenated into a single vector.

        Args:
            norm_type: The type of the used p-norm, cast to float if necessary.
                Can be ``'inf'`` for infinity norm.
            group_separator: The separator string used by the logger to group
                the gradients norms in their own subfolder instead of the logs one.

        """
        super().__init__(norm_types, group_separator)
        self.return_update_size_norm = return_update_size_norm
        self.return_relative_std_update = return_relative_std_update
        self.return_relative_norm_update = return_relative_norm_update

    def on_before_optimizer(self, estimator: Estimator, model: _FabricModule, optimizer: _FabricOptimizer) -> None:
        self._previous_params = [
            p.data.clone().detach() for _, p in model.named_parameters() if p.grad is not None and p.requires_grad
        ]

    def on_after_optimizer(self, estimator: Estimator, model: _FabricModule, optimizer: _FabricOptimizer) -> None:
        # assert len(current_params) == len(previous_params), ValueError(
        #     f"Current and previous parameter lists are not the same: {len(current_params)=} and {len(previous_params)=}"
        # )

        current_params = (
            (n, p.data.clone().detach()) for n, p in model.named_parameters() if p.grad is not None and p.requires_grad
        )

        diffs = [(n, p_after - p_before) for (n, p_after), p_before in zip(current_params, self._previous_params)]

        logs = {}
        if self.return_update_size_norm:
            update_size_norms = update_size_norm(
                differences=diffs, norm_types=self.norm_types, group_separator=self.group_separator
            )
            logs.update(update_size_norms)

        if self.return_relative_std_update:
            relative_std = relative_stdlog10_update_size(
                differences=diffs, previous_params=self._previous_params, group_separator=self.group_separator
            )
            logs.update(relative_std)

        if self.return_relative_norm_update:
            relative_norms = relative_norm_update_size(
                differences=diffs,
                previous_params=self._previous_params,
                norm_types=self.norm_types,
                group_separator=self.group_separator,
            )
            logs.update(relative_norms)

        if len(logs) > 0:
            estimator.log_dict(logs, step=estimator.tracker.global_step)

        # free memory
        self._previous_params = []
