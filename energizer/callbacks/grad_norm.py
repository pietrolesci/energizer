from typing import Union

from lighting_fabric.wrappers import _FabricModule, _FabricOptimizer
from lightning.pytorch.utilities.grads import grad_norm

from energizer.callbacks import Callback
from energizer.estimators.estimator import Estimator


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

    def on_before_optimizer_step(self, estimator: Estimator, model: _FabricModule, optimizer: _FabricOptimizer) -> None:
        norms = grad_norm(model, norm_type=self.norm_type, group_separator=self.group_separator)
        estimator.log_dict(norms, step=estimator.progress_tracker.global_step)
