from lightning.fabric.wrappers import _FabricModule, _FabricOptimizer
from energizer.callbacks import Callback
from energizer.estimator import Estimator


class LearningRateMonitor(Callback):
    def __init__(self, prefix: str = "lr_monitor/") -> None:
        super().__init__()
        self.prefix = prefix

    def on_before_optimizer(self, estimator: Estimator, model: _FabricModule, optimizer: _FabricOptimizer) -> None:
        last_lrs = [group["lr"] for group in optimizer.param_groups]

        if len(last_lrs) > 1:
            lrs = {f"{self.prefix}lr_param_group{idx}": lr for idx, lr in enumerate(last_lrs)}
            estimator.log_dict(lrs, step=estimator.tracker.global_step)
        else:
            estimator.log(f"{self.prefix}lr", next(iter(last_lrs)), step=estimator.tracker.global_step)
