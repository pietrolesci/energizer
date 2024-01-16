from pathlib import Path

import torch

from energizer.callbacks.base import Callback


class PytorchTensorboardProfiler(Callback):
    def __init__(
        self, dirpath: str | Path, wait: int = 1, warmup: int = 1, active: int = 1, repeat: int = 2, **kwargs
    ) -> None:
        self.prof = torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(dirpath)),
            **kwargs,
        )

    def on_train_epoch_start(self, *args, **kwargs) -> None:
        self.prof.start()

    def on_train_batch_end(self, *args, **kwargs) -> None:
        self.prof.step()

    def on_train_epoch_end(self, *args, **kwargs) -> None:
        self.prof.stop()
