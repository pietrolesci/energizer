from typing import List

from pytorch_lightning.callbacks import Callback
from pytorch_lightning.callbacks.progress.base import ProgressBarBase

from energizer.callbacks.tqdm_progress import TQDMProgressBarActiveLearning
from energizer.learners.hooks import CallBackActiveLearningHooks


def patch_callbacks(callbacks: List[Callback]) -> List[Callback]:
    def add_pool_hooks(callback: Callback) -> Callback:
        hook_names = [m for m in dir(CallBackActiveLearningHooks) if not m.startswith("_")]
        for name in hook_names:
            if not hasattr(callback, name):
                setattr(callback, name, getattr(CallBackActiveLearningHooks, name))
        return callback

    new_callbacks = []
    for c in callbacks:
        if isinstance(c, ProgressBarBase):
            prog_bar = TQDMProgressBarActiveLearning(process_position=c.process_position, refresh_rate=c.refresh_rate)
            prog_bar = add_pool_hooks(prog_bar)
            new_callbacks.append(prog_bar)
        else:
            new_callbacks.append(add_pool_hooks(c))

    return new_callbacks
