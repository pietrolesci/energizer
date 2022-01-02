from typing import Any


class PoolHooksMixin:
    def _call_callback_hook(self, hook_name, *args, **kwargs):
        """Call `hook_name` from each callback passed to the trainer."""
        for callback in self.trainer.callbacks:
            fn = getattr(callback, hook_name, None)
            if callable(fn):
                fn(self, *args, **kwargs)

    def on_pool_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        pass

    def on_pool_batch_end(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        pass

    def on_pool_epoch_start(self) -> None:
        pass

    def on_pool_epoch_end(self) -> None:
        pass

    def on_pool_start(self) -> None:
        pass

    def on_pool_end(self) -> None:
        pass

    def on_pool_model_eval(self) -> None:
        pass

    def on_pool_model_train(self) -> None:
        pass
