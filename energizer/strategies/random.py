from typing import Any, Optional, Tuple

import torch
from torch import Tensor

from energizer.strategies.base import EnergizerStrategy


class RandomStrategy(EnergizerStrategy):
    def __init__(self) -> None:
        """Create the RandomStrategy."""
        super().__init__(None)

    def pool_step(self, batch: Any, batch_idx: int, dataloader_idx: Optional[int] = None) -> Tuple[Tensor, Tensor]:
        pass

    def pool_step_end(self, outputs: Tuple[Tensor, Tensor]) -> None:
        pass

    def pool_epoch_end(self, outputs: Any) -> None:
        self.indices = torch.randint(
            low=0,
            high=self.trainer.datamodule.pool_size,  # type: ignore
            size=(self.query_size,),
        )
