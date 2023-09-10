from pathlib import Path
from typing import Union

from lightning.fabric.loggers.tensorboard import TensorBoardLogger as _TensorBoardLogger
from tbparse import SummaryReader


class TensorBoardLogger(_TensorBoardLogger):
    def save_to_parquet(self, path: Union[str, Path]) -> None:
        SummaryReader(str(self.log_dir)).scalars.to_parquet(path)
