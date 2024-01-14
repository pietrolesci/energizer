from pathlib import Path
from typing import Optional, Union

from lightning.fabric.loggers.tensorboard import TensorBoardLogger as _TensorBoardLogger
from tbparse import SummaryReader


class TensorBoardLogger(_TensorBoardLogger):
    LOGGER_NAME: str = "tensorboard"

    def __init__(
        self,
        root_dir: Union[str, Path],
        name: Optional[str] = "tensorboard_logs",
        version: Optional[Union[int, str]] = None,
    ) -> None:
        super().__init__(root_dir, name, version)

    @property
    def logger_name(self) -> str:
        return self.LOGGER_NAME

    def save_to_parquet(self, path: Union[str, Path]) -> None:
        SummaryReader(str(self.log_dir)).scalars.to_parquet(path)
