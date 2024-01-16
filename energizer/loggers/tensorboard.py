from pathlib import Path

from lightning.fabric.loggers.tensorboard import TensorBoardLogger as _TensorBoardLogger
from tbparse import SummaryReader


class TensorBoardLogger(_TensorBoardLogger):
    LOGGER_NAME: str = "tensorboard"

    def __init__(
        self, root_dir: str | Path, name: str | None = "tensorboard_logs", version: int | str | None = None
    ) -> None:
        super().__init__(root_dir, name, version)

    @property
    def logger_name(self) -> str:
        return self.LOGGER_NAME

    def save_to_parquet(self, path: str | Path) -> None:
        SummaryReader(str(self.log_dir)).scalars.to_parquet(path)
