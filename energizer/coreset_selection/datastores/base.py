from torch.utils.data import DataLoader

from energizer.datastores.base import Datastore
from energizer.enums import RunningStage
from pathlib import Path

class CoresetDatastore(Datastore):
    CACHE_DIR: Path = Path("./")
    CACHE_FILENAME: str = "statistics"

    def train_loader(self, *args, **kwargs) -> DataLoader:
        return self.get_loader(RunningStage.TRAIN, *args, **kwargs)  # type: ignore

    def pool_loader(self, *args, **kwargs) -> DataLoader:
        return self.get_loader(RunningStage.POOL, *args, **kwargs)  # type: ignore
    
    def record(self, buffer: list) -> None:
        if len(cache_buffer) == self.tracker.log_interval or self.tracker.is_last_epoch:
            self.write_to_cache(cache_buffer)

        if not self.CACHE_DIR.exists():
            self.CACHE_DIR.mkdir(exist_ok=True, parents=True)

        


