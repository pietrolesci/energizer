from mnist_example import MNISTModel, MNISTDataModule
from torch.utils.data import DataLoader
from energizer.strategies.strategies import EntropyStrategy
from energizer.strategies.inference import MCDropout, Deterministic
from energizer.trainer import Trainer


if __name__ == "__main__":
    datamodule = MNISTDataModule(batch_size=32)
    datamodule.setup()

    model = MNISTModel()
    adapter = MCDropout(model)
    strategy = EntropyStrategy(adapter, query_size=3)

    trainer = Trainer(max_epochs=1)
    trainer.active_fit(strategy, datamodule=datamodule)
