import torch
from datasets import load_dataset
from pytorch_lightning import LightningModule, Trainer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding


class Model(LightningModule):
    def __init__(self):
        super().__init__()
        self.backbone = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny", num_labels=4)
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, batch):
        return self.backbone(**batch).logits

    def step(self, batch, *args, **kwargs):
        y = batch.pop("labels")
        y_hat = self(batch)
        return self.loss(y_hat, y), y_hat

    def training_step(self, batch, *args, **kwargs):
        # self.print("TRAIN")
        loss, y_hat = self.step(batch, *args, **kwargs)
        self.log("train_loss", loss)

    def validation_step(self, batch, *args, **kwargs):
        # self.print("VAL")
        loss, y_hat = self.step(batch, *args, **kwargs)
        self.log("val_loss", loss)

    def test_step(self, batch, batch_idx, *args, **kwargs):
        # self.print("TEST")
        loss, y_hat = self.step(batch, *args, **kwargs)
        self.log("test_loss", loss)
        return loss

    def test_step_end(self, outputs):
        self.print(outputs)

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.01)


if __name__ == "__main__":

    tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny")
    ds = load_dataset("pietrolesci/ag_news", name="concat")
    ds = ds.map(lambda ex: tokenizer(ex["text"], return_token_type_ids=False), batched=True)
    ds = ds.with_format(columns=["input_ids", "attention_mask", "label"])

    model = Model()
    dataset = ds["test"].select(list(range(1_000)))
    dl = DataLoader(dataset, collate_fn=DataCollatorWithPadding(tokenizer), batch_size=10)
    trainer = Trainer(strategy="ddp2", accelerator="cpu", devices=2)

    trainer.test(model, dataloaders=[dl])
