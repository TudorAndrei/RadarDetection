import os

import pretty_errors
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms
from torchvision.transforms import ConvertImageDtype, Normalize, Pad, ToTensor

from data_utils import data_generator
from lightning_models import ViTLigthning

BATCH_SIZE = 128
NW = 8
EPOCHS = 100


hyps = {
    "num_classes": 5,
    "image_size": (128, 64),
    "patch_size": (32, 32),
    "lr": 0.03,
    "dim": 128,
    "depth": 15,
    "heads": 10,
    "mlp_dim": 2048,
    "dropout": 0.1,
    "emb_dropout": 0.1,
}

trans_ = transforms.Compose(
    [
        ToTensor(),
        Pad(
            [5, 0, 4, 0],
        ),
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ConvertImageDtype(torch.float),
    ]
)
if __name__ == "__main__":

    train_, val_ = data_generator(
        img_dir="./train",
        csv_path=r"train.csv",
        bs=BATCH_SIZE,
        nw=NW,
        transform=trans_,
    )
    logger = TensorBoardLogger("tb_logs", name="vit")

    model = ViTLigthning(**hyps)
    trainer = pl.Trainer(
        # fast_dev_run=True,
        gpus=1,
        logger=logger,
        max_epochs=EPOCHS,
        callbacks=[
            ModelCheckpoint(
                monitor="val/val_loss",
                mode="min",
                dirpath=f"models/vit",
                filename="radar-epoch{epoch:02d}-val_loss{val/val_loss:.2f}",
                auto_insert_metric_name=False,
            ),
            EarlyStopping(monitor="val/val_loss", patience=5),
            LearningRateMonitor(logging_interval="step"),
        ],
    )
    trainer.fit(model, train_, val_)
    os.system("notify-send 'Done Training'")
