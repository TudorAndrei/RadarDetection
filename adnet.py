import os

import pretty_errors
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms
from torchvision.transforms import ConvertImageDtype, Pad, ToTensor

from data_utils import data_generator
from lightning_models import ADNet_lightning

REGRESSION = "regression"
CLASSIFICATION = "classification"
BATCH_SIZE = 64
NW = 8
EPOCHS = 100


if __name__ == "__main__":

    trans_ = transforms.Compose(
        [
            ToTensor(),
            Pad(
                [5, 0, 4, 0],
            ),
        ]
    )
    train_, val_ = data_generator(
        img_dir="./train",
        csv_path=r"train.csv",
        bs=BATCH_SIZE,
        nw=NW,
        transform=trans_,
    )

    logger = TensorBoardLogger("tb_logs", name="adnet")
    hyp = {"lr": 0.003, "op": REGRESSION, "optim": "adam"}
    # hyp = {"lr": 0.003, "op": CLASSIFICATION, "optim": "sgd"}
    model = ADNet_lightning(**hyp)
    trainer = pl.Trainer(
        # fast_dev_run=True,
        enable_model_summary=False,
        # benchmark=True,
        logger=logger,
        terminate_on_nan=True,
        gpus=1,
        # precision=16,
        max_epochs=EPOCHS,
        callbacks=[
            ModelCheckpoint(
                monitor="val/val_loss",
                mode="min",
                dirpath=f"models/adnet",
                filename="radar-epoch{epoch:02d}-val_loss{val/val_loss:.2f}",
                auto_insert_metric_name=False,
            ),
            # EarlyStopping(monitor="val/val_loss", patience=5),
            LearningRateMonitor(logging_interval="step"),
        ],
    )
    trainer.fit(model, train_, val_)
