import logging
import os
import warnings

import pretty_errors
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision import transforms
from torchvision.transforms import Pad, ToTensor

from config import config
from data_utils import classif_report, data_generator

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
seed_everything(42)

REGRESSION = "regression"
CLASSIFICATION = "classification"
BATCH_SIZE = 64
NW = 8
EPOCHS = 100


if __name__ == "__main__":

    experiments = list(config.keys())

    # transformation operation
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
    for config_name in experiments:
        # load the model from the config file
        model = config[config_name]

        logger = TensorBoardLogger("tb_logs", name=model["name"])
        module = model["model"](**model["hyps"])
        trainer = Trainer(
            enable_model_summary=False,
            logger=logger,
            detect_anomaly=True,
            gpus=1,
            log_every_n_steps=BATCH_SIZE,
            max_epochs=EPOCHS,
            callbacks=[
                # Save the best model
                ModelCheckpoint(
                    monitor="val/val_loss",
                    mode="min",
                    dirpath=f"models/{model['name']}",
                    filename="radar-epoch{epoch:02d}-val_loss{val/val_loss:.2f}",
                    auto_insert_metric_name=False,
                ),
                EarlyStopping(monitor="val/val_loss", patience=6),
                LearningRateMonitor(logging_interval="step"),
            ],
        )
        trainer.fit(module, train_, val_)

        classif_report(module, train_, val_)

    # Notification for end of trainig
    os.system("notify-send 'Job Done ✔️'")
