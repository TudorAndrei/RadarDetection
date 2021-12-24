import logging
import os
import warnings

import pretty_errors
import torch
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.metrics import classification_report
from torchvision import transforms
from torchvision.transforms import Pad, ToTensor, Normalize
from tqdm import tqdm

from config import config
from data_utils import data_generator
from lightning_models import get_regression_output

warnings.filterwarnings("ignore")
logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)
seed_everything(42)

REGRESSION = "regression"
CLASSIFICATION = "classification"
BATCH_SIZE = 128
NW = 8
EPOCHS = 100
classif_report = True


if __name__ == "__main__":

    # model = config["conv_mixer"]
    # model = config["conv_repeater"]
    model = config["vit"]
    trans_ = transforms.Compose(
        [
            ToTensor(),
            Pad(
                [5, 0, 4, 0],
            ),
            Normalize(0.5,0.5)
        ]
    )
    train_, val_ = data_generator(
        img_dir="./train",
        csv_path=r"train.csv",
        bs=BATCH_SIZE,
        nw=NW,
        transform=trans_,
    )

    logger = TensorBoardLogger("tb_logs", name=model["name"])
    module = model["model"](**model["hyps"])
    trainer = Trainer(
        # fast_dev_run=True,
        # enable_model_summary=False,
        # benchmark=True,
        logger=logger,
        detect_anomaly=True,
        gpus=1,
        # precision=16,
        max_epochs=EPOCHS,
        callbacks=[
            ModelCheckpoint(
                monitor="val/val_loss",
                mode="min",
                dirpath=f"models/{model['name']}",
                filename="radar-epoch{epoch:02d}-val_loss{val/val_loss:.2f}",
                auto_insert_metric_name=False,
            ),
            EarlyStopping(monitor="val/val_loss", patience=6),
            LearningRateMonitor(logging_interval="step"),
            ModelSummary(max_depth=-1)
        ],
    )
    trainer.fit(module, train_, val_)
    if classif_report:
        module.eval()
        module.freeze()

        train_preds = []
        train_gt = []
        for img, label in tqdm(train_):
            if module.op == "classification":
                output = module(img)
                predict = output.argmax(1)
            else:
                label = label.float()
                output = module(img)
                output = torch.squeeze(output, dim=1)
                predict = get_regression_output(output)
            predict = predict.int()
            label = label.int()
            for p, l in zip(predict, label):
                train_preds.append(p)
                train_gt.append(l)
        print(classification_report(train_gt, train_preds))

        preds = []
        gt = []
        for img, label in tqdm(val_):
            if module.op == "classification":
                output = module(img)
                predict = output.argmax(1)
            else:
                label = label.float()
                output = module(img)
                output = torch.squeeze(output, dim=1)
                predict = get_regression_output(output)
            predict = predict.int()
            label = label.int()
            for p, l in zip(predict, label):
                preds.append(p)
                gt.append(l)
        print(classification_report(gt, preds))

    os.system("notify-send 'Job Done ✔️'")
