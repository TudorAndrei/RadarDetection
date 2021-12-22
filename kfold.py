import logging

import numpy as np
import pandas as pd
import pretty_errors
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from sklearn.model_selection import KFold
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Subset
from torchvision import transforms
from torchvision.transforms import Pad, ToTensor

from data_utils import kfold_generator
from runs import runs

logging.getLogger("pytorch_lightning").setLevel(logging.WARNING)

seed_everything(42)

REGRESSION = "regression"
CLASSIFICATION = "classification"
BATCH_SIZE = 64
NW = 1
EPOCHS = 100
trans_ = transforms.Compose(
    [
        ToTensor(),
        Pad(
            [5, 0, 4, 0],
        ),
    ]
)

if __name__ == "__main__":
    search = runs
    csv_path = r"train.csv"
    img_dir = "./train"
    train_dataset = kfold_generator(img_dir, csv_path, trans_)
    for i, (name, model) in enumerate(search.items()):
        mmae = []
        kf = KFold(n_splits=5, random_state=42, shuffle=True).split(
            range(len(train_dataset))
        )
        for i, (train_index, valid_index) in enumerate(kf):
            trainer = Trainer(
                # fast_dev_run=False,
                enable_progress_bar=False,
                # logger=TensorBoardLogger("tb_logs", name=f"kfold_{name}{i}"),
                enable_model_summary=False,
                max_epochs=EPOCHS,
                # benchmark=True,
                detect_anomaly=True,
                gpus=1,
                callbacks=[
                    EarlyStopping(monitor="val/val_loss", patience=5),
                ],
            )
            train_data = Subset(train_dataset, train_index)
            valid_data = Subset(train_dataset, valid_index)
            trainer.fit(
                model=model["model"](**model["hyps"]),
                train_dataloaders=DataLoader(
                    train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
                ),
                val_dataloaders=DataLoader(
                    valid_data, batch_size=BATCH_SIZE, num_workers=4
                ),
            )
            val_result = trainer.validate(
                model=model["model"](**model["hyps"]),
                dataloaders=DataLoader(
                    valid_data, batch_size=BATCH_SIZE, num_workers=4
                ),
                verbose=False,
            )
            mmae.append(val_result[0]["val/mae"])
        print(f"{model} : {np.mean(mmae)}")
        # break
