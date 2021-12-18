import pretty_errors
import pytorch_lightning as pl
import torch
from torchvision import transforms
from torchvision.transforms import ConvertImageDtype, Pad, ToTensor

from data_utils import data_generator
from lightning_models import CNN_lightning

BATCH_SIZE = 128
NW = 8
EPOCHS = 10

hyps = {
    "size": [3, 16, 32, 64],
    "num_blocks": 1,
    "kernel_size": 3,
    "strides": 2,
    "padding": 1,
}

if __name__ == "__main__":

    trans_ = transforms.Compose(
        [
            ToTensor(),
            Pad(
                [5, 0, 4, 0],
            ),
            ConvertImageDtype(torch.float),
        ]
    )
    train_, val_ = data_generator(
        img_dir="./train",
        csv_path=r"train.csv",
        bs=BATCH_SIZE,
        nw=NW,
        transform=trans_,
    )


    model = CNN_lightning(**hyps)
    # for img, label in val_:
    #     out = model(img)
    #     break
    trainer = pl.Trainer(
        # benchmark=True,
        gpus=1,
        # precision=16,
        max_epochs=EPOCHS,
        callbacks=[
            # ModelCheckpoint(
            #     monitor="val/val_loss",
            #     mode="min",
            #     dirpath=f"models/model{hyp_print}",
            #     filename="radar-epoch{epoch:02d}-val_loss{val/val_loss:.2f}",
            #     auto_insert_metric_name=False,
            # ),
            # EarlyStopping(monitor="val/val_loss", patience=5),
            # LearningRateMonitor(logging_interval="step"),
        ],
    )
    trainer.fit(model, train_, val_)
