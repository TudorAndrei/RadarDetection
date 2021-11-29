from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data_utils import DataGenerator
from convmixer import ConvMixerModule
import pytorch_lightning as pl

pl.seed_everything(42)

BATCH_SIZE = 8
NW = 8
EPOCHS = 200

hyps = { "size": 7,
    "num_blocks": 5,
    "kernel_size": 9,
    "patch_size": 8,
    "num_classes": 5,
}

hyp_print = ''
for key, value in hyps.items():
    hyp_print += f"_{key}_{value}"

if __name__ == "__main__":

    dataset = DataGenerator(
        img_dir="./train", path=r"train.csv", bs=BATCH_SIZE, nw=NW
    )
    train_ = dataset.get_train()
    val_ = dataset.get_val()
    main_model = ConvMixerModule(**hyps)
    trainer = pl.Trainer(
        # fast_dev_run=True,
        benchmark=True,
        gpus=1,
        max_epochs=EPOCHS,
        callbacks=[
            ModelCheckpoint(
                monitor="val_loss",
                dirpath=f"models/{hyp_print}",
                filename="radar-{epoch:02d}-{val_loss:.2f}-{val_accuracy:.2f}",
            ),
            EarlyStopping(monitor="val_loss", patience=20)
        ]
    )
    trainer.fit(main_model, train_dataloaders=train_, val_dataloaders=val_)
