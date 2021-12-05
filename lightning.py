import pretty_errors
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

# from convmixer import ConvMixerModule
from convmixer_expanded import ConvMixerModule
from data_utils import data_generator

pl.seed_everything(42)

BATCH_SIZE = 128
NW = 8
EPOCHS = 50


search_space = {
    "size": [4, 8, 16, 32, 64],
    "num_blocks": [1, 2, 3, 4, 5, 6],
    "kernel_size": [3, 5, 7, 9],
    "patch_size": [20, 25, 30],
}


if __name__ == "__main__":

    train_, val_ = data_generator(
        img_dir="./train", csv_path=r"train.csv", bs=BATCH_SIZE, nw=NW
    )


    for size in search_space['size']:
        for num_blocks in search_space['num_blocks']:
            for kernel_size in search_space['kernel_size']:
                for patch_size in search_space['patch_size']:
                    hyps = {
                        "size": size,
                        "num_blocks": num_blocks,
                        "kernel_size": kernel_size,
                        "patch_size": patch_size,
                        "num_classes": 5,
                        "lr": 0.003,
                        "res_type": "cat",
                    }
                    hyp_print = ""
                    for key, value in hyps.items():
                        hyp_print += f"_{key}_{value}"
                    print(f"Training {hyp_print}")
                    main_model = ConvMixerModule(**hyps)
                    trainer = pl.Trainer(
                        # fast_dev_run=True,
                        # benchmark=True,
                        gpus=1,
                        max_epochs=EPOCHS,
                        callbacks=[
                            ModelCheckpoint(
                                monitor="val/val_loss",
                                mode="min",
                                dirpath=f"models/model{hyp_print}",
                                filename="radar-epoch{epoch:02d}-val_loss{val/val_loss:.2f}",
                                auto_insert_metric_name=False,
                            ),
                            EarlyStopping(monitor="val/val_loss", patience=10),
                            LearningRateMonitor(logging_interval="step"),
                        ],
                    )
                    trainer.fit(main_model, train_dataloaders=train_, val_dataloaders=val_)
