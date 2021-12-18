import pretty_errors
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.metrics import classification_report

from data_utils import data_generator
from lightning_models import ConvMixerLightning

# pl.seed_everything(42)

BATCH_SIZE = 4
NW = 8
EPOCHS = 10
classif_report = False
file_name = lambda x: x.split("/")[-1]

search_space = {
    "size": [32, 64, 128],
    "num_blocks": [3, 4, 5, 6],
    "kernel_size": [3, 5, 7, 9],
    "patch_size": [20, 25, 30],
}

search_space = {
    "size": [32],
    "num_blocks": [4],
    "kernel_size": [5],
    "patch_size": [11],
}


if __name__ == "__main__":

    train_, val_ = data_generator(
        img_dir="./train", csv_path=r"train.csv", bs=BATCH_SIZE, nw=NW
    )

    for size in search_space["size"]:
        for num_blocks in search_space["num_blocks"]:
            for kernel_size in search_space["kernel_size"]:
                for patch_size in search_space["patch_size"]:
                    hyps = {
                        "size": size,
                        "num_blocks": num_blocks,
                        "kernel_size": kernel_size,
                        "patch_size": patch_size,
                        "num_classes": 5,
                        "lr": 0.3,
                        # "res_type": "add",
                        "res_type": "cat",
                        # "op": "regression",
                        "op": "classification",
                    }
                    hyp_print = ""
                    for key, value in hyps.items():
                        hyp_print += f"_{key}_{value}"
                    print(f"Training {hyp_print}")
                    model = ConvMixerLightning(**hyps)
                    trainer = pl.Trainer(
                        fast_dev_run=True,
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
                            EarlyStopping(monitor="val/val_loss", patience=5),
                            LearningRateMonitor(logging_interval="step"),
                        ],
                    )
                    trainer.fit(
                        model,
                        train_dataloaders=train_,
                        val_dataloaders=val_,
                    )

                    if classif_report:
                        model.eval()
                        model.freeze()
                        preds = []
                        gt = []
                        for batch in val_:
                            img, labels = batch
                            img = img.permute(0, 3, 1, 2).float()
                            out = model(img)
                            predictions = torch.argmax(out, 1).numpy()
                            # print(predictions)
                            # preds = [(file_name(img_path[i]), pred + 1) for i,pred in enumerate(predictions)]
                            for pred, label in zip(predictions, labels):
                                preds.append(pred)
                                gt.append(label)
                        print(classification_report(gt, preds))
