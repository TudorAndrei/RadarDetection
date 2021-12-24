import pytorch_lightning as pl
import torch
import torchmetrics as tm
from vit_pytorch import ViT

from models import CNN, ADNet, ConvMixer, ConvRepeater

REGRESSION = "regression"
CLASSIFICATION = "classification"


def get_means(array, keys):
    means = []
    for key in keys:
        means.append(torch.stack([x[key] for x in array]).mean())
    return means


def get_regression_output(output):
    output = torch.round(output)
    output = torch.clamp(output, min=0, max=4)
    return output


class BaseLightning(pl.LightningModule):
    def __init__(self, num_classes=5, lr=0.03, op=CLASSIFICATION, optim="adam"):
        super().__init__()
        self.lr = lr
        self.op = op
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.optim = optim
        self.num_outputs = num_classes
        if self.op == REGRESSION:
            self.num_outputs = 1

        self.lossf = torch.nn.CrossEntropyLoss()
        if self.op == "regression":
            self.lossf = torch.nn.MSELoss()
        # print(f"Operation is {self.op}")

    def configure_optimizers(self):
        if self.optim == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(), momentum=0.9, lr=self.lr, nesterov=True
            )
            return {"optimizer": optimizer}

        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, threshold=5, factor=0.5
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "train/train_loss",
                    "interval": "epoch",
                },
            }

    def training_step(self, batch, _):
        img, label = batch

        if self.op == "classification":
            label = label
            output = self(img)
            predict = output.argmax(1)
        else:
            label = label.float()
            output = self(img)
            output = torch.squeeze(output, dim=1)
            predict = get_regression_output(output)

        loss = self.lossf(output, label)
        predict = predict.int()
        label = label.int()
        acc = tm.functional.accuracy(predict, label)
        self.log("train/train_accuracy", acc, on_step=False, on_epoch=True)
        self.log(
            "train/train_loss",
            loss,
            # prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, _):
        img, label = batch

        if self.op == "classification":
            output = self(img)
            predict = output.argmax(1)
        else:
            label = label.float()
            output = self(img)
            output = torch.squeeze(output, dim=1)
            predict = get_regression_output(output)

        loss = self.lossf(output, label)

        predict = predict.int()
        label = label.int()
        acc = tm.functional.accuracy(predict, label)
        f1 = tm.functional.f1(predict, label)
        mae = tm.functional.mean_absolute_error(predict, label)
        weighted_acc = tm.functional.accuracy(
            predict, label, average="weighted", num_classes=self.num_classes
        )
        return {
            "val_loss": loss,
            "val_accuracy": acc,
            "val_f1": f1,
            "w_acc": weighted_acc,
            "mae": mae,
        }

    def validation_epoch_end(self, out):
        acc, loss, f1, w_acc, mae = get_means(
            out, ["val_accuracy", "val_loss", "val_f1", "w_acc", "mae"]
        )
        self.log("val/val_accuracy", acc, on_step=False, on_epoch=True)
        self.log("val/f1", f1, on_step=False, on_epoch=True)
        self.log("val/w_acc", w_acc, on_step=False, on_epoch=True)
        self.log("val/mae", mae, on_step=False, on_epoch=True)
        self.log(
            "val/val_loss",
            loss,
            # prog_bar=True,
            on_step=False,
            on_epoch=True,
        )


class ConvMixerLightning(BaseLightning):
    def __init__(
        self,
        optim,
        size=5,
        num_blocks=2,
        kernel_size=15,
        patch_size=15,
        num_classes=5,
        lr=0.003,
        res_type="add",
        op="classification",
    ):
        super().__init__(lr=lr, num_classes=num_classes, op=op, optim=optim)
        self.op = op
        if res_type == "add":
            # print("using ConvMixer")
            self.model = ConvMixer(
                size, num_blocks, kernel_size, patch_size, num_classes, self.op
            )
        elif res_type == "cat":
            # print("using ConvRepeater")
            self.model = ConvRepeater(
                size, num_blocks, kernel_size, patch_size, num_classes, self.op
            )
        else:
            print(f"{res_type} is not a valid option")

    def forward(self, x):
        return self.model(x)


class CNN_lightning(BaseLightning):
    def __init__(
        self,
        size=5,
        num_blocks=2,
        kernel_size=15,
        num_classes=5,
        strides=2,
        padding=1,
        lr=0.003,
        op="classification",
    ):
        super().__init__(lr=lr, num_classes=num_classes, op=op)
        self.op = op
        self.model = CNN(
            num_blocks=num_blocks,
            size=size,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            op=self.op,
        )

    def forward(self, x):
        return self.model(x)


class ADNet_lightning(BaseLightning):
    def __init__(self, optim, num_classes=5, lr=0.03, op=CLASSIFICATION):
        super().__init__(lr=lr, num_classes=num_classes, op=op, optim=optim)
        self.model = ADNet(num_classes=self.num_outputs)

    def forward(self, x):
        return self.model(x)


class ViTLigthning(BaseLightning):
    def __init__(
        self,
        num_classes=5,
        image_size=128,
        lr=0.03,
        patch_size=32,
        dim=1024,
        depth=6,
        heads=16,
        mlp_dim=1025,
        dropout=0.1,
        emb_dropout=0.1,
        op="classification",
    ):
        super().__init__(lr=lr, num_classes=num_classes, op=op)
        self.save_hyperparameters(ignore=["image_size", "lr"])
        self.model = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=self.num_outputs,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            pool="mean",
        )

    def forward(self, x):
        return self.model(x)
