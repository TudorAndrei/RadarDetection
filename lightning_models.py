import pytorch_lightning as pl
import torch
import torchmetrics as tm
from vit_pytorch import ViT

from vit_pytorch.cross_vit import CrossViT

from models import CNN, ADNet, ConvMixer, ConvRepeater


def get_means(array, keys):
    means = []
    for key in keys:
        means.append(torch.stack([x[key] for x in array]).mean())
    return means


class ConvMixerLightning(pl.LightningModule):
    def __init__(
        self,
        size=5,
        num_blocks=2,
        kernel_size=15,
        patch_size=15,
        num_classes=5,
        lr=0.003,
        res_type="add",
        op="classification",
    ):
        super().__init__()
        self.lr = lr
        self.op = op
        self.save_hyperparameters()
        self.num_classes = num_classes
        if res_type == "add":
            print("using ConvMixer")
            self.model = ConvMixer(
                size, num_blocks, kernel_size, patch_size, num_classes, self.op
            )
        elif res_type == "cat":
            print("using ConvRepeater")
            self.model = ConvRepeater(
                size, num_blocks, kernel_size, patch_size, num_classes, self.op
            )
        else:
            print(f"{res_type} is not a valid option")

        self.lossf = torch.nn.CrossEntropyLoss()
        if self.op == "regression":
            self.lossf = torch.nn.MSELoss()
        print(f"Operation is {self.op}")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/val_loss",
                "interval": "epoch",
            },
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        img, label = batch

        img = img.float()

        if self.op == "classification":
            label = label
            output = self.model(img)
            predict = output.argmax(1)
        else:
            label = label.float()
            output = self.model(img)
            output = torch.squeeze(output, dim=1)
            output = torch.round(output)
            output = torch.clamp(output, min=0, max=4)
            predict = output

        loss = self.lossf(output, label)
        predict = predict.int()
        label = label.int()
        acc = tm.functional.accuracy(predict, label)
        self.log("train/train_accuracy", acc, on_step=False, on_epoch=True)
        self.log(
            "train/train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, _):
        img, label = batch
        img = img.float()

        if self.op == "classification":
            label = label
            output = self.model(img)
            predict = output.argmax(1)
        else:
            label = label.float()
            output = self.model(img)
            output = torch.squeeze(output, dim=1)
            output = torch.round(output)
            output = torch.clamp(output, min=0, max=4)
            predict = output

        loss = self.lossf(output, label)

        # the metrics needs int values
        predict = predict.int()
        label = label.int()
        # metrics that are to be tracked
        acc = tm.functional.accuracy(predict, label)
        f1 = tm.functional.f1(predict, label)
        weighted_acc = tm.functional.accuracy(
            predict, label, average="weighted", num_classes=self.num_classes
        )
        return {
            "val_loss": loss,
            "val_accuracy": acc,
            "val_f1": f1,
            "w_acc": weighted_acc,
        }

    def validation_epoch_end(self, out):
        acc, loss, f1, w_acc = get_means(
            out, ["val_accuracy", "val_loss", "val_f1", "w_acc"]
        )
        self.log("val/val_accuracy", acc, on_step=False, on_epoch=True)
        self.log(
            "val/val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log("val/f1", f1, on_step=False, on_epoch=True)
        self.log("val/w_acc", w_acc, on_step=False, on_epoch=True)


class CNN_lightning(pl.LightningModule):
    def __init__(
        self,
        size=5,
        num_blocks=2,
        kernel_size=15,
        num_classes=5,
        strides=2,
        padding=1,
        lr=0.003,
    ):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.model = CNN(
            num_blocks=num_blocks,
            size=size,
            kernel_size=kernel_size,
            strides=strides,
            padding=padding,
        )

        self.lossf = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=5)
        return {
            "optimizer": optimizer,
            # "lr_scheduler": {
            #     "scheduler": scheduler,
            #     "monitor": "val/val_loss",
            #     "interval": "epoch",
            # },
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        img, label = batch

        label = label
        output = self.model(img)
        predict = output.argmax(1)

        loss = self.lossf(output, label)
        predict = predict.int()
        label = label.int()
        acc = tm.functional.accuracy(predict, label)
        self.log("train/train_accuracy", acc, on_step=False, on_epoch=True)
        self.log(
            "train/train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, _):
        img, label = batch
        img = img.float()

        label = label
        output = self.model(img)
        predict = output.argmax(1)

        # metrics
        loss = self.lossf(output, label)
        acc = tm.functional.accuracy(predict, label)
        f1 = tm.functional.f1(predict, label)
        weighted_acc = tm.functional.accuracy(
            predict, label, average="weighted", num_classes=self.num_classes
        )

        return {
            "val_loss": loss,
            "val_accuracy": acc,
            "val_f1": f1,
            "w_acc": weighted_acc,
        }

    def validation_epoch_end(self, out):
        acc, loss, f1, w_acc = get_means(
            out, ["val_accuracy", "val_loss", "val_f1", "w_acc"]
        )
        self.log("val/val_accuracy", acc, on_step=False, on_epoch=True)
        self.log(
            "val/val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log("val/f1", f1, on_step=False, on_epoch=True)
        self.log("val/w_acc", w_acc, on_step=False, on_epoch=True)


class ADNet_lightning(pl.LightningModule):
    def __init__(
        self,
        num_classes=5,
        lr=0.03,
    ):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.model = ADNet(num_classes=num_classes)

        self.lossf = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/val_loss",
                "interval": "epoch",
            },
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        img, label = batch

        label = label
        output = self.model(img)
        predict = output.argmax(1)

        loss = self.lossf(output, label)
        predict = predict.int()
        label = label.int()
        acc = tm.functional.accuracy(predict, label)
        self.log("train/train_accuracy", acc, on_step=False, on_epoch=True)
        self.log(
            "train/train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, _):
        img, label = batch
        img = img.float()

        label = label
        output = self.model(img)
        predict = output.argmax(1)

        # metrics
        loss = self.lossf(output, label)
        acc = tm.functional.accuracy(predict, label)
        f1 = tm.functional.f1(predict, label)
        weighted_acc = tm.functional.accuracy(
            predict, label, average="weighted", num_classes=self.num_classes
        )

        return {
            "val_loss": loss,
            "val_accuracy": acc,
            "val_f1": f1,
            "w_acc": weighted_acc,
        }

    def validation_epoch_end(self, out):
        acc, loss, f1, w_acc = get_means(
            out, ["val_accuracy", "val_loss", "val_f1", "w_acc"]
        )
        self.log("val/val_accuracy", acc, on_step=False, on_epoch=True)
        self.log(
            "val/val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log("val/f1", f1, on_step=False, on_epoch=True)
        self.log("val/w_acc", w_acc, on_step=False, on_epoch=True)


class ViTLigthning(pl.LightningModule):
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
    ):
        super().__init__()
        self.lr = lr
        self.save_hyperparameters(ignore=['image_size', 'lr'])
        self.num_classes = num_classes
        self.model = ViT(
            image_size=image_size,
            patch_size=patch_size,
            num_classes=num_classes,
            dim=dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            dropout=dropout,
            emb_dropout=emb_dropout,
            pool='mean'
        )
        self.lossf = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, threshold=5, factor=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "train/train_loss",
                "interval": "epoch",
            },
        }

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, _):
        img, label = batch

        label = label
        output = self.model(img)
        predict = output.argmax(1)

        loss = self.lossf(output, label)
        predict = predict.int()
        label = label.int()
        acc = tm.functional.accuracy(predict, label)
        self.log("train/train_accuracy", acc, on_step=False, on_epoch=True)
        self.log(
            "train/train_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(self, batch, _):
        img, label = batch
        img = img.float()

        label = label
        output = self.model(img)
        predict = output.argmax(1)

        # metrics
        loss = self.lossf(output, label)
        acc = tm.functional.accuracy(predict, label)
        f1 = tm.functional.f1(predict, label)
        weighted_acc = tm.functional.accuracy(
            predict, label, average="weighted", num_classes=self.num_classes
        )

        return {
            "val_loss": loss,
            "val_accuracy": acc,
            "val_f1": f1,
            "w_acc": weighted_acc,
        }

    def validation_epoch_end(self, out):
        acc, loss, f1, w_acc = get_means(
            out, ["val_accuracy", "val_loss", "val_f1", "w_acc"]
        )
        self.log("val/val_accuracy", acc, on_step=False, on_epoch=True)
        self.log(
            "val/val_loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log("val/f1", f1, on_step=False, on_epoch=True)
        self.log("val/w_acc", w_acc, on_step=False, on_epoch=True)
