import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics as tm


class ResidualBlock(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x) + x
        return out


class ConcatBlock(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = torch.cat([self.layers(x), x], dim=1)
        return out


class PatchEmbedding(nn.Module):
    def __init__(self, size, patch_size) -> None:
        super().__init__()
        self.patch = nn.Conv2d(
            3, size, kernel_size=patch_size, stride=patch_size
        )

    def forward(self, x):
        return self.patch(x)


class GeluBatch(nn.Module):
    def __init__(self, size) -> None:
        super().__init__()
        self.gelu = nn.GELU()
        self.norm = nn.BatchNorm2d(size)

    def forward(self, x):
        out = self.gelu(x)
        return self.norm(out)


class DepthConv2d(nn.Module):
    def __init__(self, channels, kernel_size, padding):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            padding=padding,
        )

    def forward(self, x):
        return self.depth_conv(x)


class PointConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.point_conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        return self.point_conv(x)


class ConvMixerLayer(nn.Module):
    def __init__(self, size, num_blocks, kernel_size=9, res_type="add"):
        super().__init__()
        if res_type == "add":
            self.residual = ResidualBlock
        elif res_type == "cat":
            self.residual = ConcatBlock
            size = [size]

            for i in range(1, num_blocks + 1):
                size.append(2 * size[i - 1])

        print(size)

        self.conv_mixer = nn.Sequential(
            *[
                nn.Sequential(
                    self.residual(
                        [
                            DepthConv2d(size[i], kernel_size, "same"),
                            GeluBatch(size[i]),
                        ]
                    ),
                    PointConv2d(size[i + 1], size[i + 1]),
                    GeluBatch(size[i + 1]),
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, x):
        return self.conv_mixer(x)


class ConvMixer(nn.Module):
    def __init__(
        self, size, num_blocks, kernel_size, patch_size, num_classes, res_type
    ):
        super().__init__()
        self.layers = nn.Sequential(
            PatchEmbedding(size, patch_size=patch_size),
            GeluBatch(size),
            ConvMixerLayer(size, num_blocks, kernel_size, res_type),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(size * (2 ** num_blocks), num_classes),
        )

    def forward(self, x):
        return self.layers(x)


class ConvMixerModule(pl.LightningModule):
    def __init__(
        self,
        size=5,
        num_blocks=2,
        kernel_size=15,
        patch_size=15,
        num_classes=5,
        lr=0.003,
        res_type="add",
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = ConvMixer(
            size, num_blocks, kernel_size, patch_size, num_classes, res_type
        )
        self.lossf = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, threshold=5
        )
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
        img = img.float().permute(0, 3, 1, 2)
        label = label
        output = self.model(img)
        loss = self.lossf(output, label)
        predict = output.argmax(1)
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
        img = img.float().permute(0, 3, 1, 2)
        label = label
        output = self.model(img)
        predict = output.argmax(1)
        acc = tm.functional.accuracy(predict, label)
        loss = self.lossf(output, label)
        return {"val_loss": loss, "val_accuracy": acc}

    def validation_epoch_end(self, out):
        acc = torch.stack([acc["val_accuracy"] for acc in out]).mean()
        loss = torch.stack([acc["val_loss"] for acc in out]).mean()
        self.log("val/val_accuracy", acc, on_step=False, on_epoch=True)
        self.log(
            "val/val_loss", loss, prog_bar=True, on_step=False, on_epoch=True
        )
        self.log("hp_metric", loss)
