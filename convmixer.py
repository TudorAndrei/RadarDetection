import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics as tm



class ResidualBlock(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) + x


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
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            padding=padding,
        )

    def forward(self, x):
        return self.conv(x)


class PointConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
        )

    def forward(self, x):
        return self.conv(x)


class ConvMixerLayer(nn.Module):
    def __init__(self, size, num_blocks, kernel_size=9):
        super().__init__()

        self.conv_mixer = nn.Sequential(
            *[
                nn.Sequential(
                    ResidualBlock(
                        [
                            DepthConv2d(size, kernel_size, "same"),
                            GeluBatch(size),
                        ]
                    ),
                    PointConv2d(size, size),
                    GeluBatch(size),
                )
                for _ in range(num_blocks)
            ]
        )

    def forward(self, x):
        return self.conv_mixer(x)


class ConvMixer(nn.Module):
    def __init__(self, size, num_blocks, kernel_size, patch_size, num_classes):
        super().__init__()
        self.layers = nn.Sequential(
            PatchEmbedding(size, patch_size=patch_size),
            GeluBatch(size),
            ConvMixerLayer(size, num_blocks, kernel_size),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(size, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


class ConvMixerModule(pl.LightningModule):
    def __init__(self, size=5, num_blocks=2, kernel_size=15, patch_size=15, num_classes=5):
        super().__init__()
        self.model = ConvMixer(
            size, num_blocks, kernel_size, patch_size, num_classes
        )
        self.lossf = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        img, label = batch
        img = img.float().permute(0, 3, 1, 2)
        label = label
        output = self.model(img)
        loss = self.lossf(output, label)
        predict = output.argmax(1)
        acc = tm.functional.accuracy(predict, label)
        self.log("train_accuracy", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        img, label = batch
        img = img.float().permute(0, 3, 1, 2)
        label = label
        output = self.model(img)
        predict = output.argmax(1)
        acc = tm.functional.accuracy(predict, label)
        loss = self.lossf(output, label)
        self.log("val_accuracy", acc)
        self.log("val_loss", loss)
