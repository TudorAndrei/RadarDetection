import torch
import torch.nn as nn
from torch.nn import (
    AdaptiveMaxPool2d,
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
)
from torch.nn.functional import softmax
from torch.nn.modules.activation import GELU


class PrintLayer(torch.nn.Module):
    def __init__(self):
        """Print the shape of the input tensor"""
        super().__init__()

    def forward(self, x):
        print(x.shape)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, layers):
        """Create a residual connection and add the two tensors

        Args:
            layers: an nn.Module which contains the layers of the block
        """
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x) + x


class ConcatBlock(nn.Module):
    def __init__(self, layers):
        """Same as ResidualBlock, but instead, concatenate the channels


        Args:
            layers: an nn.Module which contains the layers of the block
        """
        super().__init__()
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = torch.cat([self.layers(x), x], dim=1)
        return out


class PatchEmbedding(nn.Module):
    def __init__(self, size, patch_size):
        """Split the image in equay sized patches

        Args:
            size: the number of output channels
            patch_size: the size of a patch
        """
        super().__init__()
        self.patch = nn.Conv2d(3, size, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.patch(x)


class GeluBatch(nn.Module):
    def __init__(self, size, p=0.3):
        """ReLU + Batch Normalization

        The ReLu activation function can be replaced with GELU

        Args:
            size: size of the output
            p: dropout rate
        """
        super().__init__()
        self.gelu = GELU()
        # self.gelu = ReLU()
        self.drop = nn.Dropout(p)
        self.norm = nn.BatchNorm2d(size)

    def forward(self, x):
        x = self.gelu(x)
        x = self.drop(x)
        x = self.norm(x)
        return x


class DoubleConvBlock(torch.nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, strides, padding):
        """Conv2d + ReLU + Conv2d


        Args:
            in_ch: nr. of input channels
            out_ch: nr. of output channels
            kernel_size: kernel size of the conv operation
            strides: strides
            padding: padding
        """
        super().__init__()
        self.layers = nn.Sequential(
            Conv2d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
            ),
            GeluBatch(out_ch),
            Conv2d(
                out_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=strides,
                padding=padding,
            ),
            Dropout(0.3),
        )

    def forward(self, x):
        x = self.layers(x)
        return x


class DepthConv2d(nn.Module):
    def __init__(self, channels, kernel_size, p=0.3, padding="same"):
        """Depthwise convolution


        Args:
            channels: the number of input and output channels
            kernel_size: kernel size
            p: dropout rate
            padding: padding
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            groups=channels,
            padding=padding,
        )
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return self.drop(self.conv(x))


class PointConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, p=0.3):
        """Pointwise Convolution

        Args:
            in_channels: the number of input channels
            out_channels: the number of output channels
            p: dropout rate
        """
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
        )
        self.drop = nn.Dropout(p)

    def forward(self, x):
        return self.drop(self.conv(x))


class ConvMixerLayer(nn.Module):
    def __init__(self, size, num_blocks, kernel_size=9):
        """ConvMixer block, see paper

        Args:
            size: size of the channels
            num_blocks: how many blocks of the ConvMixerLayer
            kernel_size: kernel size
        """
        super().__init__()

        self.conv_mixer = nn.Sequential(
            *[
                nn.Sequential(
                    ResidualBlock(
                        [
                            DepthConv2d(size, kernel_size),
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
    def __init__(
        self,
        size,
        num_blocks,
        kernel_size,
        patch_size,
        num_classes,
    ):
        """Main class where all the blocks of the ConvMixer are put together



        Args:
            size: size of the channels
            num_blocks: number of ConvMixerLayer
            kernel_size: kernel size
            patch_size: size of the patch
            num_classes: number of output classes
            op: classification or regression
        """
        super().__init__()
        self.layers = nn.Sequential(
            PatchEmbedding(size, patch_size=patch_size),
            GeluBatch(size),
            ConvMixerLayer(size, num_blocks, kernel_size),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        self.op = nn.Linear(size, num_classes)

    def forward(self, x):
        out = self.layers(x)
        return self.op(out)


class ConvRepeaterLayer(nn.Module):
    def __init__(self, size, num_blocks, p, kernel_size=9):
        """Conv Repetear, see report

        An extension of the ConvMixer, which concatenates the output of the Depthwise conv to the residual connection

        Args:
            size: channels size
            num_blocks: number of ConvRepeaterLayer
            p: dropout rate
            kernel_size: kernel size
        """
        super().__init__()
        self.residual = ConcatBlock
        size = [size]

        # Since the results of the residual connection are doubled after each block, the size needs to increase too
        for i in range(1, num_blocks + 1):
            size.append(2 * size[i - 1])

        self.conv_mixer = nn.Sequential(
            *[
                nn.Sequential(
                    self.residual(
                        [
                            DepthConv2d(size[i], kernel_size),
                            GeluBatch(size[i], p),
                        ]
                    ),
                    PointConv2d(size[i + 1], size[i + 1]),
                    GeluBatch(size[i + 1], p),
                )
                for i in range(num_blocks)
            ]
        )

    def forward(self, x):
        return self.conv_mixer(x)


class ConvRepeater(nn.Module):
    def __init__(
        self,
        size,
        num_blocks,
        kernel_size,
        patch_size,
        num_classes,
        op="classification",
        p=0.3,
    ):
        """Main Class of ConvRepeaterLayer


        Args:
            size: channel size
            num_blocks: number of ConvRepeaterLayer
            kernel_size: kernel size
            patch_size: size of the patches
            num_classes: number of classes
            op: regression or classification
            p: dropout rate
        """
        super().__init__()
        self.layers = nn.Sequential(
            PatchEmbedding(size, patch_size=patch_size),
            GeluBatch(size, p=0.3),
            ConvRepeaterLayer(size, num_blocks, p, kernel_size),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )

        if op == "classification":
            self.op = nn.Linear(size * (2 ** num_blocks), num_classes)
        elif op == "regression":
            self.op = nn.Linear(size * (2 ** num_blocks), 1)

    def forward(self, x):
        out = self.layers(x)
        return self.op(out)


class CNN(torch.nn.Module):
    def __init__(
        self,
        size,
        kernel_size,
        strides,
        padding,
        num_classes=5,
    ):
        """Basic CNN implementation



        Args:
            size: channel sizes
            kernel_size: kernel size
            strides: strides size
            padding: padding size
            num_classes: number of classes
        """
        super().__init__()
        self.residual = ResidualBlock
        # print(size)
        self.hidden = 1024
        self.layers = nn.Sequential(
            *[
                nn.Sequential(
                    Conv2d(
                        size[i],
                        size[i + 1],
                        kernel_size=kernel_size,
                        stride=strides,
                        padding=padding,
                    ),
                    GeluBatch(size[i + 1]),
                )
                for i in range(len(size) - 1)
            ],
            Flatten(),
        )
        self.flattened = 4096
        self.linear = Linear(self.flattened, self.hidden)

        self.op = nn.Linear(self.hidden, num_classes)

    def forward(self, x):
        out = self.layers(x)
        out = self.linear(out)
        out = self.op(out)
        return out


class ADNet(torch.nn.Module):
    def __init__(
        self,
        num_classes=5,
    ):
        """Azimuth Dopler Model from MVRSS

        AD branch of the MVRSS architecture

        Args:
            num_classes: number of classes
        """
        super().__init__()
        self.in_size = 3
        self.residual = ResidualBlock
        self.layers = nn.Sequential(
            DoubleConvBlock(3, 128, kernel_size=3, strides=1, padding=1),
            MaxPool2d(2, stride=(2, 1)),
            DoubleConvBlock(128, 128, kernel_size=3, strides=1, padding=1),
            MaxPool2d(2, stride=(2, 1)),
            PointConv2d(128, 128, p=0),
            AdaptiveMaxPool2d((1, 1)),
            Flatten(),
        )
        self.flattened = 128
        self.linear = Linear(self.flattened, num_classes)

    def forward(self, x):
        out = self.layers(x)
        out = self.linear(out)
        return out
