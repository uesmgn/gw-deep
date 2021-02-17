import torch
import torch.nn as nn


__all__ = ["ResBlock", "ResBlockDec"]


class ResBlock(nn.Module):
    """Short summary.

    Parameters
    ----------
    in_channels : int
        Description of parameter `in_channels`.
    out_channels : int
        Description of parameter `out_channels`.
    stride : int
        Description of parameter `stride`.
    activation : nn.Module
        Description of parameter `activation`.

    Attributes
    ----------
    block : type
        Description of attribute `block`.
    connection : type
        Description of attribute `connection`.
    activation

    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, activation: nn.Module = None):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.connection = None
        if in_channels != out_channels or stride != 1:
            self.connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        if activation is None:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = activation

    def forward(self, x):
        identity = x
        if self.connection is not None:
            identity = self.connection(x)
        x = self.block(x) + identity
        return self.activation(x)


class ResBlockDec(nn.Module):
    """Short summary.

    Parameters
    ----------
    in_channels : type
        Description of parameter `in_channels`.
    out_channels : type
        Description of parameter `out_channels`.
    activation : type
        Description of parameter `activation`.

    Attributes
    ----------
    block : type
        Description of attribute `block`.
    connection : type
        Description of attribute `connection`.
    activation

    """

    def __init__(self, in_channels, out_channels, activation=None):
        super().__init__()
        self.block = self.block = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels != out_channels or stride != 1:
            self.connection = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        if activation is None:
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        else:
            self.activation = activation

    def forward(self, x):
        identity = x
        if self.connection is not None:
            identity = self.connection(x)
        x = self.block(x) + identity
        return self.activation(x)
