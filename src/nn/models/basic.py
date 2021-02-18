import torch
import torch.nn as nn
import math


__all__ = ["ECBlock", "DCBlock"]


class ECBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        padding = math.ceil((kernel_size - stride) / 2)
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = activation

    def forward(self, x):
        x = self.block(x)
        if self.activation is not None:
            return self.activation(x)
        return x


class DCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, activation: nn.Module = nn.ReLU(inplace=True)):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.block = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=stride),
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = activation

    def forward(self, x):
        x = self.block(x)
        if self.activation is not None:
            return self.activation(x)
        return x
