import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import abc

from .basic import *


__all__ = ["VAE"]


class VAE(BaseModule):
    def __init__(self, in_channels: int = 3, z_dim: int = 512, msize: int = 7):
        super().__init__()
        self.encoder = Encoder(in_channels, z_dim)
        self.decoder = Decoder(z_dim, in_channels, msize)
        self.weight_init()

    def forward(self, x: torch.Tensor):
        z, mean, logvar = self.encoder(x)
        x_rec = self.decoder(z)
        bce = self.bce(x_rec, x)
        kl_gauss = self.kl_gauss(mean, logvar)
        return bce, kl_gauss

    def params(self, x: torch.Tensor):
        assert not self.training
        z, mean, logvar = self.encoder(x)
        x_rec = self.decoder(z)
        return mean, x_rec

    def bce(self, x_rec: torch.Tensor, x: torch.Tensor):
        b, c, h, w = x.shape
        bce = F.binary_cross_entropy_with_logits(x_rec, x, reduction="sum")
        return bce / c / w / h

    def kl_gauss(self, mean: torch.Tensor, logvar: torch.Tensor):
        b, d = mean.shape
        kl = -0.5 * torch.sum(1 + logvar - torch.pow(mean, 2) - logvar.exp())
        return kl / d
