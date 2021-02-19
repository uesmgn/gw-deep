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

    def forward(self, x: torch.Tensor, L: int = 1):
        z, mean, logvar = self.encoder(x, L=L)
        x_rec = self.decoder(z)
        x_duplicated = x.repeat(L, 1, 1, 1) if L > 1 else x
        bce = self.bce(x_rec, x_duplicated) / L
        kl_gauss = self.kl_gauss(mean, logvar)
        return bce, kl_gauss, mean

    def bce(self, x_rec: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return F.binary_cross_entropy_with_logits(x_rec, x, reduction="none").mean(dim=[-1, -2]).sum()

    def kl_gauss(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.mean(1 + logvar - torch.pow(mean, 2) - logvar.exp(), dim=-1).sum()
