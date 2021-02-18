import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import abc

from .basic import *


class Encoder(nn.Module):
    def __init__(self, in_channels: int, out_dim: int):
        super().__init__()
        self.blocks = nn.Sequential(
            ECBlock(in_channels, 64, kernel_size=7, stride=2),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ECBlock(64, 128, stride=2),
            ECBlock(128, 256, stride=2),
            ECBlock(256, 512, stride=2),
        )
        self.logits = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, out_dim * 2),
        )

    def _reparameterize(self, mean, logvar, L=1):
        mean = mean.repeat(L, 1, 1).squeeze()
        logvar = logvar.repeat(L, 1, 1).squeeze()
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return z

    def forward(self, x, reparameterize=True, L=1):
        x = self.blocks(x)
        x = self.logits(x)
        if reparameterize:
            mean, logvar = torch.split(x, x.shape[-1] // 2, -1)
            z = self._reparameterize(mean, logvar, L).view(x.shape[0] * L, -1)
            return z, mean, logvar
        else:
            return x


class Decoder(nn.Module):
    def __init__(self, in_dim: int, out_channels: int, msize: int = 7):
        super().__init__()
        self.msize = msize
        self.head = nn.Sequential(
            nn.Linear(in_dim, in_dim * msize * msize, bias=False),
            nn.BatchNorm1d(in_dim * msize * msize),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            nn.UpsamplingNearest2d(scale_factor=2),
            DCBlock(in_dim, 256, stride=2),
            DCBlock(256, 128, stride=2),
            DCBlock(128, 64, stride=2),
            DCBlock(64, out_channels, stride=2, activation=None),
        )

    def forward(self, x):
        x = self.head(x)
        x = x.view(x.shape[0], -1, self.msize, self.msize)
        x = self.blocks(x)
        return x


class DAE(nn.Module):
    def __init__(self, in_channels: int = 3, z_dim: int = 512, msize: int = 7):
        super().__init__()
        self.encoder = Encoder(in_channels, z_dim // 2)
        self.decoder = Decoder(z_dim, in_channels, msize)

    def forward(self, x, x_):
        z = self.encoder(x_, reparameterize=False)
        x_rec = self.decoder(z)
        bce = self.bce(x_rec, x) / L
        return bce

    def bce(self, x_rec, x):
        return F.binary_cross_entropy_with_logits(x_rec, x, reduction="none").mean(dim=[-1, -2]).sum()

    def get_params(self, x):
        z = self.encoder(x, reparameterize=False)
        return z


class VAE(nn.Module):
    def __init__(self, in_channels: int = 3, z_dim: int = 512, msize: int = 7):
        super().__init__()
        self.encoder = Encoder(in_channels, z_dim)
        self.decoder = Decoder(z_dim, in_channels, msize)

    def forward(self, x, L=1):
        z, mean, logvar = self.encoder(x, L=L)
        x_rec = self.decoder(z)
        x_duplicated = x.repeat(L, 1, 1, 1) if L > 1 else x
        bce = self.bce(x_rec, x_duplicated) / L
        kl_gauss = self.kl_gauss(mean, logvar)
        return bce, kl_gauss

    def bce(self, x_rec, x):
        return F.binary_cross_entropy_with_logits(x_rec, x, reduction="none").mean(dim=[-1, -2]).sum()

    def kl_gauss(self, mean, logvar):
        return -0.5 * torch.mean(1 + logvar - torch.pow(mean, 2) - logvar.exp(), dim=-1).sum()

    def get_params(self, x):
        _, z, _ = self.encoder(x, L=1)
        return z
