import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import abc


__all__ = ["BaseModule", "ECBlock", "DCBlock", "Encoder", "Decoder"]


class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def load_state_dict_part(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            print(f"load state dict: {name}")
            own_state[name].copy_(param)

    def weight_init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_normal_(m.weight)
                try:
                    nn.init.zeros_(m.bias)
                except:
                    continue


# Conv2d: 畳み込み
# BatchNorm2d: バッチ正規化
# activation: 活性化関数, ReLU(), LeakyReLU(), Sigmoid(), Softmax()...
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


# block_1: (4, 224, 224) -> (64, 112, 112)
# pool: (64, 112, 112) -> (64, 56, 56)
# block_2: (64, 56, 56) -> (128, 28, 28)
# block_3: (128, 28, 28) -> (256, 14, 14)
# block_4: (256, 14, 14) -> (512, 7, 7)
# global_avg_pooling: (512, 7, 7) -> (512, 1, 1)
# flatten: (512, 1, 1) -> (512, )
# fc: (512, ) -> (1024, )
#   mean: (512, )
#   logvar: (512, )
# rameterization_trick: (512, ) z = mean + std * eps (eps ~ N(0, I))
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

    def reparameterize(self, mean, logvar, L=1):
        mean = mean.repeat(L, 1, 1).squeeze()
        logvar = logvar.repeat(L, 1, 1).squeeze()
        z = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return z

    def forward(self, x, L=1):
        x = self.blocks(x)
        x = self.logits(x)
        mean, logvar = torch.split(x, x.shape[-1] // 2, -1)
        z = self.reparameterize(mean, logvar, L).view(x.shape[0] * L, -1)
        return z, mean, logvar


# fc: (512, ) -> (512 * 7 * 7, )
# reshape: (512 * 7 * 7, ) -> (512, 7, 7)
# block_1: (512, 7, 7) -> (256, 14, 14)
# block_2: -
# block_3: -
# block_4: (64, 112, 112) -> (4, 224, 224)
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
