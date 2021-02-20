import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import abc

from .basic import *


__all__ = ["IIC"]


class IIC(BaseModule):
    def __init__(self, in_channels: int = 3, num_classes: int = 10, num_classes_over: int = 100, z_dim=512, num_heads=10):
        super().__init__()
        self.use_multi_heads = num_heads > 1
        self.num_heads = num_heads
        self.encoder = Encoder(in_channels, z_dim)

        if self.use_multi_heads:
            self.classifier = nn.ModuleList([self.gen_classifier(z_dim, num_classes) for _ in range(self.num_heads)])
        else:
            self.classifier = self.gen_classifier(z_dim, num_classes)

        if self.use_multi_heads:
            self.over_classifier = nn.ModuleList([self.gen_classifier(z_dim, num_classes_over) for _ in range(self.num_heads)])
        else:
            self.over_classifier = self.gen_classifier(z_dim, num_classes_over)

        self.weight_init()

    def gen_classifier(self, in_dim, out_dim):
        return nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.BatchNorm1d(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

    def forward(self, x, *args, lam=1.0, z_detach=False):
        _, z_x, _ = self.encoder(x)
        if z_detach:
            z_x = z_x.detach()
        w_x = self.clustering(z_x)
        w_x_over = self.over_clustering(z_x)
        if self.training:
            mi, mi_over, n = 0, 0, 0
            for y in args:
                _, z_y, _ = self.encoder(y)
                if z_detach:
                    z_y = z_y.detach()
                w_y = self.clustering(z_y)
                w_y_over = self.over_clustering(z_y)
                mi += self.mutual_info(w_x, w_y, lam=lam)
                mi_over += self.mutual_info(w_x_over, w_y_over, lam=lam)
                n += 1
            mi, mi_over = mi / n, mi_over / n
            return mi, mi_over
        else:
            return z_x, w_x, w_x_over

    def clustering(self, x):
        if self.use_multi_heads:
            tmp = []
            for classifier in self.classifier:
                w = F.softmax(classifier(x), dim=-1)
                tmp.append(w)
            return torch.stack(tmp, dim=-1)
        else:
            w = F.softmax(self.classifier(x), dim=-1)
            return w

    def over_clustering(self, x):
        if self.use_multi_heads:
            tmp = []
            for classifier in self.over_classifier:
                w = F.softmax(classifier(x), dim=-1)
                tmp.append(w)
            return torch.stack(tmp, dim=-1)
        else:
            w = F.softmax(self.over_classifier(x), dim=-1)
            return w

    def mutual_info(self, x, y, lam=1.0, eps=1e-8):
        if self.use_multi_heads:
            p = (x.unsqueeze(2) * y.unsqueeze(1)).sum(0)
            p = ((p + p.permute(1, 0, 2)) / 2) / p.sum()
            p[(p < eps).data] = eps
            _, k, m = x.shape
            pi = p.sum(dim=1).view(k, -1).expand(k, k, m).pow(lam)
            pj = p.sum(dim=0).view(k, -1).expand(k, k, m).pow(lam)
        else:
            m = 1
            p = (x.unsqueeze(2) * y.unsqueeze(1)).sum(0)
            p = ((p + p.t()) / 2) / p.sum()
            _, k = x.shape
            p[(p < eps).data] = eps
            pi = p.sum(dim=1).view(k, 1).expand(k, k).pow(lam)
            pj = p.sum(dim=0).view(1, k).expand(k, k).pow(lam)
        return (p * (torch.log(pi) + torch.log(pj) - torch.log(p))).sum() / m
