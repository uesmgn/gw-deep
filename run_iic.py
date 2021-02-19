import os
import hydra
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as tf
from tqdm import tqdm
from collections import defaultdict

from sklearn.manifold import TSNE

import src.data.datasets as datasets
import src.nn.models as models
import src.utils.transforms as transforms
import src.utils.functional as F

plt.style.use("seaborn-poster")
plt.rcParams["lines.markersize"] = 6.0
plt.rcParams["text.usetex"] = True
plt.rc("legend", fontsize=10)


@hydra.main(config_path="config", config_name="iic")
def main(args):
    transform = tf.Compose(
        [
            tf.CenterCrop(224),
        ]
    )

    # 24 / 272 ≒ 0.088
    augment = tf.Compose(
        [
            tf.RandomAffine(0, translate=(0.088, 0), fillcolor=None),
            tf.CenterCrop(224),
        ]
    )
    target_transform = transforms.ToIndex(args.labels)

    num_classes = len(args.labels)

    dataset = datasets.HDF5(args.dataset_path, transform=transform, target_transform=target_transform)
    train_set, test_set = dataset.split(train_size=0.8, random_state=args.random_state, stratify=dataset.targets)
    train_set = train_set.co(augment)

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),
        sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=12800),
        pin_memory=True,
        drop_last=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=False,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")

    model = models.IIC(4, num_classes=50, num_classes_over=250, z_dim=512, num_heads=10).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)
    stats_train, stats_test = defaultdict(list), defaultdict(list)
    for epoch in range(100):
        print(f"----- training at epoch {epoch} -----")
        model.train()
        num_samples = 0
        loss_dict_train = defaultdict(lambda: 0)
        for (x, x_), _ in tqdm(train_loader):
            x, x_ = x.to(device, non_blocking=True), x_.to(device, non_blocking=True)
            mi, mi_over = model(x, x_, lam=args.lam)
            loss = sum([mi, mi_over])
            optim.zero_grad()
            loss.backward()
            optim.step()
            loss_dict_train["total"] += loss.item()
            loss_dict_train["mutual_info"] += bce.item()
            loss_dict_train["mutual_info_over_clustering"] += mi_over.item()
            num_samples += len(x)

        for key, value in loss_dict_train.items():
            value /= num_samples
            loss_dict_train[key] = value
            print(f"{key}: {value:.3f} at epoch: {epoch}")
            stats_train[key].append(value)

        if epoch % 5 == 0:
            print(f"----- evaluating at epoch {epoch} -----")
            model.eval()
            loss_dict_test = defaultdict(lambda: 0)
            params = defaultdict(list)

            with torch.no_grad():
                for x, target in tqdm(test_loader):
                    x = x.to(device, non_blocking=True)
                    z, w, w_over = model(x)
                    params["y"].append(target)
                    params["w"].append(w)
                    params["w_over"].append(w_over)

            y = torch.cat(params["y"]).int().numpy()
            w = torch.cat(params["w"]).int().numpy()
            w_over = torch.cat(params["w_over"]).int().numpy()

            for key, value in stats_train.items():
                xx = np.linspace(0, epoch, len(value))
                plt.plot(xx, value)
                plt.ylabel(key.replace("_", " "))
                plt.xlabel("epoch")
                plt.title(key.replace("_", " "))
                plt.xlim(0, epoch)
                plt.tight_layout()
                plt.savefig(f"{key}_train_e{epoch}.png")
                plt.close()

        if epoch % 50 == 0:
            torch.save(model.state_dict(), args.model_path)


if __name__ == "__main__":
    main()
