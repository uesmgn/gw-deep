import os
import torch
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
target_transform = transforms.ToIndex(
    [
        "1080Lines",
        "1400Ripples",
        "Air_Compressor",
        "Blip",
        "Chirp",
        "Extremely_Loud",
        "Helix",
        "Koi_Fish",
        "Light_Modulation",
        "Low_Frequency_Burst",
        "Low_Frequency_Lines",
        "No_Glitch",
        "None_of_the_Above",
        "Paired_Doves",
        "Power_Line",
        "Repeating_Blips",
        "Scattered_Light",
        "Scratchy",
        "Tomte",
        "Violin_Mode",
        "Wandering_Line",
        "Whistle",
    ]
)

random_state = 123
batch_size = 128
num_classes = 22
dataset_root = "/home/gen.ueshima/gen/workspace/github/GravitySpy/processed/dataset_small.h5"

dataset = datasets.HDF5(dataset_root, transform=transform, target_transform=target_transform)
train_set, test_set = dataset.split(train_size=0.8, random_state=random_state, stratify=dataset.targets)
train_set.transform = augment

train_loader = torch.utils.data.DataLoader(
    train_set,
    batch_size=batch_size,
    num_workers=os.cpu_count(),
    sampler=torch.utils.data.RandomSampler(train_set, replacement=True, num_samples=12800),
    pin_memory=True,
    drop_last=True,
)
test_loader = torch.utils.data.DataLoader(
    test_set,
    batch_size=batch_size,
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

model = models.VAE(4, 512).to(device)
optim = torch.optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(100):
    print(f"----- training at epoch {epoch} -----")
    model.train()
    num_samples = 0
    loss_dict_train = defaultdict(lambda: 0)
    for x, _ in tqdm(train_loader):
        x = x.to(device, non_blocking=True)
        bce, kl_gauss = model(x)
        loss = sum([bce, kl_gauss])
        optim.zero_grad()
        loss.backward()
        optim.step()
        loss_dict_train["total"] += loss.item()
        loss_dict_train["binary_cross_entropy"] += bce.item()
        loss_dict_train["kl_divergence"] += kl_gauss.item()
        num_samples += len(x)

    for key, value in loss_dict_train.items():
        value /= num_samples
        loss_dict_train[key] = value
        print(f"{key}: {value:.3f} at epoch: {epoch}")

    if epoch % 5 == 0:
        print(f"----- evaluating at epoch {epoch} -----")
        model.eval()
        loss_dict_test = defaultdict(lambda: 0)
        params = defaultdict(list)

        with torch.no_grad():
            for x, target in tqdm(test_loader):
                x = x.to(device, non_blocking=True)
                bce, kl_gauss = model(x)
                z = model.get_params(x)
                params["y"].append(target)
                params["z"].append(z)
                loss_dict_test["total"] += loss.item()
                loss_dict_test["binary_cross_entropy"] += bce.item()
                loss_dict_test["kl_divergence"] += kl_gauss.item()
                num_samples += len(x)

        for key, value in loss_dict_test.items():
            value /= num_samples
            loss_dict_test[key] = value
            print(f"{key}: {value:.3f} at epoch: {epoch}")

        y = np.array(params["y"]).astype(int)
        z = np.concatenate(params["z"])

        print("t-SNE decomposing...")
        qz_tsne = TSNE(n_components=2, metric="cosine", random_state=args.seed).fit(z).embedding_

        print(f"Plotting 2D latent features with true labels...")
        fig, ax = plt.subplots()
        cmap = F.segmented_cmap(num_classes, "tab10")
        for i in range(num_classes):
            idx = np.where(y == i)[0]
            if len(idx) > 0:
                c = cmap(i)
                ax.scatter(qz_tsne[idx, 0], qz_tsne[idx, 1], color=c, label=i)
            ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
            ax.set_title(f"t-SNE 2D plot of latent code at epoch {epoch}")
            ax.set_aspect(1.0 / ax.get_data_ratio())
            plt.tight_layout()
            plt.savefig(f"z_true_e{epoch}.png")
            plt.close()
