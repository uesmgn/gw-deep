import os
import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as tf
from collections import defaultdict
from tqdm import tqdm
import seaborn

from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_samples, confusion_matrix
from sklearn.preprocessing import normalize

import src.data.datasets as datasets
import src.nn.models as models
import src.utils.transforms as transforms
import src.utils.functional as F
import src.utils.logging as logging

plt.style.use("seaborn-poster")
plt.rcParams["lines.markersize"] = 6.0
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

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=os.cpu_count(),
        pin_memory=True,
        drop_last=False,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.set_device(args.gpu.eval)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")

    model = models.IIC(
        args.in_channels, num_classes=args.num_classes, num_classes_over=args.num_classes_over, z_dim=args.z_dim, num_heads=args.num_heads
    ).to(device)
    model_file = os.path.join(args.model_dir, args.trained_model_file)
    try:
        model.load_state_dict_part(torch.load(model_file))
    except:
        raise FileNotFoundError(f"Model file does not exist: {model_file}")

    py, pw, z, y = [], [], [], []
    model.eval()
    with torch.no_grad():
        for x, target in tqdm(test_loader):
            x = x.to(device, non_blocking=True)
            py_m, pw_m, z_m = model.params(x)
            py.append(py_m)
            pw.append(pw_m)
            z.append(z_m)
            y.append(target)
    py = torch.cat(py).argmax(1).cpu().numpy().astype(int)  # shape: (N, num_heads), values: [0, num_classes-1]
    pw = torch.cat(pw).argmax(1).cpu().numpy().astype(int)  # shape: (N, num_heads), values: [0, num_classes_over-1]
    print(py.shape, pw.shape)
    z = torch.cat(z).cpu().numpy()
    y = torch.cat(y).cpu().numpy().astype(int)

    for i in range(args.num_heads):
        py_i, pw_i = py[:, i], pw[:, i]
        cm_y = confusion_matrix(y, py_i, labels=list(range(args.num_classes)))[: args.num_classes, :]
        cm_w = confusion_matrix(y, pw_i, labels=list(range(args.num_classes_over)))[: args.num_classes, :]

        fig, ax = plt.subplots()
        seaborn.heatmap(
            normalize(cm_y, axis=0),
            ax=ax,
            annot=cm_y,
            linewidths=0.1,
            linecolor="gray",
            cmap="afmhot_r",
            cbar=True,
            cbar_kws={"aspect": 50, "pad": 0.01, "anchor": (0, 0.05)},
        )
        plt.yticks(rotation=45)
        plt.xlabel("new labels")
        plt.ylabel("true labels")
        plt.tight_layout()
        plt.savefig(f"cm_{i}_e{epoch}.png")
        plt.close()

        fig, ax = plt.subplots()
        seaborn.heatmap(
            normalize(cm_w, axis=0),
            ax=ax,
            annot=cm_w,
            linewidths=0.1,
            linecolor="gray",
            cmap="afmhot_r",
            cbar=True,
            cbar_kws={"aspect": 50, "pad": 0.01, "anchor": (0, 0.05)},
        )
        plt.xlabel("new labels (overclustering)")
        plt.ylabel("true labels")
        plt.tight_layout()
        plt.savefig(f"cm_over_{i}_e{epoch}.png")
        plt.close()

    # print(f"Plotting 2D latent features with true labels...")
    # z_tsne = TSNE(n_components=2, random_state=args.random_state).fit(z).embedding_
    # fig, ax = plt.subplots()
    # cmap = F.segmented_cmap("tab10", num_classes)
    # for i in range(num_classes):
    #     idx = np.where(y == i)[0]
    #     if len(idx) > 0:
    #         c = cmap(i)
    #         ax.scatter(z_tsne[idx, 0], z_tsne[idx, 1], color=c, label=args.labels[i], edgecolors=F.darken(c))
    # ax.legend(bbox_to_anchor=(1.01, 1.0), loc="upper left")
    # ax.set_aspect(1.0 / ax.get_data_ratio())
    # plt.tight_layout()
    # plt.savefig(f"z_tsne_true.png")
    # plt.close()


if __name__ == "__main__":
    main()
