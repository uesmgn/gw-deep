import os
import hydra
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision.transforms as tf
from collections import defaultdict

from sklearn.manifold import TSNE

import src.data.datasets as datasets
import src.nn.models as models
import src.utils.transforms as transforms
import src.utils.functional as F
import src.utils.logging as logging


@hydra.main(config_path="config", config_name="vae")
def main(args):
    if args.verbose:
        from tqdm import tqdm
    else:
        tqdm = lambda x: x

    transform = tf.Compose(
        [
            tf.CenterCrop(224),
        ]
    )
    target_transform = transforms.ToIndex(args.labels)

    dataset = datasets.HDF5(args.dataset_path, transform=transform, target_transform=target_transform)
    train_set, test_set = dataset.split(train_size=args.train_size, random_state=args.random_state, stratify=dataset.targets)

    test_loader = torch.utils.data.DataLoader(
        test_set,
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

    model = models.VAE(args.in_channels, args.z_dim).to(device)
    model_file = f"vae_e{args.eval_epoch}.pt"
    try:
        model.load_state_dict_part(torch.load(os.path.join(args.model_dir, model_file)))
    except:
        raise FileNotFoundError(f"Model file does not exist: {os.path.join(args.model_dir, model_file)}")

    model.eval()
    with torch.no_grad():
        for x, target in tqdm(test_loader):
            x = x.to(device, non_blocking=True)
            z, x_recon = model.get_params(x)


if __name__ == "__main__":
    main()
