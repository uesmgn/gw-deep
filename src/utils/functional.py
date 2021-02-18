import torch
import matplotlib.colors as colors
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def get_mean_std(loader):
    channels_sum, channels_squared_sum, n = 0, 0, 0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        n += 1
    mean = channels_sum / n
    std = (channels_squared_sum / n - mean ** 2) ** 0.5
    return mean, std


def segmented_cmap(cmap, num_split=10):
    cmap = plt.get_cmap(cmap)
    norm = colors.Normalize(vmin=0, vmax=cmap.N)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)
    tmp = [mapper.to_rgba(i) for i in range(cmap.N)]
    cmap = colors.LinearSegmentedColormap.from_list("custom_cmap", tmp, N=num_split)
    return cmap
