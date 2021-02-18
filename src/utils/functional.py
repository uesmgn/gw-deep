import torch


def get_mean_std(loader):
    channels_sum, channels_squared_sum, n = 0, 0, 0
    for data, _ in loader:
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        n += 1
    mean = channels_sum / n
    std = (channels_squared_sum / n - mean ** 2) ** 0.5
    return mean, std
