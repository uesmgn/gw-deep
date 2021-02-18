import os
import torch.utils.data as data
import h5py
import torch
from collections import abc, defaultdict
import numpy as np
from sklearn.model_selection import train_test_split
import copy

__class__ = ["HDF5"]


class HDF5(torch.utils.data.Dataset):
    def __init__(self, root, transform=None, target_transform=None, target_tag="target"):
        super().__init__()
        self.root = os.path.abspath(root)
        if not os.path.isfile(self.root):
            raise FileNotFoundError(f"The file doesn't exist: `{self.root}`")
        self.transform = transform
        self.target_transform = target_transform
        self.target_tag = target_tag
        self.cache = []
        try:
            self.init_cache()
        except:
            raise RuntimeError(f"Failed to load dataset from `{self.root}`")
        else:
            print(f"Succesfully loaded {len(self)} items.")

    @property
    def targets(self):
        return [t for _, t in self.cache]

    @property
    def counter(self):
        cnt = defaultdict(lambda: 0)
        for _, target in self.cache:
            cnt[target] += 1
        cnt = sorted(cnt.items(), key=lambda x: x[0])
        return dict(cnt)

    def __getitem__(self, i):
        ref, target = self.cache[i]
        with self.open(self.root) as f:
            x = self.load_data(f[ref])
        if callable(self.transform):
            x = self.transform(x)
        if callable(self.target_transform):
            target = self.target_transform(target)
        return x, target

    def __len__(self):
        return len(self.cache)

    def split(self, train_size=0.7, random_state=None, stratify=None):
        idx = np.arange(len(self))
        train_idx, test_idx = train_test_split(idx, train_size=train_size, random_state=random_state, stratify=stratify)
        train_set = copy.copy(self).select(train_idx)
        test_set = copy.copy(self).select(test_idx)
        return train_set, test_set

    def select(self, idx):
        self.cache = [self.cache[i] for i in idx]
        return self

    def init_cache(self):
        self.cache = []
        with self.open(self.root) as f:
            self.cache = self.load_cache_recursive(f)

    def load_cache_recursive(self, f):
        cache = []
        for key, el in f.items():
            if isinstance(el, abc.MutableMapping):
                cache.extend(self.load_cache_recursive(el))
            else:
                target = el.attrs.get(self.target_tag)
                cache.append((el.ref, target))
        return cache

    def open(self, path):
        return h5py.File(path, "r", libver="latest")

    def load_data(self, el):
        x = np.array(el[:])
        x = torch.from_numpy(x).float()
        x = torch.clamp(x / 255.0, 0.0, 1.0)
        return x

    def co(self, augment):
        self.transform = lambda x: (self.transform(x), augment(x))
        return self
