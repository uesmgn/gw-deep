import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def setup(ax, **kwargs):
    plt.style.use("seaborn-poster")
    plt.rc("legend", fontsize=10)
    for k, v in kwargs.items():
        if k == "xlabel":
            ax.set_xlabel(v)
        elif k == "ylabel":
            ax.set_ylabel(v)
        elif k == "xlim":
            ax.set_xlim(v)
        elif k == "ylim":
            ax.set_ylim(v)
        elif k == "title":
            ax.set_title(v)


class LossLogger:
    def __init__(self):
        self.stats = defaultdict(list)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            self.stats[key].append(value)

    def items(self):
        return self.stats.items()

    def save(self, key, epoch, out, **kwargs):
        if epoch > 1:
            fig, ax = plt.subplots()
            yy = self.stats[key]
            xx = np.linspace(0, epoch, len(yy))
            ax.plot(xx, yy)
            setup(ax, **kwargs)
            plt.tight_layout()
            plt.savefig(out)
            plt.close()
