import numpy as np
import matplotlib.pyplot as plt


def setup(ax, **kwargs):
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
        self.stats = {}

    def update(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self.stats:
                self.stats[key] = defaultdict(list)
            self.stats[key].append(value)

    def items(self):
        return self.stats.items()

    def save(self, values, epoch, out, **kwargs):
        if epoch > 1:
            fig, ax = plt.subplots()
            xx = np.linspace(0, epoch, len(values))
            ax.plot(xx, values)
            setup(ax, **kwargs)
            plt.tight_layout()
            plt.savefig(out)
            plt.close()
