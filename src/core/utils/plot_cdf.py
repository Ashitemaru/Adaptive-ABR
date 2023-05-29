import matplotlib.pyplot as plt
import numpy as np


def rgb_to_hex(r, g, b):
    return "#%02x%02x%02x" % (r, g, b)


def plot_cdf(bases, labels, xlabel, path, xmin=None, xmax=None):
    assert len(bases) == len(labels)

    lines = ["-", ":", "-.", "--", "--"]
    colors = [
        rgb_to_hex(102, 49, 160),
        rgb_to_hex(237, 65, 29),
        rgb_to_hex(255, 192, 0),
        rgb_to_hex(29, 29, 29),
        rgb_to_hex(0, 212, 97),
    ]

    plt.figure()
    plt.grid(True)
    for data, color, line, label in zip(bases, colors, lines, labels):
        values, base = np.histogram(data, bins=1000)
        cumulative = np.cumsum(values)
        cumulative = cumulative / np.max(cumulative)
        plt.plot(base[:-1], cumulative, line, color=color, lw=1, label=label)

    plt.legend(loc="best")
    plt.xlabel(xlabel)
    plt.ylabel("CDF")
    plt.ylim(0, 1)
    if xmin is not None:
        plt.xlim(xmin=xmin)
    if xmax is not None:
        plt.xlim(xmax=xmax)

    plt.savefig(path)


if __name__ == "__main__":
    pass
