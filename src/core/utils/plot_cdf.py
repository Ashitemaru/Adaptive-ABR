import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d

BITRATE_MAP = {
    20000: 10000,
    40000: 20000,
    60000: 30000,
    80000: 120000,
    110000: 150000,
    160000: 200000,
}


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


def load_reward(data_path, skip_tail=False):
    data = []
    for name in os.listdir(data_path):
        with open(os.path.join(data_path, name), "r") as handler:
            lines = handler.readlines()
            if skip_tail:
                lines = lines[:-1]

            prev_bitrate = -1
            for line in lines:
                line = line.replace("\n", "")
                if line == "":
                    continue

                raw = line.split("\t")
                bitrate = int(raw[0])
                rebuffer = float(raw[1])

                if prev_bitrate == -1:
                    prev_bitrate = bitrate
                    continue

                smooth = abs(prev_bitrate - bitrate)
                prev_bitrate = bitrate

                reward = bitrate / 1000 - 320 * rebuffer - smooth / 1000
                data.append(reward)

    return np.array(data)


if __name__ == "__main__":
    pensieve = load_reward(
        "./final/pensieve(train:norm,test:nyu-mets)",
        skip_tail=True,
    )
    oboe = load_reward(
        "./final/oboe(test:nyu-mets)",
        skip_tail=False,
    )
    grbal = load_reward(
        "./final/grbal(train:lumos,test:nyu-mets)",
        # "./log/grbal-2023-05-31-09:03:03",
        skip_tail=True,
    )
    bba = load_reward(
        "./final/bba(test:nyu-mets)",
        skip_tail=True,
    )

    print(np.mean(bba), np.mean(oboe), np.mean(pensieve), np.mean(grbal))
    plot_cdf(
        [bba, oboe, pensieve, grbal],
        ["BBA", "Oboe", "Pensieve", "Ours"],
        xlabel="QoE",
        path="./image/nyu_mets_fluent.png",
        xmin=-100,
        xmax=160,
    )
