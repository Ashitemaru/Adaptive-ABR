import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import sys

if __name__ == "__main__":
    reward_lists = []
    for exp in sys.argv[1:]:
        rewards = []
        with open(f"./data/{exp}/reward.log", "r") as f:
            for line in f:
                rewards.append(float(line.replace("\n", "")))

        reward_lists.append(rewards)

    plt.figure()
    plt.title("Average reward (QoE)")
    for rewards in reward_lists:
        plt.plot(rewards)
    plt.legend(sys.argv[1:])
    plt.xlabel("Iteration")
    plt.ylabel("Average reward")
    plt.savefig("./image/reward.png")
