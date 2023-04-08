import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

if __name__ == "__main__":
    rewards = []
    with open("./data/grbal_pensieve/reward.log", "r") as f:
        for line in f:
            rewards.append(float(line.replace("\n", "")))

    print(f"Average reward: {np.mean(rewards)}")

    plt.figure()
    plt.plot(rewards)
    plt.xlabel("Iteration")
    plt.ylabel("Average reward")
    plt.savefig("./image/reward.png")
