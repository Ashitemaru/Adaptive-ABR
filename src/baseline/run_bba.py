import numpy as np
from tqdm import tqdm
import os
from datetime import datetime
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, ".."))

from experiment_utils.sim_abr_policy import rollout
from core.envs.abr_env import ABREnv
from core.utils.constants import ACTION_DIM


class PolicyWrapper:
    def __init__(self, cushion, reservior, n_action):
        self.cushion = cushion
        self.reservior = reservior
        self.n_action = n_action

    def reset(self):
        pass

    def get_action(self, observation):
        buffer_size = observation[2]  # When the env is not fully observed
        if buffer_size < self.cushion:
            action = 0
        elif buffer_size > self.reservior + self.cushion:
            action = self.n_action - 1
        else:
            action = int(
                (self.n_action - 1) * (buffer_size - self.cushion) / self.reservior
            )

        return np.array([[action]]), {}


def sim_bba_policy(cushion, reservior):
    log_path = os.getcwd() + "/log/bba-" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    env = ABREnv(mode="test-nyu-mets", full_observation=False)

    rewards = []
    for i in tqdm(range(env.trace_num)):
        path = rollout(
            env=env,
            policy=PolicyWrapper(cushion, reservior, ACTION_DIM),
            trace_idx=i,
            adapt_batch_size=None,
            logger_handle=open(f"{log_path}/trace_{i}.log", "w"),
        )
        rewards.append(np.mean(path[-1]["rewards"]))

    avg_reward = np.mean(rewards)
    print(f"TestAverageReward = {avg_reward}")


if __name__ == "__main__":
    sim_bba_policy(cushion=float(sys.argv[1]), reservior=float(sys.argv[2]))
