import numpy as np
import tensorflow as tf
from tqdm import tqdm
import os
from datetime import datetime
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, ".."))
tf.logging.set_verbosity(tf.logging.ERROR)

from baseline.pensieve_ppo import PensievePPO
from baseline.run_pensieve import convert_observation
from experiment_utils.sim_abr_policy import rollout
from core.envs.abr_env import ABREnv
from core.utils.constants import (
    ACTION_DIM,
    STATE_INFO,
    STATE_LEN,
    ACTOR_LR,
)


class PolicyWrapper:
    def __init__(self, core):
        self.core = core

    def reset(self):
        pass

    def get_action(self, observation):
        observation = convert_observation(observation)
        action_probability = self.core.predict(observation)
        noise = np.random.gumbel(size=len(action_probability))
        action = np.argmax(np.log(action_probability) + noise)

        return np.array([[action]]), {}


def sim_pensieve_policy(model_path, epoch):
    np.random.seed(seed=42)

    log_path = (
        os.getcwd() + "/log/pensieve-" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    )
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    env = ABREnv(mode="test-lumos", full_observation=True)

    with tf.Session() as session:
        actor = PensievePPO(
            session=session,
            n_state=[STATE_INFO, STATE_LEN],
            n_action=ACTION_DIM,
            lr=ACTOR_LR,
        )
        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(session, model_path)

        rewards = []
        for i in tqdm(range(env.trace_num)):
            path = rollout(
                env=env,
                policy=PolicyWrapper(actor),
                trace_idx=i,
                adapt_batch_size=None,
                logger_handle=open(f"{log_path}/epoch_{epoch}_trace_{i}.log", "w"),
            )
            rewards.append(np.mean(path[-1]["rewards"]))

        avg_reward = np.mean(rewards)
        print(f"TestAverageReward = {avg_reward}")

        if not os.path.exists("./data/pensieve"):
            os.makedirs("./data/pensieve")
        with open("./data/pensieve/reward.log", "a") as f:
            f.write(f"{avg_reward}\n")


if __name__ == "__main__":
    sim_pensieve_policy(model_path=sys.argv[1], epoch=int(sys.argv[2]))
