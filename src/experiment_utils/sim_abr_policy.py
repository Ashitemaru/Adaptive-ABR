import os
import sys
import joblib
import tensorflow as tf
import argparse
from tqdm import tqdm
import json
import numpy as np
from datetime import datetime

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, ".."))

from core.envs.abr_env import ABREnv


def rollout(
    env,
    policy,
    trace_idx,
    max_path_length=np.inf,
    ignore_done=False,
    adapt_batch_size=None,
    logger_handle=None,
):
    # Get wrapped env
    wrapped_env = env
    while hasattr(wrapped_env, "_wrapped_env"):
        wrapped_env = wrapped_env._wrapped_env

    paths = []
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []

    observation = env.reset(trace_idx)
    policy.reset()
    path_length = 0

    while path_length < max_path_length:
        if adapt_batch_size is not None and len(observations) > adapt_batch_size + 1:
            adapt_observation = observations[-adapt_batch_size - 1 : -1]
            adapt_action = actions[-adapt_batch_size - 1 : -1]
            adapt_next_observation = observations[-adapt_batch_size:]
            policy.dynamics_model.switch_to_pre_adapt()
            policy.dynamics_model.adapt(
                [np.array(adapt_observation)],
                [np.array(adapt_action)],
                [np.array(adapt_next_observation)],
            )
        action, agent_info = policy.get_action(observation)
        next_observation, reward, done, env_info = env.step(action[0])
        observations.append(observation)
        rewards.append(reward)
        actions.append(action[0])
        agent_infos.append(agent_info)
        env_infos.append(env_info)
        path_length += 1

        if logger_handle is not None:
            logger_handle.write(
                str(env_info["bitrate"])
                + "\t"
                + str(env_info["stall_time"])
                + "\t"
                + str(env_info["buffer_size"])
                + "\t"
                + str(env_info["chunk_size"])
                + "\t"
                + str(env_info["delay"])
                + "\t"
                + str(reward)
                + "\n"
            )

        if done and not ignore_done:  # and not animated:
            break
        observation = next_observation

        paths.append(
            dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                agent_infos=agent_infos,
                env_infos=env_infos,
            )
        )

    logger_handle.write(f"Average reward: {np.mean(rewards)}\n")
    return paths


def sim_abr_policy(itr, pkl_path, json_path, mode):
    log_path = os.getcwd() + "/log/" + datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    with tf.Session() as sess:
        print("Testing policy %s with mode %s" % (pkl_path, mode))
        json_params = json.load(open(json_path, "r"))
        data = joblib.load(pkl_path)
        policy = data["policy"]
        env = ABREnv(mode="test-" + mode)

        rewards = []
        for i in tqdm(range(env.trace_num)):
            path = rollout(
                env,
                policy,
                trace_idx=i,
                max_path_length=1000,
                ignore_done=False,
                adapt_batch_size=json_params.get("adapt_batch_size", None),
                logger_handle=open(
                    f"{log_path}/iter_{itr}_mode_{mode}_trace_{i}.log", "w"
                ),
            )
            rewards.append(np.mean([np.mean(p["rewards"]) for p in path]))

        avg_reward = np.mean(rewards)
        print(f"TestAverageReward({mode}) = {avg_reward}")

        with open("./data/grbal_abr/reward.log", "a") as f:
            f.write(f"{avg_reward}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", "-i", type=int, help="Iteration")
    parser.add_argument("--json", "-j", type=str, help="JSON file path")
    parser.add_argument("--mode", "-m", type=str, help="Test mode")
    args = parser.parse_args()

    sim_abr_policy(
        args.iteration,
        f"./data/grbal_abr/itr_{args.iteration}.pkl",
        args.json,
        args.mode,
    )
