# TODO: This file should be rewinded after the project

import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, ".."))

import joblib
import tensorflow as tf
import argparse
import os.path as osp
from core.samplers.utils import rollout
from core.logger import logger
from core.envs.pensieve_env_park import PenseieveEnvPark
import json
import numpy as np


def sim_policy_for_pensieve(itr, pkl_path, json_path, mode):
    with tf.Session() as sess:
        print("Testing policy %s with mode %s" % (pkl_path, mode))
        json_params = json.load(open(json_path, "r"))
        data = joblib.load(pkl_path)
        policy = data["policy"]
        # env = data["env"]
        env = PenseieveEnvPark(mode="test-" + mode)

        rewards = []
        for i in range(10):
            path = rollout(
                env,
                policy,
                max_path_length=1000,
                animated=False,
                ignore_done=False,
                adapt_batch_size=json_params.get("adapt_batch_size", None),
                logger_handle=open(f"./log/iter_{itr}_mode_{mode}_trace_{i}.log", "w"),
            )
            rewards.append(np.mean([np.mean(p["rewards"]) for p in path]))

        avg_reward = np.mean(rewards)
        print(f"TestAverageReward({mode}) = {avg_reward}")

        with open("./data/grbal_pensieve/reward.log", "a") as f:
            f.write(f"{avg_reward}\n")


if __name__ == "__main__":
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("param", type=str, help="Directory with the pkl and json file")
    parser.add_argument(
        "--max_path_length", "-l", type=int, default=1000, help="Max length of rollout"
    )
    parser.add_argument(
        "--num_rollouts", "-n", type=int, default=10, help="Max length of rollout"
    )
    parser.add_argument("--speedup", type=float, default=1, help="Speedup")
    parser.add_argument("--video_filename", type=str, help="path to the out video file")
    parser.add_argument(
        "--prompt",
        type=bool,
        default=False,
        help="Whether or not to prompt for more sim",
    )
    parser.add_argument(
        "--ignore_done",
        action="store_true",
        help="Whether stop animation when environment done or continue anyway",
    )
    args = parser.parse_args()

    with tf.Session() as sess:
        pkl_path = osp.join(args.param, "params.pkl")
        json_path = osp.join(args.param, "params.json")
        print("Testing policy %s" % pkl_path)
        json_params = json.load(open(json_path, "r"))
        data = joblib.load(pkl_path)
        policy = data["policy"]
        env = data["env"]
        for _ in range(args.num_rollouts):
            path = rollout(
                env,
                policy,
                max_path_length=args.max_path_length,
                animated=True,
                ignore_done=args.ignore_done,
                adapt_batch_size=json_params.get("adapt_batch_size", None),
            )
            print(np.mean([np.mean(p["rewards"]) for p in path]))
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--iteration", "-i", type=int, help="Iteration")
    parser.add_argument("--json", "-j", type=str, help="JSON file path")
    parser.add_argument("--mode", "-m", type=str, help="Test mode")
    args = parser.parse_args()

    sim_policy_for_pensieve(
        args.iteration,
        f"./data/grbal_pensieve/itr_{args.iteration}.pkl",
        args.json,
        args.mode,
    )
