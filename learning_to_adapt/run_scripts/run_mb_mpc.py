import os
import sys

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, ".."))

from core.dynamics.mlp_dynamics import MLPDynamicsModel
from core.trainers.mb_trainer import Trainer
from core.policies.mpc_controller import MPCController
from core.samplers.sampler import Sampler
from core.logger import logger
from core.envs.normalized_env import normalize
from core.utils.utils import ClassEncoder
from core.samplers.model_sample_processor import ModelSampleProcessor
from core.envs import *
import json
import os

EXP_NAME = "mb_mpc"


def run_experiment(config):
    exp_dir = os.getcwd() + "/data/" + EXP_NAME + "/" + config.get("exp_name", "")
    logger.configure(
        dir=exp_dir, format_strs=["stdout", "log", "csv"], snapshot_mode="last"
    )
    json.dump(
        config,
        open(exp_dir + "/params.json", "w"),
        indent=2,
        sort_keys=True,
        cls=ClassEncoder,
    )

    env = normalize(config["env"](reset_every_episode=True, task=config["task"]))

    dynamics_model = MLPDynamicsModel(
        name="dyn_model",
        env=env,
        learning_rate=config["learning_rate"],
        hidden_sizes=config["hidden_sizes"],
        valid_split_ratio=config["valid_split_ratio"],
        rolling_average_persistency=config["rolling_average_persistency"],
        hidden_nonlinearity=config["hidden_nonlinearity"],
        batch_size=config["batch_size"],
    )

    policy = MPCController(
        name="policy",
        env=env,
        dynamics_model=dynamics_model,
        discount=config["discount"],
        n_candidates=config["n_candidates"],
        horizon=config["horizon"],
        use_cem=config["use_cem"],
        num_cem_iters=config["num_cem_iters"],
    )

    sampler = Sampler(
        env=env,
        policy=policy,
        num_rollouts=config["num_rollouts"],
        max_path_length=config["max_path_length"],
        n_parallel=config["n_parallel"],
    )

    sample_processor = ModelSampleProcessor(recurrent=False)

    algo = Trainer(
        env=env,
        policy=policy,
        dynamics_model=dynamics_model,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config["n_itr"],
        initial_random_samples=config["initial_random_samples"],
        dynamics_model_max_epochs=config["dynamic_model_epochs"],
    )
    algo.train()


if __name__ == "__main__":
    # -------------------- Define Variants -----------------------------------

    config = {
        # Environment
        "env": HalfCheetahEnv,
        "task": None,
        # Policy
        "n_candidates": 2000,
        "horizon": 20,
        "use_cem": False,
        "num_cem_iters": 5,
        "discount": 1.0,
        # Sampling
        "max_path_length": 100,
        "num_rollouts": 10,
        "initial_random_samples": True,
        # Training
        "n_itr": 50,
        "learning_rate": 0.001,
        "batch_size": 128,
        "dynamic_model_epochs": 100,
        "valid_split_ratio": 0.1,
        "rolling_average_persistency": 0.99,
        # Dynamics Model
        "hidden_sizes": (512, 512),
        "hidden_nonlinearity": "relu",
        # Other
        "n_parallel": 2,
    }

    run_experiment(config)
