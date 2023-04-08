import os
import sys
import json

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, ".."))

from core.utils.utils import ClassEncoder
from core.envs.normalized_env import normalize
from core.policies.mpc_controller import MPCController
from core.samplers.sampler import Sampler
from core.samplers.model_sample_processor import ModelSampleProcessor
from core.logger import logger
from core.dynamics.meta_mlp_dynamics import MetaMLPDynamicsModel
from core.trainers.mb_trainer import Trainer
from core.envs.pensieve_env_park import PenseieveEnvPark

EXP_NAME = "grbal_pensieve"


def run_experiment(config):
    exp_dir = os.getcwd() + "/data/" + EXP_NAME + "/" + config.get("exp_name", "")
    logger.configure(
        dir=exp_dir, format_strs=["stdout", "log", "csv"], snapshot_mode="all"
    )
    json.dump(
        config,
        open(exp_dir + "/params.json", "w"),
        indent=2,
        sort_keys=True,
        cls=ClassEncoder,
    )

    env = normalize(config["env"](mode="train-1"))

    dynamics_model = MetaMLPDynamicsModel(
        name="dyn_model",
        env=env,
        meta_batch_size=config["meta_batch_size"],
        inner_learning_rate=config["inner_learning_rate"],
        learning_rate=config["learning_rate"],
        hidden_sizes=config["hidden_sizes_model"],
        valid_split_ratio=config["valid_split_ratio"],
        rolling_average_persistency=config["rolling_average_persistency"],
        hidden_nonlinearity=config["hidden_nonlinearity_model"],
        batch_size=config["adapt_batch_size"],
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
        n_parallel=config["n_parallel"],
        max_path_length=config["max_path_length"],
        num_rollouts=config["num_rollouts"],
        adapt_batch_size=config[
            "adapt_batch_size"
        ],  # Comment this out and it won't adapt during rollout
    )

    sample_processor = ModelSampleProcessor(recurrent=True)

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
        "env": PenseieveEnvPark,
        "max_path_length": 1000,
        "task": None,
        "normalize": True,
        "n_itr": 50,
        "discount": 1.0,
        # Policy
        "n_candidates": 500,
        "horizon": 10,
        "use_cem": False,
        "num_cem_iters": 5,
        # Training
        "num_rollouts": 5,
        "valid_split_ratio": 0.1,
        "rolling_average_persistency": 0.99,
        "initial_random_samples": True,
        # Dynamics Model
        "meta_batch_size": 10,
        "hidden_nonlinearity_model": "relu",
        "learning_rate": 1e-3,
        "inner_learning_rate": 0.001,
        "hidden_sizes_model": (512, 512, 512),
        "dynamic_model_epochs": 100,
        "adapt_batch_size": 16,
        # Other
        "n_parallel": 5,
    }

    run_experiment(config)
