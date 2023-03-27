# Learning to Adapt in Dynamic, Real-World Environment through Meta-Reinforcement Learning

Implementation of [Learning to Adapt in Dynamic, Real-World Environment through Meta-Reinforcement Learning](https://arxiv.org/abs/1803.11347).

The code is written in Python 3 and builds on Tensorflow. The environments require the Mujoco131 physics engine.

## Usage

Run experiment (GrBAL as the example, ReBAL is the same):

```shell
python run_scripts/run_grbal.py
```

We have also implement a non-adaptive model-based method that uses random shooting or cross-entropy for planning. You can run this baseline by executing the command:

```shell
python run_scripts/run_mb_mpc.py
```

When running experiments, the data will be stored in `data/$EXPERIMENT_NAME`. You can visualize the learning process by using the visualization kit:

```shell
python viskit/frontend.py data/$EXPERIMENT_NAME
```

In order to visualize and test a learned policy run:

```shell
python experiment_utils/sim_policy data/$EXPERIMENT_NAME
```
