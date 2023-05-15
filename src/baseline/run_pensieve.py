import numpy as np
import tensorflow as tf
import multiprocessing as mp
import sys
import os
from tqdm import tqdm

current_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_path, ".."))
tf.logging.set_verbosity(tf.logging.ERROR)

from baseline.pensieve_ppo import PensievePPO
from core.envs.abr_env import ABREnv
from core.utils.constants import (
    AGENT_NUM,
    STATE_INFO,
    STATE_LEN,
    ACTION_DIM,
    ACTOR_LR,
    TRAIN_EPOCH,
    PPO_TRAIN_EPOCH,
    PPO_SUMMARY_DIR,
    TRAIN_SEQUENCE_LEN,
    MODEL_SAVE_INTERVAL,
    VIDEO_BIT_RATE,
    BUFFER_NORM_FACTOR,
)


def convert_observation(observation):
    converted_observation = np.zeros((STATE_INFO, STATE_LEN))
    converted_observation[0, -1] = VIDEO_BIT_RATE[int(observation[18])] / float(
        np.max(VIDEO_BIT_RATE)
    )
    converted_observation[1, -1] = observation[16] / BUFFER_NORM_FACTOR
    converted_observation[2, :] = observation[0:8] / 1e6
    converted_observation[3, :] = observation[8:16] / 10
    converted_observation[4, :ACTION_DIM] = observation[-6:] / 1e6
    converted_observation[5, -1] = observation[17]

    return converted_observation


def model_test(epoch, model_path):
    os.system(f"python ./src/baseline/sim_baseline_abr_policy.py {model_path} {epoch}")

    with open("./data/pensieve/reward.log", "r") as f:
        rewards_mean = float(f.readlines()[-1].replace("\n", ""))

    return rewards_mean


def central_agent(network_param_queues, experience_queues):
    assert len(network_param_queues) == AGENT_NUM
    assert len(experience_queues) == AGENT_NUM
    tf_config = tf.ConfigProto(
        intra_op_parallelism_threads=5, inter_op_parallelism_threads=5
    )

    with tf.Session(config=tf_config) as session:
        actor = PensievePPO(
            session=session,
            n_state=[STATE_INFO, STATE_LEN],
            n_action=ACTION_DIM,
            lr=ACTOR_LR,
        )

        session.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=1000)  # Save neural network parameters

        max_reward = -1e4
        tick_gap = 0
        for epoch in tqdm(range(TRAIN_EPOCH)):
            # Synchronize the network parameters of work agent
            actor_network_params = actor.get_params()
            for i in range(AGENT_NUM):
                network_param_queues[i].put(actor_network_params)

            # Generate training batch from child agent experience
            state_batch = []
            action_batch = []
            policy_batch = []
            value_batch = []
            for i in range(AGENT_NUM):
                states, actions, policies, values = experience_queues[i].get()
                state_batch += states
                action_batch += actions
                policy_batch += policies
                value_batch += values
            state_batch = np.stack(state_batch, axis=0)
            action_batch = np.vstack(action_batch)
            policy_batch = np.vstack(policy_batch)
            value_batch = np.vstack(value_batch)

            # Why train 5 epoch here? Increase the reward?
            for _ in range(PPO_TRAIN_EPOCH):
                actor.train(state_batch, action_batch, policy_batch, value_batch, epoch)

            if epoch % MODEL_SAVE_INTERVAL == 0:
                # Test the network & save the neural network parameters to disk
                model_path = PPO_SUMMARY_DIR + "ppo_epoch_" + str(epoch) + ".ckpt"
                saver.save(session, model_path)

                avg_reward = model_test(
                    epoch,
                    model_path,
                )

                if avg_reward > max_reward:
                    max_reward = avg_reward
                    tick_gap = 0
                else:
                    tick_gap += 1

                if tick_gap >= 5:
                    actor.decay_entropy()
                    tick_gap = 0


def agent(network_param_queue, experience_queue):
    env = ABREnv(mode="train-lumos", full_observation=True)
    with tf.Session() as session:
        actor = PensievePPO(
            session=session,
            n_state=[STATE_INFO, STATE_LEN],
            n_action=ACTION_DIM,
            lr=ACTOR_LR,
        )

        # Initial synchronization of the network parameters from the coordinator
        actor.set_params(network_param_queue.get())

        for _ in range(TRAIN_EPOCH):
            observation = convert_observation(env.reset())

            state_batch = []
            action_batch = []
            policy_batch = []
            reward_batch = []
            for _ in range(TRAIN_SEQUENCE_LEN):
                state_batch.append(observation)
                action_probability = actor.predict(
                    np.reshape(observation, (1, STATE_INFO, STATE_LEN))
                )

                # Gumbel noise
                noise = np.random.gumbel(size=len(action_probability))
                bitrate = np.argmax(np.log(action_probability) + noise)

                observation, reward, done, _ = env.step(np.array([bitrate]))
                observation = convert_observation(observation)

                action = np.zeros(ACTION_DIM)
                action[bitrate] = 1

                action_batch.append(action)
                reward_batch.append(reward)
                policy_batch.append(action_probability)

                if done:
                    break

            value_batch = actor.compute_value(state_batch, reward_batch, done)
            experience_queue.put([state_batch, action_batch, policy_batch, value_batch])

            actor.set_params(network_param_queue.get())


def main():
    np.random.seed(seed=42)

    if not os.path.exists(PPO_SUMMARY_DIR):
        os.makedirs(PPO_SUMMARY_DIR)

    # Communication queues between agents
    network_param_queues = []
    experience_queues = []
    for _ in range(AGENT_NUM):
        network_param_queues.append(mp.Queue(1))
        experience_queues.append(mp.Queue(1))

    # Create a coordinator and multiple agent processes
    # @note: Threading is not desirable due to python GIL
    coordinator = mp.Process(
        target=central_agent, args=(network_param_queues, experience_queues)
    )
    coordinator.start()

    agents = []
    for i in range(AGENT_NUM):
        agents.append(
            mp.Process(
                target=agent, args=(network_param_queues[i], experience_queues[i])
            )
        )
    for i in range(AGENT_NUM):
        agents[i].start()

    # Wait until training is done
    coordinator.join()


if __name__ == "__main__":
    main()
