import tensorflow as tf
import tflearn
import numpy as np

from core.utils.constants import (
    HIDDEN_LAYER_SIZE,
    CONV_FILTER_NUM,
    CONV_FILTER_SIZE,
    ACTION_EPSILON,
    PPO_EPSILON,
    REWARD_DECAY,
)


class PensievePPO:
    def __init__(self, n_state, n_action, lr, session):
        # PPO entropy
        self._entropy = 5

        # Basic dimension info & learning rate
        self.n_state = n_state
        self.n_action = n_action
        self.lr = lr

        # Tensorflow session
        self.session = session

        # Input placeholders
        self.reward_placeholder = tf.placeholder(tf.float32, [None, 1])
        self.action_placeholder = tf.placeholder(tf.float32, [None, self.n_action])
        self.state_placeholder = tf.placeholder(
            tf.float32, [None, self.n_state[0], self.n_state[1]]
        )
        self.entropy_weight_placeholder = tf.placeholder(tf.float32)

        # Policy & value placeholders
        self.prev_policy_placeholder = tf.placeholder(tf.float32, [None, self.n_action])
        self.policy, self.value = self.__create_network(
            input_placeholder=self.state_placeholder
        )
        self.clipped_policy = tf.clip_by_value(
            self.policy, ACTION_EPSILON, 1 - ACTION_EPSILON
        )

        # Derived variables
        self.log_prob = tf.log(
            tf.reduce_sum(
                tf.multiply(self.clipped_policy, self.action_placeholder),
                reduction_indices=1,
                keepdims=True,
            )
        )
        self.entropy = tf.multiply(self.clipped_policy, tf.log(self.clipped_policy))
        self.advantage = tf.stop_gradient(self.reward_placeholder - self.value)
        self.ppo_loss = tf.minimum(
            self.__ratio_function(
                self.clipped_policy,
                self.prev_policy_placeholder,
                self.action_placeholder,
            )
            * self.advantage,
            tf.clip_by_value(
                self.__ratio_function(
                    self.clipped_policy,
                    self.prev_policy_placeholder,
                    self.action_placeholder,
                ),
                clip_value_min=1 - PPO_EPSILON,
                clip_value_max=1 + PPO_EPSILON,
            )
            * self.advantage,
        )
        self.dual_loss = (
            tf.cast(tf.less(self.advantage, 0), dtype=tf.float32)
            * tf.maximum(self.ppo_loss, 3 * self.advantage)
            + tf.cast(tf.greater_equal(self.advantage, 0), dtype=tf.float32)
            * self.ppo_loss
        )

        # Get all network parameters
        self.network_params = tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor"
        )
        self.network_params += tf.get_collection(
            tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic"
        )

        # Set all network parameters
        self.network_param_placeholders = []
        for param in self.network_params:
            self.network_param_placeholders.append(
                tf.placeholder(tf.float32, shape=param.get_shape())
            )
        self.assign_network_params_operators = []
        for idx, param in enumerate(self.network_param_placeholders):
            self.assign_network_params_operators.append(
                self.network_params[idx].assign(param)
            )

        # Final losses & optimizer
        self.loss = -tf.reduce_sum(
            self.dual_loss
        ) + self.entropy_weight_placeholder * tf.reduce_sum(self.entropy)
        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        self.value_loss = tflearn.mean_square(self.value, self.reward_placeholder)
        self.value_optimizer = tf.train.AdamOptimizer(self.lr * 10).minimize(
            self.value_loss
        )

    def __ratio_function(self, policy, prev_policy, action):
        return tf.reduce_sum(
            tf.multiply(policy, action), reduction_indices=1, keepdims=True
        ) / tf.reduce_sum(
            tf.multiply(prev_policy, action), reduction_indices=1, keepdims=True
        )

    def __create_network(self, input_placeholder):
        with tf.variable_scope("actor"):
            split_0 = tflearn.fully_connected(
                input_placeholder[:, 0:1, -1],
                n_units=HIDDEN_LAYER_SIZE,
                activation="relu",
            )
            split_1 = tflearn.fully_connected(
                input_placeholder[:, 1:2, -1],
                n_units=HIDDEN_LAYER_SIZE,
                activation="relu",
            )
            split_2 = tflearn.conv_1d(
                input_placeholder[:, 2:3, :],
                nb_filter=CONV_FILTER_NUM,
                filter_size=CONV_FILTER_SIZE,
                activation="relu",
            )
            split_3 = tflearn.conv_1d(
                input_placeholder[:, 3:4, :],
                nb_filter=CONV_FILTER_NUM,
                filter_size=CONV_FILTER_SIZE,
                activation="relu",
            )
            split_4 = tflearn.conv_1d(
                input_placeholder[:, 4:5, : self.n_action],
                nb_filter=CONV_FILTER_NUM,
                filter_size=CONV_FILTER_SIZE,
                activation="relu",
            )
            split_5 = tflearn.fully_connected(
                input_placeholder[:, 5:6, -1],
                n_units=HIDDEN_LAYER_SIZE,
                activation="relu",
            )

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)

            merged_input = tflearn.merge(
                [split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5],
                "concat",
            )
            raw_policy = tflearn.fully_connected(
                merged_input, n_units=HIDDEN_LAYER_SIZE, activation="relu"
            )
            policy = tflearn.fully_connected(
                raw_policy, n_units=self.n_action, activation="softmax"
            )

        with tf.variable_scope("critic"):
            split_0 = tflearn.fully_connected(
                input_placeholder[:, 0:1, -1],
                n_units=HIDDEN_LAYER_SIZE,
                activation="relu",
            )
            split_1 = tflearn.fully_connected(
                input_placeholder[:, 1:2, -1],
                n_units=HIDDEN_LAYER_SIZE,
                activation="relu",
            )
            split_2 = tflearn.conv_1d(
                input_placeholder[:, 2:3, :],
                nb_filter=CONV_FILTER_NUM,
                filter_size=CONV_FILTER_SIZE,
                activation="relu",
            )
            split_3 = tflearn.conv_1d(
                input_placeholder[:, 3:4, :],
                nb_filter=CONV_FILTER_NUM,
                filter_size=CONV_FILTER_SIZE,
                activation="relu",
            )
            split_4 = tflearn.conv_1d(
                input_placeholder[:, 4:5, : self.n_action],
                nb_filter=CONV_FILTER_NUM,
                filter_size=CONV_FILTER_SIZE,
                activation="relu",
            )
            split_5 = tflearn.fully_connected(
                input_placeholder[:, 5:6, -1],
                n_units=HIDDEN_LAYER_SIZE,
                activation="relu",
            )

            split_2_flat = tflearn.flatten(split_2)
            split_3_flat = tflearn.flatten(split_3)
            split_4_flat = tflearn.flatten(split_4)

            merged_input = tflearn.merge(
                [split_0, split_1, split_2_flat, split_3_flat, split_4_flat, split_5],
                "concat",
            )
            raw_value = tflearn.fully_connected(
                merged_input, n_units=HIDDEN_LAYER_SIZE, activation="relu"
            )
            value = tflearn.fully_connected(raw_value, n_units=1, activation="linear")

        return policy, value

    def predict(self, inputs):
        inputs = np.reshape(inputs, (-1, self.n_state[0], self.n_state[1]))
        action = self.session.run(
            self.clipped_policy, feed_dict={self.state_placeholder: inputs}
        )
        return action[0]

    def get_params(self):
        return self.session.run(self.network_params)

    def set_params(self, network_params):
        self.session.run(
            self.assign_network_params_operators,
            feed_dict={
                key: val
                for key, val in zip(self.network_param_placeholders, network_params)
            },
        )

    def decay_entropy(self, decay=0.6):
        self._entropy *= decay

    @property
    def clipped_entropy(self):
        return np.clip(self._entropy, 0.1, 5)

    def train(self, state_batch, action_batch, policy_batch, value_batch, epoch):
        (
            state_batch,
            action_batch,
            policy_batch,
            value_batch,
        ) = tflearn.data_utils.shuffle(
            state_batch, action_batch, policy_batch, value_batch
        )
        self.session.run(
            [self.optimizer, self.value_optimizer],
            feed_dict={
                self.state_placeholder: state_batch,
                self.action_placeholder: action_batch,
                self.reward_placeholder: value_batch,
                self.prev_policy_placeholder: policy_batch,
                self.entropy_weight_placeholder: self.clipped_entropy,
            },
        )

    def compute_value(self, state_batch, reward_batch, terminal):
        batch_size = len(state_batch)
        accumulated_reward_batch = np.zeros([len(reward_batch), 1])

        if terminal:
            accumulated_reward_batch[-1, 0] = 0
        else:
            value_batch = self.session.run(
                self.value,
                feed_dict={
                    self.state_placeholder: state_batch,
                },
            )
            accumulated_reward_batch[-1, 0] = value_batch[-1, 0]
        for t in reversed(range(batch_size - 1)):
            accumulated_reward_batch[t, 0] = (
                reward_batch[t] + REWARD_DECAY * accumulated_reward_batch[t + 1, 0]
            )

        return list(accumulated_reward_batch)


if __name__ == "__main__":
    pass
