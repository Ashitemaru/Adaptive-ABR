import numpy as np

from learning_to_adapt.core.utils.serializable import Serializable
from learning_to_adapt.core.envs.base import Env
from src.envs.pensieve_video_server import PensieveVideoServer
from src.constants import (
    TRACE_FILE,
    DEFAULT_QUALITY,
    STATE_INFO,
    STATE_LEN,
    VIDEO_BIT_RATE,
    BUFFER_NORM_FACTOR,
    REBUF_PENALTY,
    SMOOTH_PENALTY,
    M_IN_K,
    ACTION_DIM,
    CHUNK_TIL_VIDEO_END_CAP,
)


class PensieveEnv(Env, Serializable):
    def __init__(self, random_seed=42):
        Serializable.quick_init(self, locals())

        np.random.seed(random_seed)

        self.server = PensieveVideoServer(TRACE_FILE, random_seed=random_seed)
        self.last_bitrate = DEFAULT_QUALITY
        self.buffer_size = 0
        self.state = np.zeros((STATE_INFO, STATE_LEN))

    def step(self, action):
        bitrate = int(action)

        # The action is from the last decision
        # This is to make the framework similar to the real
        (
            delay,
            sleep_time,
            self.buffer_size,
            rebuf,
            video_chunk_size,
            next_video_chunk_sizes,
            end_of_video,
            video_chunk_remain,
        ) = self.server.get_video_chunk(bitrate)

        self.time_stamp += delay  # In ms
        self.time_stamp += sleep_time  # In ms

        # @formula: reward = video quality - rebuffer penalty - smooth penalty
        reward = (
            VIDEO_BIT_RATE[bitrate] / M_IN_K
            - REBUF_PENALTY * rebuf
            - SMOOTH_PENALTY
            * np.abs(VIDEO_BIT_RATE[bitrate] - VIDEO_BIT_RATE[self.last_bitrate])
            / M_IN_K
        )

        self.last_bitrate = bitrate
        state = np.roll(self.state, -1, axis=1)

        # This should be STATE_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bitrate] / float(
            np.max(VIDEO_BIT_RATE)
        )  # Last quality
        state[1, -1] = self.buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = (
            float(video_chunk_size) / float(delay) / M_IN_K
        )  # Kilo byte per ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :ACTION_DIM] = (
            np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
        )  # Mega byte
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(
            CHUNK_TIL_VIDEO_END_CAP
        )

        self.state = state
        return (
            state,
            reward,
            end_of_video,
            {"bitrate": VIDEO_BIT_RATE[bitrate], "rebuffer": rebuf},
        )

    def reset(self):
        self.time_stamp = 0
        self.last_bitrate = DEFAULT_QUALITY
        self.state = np.zeros((STATE_INFO, STATE_LEN))
        self.buffer_size = 0

        bitrate = self.last_bitrate
        (
            delay,
            sleep_time,
            self.buffer_size,
            rebuf,
            video_chunk_size,
            next_video_chunk_sizes,
            end_of_video,
            video_chunk_remain,
        ) = self.server.get_video_chunk(bitrate)
        state = np.roll(self.state, -1, axis=1)

        # This should be STATE_INFO number of terms
        state[0, -1] = VIDEO_BIT_RATE[bitrate] / float(
            np.max(VIDEO_BIT_RATE)
        )  # Last quality
        state[1, -1] = self.buffer_size / BUFFER_NORM_FACTOR  # 10 sec
        state[2, -1] = (
            float(video_chunk_size) / float(delay) / M_IN_K
        )  # Kilo byte per ms
        state[3, -1] = float(delay) / M_IN_K / BUFFER_NORM_FACTOR  # 10 sec
        state[4, :ACTION_DIM] = (
            np.array(next_video_chunk_sizes) / M_IN_K / M_IN_K
        )  # Mega byte
        state[5, -1] = np.minimum(video_chunk_remain, CHUNK_TIL_VIDEO_END_CAP) / float(
            CHUNK_TIL_VIDEO_END_CAP
        )
        self.state = state

        return state

    def render(self):
        return

    def log_diagnostics(self, paths):
        print(f"Trying to log to {paths}")

    @property
    def action_dim(self):
        return ACTION_DIM

    @property
    def observation_space(self):
        return super().observation_space

    @property
    def action_space(self):
        return super().action_space


if __name__ == "__main__":
    pass
