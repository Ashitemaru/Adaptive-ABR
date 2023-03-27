import numpy as np
from collections import deque

from core.utils.serializable import Serializable
from core.envs.base import Env
from core.spaces.box import Box
from core.logger import logger
from core.utils.trace_loader import DataLoader
from core.utils.constants import (
    ACTION_DIM,
    BITRATE_LEVELS,
    VIDEO_SIZE_FILE,
    STATE_LEN,
    VIDEO_BIT_RATE,
    REBUF_PENALTY,
    SMOOTH_PENALTY,
    VIDEO_CHUNCK_LEN,
    BUFFER_THRESH,
)


class PenseieveEnvPark(Env, Serializable):
    def __init__(self, random_seed=42):
        Serializable.quick_init(self, locals())

        self.random_seed = random_seed
        np.random.seed(random_seed)

        self.__setup_space()
        self.__load_video_size()
        self.__load_traces()

    def __setup_space(self):
        self.observation_low = np.array([0] * 11)
        self.observation_high = np.array(
            [
                10e8,  # Past chunk throughput, in bytes per second
                100,  # Past chunk download time, in second
                100,  # Buffer size, in second
                500,  # Video chunk left
                5,  # Past action
                10e7,  # Chunk sizes for 6 bitrate, in bytes
                10e7,
                10e7,
                10e7,
                10e7,
                10e7,
            ]
        )
        self._observation_space = Box(
            low=self.observation_low, high=self.observation_high
        )
        self._action_space = Box(low=np.array([0]), high=np.array([ACTION_DIM]))

    def __convert_action(self, action):
        action = int(action[0])
        if action == ACTION_DIM:
            return ACTION_DIM - 1
        else:
            return action

    def __load_video_size(self):
        self.chunk_size = []
        for bitrate in range(BITRATE_LEVELS):
            chunk_size_bitrate = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    chunk_size_bitrate.append(int(line.split()[0]))
            self.chunk_size.append(chunk_size_bitrate)

        self.chunk_size = np.array(self.chunk_size)

        assert len(np.unique([len(chunk_size) for chunk_size in self.chunk_size])) == 1
        self.total_num_chunks = len(self.chunk_size[0])

        # assert self.total_num_chunks == TOTAL_VIDEO_CHUNCK

    def __load_traces(self):
        loader = DataLoader()
        self.all_trace = loader.get_trace_set_by_env(
            "4g-walking"
        ) + loader.get_trace_set_by_env("4g-driving")

    def __sample_trace(self):
        trace_len = [len(trace) for trace in self.all_trace]
        total_len = float(sum(trace_len))
        trace_probability = [l / total_len for l in trace_len]

        # Sample a trace
        trace_idx = np.random.choice(len(trace_len), p=trace_probability)

        # Sample a starting point
        init_t_idx = np.random.choice(len(self.all_trace[trace_idx]))

        return self.all_trace[trace_idx], init_t_idx

    def __get_chunk_time(self):
        if self.curr_t_idx == len(self.trace) - 1:  # At the end of the trace
            return 1
        else:
            return self.trace[self.curr_t_idx + 1][0] - self.trace[self.curr_t_idx][0]

    def log_diagnostics(self, paths, prefix):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3] for path in paths
        ]
        logger.logkv(prefix + "AverageForwardProgress", np.mean(progs))
        logger.logkv(prefix + "MaxForwardProgress", np.max(progs))
        logger.logkv(prefix + "MinForwardProgress", np.min(progs))
        logger.logkv(prefix + "StdForwardProgress", np.std(progs))

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def observe(self):
        if self.chunk_idx < self.total_num_chunks:
            valid_chunk_idx = self.chunk_idx
        else:
            valid_chunk_idx = 0

        if self.past_action is not None:
            valid_past_action = self.past_action
        else:
            valid_past_action = 0

        obs_arr = [
            self.past_chunk_throughputs[-1],
            self.past_chunk_download_times[-1],
            self.buffer_size,
            self.total_num_chunks - self.chunk_idx,
            valid_past_action,
        ]

        obs_arr.extend(self.chunk_size[i][valid_chunk_idx] for i in range(6))

        for i in range(len(obs_arr)):
            if obs_arr[i] > self.observation_high[i]:
                print(
                    f"[WARN] Observation at index {i} at chunk index {self.chunk_idx} has value {obs_arr[i]}, higher than obs_high {self.observation_high[i]}"
                )
                obs_arr[i] = self.observation_high[i]

        obs_arr = np.array(obs_arr)
        assert self.observation_space.contains(obs_arr)

        return obs_arr

    def step(self, action):
        assert self.action_space.contains(action)
        action = self.__convert_action(action)

        chunk_size = self.chunk_size[action][self.chunk_idx]
        delay = 0  # In seconds

        # Keep downloading the chunk through the network trace
        while chunk_size > 1e-8:
            throughput = (
                self.trace[self.curr_t_idx][1] / 8.0 * 1e6
            )  # In bytes per second
            chunk_time_used = min(
                self.chunk_time_left,
                chunk_size / throughput if abs(throughput) > 1e-6 else 1e10,
            )
            chunk_size -= throughput * chunk_time_used
            self.chunk_time_left -= chunk_time_used
            delay += chunk_time_used

            if self.chunk_time_left == 0:
                self.curr_t_idx += 1
                if self.curr_t_idx == len(self.trace):
                    self.curr_t_idx = 0

                self.chunk_time_left = self.__get_chunk_time()

        rebuffer_time = max(delay - self.buffer_size, 0)
        self.buffer_size = max(self.buffer_size - delay, 0)
        self.buffer_size += (
            VIDEO_CHUNCK_LEN / 1000
        )  # Each chunk means 4 seconds of video
        self.buffer_size = min(
            self.buffer_size, BUFFER_THRESH / 1000
        )  # Clip the buffer size

        if self.past_action is None:
            bitrate_change = 0
        else:
            bitrate_change = np.abs(
                VIDEO_BIT_RATE[action] - VIDEO_BIT_RATE[self.past_action]
            )

        reward = (
            VIDEO_BIT_RATE[action] / 1000
            - REBUF_PENALTY * rebuffer_time
            - SMOOTH_PENALTY * bitrate_change / 1000
        )

        self.past_action = action
        self.past_chunk_throughputs.append(
            self.chunk_size[action][self.chunk_idx] / float(delay)
        )
        self.past_chunk_download_times.append(delay)

        self.chunk_idx += 1
        done = self.chunk_idx == self.total_num_chunks

        return (
            self.observe(),
            reward,
            done,
            {
                "bitrate": VIDEO_BIT_RATE[action],
                "stall_time": rebuffer_time,
                "bitrate_change": bitrate_change,
            },
        )

    def reset(self):
        self.trace, self.curr_t_idx = self.__sample_trace()
        self.chunk_time_left = self.__get_chunk_time()
        self.chunk_idx = 0
        self.buffer_size = 0
        self.past_action = None
        self.past_chunk_throughputs = deque(maxlen=STATE_LEN)
        self.past_chunk_download_times = deque(maxlen=STATE_LEN)
        for _ in range(STATE_LEN):
            self.past_chunk_throughputs.append(0)
            self.past_chunk_download_times.append(0)

        return self.observe()

    def reward(self, observation, action, next_observation):
        def calc(_observation, _action, _next_observation):
            _action = self.__convert_action(_action)
            _past_action = int(_observation[4])
            if _past_action == ACTION_DIM:
                _past_action = ACTION_DIM - 1

            if _past_action is None:
                bitrate_change = 0
            else:
                bitrate_change = np.abs(
                    VIDEO_BIT_RATE[_action] - VIDEO_BIT_RATE[_past_action]
                )

            delay = _next_observation[1]
            buffer_size = _observation[2]
            rebuffer_time = max(delay - buffer_size, 0)
            return (
                VIDEO_BIT_RATE[_action] / 1000
                - REBUF_PENALTY * rebuffer_time
                - SMOOTH_PENALTY * bitrate_change / 1000
            )

        return np.array([calc(*t) for t in zip(observation, action, next_observation)])


if __name__ == "__main__":
    pass
