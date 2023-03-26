import numpy as np

from learning_to_adapt.core.utils.serializable import Serializable
from learning_to_adapt.core.envs.base import Env
from learning_to_adapt.core.spaces.box import Box
from learning_to_adapt.core.spaces.discrete import Discrete
from src.data.loader import DataLoader
from src.constants import (
    ACTION_DIM,
    BITRATE_LEVELS,
    VIDEO_SIZE_FILE,
)


class PenseieveEnvPark(Env, Serializable):
    def __init__(self, random_seed=42):
        np.random.seed(random_seed)

        self.__setup_space()
        self.__load_video_size()
        self.__load_traces()

    def __setup_space(self):
        self.observation_low = np.array([0] * 11)
        self.observation_high = np.array(
            [10e6, 100, 100, 500, 5, 10e6, 10e6, 10e6, 10e6, 10e6, 10e6]
        )
        self._observation_space = Box(
            low=self.observation_low, high=self.observation_high
        )
        self._action_space = Discrete(ACTION_DIM)

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

    def __load_traces(self):
        self.data_loader = DataLoader()

    def log_diagnostics(self, paths):
        print(f"Log to {paths}")

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def step(self, action):
        assert self.action_space.contains(action)

        chunk_size = self.chunk_size[action][self.chunk_idx]
        delay = 0  # In seconds

        while chunk_size > 1e-8:
            throughput = 0

    def reset(self):
        pass


if __name__ == "__main__":
    pass
