import os
import numpy as np
import random
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

TRAIN_DATA_DIR = "./src/data/env_train_traces"
TEST_DATA_DIR = "./src/data/env_test_traces"


class TraceHelper:
    @classmethod
    def down_sample(cls, trace, interval=2):
        return [t for i, t in enumerate(trace) if i % interval == 0]

    @classmethod
    def slim_trace(cls, trace):
        return [t for t in trace if t[1] is not None]

    @classmethod
    def fill_trace(cls, trace):
        filled_trace = []

        for i in range(0, int(trace[0][0])):
            filled_trace.append((i, None))
        filled_trace.append(trace[0])

        for t in trace[1:]:
            for i in range(1 + int(filled_trace[-1][0]), int(t[0])):
                filled_trace.append((i, None))
            filled_trace.append(t)

        return filled_trace

    @classmethod
    def merge_trace(cls, trace_a, trace_b, split_time_list):
        """
        This function will create a switching trace from `trace_a, trace_b`, the switching time points are passed through param `split_time_list`.

        It should be noticed that the final trace will not contain `None`-s, but when there are too many `None`-s in the original trace after filling which makes a slice of the final trace is all `None`-s, this split time point will be ignored.

        e.g. `trace_a` is `[(3, 3), (4, 4)]`, `trace_b` is `[(0, 1), (1, 3), (2, 4), (3, 4)]`. After filling, `trace_a` is `[(0, None), (1, None), (2, None), (3, 3), (4, 4)]` and `trace_b` keeps unchanged.

        If we choose the split time point at 2, this will make the final trace be `[(0, None), (1, None), (2, 4), (3, 4)]`, after removing all the `None`-s, the final trace will be `[(2, 4), (3, 4)]`, the split time point 2 takes no effect here and will be ignored.
        """

        trace_a = TraceHelper.fill_trace(trace_a)
        trace_b = TraceHelper.fill_trace(trace_b)
        end_time = min(trace_a[-1][0], trace_b[-1][0])

        prev_index = 0
        final_trace = []
        final_split_time_list = []
        for i, index in enumerate(split_time_list + [end_time]):
            assert (i < len(split_time_list) and 0 < index < end_time) or (
                i == len(split_time_list) and index == end_time
            )
            select_from_a = i % 2 == 0

            trace_slice = TraceHelper.slim_trace(
                (trace_a if select_from_a else trace_b)[int(prev_index) : int(index)]
            )
            if len(trace_slice) > 0:  # Not all None
                final_trace += trace_slice
                if index != end_time:
                    final_split_time_list.append(index)

            prev_index = index

        return final_trace, final_split_time_list

    @classmethod
    def load_trace(cls, path, enable_none=False):
        trace = []
        prev_time = -1
        overlap_cnt = 0
        with open(path, "rb") as f:
            for line in f:
                parse = line.split()
                time = float(parse[0])

                if parse[1] == "None" and enable_none:
                    trace.append((time, None))
                elif parse[1] == "None" and not enable_none:
                    continue

                # If overlap happens (overlap means 2 timestamps are same),
                # we directly overwrite the previous bandwidth with current one.
                elif abs(prev_time - time) < 1e-6:
                    trace[-1] = (trace[-1][0], float(parse[1]))
                    overlap_cnt += 1
                elif time > prev_time:
                    trace.append((time, float(parse[1])))

                prev_time = time
                assert time % 1 == 0

        return trace, overlap_cnt

    @classmethod
    def dump_trace(cls, trace, path, enable_none=False):
        with open(path, "w") as f:
            for t in trace:
                if t[1] is None and not enable_none:
                    continue

                f.write(f"{t[0]} {t[1]}\n")

    @classmethod
    def plot_trace(cls, trace, split_time_list, trace_name, image_path):
        plt.figure()
        plt.xlabel("Time (s)")
        plt.ylabel("Bandwidth (Mbps)")
        plt.grid(True)
        plt.title(f"Network trace ({trace_name})")
        plt.plot([t[0] for t in trace], [t[1] for t in trace])
        for split_time in split_time_list:
            plt.axvline(split_time, linestyle="--", color="gray")
        plt.savefig(image_path)


class DataLoader:
    def __init__(self):
        self.driving_4g_train_trace = []
        self.walking_4g_train_trace = []
        self.driving_5g_train_trace = []
        self.walking_5g_train_trace = []
        self.driving_5g_test_trace = []

        self.__load_trace(mode="train")
        self.__load_trace(mode="test")

    def __load_trace(self, mode="train"):
        data_root = TRAIN_DATA_DIR if mode == "train" else TEST_DATA_DIR
        trace_file_list = os.listdir(data_root)

        tot_overlap_cnt = 0
        for file_name in trace_file_list:
            file_path = os.path.join(data_root, file_name)
            trace, overlap_cnt = TraceHelper.load_trace(file_path, enable_none=False)
            tot_overlap_cnt += overlap_cnt

            if mode == "test":
                assert file_name.startswith("5g_trace") and file_name.endswith(
                    "driving"
                )
                self.driving_5g_test_trace.append(trace)
            else:
                if file_name.startswith("4g_trace_driving"):
                    self.driving_4g_train_trace.append(trace)
                elif file_name.startswith("4g_trace_walking"):
                    self.walking_4g_train_trace.append(trace)
                elif file_name.startswith("5g_trace") and file_name.endswith("driving"):
                    self.driving_5g_train_trace.append(trace)
                elif file_name.startswith("5g_trace") and file_name.endswith("walking"):
                    self.walking_5g_train_trace.append(trace)
                else:
                    raise ValueError("Corrupted dataset")

        print(f"Detected {tot_overlap_cnt} overlap(s) totally for mode {mode}.")

    def __sample_trace_by_env(self, env):
        trace_set = []
        if env == "4g-walking":
            trace_set = self.walking_4g_train_trace
        elif env == "4g-driving":
            trace_set = self.driving_4g_train_trace
        elif env == "5g-walking":
            trace_set = self.walking_5g_train_trace
        elif env == "5g-driving-train":
            trace_set = self.driving_5g_train_trace
        elif env == "5g-driving-test":
            trace_set = self.driving_5g_test_trace
        else:
            raise ValueError("Invalid environment")

        index = np.random.randint(len(trace_set))
        return trace_set[index]

    def sample_switch_trace(
        self,
        env_a="4g-walking",
        env_b="4g-driving",
        switch_mode="single",
        **kwarg,
    ):
        """
        `switch_mode` can be `"single", "interval", "random"`.

        - `"single"` means attaching a trace of `env_b` directly after a trace of `env_a`, and traces will be downsampled to get a reasonable trace length.
        - `"interval"` means switching intervally between 2 traces, switching time should be passed through param `n_switch`, default to 2.
        - `"random"` means randomly switching between 2 traces.
        """

        trace_a = TraceHelper.down_sample(self.__sample_trace_by_env(env_a))
        trace_b = TraceHelper.down_sample(self.__sample_trace_by_env(env_b))
        if switch_mode == "single":
            return (
                trace_a + [(time + trace_a[-1][0] + 1, bw) for time, bw in trace_b],
                [trace_a[-1][0]],
            )

        elif switch_mode == "interval":
            n_switch = kwarg.get("n_switch", 2)
            assert isinstance(n_switch, int) and n_switch > 0
            end_time = min(trace_a[-1][0], trace_b[-1][0])
            split_time_list = list(np.linspace(0, end_time, n_switch + 2)[1:-1])
            return TraceHelper.merge_trace(trace_a, trace_b, split_time_list)

        elif switch_mode == "random":
            n_switch = np.random.randint(1, 10)
            end_time = min(trace_a[-1][0], trace_b[-1][0])
            split_time_list = list(random.sample(range(1, int(end_time)), k=n_switch))
            split_time_list.sort()
            return TraceHelper.merge_trace(trace_a, trace_b, split_time_list)

        else:
            raise ValueError("Invalid switching mode")

    @property
    def test_set(self):
        return self.driving_5g_test_trace

    @property
    def train_set(self):
        return (
            self.driving_4g_train_trace
            + self.driving_5g_train_trace
            + self.walking_4g_train_trace
            + self.walking_5g_train_trace
        )


if __name__ == "__main__":
    loader = DataLoader()
    trace, split_time_list = loader.sample_switch_trace(switch_mode="random")
    TraceHelper.plot_trace(trace, split_time_list, "TEST", "./image/test.png")
