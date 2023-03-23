import os

TRAIN_DATA_DIR = "./src/data/env_train_traces"
TEST_DATA_DIR = "./src/data/env_test_traces"


class DataLoader:
    def __init__(self, mode="train"):
        self.mode = mode

        data_root = TRAIN_DATA_DIR if mode == "train" else TEST_DATA_DIR
        trace_file_list = os.listdir(data_root)

        self.driving_4g_train_time = []
        self.walking_4g_train_time = []
        self.driving_5g_train_time = []
        self.walking_5g_train_time = []
        self.driving_5g_test_time = []

        self.driving_4g_train_bw = []
        self.walking_4g_train_bw = []
        self.driving_5g_train_bw = []
        self.walking_5g_train_bw = []
        self.driving_5g_test_bw = []

        for file_name in trace_file_list:
            file_path = os.path.join(data_root, file_name)

            time = []
            bandwidth = []
            with open(file_path, "rb") as f:
                for line in f:
                    parse = line.split()
                    time.append(float(parse[0]))
                    bandwidth.append(float(parse[1]))

            if mode == "test":
                assert file_name.startswith("5g_trace") and file_name.endswith(
                    "driving"
                )
                self.driving_5g_test_time.append(time)
                self.driving_5g_test_bw.append(bandwidth)
            else:
                if file_name.startswith("4g_trace_driving"):
                    self.driving_4g_train_time.append(time)
                    self.driving_4g_train_bw.append(bandwidth)
                elif file_name.startswith("4g_trace_walking"):
                    self.walking_4g_train_time.append(time)
                    self.walking_4g_train_bw.append(bandwidth)
                elif file_name.startswith("5g_trace") and file_name.endswith("driving"):
                    self.driving_5g_train_time.append(time)
                    self.driving_5g_train_bw.append(bandwidth)
                elif file_name.startswith("5g_trace") and file_name.endswith("walking"):
                    self.walking_5g_train_time.append(time)
                    self.walking_5g_train_bw.append(bandwidth)
                else:
                    assert False

        print(min(list(map(len, self.driving_5g_train_bw))))
        print(min(list(map(len, self.driving_4g_train_bw))))
        print(min(list(map(len, self.walking_5g_train_bw))))
        print(min(list(map(len, self.walking_4g_train_bw))))

    def sample_trace(
        self, net_env="4g", status="driving", switch_mode="interval", mode="train"
    ):
        """Sample a network trace from the dataset.

        Args:
            net_env (str, optional): "4g" or "5g" or "mixed". Defaults to "4g".
            status (str, optional): "driving" or "walking" or "mixed". Defaults to "driving".
            switch_mode (str, optional): "interval" or "random". Defaults to "interval".
            mode (str, optional): "train" or "test". Defaults to "train".
        """
        pass


if __name__ == "__main__":
    loader = DataLoader()
