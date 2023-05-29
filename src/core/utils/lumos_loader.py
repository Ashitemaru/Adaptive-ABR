import os

LUMOS_PATH = "./src/data/lumos/Lumos5G-v1.0.csv"
LUMOS_TRAIN_TRACE_DIR = "./src/data/lumos_traces"
LUMOS_TEST_TRACE_DIR = "./src/data/lumos_test_traces"


def main():
    """
    LUMOS 5G Dataset, CSV format, header:
    - run_num (0)
    - seq_num (1)
    - abstractSignalStr
    - latitude
    - longitude
    - movingSpeed
    - compassDirection
    - nrStatus (7)
    - lte_rssi
    - lte_rsrp
    - lte_rsrq
    - lte_rssnr
    - nr_ssRsrp
    - nr_ssRsrq
    - nr_ssSinr
    - Throughput (15)
    - mobility_mode (16)
    - trajectory_direction
    - tower_id
    """

    trace_dict = {}
    with open(LUMOS_PATH, "r") as handler:
        for line in handler:
            raw_list = line.split(",")
            run_num = raw_list[0]
            seq_num = raw_list[1]
            nr_status = raw_list[7]
            throughput = raw_list[15]
            mobility_mode = raw_list[16]

            if run_num == "run_num":  # Skip the header
                continue
            run_num = int(run_num)

            trace_item = {
                "time": float(seq_num),
                "status": "5G" if nr_status == "CONNECTED" else "4G",
                "throughput": float(throughput) / 8,
                "mobility_mode": mobility_mode,
            }
            if run_num in trace_dict:
                trace_dict[run_num].append(trace_item)
            else:
                trace_dict[run_num] = [trace_item]

    # Dump the trace
    if not os.path.exists(LUMOS_TRAIN_TRACE_DIR):
        os.makedirs(LUMOS_TRAIN_TRACE_DIR)
    if not os.path.exists(LUMOS_TEST_TRACE_DIR):
        os.makedirs(LUMOS_TEST_TRACE_DIR)
    for run_num, trace in trace_dict.items():
        assert (
            len(set([x["mobility_mode"] for x in trace])) == 1
        )  # Mobility mode should not change within a trace
        assert (
            len(set([x["status"] for x in trace])) > 1
        )  # All traces have at least one handoff

        with open(
            f"{LUMOS_TEST_TRACE_DIR if run_num < 25 else LUMOS_TRAIN_TRACE_DIR}/lumos_trace_{run_num}.log",
            "w",
        ) as handler:
            for trace_item in trace:
                handler.write(
                    str(trace_item["time"]) + " " + str(trace_item["throughput"]) + "\n"
                )


if __name__ == "__main__":
    main()
