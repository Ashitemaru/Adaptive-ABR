import os
import time

UCC_PATH = "./src/data/ucc_5g/"
SERVICE = [
    # {"name": "Amazon_Prime", "file": ["animated-AdventureTime", "Season3-TheExpanse"]},
    {"name": "Download", "file": [""]},
    {"name": "Netflix", "file": ["animated-RickandMorty", "Season3-StrangerThings"]},
]
STATE = ["Driving", "Static"]
UCC_DRIVING_TRACE_DIR = "./src/data/ucc_5g_driving_traces"
UCC_STATIC_TRACE_DIR = "./src/data/ucc_5g_static_traces"


def main():
    driving_traces = []
    static_traces = []
    for state in STATE:
        for service in SERVICE:
            for file in service["file"]:
                root_path = UCC_PATH + service["name"] + "/" + state + "/" + file
                for name in os.listdir(root_path):
                    with open(os.path.join(root_path, name), "r") as handler:
                        first_line = True
                        for line in handler:
                            raw_list = line.split(",")
                            if raw_list[0] == "Timestamp":  # Skip the header
                                continue
                            if raw_list[6] != "5G":
                                continue

                            trace_item = {
                                "time": int(
                                    time.mktime(
                                        time.strptime(raw_list[0], "%Y.%m.%d_%H.%M.%S")
                                    )
                                ),
                                "throughput": float(raw_list[12]) / 1e3,
                            }

                            if first_line:
                                (
                                    driving_traces
                                    if state == "Driving"
                                    else static_traces
                                ).append([])
                                first_line = False
                            if trace_item["throughput"] > 10:
                                (
                                    driving_traces
                                    if state == "Driving"
                                    else static_traces
                                )[-1].append(trace_item)

    # Dump the trace
    if not os.path.exists(UCC_DRIVING_TRACE_DIR):
        os.makedirs(UCC_DRIVING_TRACE_DIR)
    if not os.path.exists(UCC_STATIC_TRACE_DIR):
        os.makedirs(UCC_STATIC_TRACE_DIR)
    for idx, trace in enumerate(driving_traces):
        if len(trace) == 0:
            continue
        with open(
            f"{UCC_DRIVING_TRACE_DIR}/ucc_trace_{idx}.log",
            "w",
        ) as handler:
            for trace_item in trace:
                handler.write(
                    str(trace_item["time"]) + " " + str(trace_item["throughput"]) + "\n"
                )
    for idx, trace in enumerate(static_traces):
        if len(trace) == 0:
            continue
        with open(
            f"{UCC_STATIC_TRACE_DIR}/ucc_trace_{idx}.log",
            "w",
        ) as handler:
            for trace_item in trace:
                handler.write(
                    str(trace_item["time"]) + " " + str(trace_item["throughput"]) + "\n"
                )


if __name__ == "__main__":
    main()
