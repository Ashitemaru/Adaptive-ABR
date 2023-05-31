import os

METS_PATH = "./src/data/nyu_mets"
METS_TARGET_PATH = "./src/data/nyu_mets_traces"


def main():
    traces = []
    for status in os.listdir(METS_PATH):
        for name in os.listdir(os.path.join(METS_PATH, status)):
            if ".csv" not in name:
                continue

            trace = []
            with open(os.path.join(METS_PATH, status, name)) as handler:
                for time, line in enumerate(handler.readlines()[1:]):
                    line = line.replace("\n", "")
                    try:
                        bandwidth = float(line.split(",")[0]) * 4
                    except:
                        pass

                    trace.append((time, bandwidth))

            traces.append(trace)

    # Dump the trace
    if not os.path.exists(METS_TARGET_PATH):
        os.makedirs(METS_TARGET_PATH)
    for idx, trace in enumerate(traces):
        with open(
            f"{METS_TARGET_PATH}/nyu_mets_trace_{idx}.log",
            "w",
        ) as handler:
            for trace_item in trace:
                handler.write(str(trace_item[0]) + " " + str(trace_item[1]) + "\n")


if __name__ == "__main__":
    main()
