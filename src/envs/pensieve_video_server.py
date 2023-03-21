import os
import numpy as np

from src.constants import (
    BITRATE_LEVELS,
    VIDEO_SIZE_FILE,
    B_IN_MB,
    BITS_IN_BYTE,
    PACKET_PAYLOAD_PORTION,
    MILLISECONDS_IN_SECOND,
    TOTAL_VIDEO_CHUNCK,
    LINK_RTT,
    BUFFER_THRESH,
    DRAIN_BUFFER_SLEEP_TIME,
    VIDEO_CHUNCK_LEN,
    NOISE_LOW,
    NOISE_HIGH,
)


def load_trace(cooked_trace_folder):
    cooked_files = os.listdir(cooked_trace_folder)
    all_cooked_time = []
    all_cooked_bw = []
    all_file_names = []
    for cooked_file in cooked_files:
        file_path = cooked_trace_folder + cooked_file
        cooked_time = []
        cooked_bw = []

        with open(file_path, "rb") as f:
            for line in f:
                parse = line.split()
                cooked_time.append(float(parse[0]))
                cooked_bw.append(float(parse[1]))
        all_cooked_time.append(cooked_time)
        all_cooked_bw.append(cooked_bw)
        all_file_names.append(cooked_file)

    return all_cooked_time, all_cooked_bw, all_file_names


class PensieveVideoServer:
    def __init__(self, data_path, random_seed=42):
        np.random.seed(random_seed)

        all_cooked_time, all_cooked_bw, _ = load_trace(data_path)
        assert len(all_cooked_time) == len(all_cooked_bw)

        self.all_cooked_time = all_cooked_time
        self.all_cooked_bw = all_cooked_bw

        self.video_chunk_counter = 0
        self.buffer_size = 0

        # Pick a random trace file
        self.trace_idx = np.random.randint(len(self.all_cooked_time))
        self.cooked_time = self.all_cooked_time[self.trace_idx]
        self.cooked_bw = self.all_cooked_bw[self.trace_idx]

        # Randomize the start point of the trace
        # @note: trace file starts with time 0
        self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
        self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        self.video_size = {}  # In bytes
        for bitrate in range(BITRATE_LEVELS):
            self.video_size[bitrate] = []
            with open(VIDEO_SIZE_FILE + str(bitrate)) as f:
                for line in f:
                    self.video_size[bitrate].append(int(line.split()[0]))

    def get_video_chunk(self, quality):
        assert quality >= 0
        assert quality < BITRATE_LEVELS

        video_chunk_size = self.video_size[quality][self.video_chunk_counter]

        # Use the delivery opportunity in mahimahi
        delay = 0.0  # In ms
        video_chunk_counter_sent = 0  # In bytes

        while True:  # Download video chunk over mahimahi
            throughput = self.cooked_bw[self.mahimahi_ptr] * B_IN_MB / BITS_IN_BYTE
            duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time

            packet_payload = throughput * duration * PACKET_PAYLOAD_PORTION

            if video_chunk_counter_sent + packet_payload > video_chunk_size:
                fractional_time = (
                    (video_chunk_size - video_chunk_counter_sent)
                    / throughput
                    / PACKET_PAYLOAD_PORTION
                )
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                assert self.last_mahimahi_time <= self.cooked_time[self.mahimahi_ptr]
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.cooked_bw):
                # Loop back in the beginning
                # @note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0

        delay *= MILLISECONDS_IN_SECOND
        delay += LINK_RTT

        # Add a multiplicative noise to the delay
        delay *= np.random.uniform(NOISE_LOW, NOISE_HIGH)

        # Rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)

        # Update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # Add in the new chunk
        self.buffer_size += VIDEO_CHUNCK_LEN

        # Sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > BUFFER_THRESH:
            # Exceed the buffer limit
            # We need to skip some network bandwidth here
            # But do not add up the delay
            drain_buffer_time = self.buffer_size - BUFFER_THRESH
            sleep_time = (
                np.ceil(drain_buffer_time / DRAIN_BUFFER_SLEEP_TIME)
                * DRAIN_BUFFER_SLEEP_TIME
            )
            self.buffer_size -= sleep_time

            while True:
                duration = self.cooked_time[self.mahimahi_ptr] - self.last_mahimahi_time
                if duration > sleep_time / MILLISECONDS_IN_SECOND:
                    self.last_mahimahi_time += sleep_time / MILLISECONDS_IN_SECOND
                    break
                sleep_time -= duration * MILLISECONDS_IN_SECOND
                self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.cooked_bw):
                    # Loop back in the beginning
                    # @note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0

        # The "last buffer size" return to the controller
        # @note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least one chunk of video
        return_buffer_size = self.buffer_size

        self.video_chunk_counter += 1
        video_chunk_remain = TOTAL_VIDEO_CHUNCK - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= TOTAL_VIDEO_CHUNCK:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0

            # Pick a random trace file
            self.trace_idx = np.random.randint(len(self.all_cooked_time))
            self.cooked_time = self.all_cooked_time[self.trace_idx]
            self.cooked_bw = self.all_cooked_bw[self.trace_idx]

            # Randomize the start point of the video
            # @note: trace file starts with time 0
            self.mahimahi_ptr = np.random.randint(1, len(self.cooked_bw))
            self.last_mahimahi_time = self.cooked_time[self.mahimahi_ptr - 1]

        next_video_chunk_sizes = []
        for i in range(BITRATE_LEVELS):
            next_video_chunk_sizes.append(self.video_size[i][self.video_chunk_counter])

        return (
            delay,
            sleep_time,
            return_buffer_size / MILLISECONDS_IN_SECOND,
            rebuf / MILLISECONDS_IN_SECOND,
            video_chunk_size,
            next_video_chunk_sizes,
            end_of_video,
            video_chunk_remain,
        )


if __name__ == "__main__":
    pass
