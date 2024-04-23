import time
import statistics

class Profiler:

    def __init__(self):
        self.samples = []
        self.last_start = None

    def start(self):
        self.last_start = time.perf_counter_ns()

    def end(self):
        if self.last_start is None:
            raise Exception("Profiler not started")
        self.samples.append(time.perf_counter_ns() - self.last_start)
        self.last_start = None

    def summary(self):
        # Print the statistics of the samples
        print(f"Model Median: {statistics.median(self.samples)/1000:0f}")
        print(f"Max: {max(self.samples)/1000:0f}")
        print(f"Min: {min(self.samples)/1000:0f}")



