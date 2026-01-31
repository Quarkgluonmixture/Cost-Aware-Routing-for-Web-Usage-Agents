import time
from contextlib import contextmanager

class Timer:
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.duration_ms = 0.0

    def start(self):
        self.start_time = time.perf_counter()
        return self

    def stop(self):
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        return self.duration_ms

    @contextmanager
    def time_block(self):
        self.start()
        try:
            yield
        finally:
            self.stop()
