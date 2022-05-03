class TimerError(Exception):
    """Custom exception for timer error"""


class Timer:
    def __init__(self):
        self._start_time = None

    def start(self):
        import time

        if self._start_time is not None:
            raise TimerError(f'Timer is !\nUse .stop() to stop it.')

        self._start_time = time.perf_counter()

    def stop(self):
        import time

        if self._start_time is None:
            raise TimerError(f'Timer is not running!\n Use .start() to start it.')

        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f'Elapsed time: {elapsed_time:.4f} seconds')
