from pathlib import Path


class Averager:
    def __init__(self) -> None:
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value: float) -> None:
        self.current_total += value
        self.iterations += 1

    @property
    def value(self) -> float:
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self) -> None:
        self.current_total = 0.0
        self.iterations = 0.0


def collate_fn(batch) -> tuple:
    return tuple(zip(*batch))


def get_root_path() -> Path:
    """Get the application path"""
    return Path(__file__).parent.parent.parent
