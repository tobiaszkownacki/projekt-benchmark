from dataclasses import dataclass


@dataclass(frozen=True)
class LeaderboardEntry:
    rank: int
    optimizer: str
    dataset: str
    score: float
    submitted_at: str


@dataclass(frozen=True)
class RunResult:
    run_id: str
    run_name: str
    status: str
    created_at: str


OPTIMIZERS: list[str] = [
    "Adam",
    "AdamW",
    "SGD",
    "RMSprop",
    "Adagrad",
    "Lion",
]

DATASETS: list[str] = [
    "CIFAR-10",
    "CIFAR-100",
    "MNIST",
    "ImageNet",
    "Fashion-MNIST",
]


LEADERBOARD: list[LeaderboardEntry] = [
    LeaderboardEntry(1, "Lion", "ImageNet", 94.2, "2026-07-10"),
    LeaderboardEntry(2, "AdamW", "CIFAR-100", 92.8, "2026-07-09"),
    LeaderboardEntry(3, "Adam", "CIFAR-10", 90.5, "2026-07-08"),
    LeaderboardEntry(4, "RMSprop", "MNIST", 88.1, "2026-07-06"),
    LeaderboardEntry(5, "SGD", "Fashion-MNIST", 86.7, "2026-07-05"),
    LeaderboardEntry(6, "Adagrad", "CIFAR-10", 84.3, "2026-07-03"),
    LeaderboardEntry(7, "AdamW", "MNIST", 82.0, "2026-07-01"),
    LeaderboardEntry(8, "Adam", "Fashion-MNIST", 79.4, "2026-06-28"),
    LeaderboardEntry(9, "SGD", "CIFAR-100", 77.9, "2026-06-25"),
    LeaderboardEntry(10, "RMSprop", "ImageNet", 71.2, "2026-06-20"),
]


RUN_HISTORY: list[RunResult] = [
    RunResult(
        run_id="run-2041",
        run_name="lion-imagenet-sweep",
        status="completed",
        created_at="2026-07-12 14:31",
    ),
    RunResult(
        run_id="run-2038",
        run_name="adamw-cifar100-baseline",
        status="completed",
        created_at="2026-07-11 09:12",
    ),
    RunResult(
        run_id="run-2033",
        run_name="sgd-mnist-lr-search",
        status="failed",
        created_at="2026-07-09 18:47",
    ),
    RunResult(
        run_id="run-2027",
        run_name="rmsprop-fashion-tune",
        status="completed",
        created_at="2026-07-06 11:05",
    ),
]
