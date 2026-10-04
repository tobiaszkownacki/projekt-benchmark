"""What a finished run leaves behind for the download phase to pick up.

A compute node has no route to the database, so a run reports its numbers as a
file next to its artifacts and the downloader is what turns that file into a
row. Every executor writes the same manifest, which is why this lives here and
not in an adapter.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

MANIFEST_NAME = "result.json"


@dataclass(frozen=True)
class RunSeries:
    epochs: list[int] = field(default_factory=list)
    loss: list[float] = field(default_factory=list)
    accuracy: list[float] = field(default_factory=list)
    gradient_count: list[int] = field(default_factory=list)
    database_reaches: list[int] = field(default_factory=list)
    wall_time_seconds: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class RunResult:
    stop_reason: str
    gradient_count: int
    database_reaches: int
    final_loss: float | None = None
    final_accuracy: float | None = None
    total_steps: int | None = None
    total_epochs: int | None = None
    wall_time_seconds: float | None = None
    series: RunSeries = field(default_factory=RunSeries)

    @classmethod
    def from_benchmark_result(cls, result) -> "RunResult":
        """Read what the engine measured, without importing the engine.

        The pipeline image has no torch in it, so the mapping is by attribute
        rather than by type.
        """
        return cls(
            stop_reason=result.stop_reason.name,
            gradient_count=result.gradient_count,
            database_reaches=result.database_reaches,
            final_loss=result.final_loss,
            final_accuracy=result.final_accuracy,
            total_steps=result.total_steps,
            total_epochs=result.total_epochs,
            wall_time_seconds=result.wall_time_seconds,
            series=RunSeries(
                epochs=list(range(1, len(result.loss_history) + 1)),
                loss=list(result.loss_history),
                accuracy=list(result.accuracy_history),
                gradient_count=list(result.gradient_history),
                database_reaches=list(result.database_reaches_history),
                wall_time_seconds=list(result.time_history),
            ),
        )

    @classmethod
    def from_manifest(cls, path: Path) -> "RunResult":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{**payload, "series": RunSeries(**payload.get("series", {}))})

    def write_manifest(self, directory: Path) -> Path:
        path = Path(directory) / MANIFEST_NAME
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")
        return path


def find_manifest(files: list[str]) -> Path | None:
    for name in files:
        if Path(name).name == MANIFEST_NAME:
            return Path(name)
    return None
