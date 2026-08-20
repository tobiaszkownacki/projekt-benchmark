"""Convergence series: downsampling, alignment and quantile aggregation.

Two things here are easy to get wrong in ways that produce a chart which looks
convincing and is not true, so both are handled explicitly.

*Downsampling.* Taking every n-th point is cheap and loses spikes. In a loss
curve a spike is a measurement, not noise, so LTTB is used instead: it keeps the
points that carry the visual shape.

*Aggregating across seeds.* Runs do not measure at the same budgets -- CMA-ES
with a population of 20 spends twenty times the samples per epoch that SGD does
-- so the median at a given budget requires putting every run on a shared grid
first. That interpolation is step-wise, never linear: the loss at budget b is
the last value actually observed at or below b. Linear interpolation would
invent measurements that were never taken.
"""

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

X_AXES = {
    "gradient_count": "gradient_count",
    "database_reaches": "database_reaches",
    "epoch": "epochs",
}
METRICS = {"loss": "loss", "accuracy": "accuracy"}

MAX_POINTS = 5000
DEFAULT_POINTS = 1000


def lttb(points: Sequence[tuple[float, float]], threshold: int) -> list[tuple[float, float]]:
    """Largest-Triangle-Three-Buckets downsampling.

    Keeps the first and last point, then picks from each bucket the point
    forming the largest triangle with the previous pick and the next bucket's
    average -- which is a cheap proxy for "carries the shape".
    """
    n = len(points)
    if threshold >= n or threshold < 3:
        return list(points)

    sampled = [points[0]]
    every = (n - 2) / (threshold - 2)
    a = 0

    for i in range(threshold - 2):
        start = int((i + 1) * every) + 1
        end = min(int((i + 2) * every) + 1, n - 1)
        if start >= end:
            start, end = end - 1, end
        bucket = points[start:end] or [points[min(start, n - 1)]]
        avg_x = sum(p[0] for p in bucket) / len(bucket)
        avg_y = sum(p[1] for p in bucket) / len(bucket)

        range_start = int(i * every) + 1
        range_end = min(int((i + 1) * every) + 1, n - 1)
        candidates = points[range_start:range_end] or [points[min(range_start, n - 1)]]

        ax, ay = points[a]
        best = max(
            candidates,
            key=lambda p: abs((ax - avg_x) * (p[1] - ay) - (ax - p[0]) * (avg_y - ay)),
        )
        sampled.append(best)
        a = points.index(best, range_start if range_start < n else 0)

    sampled.append(points[-1])
    return sampled


def series_points(
    row: dict, x_axis: str, metric: str
) -> list[tuple[float, float]]:
    """Zip a result_series row into (x, y) pairs, dropping missing values."""
    x_column = X_AXES.get(x_axis, "gradient_count")
    y_column = METRICS.get(metric, "loss")
    xs = row.get(x_column) or []
    ys = row.get(y_column) or []
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if x is not None and y is not None
    ]
    pairs.sort(key=lambda p: p[0])
    return pairs


def step_value_at(points: Sequence[tuple[float, float]], budget: float) -> Optional[float]:
    """Last observed value at or below `budget`.

    None outside the range the run actually covers -- on both sides. The upper
    guard is the one that matters: without it a run that stopped early keeps
    contributing its final value to every later grid point, so it silently props
    up the median and narrows the band exactly where it should be widening. That
    is the failure §9.2 describes, and it flatters whichever method gave up
    soonest.
    """
    if not points or budget < points[0][0] or budget > points[-1][0]:
        return None
    lo, hi = 0, len(points) - 1
    best = None
    while lo <= hi:
        mid = (lo + hi) // 2
        if points[mid][0] <= budget:
            best = points[mid][1]
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def quantile(values: Sequence[float], q: float) -> float:
    """Linear-interpolation quantile, matching numpy's default method."""
    if not values:
        raise ValueError("empty")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    weight = position - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def build_grid(
    runs: Iterable[Sequence[tuple[float, float]]], points: int, logarithmic: bool
) -> list[float]:
    """A shared X grid spanning the budgets every run actually covers."""
    starts, ends = [], []
    for run in runs:
        if run:
            starts.append(run[0][0])
            ends.append(run[-1][0])
    if not starts:
        return []

    lo, hi = min(starts), max(ends)
    if hi <= lo:
        return [lo]

    if logarithmic and lo > 0:
        import math

        log_lo, log_hi = math.log10(lo), math.log10(hi)
        step = (log_hi - log_lo) / (points - 1)
        return [10 ** (log_lo + i * step) for i in range(points)]

    step = (hi - lo) / (points - 1)
    return [lo + i * step for i in range(points)]


@dataclass
class AggregatedSeries:
    label: str
    family: Optional[str]
    n_runs: int
    x: list[float] = field(default_factory=list)
    median: list[Optional[float]] = field(default_factory=list)
    q1: list[Optional[float]] = field(default_factory=list)
    q3: list[Optional[float]] = field(default_factory=list)
    n_at_x: list[int] = field(default_factory=list)
    full_until_index: int = 0

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "family": self.family,
            "n_runs": self.n_runs,
            "x": self.x,
            "median": self.median,
            "q1": self.q1,
            "q3": self.q3,
            "n_at_x": self.n_at_x,
            "full_until_index": self.full_until_index,
        }


def aggregate(
    label: str,
    runs: Sequence[Sequence[tuple[float, float]]],
    family: Optional[str] = None,
    points: int = 200,
    logarithmic: bool = False,
) -> AggregatedSeries:
    """Median and interquartile band for one group of runs.

    ``full_until_index`` marks the last grid point where every run in the group
    still had data. Past it the band is computed from a shrinking number of runs,
    which narrows it for a reason that has nothing to do with agreement between
    them -- so the client draws that part dashed and shows ``n``. Without this the
    chart quietly flatters whichever method stopped earliest.
    """
    usable = [r for r in runs if r]
    if not usable:
        return AggregatedSeries(label=label, family=family, n_runs=0)

    grid = build_grid(usable, points, logarithmic)
    result = AggregatedSeries(label=label, family=family, n_runs=len(usable))
    full_until = -1

    for index, budget in enumerate(grid):
        values = [
            v
            for v in (step_value_at(run, budget) for run in usable)
            if v is not None
        ]
        result.x.append(budget)
        result.n_at_x.append(len(values))
        if not values:
            result.median.append(None)
            result.q1.append(None)
            result.q3.append(None)
            continue
        result.median.append(quantile(values, 0.5))
        result.q1.append(quantile(values, 0.25))
        result.q3.append(quantile(values, 0.75))
        if len(values) == len(usable):
            full_until = index

    result.full_until_index = max(full_until, 0)
    return result


def downsample_pairs(
    points: Sequence[tuple[float, float]], target: int
) -> tuple[list[tuple[float, float]], bool]:
    if target <= 0 or len(points) <= target:
        return list(points), False
    return lttb(points, target), True
