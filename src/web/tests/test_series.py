"""Tests for downsampling and for the median/IQR aggregation.

The aggregation tests matter more than they look. §9.2 describes a specific way
to draw a median-and-band chart that is wrong while looking convincing, and the
failure is invisible on the chart itself -- so it has to be caught here.
"""

import pytest

from app.services import series


def test_lttb_keeps_endpoints_and_hits_the_target():
    points = [(float(i), float(i % 17)) for i in range(2000)]
    sampled = series.lttb(points, 100)
    assert len(sampled) == 100
    assert sampled[0] == points[0]
    assert sampled[-1] == points[-1]


def test_lttb_is_a_no_op_below_the_threshold():
    points = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    assert series.lttb(points, 50) == points


def test_lttb_preserves_a_spike_that_naive_sampling_would_drop():
    """A spike in a loss curve is a measurement, not noise.

    Every tenth point misses this one; LTTB is chosen precisely so it does not.
    """
    points = [(float(i), 1.0) for i in range(500)]
    points[247] = (247.0, 95.0)

    every_tenth = points[::10]
    assert (247.0, 95.0) not in every_tenth

    assert (247.0, 95.0) in series.lttb(points, 50)


def test_step_interpolation_does_not_invent_measurements():
    """Between observations the value is the last one seen, not a blend."""
    run = [(0.0, 10.0), (100.0, 5.0), (200.0, 2.0)]
    assert series.step_value_at(run, 0.0) == 10.0
    assert series.step_value_at(run, 99.0) == 10.0
    assert series.step_value_at(run, 100.0) == 5.0
    assert series.step_value_at(run, 150.0) == 5.0
    # Linear interpolation would answer 3.5 here, which was never measured.
    assert series.step_value_at(run, 150.0) != 3.5


def test_step_interpolation_is_undefined_outside_the_observed_range():
    run = [(10.0, 4.0), (20.0, 3.0)]
    assert series.step_value_at(run, 5.0) is None
    assert series.step_value_at(run, 25.0) is None


def test_quantiles_match_the_usual_definition():
    values = [1.0, 2.0, 3.0, 4.0]
    assert series.quantile(values, 0.5) == 2.5
    assert series.quantile(values, 0.25) == 1.75
    assert series.quantile(values, 0.75) == 3.25
    assert series.quantile([7.0], 0.5) == 7.0


def test_band_stops_being_full_where_the_first_run_ends():
    """The failure §9.2 warns about, stated as a test.

    Three runs, one of which stops at a quarter of the budget. Past that point
    the band is computed from fewer runs, so it narrows for a reason that has
    nothing to do with agreement between them. If full_until_index did not move,
    the chart would quietly flatter whichever method gave up first.
    """
    long_a = [(float(i), 10.0 - i * 0.01) for i in range(0, 400, 10)]
    long_b = [(float(i), 11.0 - i * 0.01) for i in range(0, 400, 10)]
    short = [(float(i), 3.0) for i in range(0, 100, 10)]

    result = series.aggregate("group", [long_a, long_b, short], points=40)

    assert result.n_runs == 3
    assert result.n_at_x[0] == 3
    assert result.n_at_x[-1] == 2

    boundary = result.full_until_index
    assert result.n_at_x[boundary] == 3
    assert result.n_at_x[boundary + 1] < 3
    # The short run stops at 100 of a 390-wide axis.
    assert result.x[boundary] <= 100.0


def test_aggregate_reports_median_and_quartiles():
    runs = [
        [(0.0, 1.0), (10.0, 1.0)],
        [(0.0, 2.0), (10.0, 2.0)],
        [(0.0, 3.0), (10.0, 3.0)],
        [(0.0, 4.0), (10.0, 4.0)],
    ]
    result = series.aggregate("g", runs, points=3)
    assert result.median[0] == pytest.approx(2.5)
    assert result.q1[0] == pytest.approx(1.75)
    assert result.q3[0] == pytest.approx(3.25)


def test_aggregate_handles_an_empty_group():
    result = series.aggregate("empty", [[], []], points=10)
    assert result.n_runs == 0
    assert result.x == []


def test_log_grid_is_geometric():
    runs = [[(1.0, 1.0), (1000.0, 0.1)]]
    grid = series.build_grid(runs, points=4, logarithmic=True)
    assert grid[0] == pytest.approx(1.0)
    assert grid[-1] == pytest.approx(1000.0)
    assert grid[1] == pytest.approx(10.0)


def test_series_points_zips_and_sorts():
    row = {
        "epochs": [1, 2, 3],
        "loss": [3.0, 2.0, 1.0],
        "gradient_count": [30, 10, 20],
        "database_reaches": [300, 100, 200],
        "accuracy": [10.0, 20.0, 30.0],
    }
    pairs = series.series_points(row, "gradient_count", "loss")
    assert [p[0] for p in pairs] == [10.0, 20.0, 30.0]
    assert [p[1] for p in pairs] == [2.0, 1.0, 3.0]


def test_downsample_reports_whether_it_did_anything():
    small = [(float(i), 1.0) for i in range(10)]
    points, truncated = series.downsample_pairs(small, 100)
    assert truncated is False and len(points) == 10

    large = [(float(i), float(i % 7)) for i in range(5000)]
    points, truncated = series.downsample_pairs(large, 500)
    assert truncated is True and len(points) == 500
