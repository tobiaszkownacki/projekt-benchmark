import numpy as np
import pytest

from benchmark_core import dataset_files as df


@pytest.fixture
def arrays():
    rng = np.random.default_rng(0)
    return rng.normal(size=(50, 4)).astype(np.float32), rng.integers(0, 3, size=50).astype(np.int64)


def test_roundtrip(tmp_path, arrays):
    X, y = arrays
    df.save_arrays(tmp_path / "ds01.npz", X, y)
    X2, y2, n_classes = df.load_arrays(tmp_path / "ds01.npz")
    np.testing.assert_array_equal(X, X2)
    np.testing.assert_array_equal(y, y2)
    assert n_classes == int(y.max()) + 1


def test_invalid_arrays_are_not_written(tmp_path, arrays):
    X, y = arrays
    path = tmp_path / "ds01.npz"
    X_nan = X.copy()
    X_nan[0, 0] = np.nan
    for bad_X, bad_y, message in (
        (X.astype(np.float64), y, "float32"),
        (X, y.astype(np.int32), "int64"),
        (X_nan, y, "NaN"),
        (X, y[:-1], "rows"),
        (X, y - 5, "non-negative"),
        (X[:0], y[:0], "empty"),
    ):
        with pytest.raises(ValueError, match=message):
            df.save_arrays(path, bad_X, bad_y)
    assert not path.exists()


def test_file_that_is_not_a_dataset(tmp_path, arrays):
    path = tmp_path / "ds01.npz"
    path.write_bytes(b"plain,csv\n1,2\n")
    with pytest.raises(ValueError, match="ds01.npz: not a dataset file"):
        df.load_arrays(path)
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="not a dataset file"):
        df.load_arrays(path)
    np.savez(path, X=arrays[0])
    with pytest.raises(ValueError, match="not a dataset file"):
        df.load_arrays(path)


def test_truncated_file(tmp_path, arrays):
    path = tmp_path / "ds01.npz"
    df.save_arrays(path, *arrays)
    path.write_bytes(path.read_bytes()[:-100])
    with pytest.raises(ValueError, match="not a dataset file"):
        df.load_arrays(path)


def test_invalid_arrays_in_a_file_name_the_file(tmp_path, arrays):
    path = tmp_path / "ds01.npz"
    np.savez(path, X=arrays[0].astype(np.float64), y=arrays[1], n_classes=3)
    with pytest.raises(ValueError, match="ds01.npz: X must be"):
        df.load_arrays(path)
