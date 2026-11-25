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


@pytest.mark.parametrize("n_classes", [2, 0, -5])
def test_n_classes_that_does_not_cover_the_labels_is_rejected(tmp_path, arrays, n_classes):
    path = tmp_path / "ds01.npz"
    np.savez(path, X=arrays[0], y=arrays[1], n_classes=n_classes)
    with pytest.raises(ValueError, match="ds01.npz: label 2 does not fit"):
        df.load_arrays(path)


def test_n_classes_may_exceed_the_labels_present(tmp_path, arrays):
    path = tmp_path / "ds01.npz"
    np.savez(path, X=arrays[0], y=arrays[1], n_classes=4)
    assert df.load_arrays(path)[2] == 4


@pytest.mark.parametrize(
    ("labels", "expected"),
    [
        ([0, 1, 2, 1], [0, 1, 2, 1]),
        ([0.0, 1.0, 2.0, 1.0], [0, 1, 2, 1]),
        ([0.2, 0.3, 1.7, 0.3], [0, 1, 2, 1]),
        ([3, 5, 8, 5], [0, 1, 2, 1]),
        ([-1, 1, -1, 1], [0, 1, 0, 1]),
        (["bad", "good", "bad", "ok"], [0, 1, 0, 2]),
        ([True, False, True, False], [1, 0, 1, 0]),
    ],
)
def test_every_distinct_label_becomes_a_class(labels, expected):
    y = df.to_labels(labels)
    assert y.dtype == np.int64
    assert y.tolist() == expected


@pytest.mark.parametrize(
    "labels",
    [
        [0.2, 1.7, np.nan],
        [0.0, 1.0, np.inf],
        np.array(["a", None, "b"], dtype=object),
        np.array(["a", np.nan, "b"], dtype=object),
    ],
)
def test_missing_labels_are_rejected(labels):
    with pytest.raises(ValueError, match="missing"):
        df.to_labels(labels)


def test_labels_must_be_one_dimensional():
    with pytest.raises(ValueError, match="1-D"):
        df.to_labels([[0], [1]])


def test_standardise():
    rng = np.random.default_rng(0)
    X = (rng.normal(size=(500, 3)) * [0.01, 1, 5000] + [3, 0, -40000]).astype(np.float32)
    Z = df.standardise(X)
    assert Z.dtype == np.float32 and Z.shape == X.shape
    np.testing.assert_allclose(Z.mean(axis=0), 0, atol=1e-3)
    np.testing.assert_allclose(Z.std(axis=0), 1, atol=1e-3)


def test_standardise_constant_column_becomes_zero():
    X = np.full((1000, 2), 0.1, dtype=np.float32)
    X[:, 1] = np.arange(1000)
    Z = df.standardise(X)
    np.testing.assert_allclose(Z[:, 0], 0, atol=1e-6)


def test_standardise_values_whose_square_overflows_float32():
    X = np.zeros((100, 1), dtype=np.float32)
    X[0] = -3.3e26
    Z = df.standardise(X)
    assert np.isfinite(Z).all() and Z[0, 0] < -9


@pytest.mark.parametrize("dataset_id", ["ds01", "wine-quality", "wine_quality", "01"])
def test_valid_dataset_ids(dataset_id):
    df.check_dataset_id(dataset_id)


@pytest.mark.parametrize("dataset_id", ["", "a.b", "../x", "a/b", "a b", "__init__", "-x", "ds01\n"])
def test_invalid_dataset_ids(dataset_id):
    with pytest.raises(ValueError, match="dataset id"):
        df.check_dataset_id(dataset_id)
