"""Reading and writing a dataset as its preprocessed arrays.

A dataset is one ``.npz`` file with ``X`` (float32, ``(n_samples, n_features)``),
``y`` (int64 class labels from 0, ``(n_samples,)``) and ``n_classes``. Both
directions check the arrays, so a mistake in the preprocessing shows up when
the file is written rather than in the middle of a run. ``standardise`` and
``to_labels`` turn what a preprocessing script returns into such arrays.

Deliberately free of torch, so it can be tested on a CPU-only runner.
"""

import re
import zipfile
from pathlib import Path

import numpy as np

# The id names the data file, the --dataset choice and the optional module
# networks/custom/<id>.py, so it must stay clear of dots and path separators.
_DATASET_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def check_dataset_id(dataset_id: str) -> None:
    if not _DATASET_ID.fullmatch(dataset_id):
        raise ValueError(
            f"{dataset_id!r} is not a valid dataset id: use letters, digits, '_' and '-', "
            "starting with a letter or a digit, e.g. ds01"
        )


def standardise(X: np.ndarray) -> np.ndarray:
    """Every feature with mean 0 and standard deviation 1, as float32; a constant feature becomes 0."""
    X = np.asarray(X, dtype=np.float64)
    scale = X.std(axis=0)
    scale[X.min(axis=0) == X.max(axis=0)] = 1
    return np.ascontiguousarray((X - X.mean(axis=0)) / scale, dtype=np.float32)


def _is_missing(label) -> bool:
    try:
        return label is None or bool(label != label)
    except TypeError:  # pandas.NA compares to neither true nor false
        return True


def to_labels(y) -> np.ndarray:
    """Numbers the distinct values of ``y`` (of any type) as int64 classes 0, 1, 2, ... in sorted order."""
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"labels must be a 1-D array, got {y.ndim}-D")
    if y.dtype.kind == "f":
        missing = not np.isfinite(y).all()
    else:
        missing = y.dtype.kind == "O" and any(_is_missing(label) for label in y)
    if missing:
        raise ValueError("labels contain missing or infinite values")
    return np.unique(y, return_inverse=True)[1].astype(np.int64)


def _validate(X: np.ndarray, y: np.ndarray) -> None:
    if X.ndim != 2 or X.dtype != np.float32:
        raise ValueError(f"X must be a 2-D float32 array, got {X.ndim}-D {X.dtype}")
    if y.ndim != 1 or y.dtype != np.int64:
        raise ValueError(f"y must be a 1-D int64 array, got {y.ndim}-D {y.dtype}")
    if len(X) != len(y):
        raise ValueError(f"X has {len(X)} rows but y has {len(y)}")
    if len(y) == 0:
        raise ValueError("dataset is empty")
    if not np.isfinite(X).all():
        raise ValueError("X contains NaN or infinite values")
    if y.min() < 0:
        raise ValueError("class labels must be non-negative")


def save_arrays(path: str | Path, X: np.ndarray, y: np.ndarray) -> None:
    _validate(X, y)
    np.savez_compressed(path, X=X, y=y, n_classes=np.int64(y.max() + 1))


def load_arrays(path: str | Path) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns ``(X, y, n_classes)``. Raises ``ValueError`` for anything that is not a dataset file."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as arrays:
            X, y, n_classes = arrays["X"], arrays["y"], int(arrays["n_classes"])
    except (KeyError, TypeError, AttributeError, EOFError, ValueError, zipfile.BadZipFile) as exc:
        raise ValueError(f"{path.name}: not a dataset file ({exc})") from exc
    try:
        _validate(X, y)
        if y.max() >= n_classes:
            raise ValueError(f"label {y.max()} does not fit n_classes={n_classes} (labels start at 0)")
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from None
    return X, y, n_classes
