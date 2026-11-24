"""Reading and writing a dataset as its preprocessed arrays.

A dataset is one ``.npz`` file with ``X`` (float32, ``(n_samples, n_features)``),
``y`` (int64 class labels from 0, ``(n_samples,)``) and ``n_classes``. Both
directions check the arrays, so a mistake in the preprocessing shows up when
the file is written rather than in the middle of a run.

Deliberately free of torch, so it can be tested on a CPU-only runner.
"""

import zipfile
from pathlib import Path

import numpy as np


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
    except ValueError as exc:
        raise ValueError(f"{path.name}: {exc}") from None
    return X, y, n_classes
