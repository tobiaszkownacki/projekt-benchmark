"""Encrypted, preprocessed datasets that are only ever decrypted in memory.

A dataset is stored as its already-preprocessed arrays (``X`` float32 of shape
``(n_samples, n_features)``, ``y`` int64 of shape ``(n_samples,)``) so that
neither the raw file nor its column names ship with the benchmark. Knowing the
data would let a participant precompute its minimum and start an optimizer
there; encryption is one more layer that makes recovering it harder.

File layout::

    MAGIC (4 B) | nonce (12 B) | AES-256-GCM ciphertext + tag

The dataset id is bound as associated data, so a blob cannot be renamed to
stand in for another dataset. Decryption goes through ``io.BytesIO`` only:
plaintext never touches the disk.

Deliberately free of torch, so it can be tested on a CPU-only runner.
"""

import base64
import binascii
import io
import os
import secrets
from pathlib import Path

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

MAGIC = b"BDS1"
NONCE_SIZE = 12
KEY_SIZE = 32
KEY_ENV_VAR = "BENCHMARK_DATA_KEY"

_cached_key: bytes | None = None


def generate_key() -> str:
    """A new random AES-256 key, as the base64 text stored in the key file."""
    return base64.urlsafe_b64encode(secrets.token_bytes(KEY_SIZE)).decode("ascii")


def decode_key(text: str) -> bytes:
    try:
        key = base64.urlsafe_b64decode(text.strip().encode("ascii"))
    except (binascii.Error, ValueError) as exc:
        raise ValueError("dataset key is not valid base64") from exc
    if len(key) != KEY_SIZE:
        raise ValueError(f"dataset key must be {KEY_SIZE} bytes, got {len(key)}")
    return key


def _bundled_key() -> str | None:
    """The key compiled into a build, if this is one.

    ``benchmark_core/_bundled_key.py`` is generated from data/private/key by
    ``scripts/bundle_data.py`` (``just bundle``) and is never committed, so a
    development checkout has none and falls back to the environment.
    """
    try:
        from benchmark_core._bundled_key import KEY
    except ImportError:
        return None
    return KEY


def get_key() -> bytes:
    """The dataset key: the bundled one in a build, otherwise ``BENCHMARK_DATA_KEY``.

    The environment variable is removed whenever it is read, which keeps it out
    of the optimizer process: that one is spawned later and inherits
    ``os.environ``.
    """
    global _cached_key
    text = os.environ.pop(KEY_ENV_VAR, None)
    if _cached_key is None:
        bundled = _bundled_key()
        if bundled is not None:
            text = bundled
        if text is None:
            raise ValueError(f"no dataset key: set {KEY_ENV_VAR} or bundle one with `just bundle`")
        _cached_key = decode_key(text)
    return _cached_key


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


def encrypt_arrays(X: np.ndarray, y: np.ndarray, dataset_id: str, key: bytes) -> bytes:
    _validate(X, y)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, X=X, y=y, n_classes=np.int64(y.max() + 1))
    nonce = secrets.token_bytes(NONCE_SIZE)
    ciphertext = AESGCM(key).encrypt(nonce, buffer.getvalue(), dataset_id.encode("utf-8"))
    return MAGIC + nonce + ciphertext


def decrypt_arrays(blob: bytes, dataset_id: str, key: bytes) -> tuple[np.ndarray, np.ndarray, int]:
    """Returns ``(X, y, n_classes)``. Raises ``ValueError`` on any mismatch."""
    if len(blob) <= len(MAGIC) + NONCE_SIZE or not blob.startswith(MAGIC):
        raise ValueError(f"dataset {dataset_id!r}: not an encrypted dataset file")
    nonce = blob[len(MAGIC) : len(MAGIC) + NONCE_SIZE]
    try:
        payload = AESGCM(key).decrypt(nonce, blob[len(MAGIC) + NONCE_SIZE :], dataset_id.encode("utf-8"))
    except InvalidTag:
        raise ValueError(
            f"dataset {dataset_id!r}: decryption failed (wrong key, wrong dataset id or corrupted file)"
        ) from None
    with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
        X, y, n_classes = arrays["X"], arrays["y"], int(arrays["n_classes"])
    _validate(X, y)
    return X, y, n_classes


def load_encrypted(path: str | Path, dataset_id: str, key: bytes | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Decrypts the dataset file at ``path`` into memory."""
    return decrypt_arrays(Path(path).read_bytes(), dataset_id, key if key is not None else get_key())
