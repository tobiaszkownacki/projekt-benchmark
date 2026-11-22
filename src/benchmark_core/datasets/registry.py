"""Finds the encrypted datasets and registers them with their networks.

Adding a dataset means dropping its ``<id>.bin`` into ``DATA_DIR``: the id
becomes the ``--dataset`` choice, its data loads from the file and every
preset network (plus any ``networks/custom/<id>.py``) fits it. No file names a
dataset, so the repository stays free of what the data is.

Discovery only lists files. Decryption, and with it reading the key, happens
the first time a dataset is used -- in ``BenchmarkRunner.__init__``, before the
optimizer process is spawned. That matters because the spawned process imports
this module too and must find nothing to decrypt and no key on its way in.
"""

import importlib
import importlib.util
from collections.abc import Callable
from pathlib import Path

import numpy as np
import torch
from torch.nn import Module
from torch.utils.data import Dataset, TensorDataset

from benchmark_core import dataset_crypto
from benchmark_core.networks.presets import ModelFactory, preset_factories

DATA_DIR = Path(__file__).resolve().parent / "data"
CUSTOM_NETWORKS = "benchmark_core.networks.custom"

# Keyed by dataset id. "data_set" is called once, when a runner is built.
DATA_SETS: dict[str, dict[str, Callable[[], Dataset]]] = {}

# Keyed by dataset id, then by the runner's `model_name` argument.
MODELS: dict[str, dict[str, Callable[[], Module]]] = {}


class EncryptedDataset:
    """One encrypted file, decrypted on first use and kept in memory after."""

    def __init__(self, dataset_id: str, path: Path):
        self.dataset_id = dataset_id
        self.path = path
        self._arrays: tuple[np.ndarray, np.ndarray, int] | None = None
        self._dataset: TensorDataset | None = None

    def _load(self) -> tuple[np.ndarray, np.ndarray, int]:
        if self._arrays is None:
            self._arrays = dataset_crypto.load_encrypted(self.path, self.dataset_id)
        return self._arrays

    def dataset(self) -> TensorDataset:
        if self._dataset is None:
            X, y, _ = self._load()
            self._dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        return self._dataset

    def shape(self) -> tuple[int, int]:
        """(feature count, class count), what a network needs to fit the data."""
        X, _, n_classes = self._load()
        return X.shape[1], n_classes


def custom_networks(dataset_id: str) -> dict[str, ModelFactory]:
    """The ``MODELS`` of ``networks/custom/<dataset_id>.py``, or nothing."""
    name = f"{CUSTOM_NETWORKS}.{dataset_id}"
    if importlib.util.find_spec(name) is None:
        return {}
    models = getattr(importlib.import_module(name), "MODELS", None)
    if not isinstance(models, dict) or not all(callable(f) for f in models.values()):
        raise TypeError(f"{name} must define MODELS = {{name: factory(n_features, n_classes)}}")
    return models


def register(dataset_id: str, path: Path) -> None:
    source = EncryptedDataset(dataset_id, path)
    factories = {**preset_factories(), **custom_networks(dataset_id)}
    DATA_SETS[dataset_id] = {"data_set": source.dataset}
    # The runner calls these without arguments; the shape comes from the data,
    # which the runner has already loaded by then, so nothing is decrypted twice.
    MODELS[dataset_id] = {name: (lambda f=factory: f(*source.shape())) for name, factory in factories.items()}


def discover(data_dir: Path = DATA_DIR) -> list[str]:
    """Registers every ``*.bin`` in ``data_dir``; returns the ids found."""
    ids = []
    for path in sorted(data_dir.glob("*.bin")):
        if not path.stem.isidentifier():
            raise ValueError(f"{path.name}: a dataset id must be a valid Python identifier, e.g. ds01")
        register(path.stem, path)
        ids.append(path.stem)
    return ids
