"""Public datasets and matching models for the local CPU backend.

These are the toy datasets bundled with scikit-learn, used to produce genuine
convergence curves from genuine optimizers for development and testing.

They are not the benchmark's own datasets, which stay out of the repository, and
they must never be presented as competition problems.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_breast_cancer, load_digits, load_wine
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    loader: Callable
    description: str


PUBLIC_DATASETS = {
    "digits": DatasetSpec("digits", load_digits, "scikit-learn digits, 1797x64, 10 klas"),
    "wine": DatasetSpec("wine", load_wine, "scikit-learn wine, 178x13, 3 klasy"),
    "breast_cancer": DatasetSpec("breast_cancer", load_breast_cancer, "scikit-learn breast cancer, 569x30, 2 klasy"),
}


def build_dataset(name: str, seed: int = 0) -> tuple[TensorDataset, int, int]:
    spec = PUBLIC_DATASETS[name]
    bundle = spec.loader()
    features = StandardScaler().fit_transform(bundle.data.astype(np.float64))
    targets = bundle.target.astype(np.int64)

    generator = np.random.default_rng(seed)
    order = generator.permutation(len(targets))
    features, targets = features[order], targets[order]

    dataset = TensorDataset(
        torch.tensor(features, dtype=torch.float32),
        torch.tensor(targets, dtype=torch.long),
    )
    return dataset, features.shape[1], int(targets.max()) + 1


class MLP(nn.Module):
    """Small enough to produce a real optimizer trajectory on a CPU in seconds."""

    def __init__(self, in_features: int, classes: int, hidden: tuple[int, ...]):
        super().__init__()
        layers: list[nn.Module] = []
        previous = in_features
        for width in hidden:
            layers += [nn.Linear(previous, width), nn.ReLU()]
            previous = width
        layers.append(nn.Linear(previous, classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


MODEL_SHAPES = {
    "mlp-1x16": (16,),
    "mlp-2x32": (32, 32),
    "mlp-3x64": (64, 64, 64),
}


def build_model(model_name: str, in_features: int, classes: int) -> nn.Module:
    return MLP(in_features, classes, MODEL_SHAPES[model_name])


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
