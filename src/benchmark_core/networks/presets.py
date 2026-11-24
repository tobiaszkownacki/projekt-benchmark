"""Networks that fit any dataset, sized from the data itself.

Every dataset gets these architectures without a file of its own: the input
and output sizes come from the dataset file (feature count and the class count
stored with it), so registering a dataset needs no model code.

Like the earlier per-dataset networks they start with ``BatchNorm1d`` over the
inputs, since the features are not scaled, and initialise every linear layer
with Xavier-uniform weights and zero biases. The initialisation is the
optimisation's starting point, so it is fixed rather than left to PyTorch's
default; with the runner's seed every optimizer starts from the same weights.
"""

from collections.abc import Callable

import torch.nn as nn

# Hidden layer widths. "default" is what BenchmarkRunner selects when no model
# is named; the CLI's own default is "light", so the two agree.
PRESETS: dict[str, tuple[int, ...]] = {
    "light": (32,),
    "medium": (64, 64),
    "heavy": (128, 128, 64),
}
PRESETS["default"] = PRESETS["light"]

ModelFactory = Callable[[int, int], nn.Module]


def init_xavier(model: nn.Module) -> nn.Module:
    """Xavier-uniform weights and zero biases for every linear layer."""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    return model


def mlp(n_features: int, n_classes: int, hidden: tuple[int, ...]) -> nn.Sequential:
    layers: list[nn.Module] = [nn.BatchNorm1d(n_features)]
    width = n_features
    for size in hidden:
        layers += [nn.Linear(width, size), nn.ReLU()]
        width = size
    layers.append(nn.Linear(width, n_classes))
    return init_xavier(nn.Sequential(*layers))


def preset_factories() -> dict[str, ModelFactory]:
    return {
        name: (lambda n_features, n_classes, h=hidden: mlp(n_features, n_classes, h))
        for name, hidden in PRESETS.items()
    }
