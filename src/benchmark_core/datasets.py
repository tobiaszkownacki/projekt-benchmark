"""The datasets and networks a benchmark run is executed against.

``BenchmarkRunner`` reads both registries by dataset name. They are empty here
on purpose: the competition networks and datasets are not part of this
repository and are supplied by whoever deploys the engine, which is also what
keeps them out of a participant's reach. Populate them in place before
constructing a runner:

    from benchmark_core import datasets

    datasets.DATA_SETS["my_set"] = {"data_set": load_my_set}
    datasets.MODELS["my_set"] = {"default": MyNetwork}

``tools/local_backend/`` deliberately uses neither: it drives the evaluator over
public scikit-learn data, so development needs no competition asset.
"""

from collections.abc import Callable

from torch.nn import Module
from torch.utils.data import Dataset

# Keyed by dataset name. "data_set" is called once, when a runner is built.
DATA_SETS: dict[str, dict[str, Callable[[], Dataset]]] = {}

# Keyed by dataset name, then by the runner's `model_name` argument -- "default"
# unless the caller asks for another architecture of the same dataset.
MODELS: dict[str, dict[str, Callable[[], Module]]] = {}
