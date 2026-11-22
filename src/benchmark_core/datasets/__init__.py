"""The datasets and networks a benchmark run is executed against.

``BenchmarkRunner`` reads both registries by dataset id. They fill themselves
on import from the encrypted files in ``datasets/data/`` (see ``registry``):
a file ``ds03.bin`` there is the dataset ``ds03``, with the preset networks
from ``benchmark_core.networks`` and any ``networks/custom/ds03.py``. The files
are not part of this repository; ``just bundle`` copies them in.

A dataset that is not encrypted can be registered by hand at the end of this
file, after ``discover()``, so that the CLI sees it too:

    DATA_SETS["my_set"] = {"data_set": load_my_set}
    MODELS["my_set"] = {"light": MyNetwork}

Only public data belongs here, never a competition dataset.
"""

from benchmark_core.datasets.registry import DATA_DIR, DATA_SETS, MODELS, discover

__all__ = ["DATA_DIR", "DATA_SETS", "MODELS", "discover"]

discover()
