"""The datasets and networks a benchmark run is executed against.

``BenchmarkRunner`` reads both registries by dataset id. They fill themselves
on import from the files in ``datasets/data/`` (see ``registry``): a file
``ds03.npz`` there is the dataset ``ds03``, with the preset networks from
``benchmark_core.networks`` and any ``networks/custom/ds03.py``. The files are
not part of this repository; ``just prepare`` writes them.

A dataset that does not come from such a file can be registered by hand at the
end of this module, after ``discover()``, so that the CLI sees it too:

    DATA_SETS["my_set"] = {"data_set": load_my_set}
    MODELS["my_set"] = {"light": MyNetwork}
"""

from benchmark_core.datasets.registry import DATA_DIR, DATA_SETS, MODELS, discover

__all__ = ["DATA_DIR", "DATA_SETS", "MODELS", "discover"]

discover()
