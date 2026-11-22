"""Optional per-dataset networks, discovered by file name.

A file ``<dataset id>.py`` here (for example ``ds03.py``) that defines::

    MODELS = {"wide": lambda n_features, n_classes: MyNetwork(n_features, n_classes)}

adds those architectures to that dataset, next to the presets, and may override
one of them by reusing its name. Each factory receives the dataset's feature and
class counts, so nothing here needs to know what the data is.
"""
