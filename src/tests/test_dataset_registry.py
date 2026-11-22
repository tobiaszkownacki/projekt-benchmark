import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # the CI engine job runs without torch

from benchmark_core import dataset_crypto as dc  # noqa: E402
from benchmark_core.datasets import registry  # noqa: E402
from benchmark_core.networks.presets import PRESETS  # noqa: E402

N_FEATURES, N_CLASSES = 5, 3


@pytest.fixture
def key(monkeypatch):
    key = dc.decode_key(dc.generate_key())
    monkeypatch.setitem(sys.modules, "benchmark_core._bundled_key", None)
    monkeypatch.setattr(dc, "_cached_key", key)
    return key


@pytest.fixture(autouse=True)
def empty_registries(monkeypatch):
    monkeypatch.setattr(registry, "DATA_SETS", {})
    monkeypatch.setattr(registry, "MODELS", {})


@pytest.fixture
def data_dir(tmp_path, key):
    rng = np.random.default_rng(0)
    for dataset_id in ("ds01", "ds02"):
        X = rng.normal(size=(40, N_FEATURES)).astype(np.float32)
        y = (np.arange(40) % N_CLASSES).astype(np.int64)
        (tmp_path / f"{dataset_id}.bin").write_bytes(dc.encrypt_arrays(X, y, dataset_id, key))
    return tmp_path


def test_discover_registers_every_file(data_dir):
    assert registry.discover(data_dir) == ["ds01", "ds02"]
    assert set(registry.DATA_SETS) == {"ds01", "ds02"}
    assert set(registry.MODELS["ds01"]) == set(PRESETS)


def test_discover_neither_decrypts_nor_reads_the_key(data_dir, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("discovery must not decrypt")

    monkeypatch.setattr(dc, "get_key", forbidden)
    monkeypatch.setattr(dc, "load_encrypted", forbidden)
    registry.discover(data_dir)


def test_missing_data_dir_registers_nothing(tmp_path):
    assert registry.discover(tmp_path / "absent") == []


def test_loader_returns_the_data_and_decrypts_once(data_dir, monkeypatch):
    calls = []
    real = dc.load_encrypted
    monkeypatch.setattr(dc, "load_encrypted", lambda *a, **k: calls.append(a) or real(*a, **k))
    registry.discover(data_dir)

    dataset = registry.DATA_SETS["ds01"]["data_set"]()
    inputs, target = dataset[0]
    assert len(dataset) == 40 and inputs.shape == (N_FEATURES,) and target.dtype == torch.long
    registry.MODELS["ds01"]["heavy"]()
    assert len(calls) == 1


@pytest.mark.parametrize("name", list(PRESETS))
def test_presets_fit_the_data(data_dir, name):
    registry.discover(data_dir)
    model = registry.MODELS["ds02"][name]()
    assert model(torch.zeros(2, N_FEATURES)).shape == (2, N_CLASSES)


def test_presets_normalise_inputs_and_use_xavier(data_dir):
    registry.discover(data_dir)
    model = registry.MODELS["ds01"]["medium"]()
    assert isinstance(model[0], torch.nn.BatchNorm1d) and model[0].num_features == N_FEATURES
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    for layer in linears:
        fan_out, fan_in = layer.weight.shape
        assert layer.weight.abs().max() <= (6 / (fan_in + fan_out)) ** 0.5
        assert torch.count_nonzero(layer.bias) == 0


def test_presets_grow_and_default_is_light(data_dir):
    registry.discover(data_dir)
    sizes = {name: sum(p.numel() for p in registry.MODELS["ds01"][name]().parameters()) for name in PRESETS}
    assert sizes["light"] < sizes["medium"] < sizes["heavy"]
    assert sizes["default"] == sizes["light"]


@pytest.fixture
def custom_package(tmp_path, monkeypatch):
    package = tmp_path / "pkg" / "my_custom"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    monkeypatch.syspath_prepend(str(tmp_path / "pkg"))
    monkeypatch.setattr(registry, "CUSTOM_NETWORKS", "my_custom")
    yield package
    for name in [m for m in sys.modules if m == "my_custom" or m.startswith("my_custom.")]:
        del sys.modules[name]


def test_custom_network_is_added_for_its_dataset_only(data_dir, custom_package):
    (custom_package / "ds01.py").write_text(
        "import torch.nn as nn\nMODELS = {'linear': lambda n_features, n_classes: nn.Linear(n_features, n_classes)}\n"
    )
    registry.discover(data_dir)
    assert isinstance(registry.MODELS["ds01"]["linear"](), torch.nn.Linear)
    assert "linear" not in registry.MODELS["ds02"]


def test_custom_network_without_models_dict_is_rejected(data_dir, custom_package):
    (custom_package / "ds01.py").write_text("MODELS = 3\n")
    with pytest.raises(TypeError, match="MODELS"):
        registry.discover(data_dir)


def test_invalid_file_name_is_rejected(tmp_path):
    (tmp_path / "not-an-id.bin").write_bytes(b"")
    with pytest.raises(ValueError, match="identifier"):
        registry.discover(tmp_path)
