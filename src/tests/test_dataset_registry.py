import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # the CI engine job runs without torch

from benchmark_core import dataset_files  # noqa: E402
from benchmark_core.datasets import registry  # noqa: E402
from benchmark_core.networks.presets import PRESETS  # noqa: E402

N_FEATURES, N_CLASSES = 5, 3


@pytest.fixture(autouse=True)
def empty_registries(monkeypatch):
    monkeypatch.setattr(registry, "DATA_SETS", {})
    monkeypatch.setattr(registry, "MODELS", {})


@pytest.fixture
def data_dir(tmp_path):
    rng = np.random.default_rng(0)
    for dataset_id in ("ds01", "ds02"):
        X = rng.normal(size=(40, N_FEATURES)).astype(np.float32)
        y = (np.arange(40) % N_CLASSES).astype(np.int64)
        dataset_files.save_arrays(tmp_path / f"{dataset_id}.npz", X, y)
    return tmp_path


def test_discover_registers_every_file(data_dir):
    assert registry.discover(data_dir) == ["ds01", "ds02"]
    assert set(registry.DATA_SETS) == {"ds01", "ds02"}
    assert set(registry.MODELS["ds01"]) == set(PRESETS)


def test_discover_does_not_read_the_data(data_dir, monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("discovery must not read the data")

    monkeypatch.setattr(dataset_files, "load_arrays", forbidden)
    registry.discover(data_dir)


def test_missing_data_dir_registers_nothing(tmp_path):
    assert registry.discover(tmp_path / "absent") == []


def test_loader_returns_the_data_and_reads_it_once(data_dir, monkeypatch):
    calls = []
    real = dataset_files.load_arrays
    monkeypatch.setattr(dataset_files, "load_arrays", lambda *a, **k: calls.append(a) or real(*a, **k))
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


def test_presets_use_xavier(data_dir):
    registry.discover(data_dir)
    model = registry.MODELS["ds01"]["medium"]()
    linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
    for layer in linears:
        fan_out, fan_in = layer.weight.shape
        assert layer.weight.abs().max() <= (6 / (fan_in + fan_out)) ** 0.5
        assert torch.count_nonzero(layer.bias) == 0


@pytest.mark.parametrize("name", list(PRESETS))
def test_presets_are_the_same_function_in_train_and_eval_mode(data_dir, name):
    """Gradients are computed in train mode and plain losses in eval mode."""
    registry.discover(data_dir)
    model = registry.MODELS["ds01"][name]()
    inputs = torch.randn(8, N_FEATURES) * 100 + 50
    assert torch.equal(model.train()(inputs), model.eval()(inputs))


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
    (tmp_path / "a.b.npz").write_bytes(b"")
    with pytest.raises(ValueError, match="dataset id"):
        registry.discover(tmp_path)


def test_id_with_a_hyphen_registers_with_its_custom_network(data_dir, custom_package):
    (data_dir / "ds01.npz").rename(data_dir / "wine-quality.npz")
    (custom_package / "wine-quality.py").write_text(
        "import torch.nn as nn\nMODELS = {'linear': lambda n_features, n_classes: nn.Linear(n_features, n_classes)}\n"
    )
    assert registry.discover(data_dir) == ["ds02", "wine-quality"]
    assert isinstance(registry.MODELS["wine-quality"]["linear"](), torch.nn.Linear)


def test_unknown_model_is_reported_before_the_data_is_read(data_dir, monkeypatch):
    from benchmark_core.optimization_engine import BenchmarkRunner, StopCondition, runner

    registry.discover(data_dir)
    monkeypatch.setattr(runner, "DATA_SETS", registry.DATA_SETS)
    monkeypatch.setattr(runner, "MODELS", registry.MODELS)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("the data must not be read")

    monkeypatch.setattr(dataset_files, "load_arrays", forbidden)
    with pytest.raises(ValueError, match=r"Unknown model for ds01: nope\. Available: \['default', 'heavy'"):
        BenchmarkRunner("ds01", StopCondition(max_epochs=1), model_name="nope", device="cpu")


@pytest.fixture
def make_runner(tmp_path, monkeypatch):
    """Builds a real runner over a dataset file of ``n`` rows."""
    from benchmark_core.optimization_engine import BenchmarkRunner, StopCondition, runner

    def make(n, **kwargs):
        rng = np.random.default_rng(0)
        data = tmp_path / "data"
        data.mkdir()
        X = rng.normal(size=(n, N_FEATURES)).astype(np.float32)
        dataset_files.save_arrays(data / "ds01.npz", X, (np.arange(n) % N_CLASSES).astype(np.int64))
        registry.discover(data)
        monkeypatch.setattr(runner, "DATA_SETS", registry.DATA_SETS)
        monkeypatch.setattr(runner, "MODELS", registry.MODELS)
        return BenchmarkRunner(
            "ds01", StopCondition(max_epochs=1), device="cpu", report_dir=str(tmp_path / "reports"), **kwargs
        )

    return make


@pytest.mark.parametrize(("n", "batch_size"), [(65, 32), (8, 1)])
def test_gradient_run_survives_a_batch_of_one_sample(make_runner, n, batch_size):
    from benchmark_core.optimization_engine.optimizers.numpy.numpy_adam import NumpyAdam

    result = make_runner(n, batch_size=batch_size).run(NumpyAdam, lr=0.001)
    assert result.total_epochs == 1 and np.isfinite(result.final_loss)


def test_compared_optimizers_start_from_the_same_weights(make_runner):
    from benchmark_core.optimization_engine.optimizers.numpy.numpy_adam import NumpyAdam

    runner = make_runner(64)
    starts = []
    factory = runner.model_factory

    def recording_factory():
        model = factory()
        starts.append(torch.nn.utils.parameters_to_vector(model.parameters()).detach().clone())
        return model

    runner.model_factory = recording_factory
    results = runner.compare({"a": (NumpyAdam, {"lr": 0.001}), "b": (NumpyAdam, {"lr": 0.001})})
    assert torch.equal(*starts)
    assert results["a"].loss_history == results["b"].loss_history
