# Optimizer Benchmark Suite [![Flake8 Linting](https://github.com/Dnafivuq/golem_template/actions/workflows/lint.yml/badge.svg)](https://github.com/Dnafivuq/golem_template/actions/workflows/lint.yml) [![Pytest](https://github.com/Dnafivuq/golem_template/actions/workflows/test.yml/badge.svg)](https://github.com/Dnafivuq/golem_template/actions/workflows/test.yml) <a target="_blank" href="https://cookiecutter-data-science.drivendata.org/"><img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" /></a>

## 1. Overview

This repository contains a benchmark suite for testing, comparing, and analyzing various optimization algorithms (e.g., Adam, SGD, CMA-ES, Differential Evolution) on different datasets and neural network models using PyTorch. The framework tracks metrics like gradient evaluations, database reaches, and standard loss/accuracy over time, providing a comprehensive toolkit to evaluate the efficiency and convergence of both gradient-based and gradient-free optimizers. It is designed to be modular and independent dependency-wise, which yields simple addition of custom optimization algorithms, new datasets, and seamless swapping of neural network architectures for cross-comparison.

## 2. Installation & Setup

> **Running the web control plane?** See **[docs/LOCAL_SETUP.md](docs/LOCAL_SETUP.md)**.
> It covers generating your own `.env` (no secret needs to be sent to you),
> starting the stack, seeding real measured runs, and what does not work
> without cluster access. The steps below cover the benchmark CLI only.

To get started with this project, follow these steps:

1. **Clone the repository**

   ```sh
   git clone <repository-url>
   cd projekt-benchmark
   ```

2. **Install dependencies**

   Using **uv** (recommended):
   ```sh
   uv sync               # core dependencies
   uv sync --extra ci    # + flake8, pytest
   ```

   Using **pip**:
   ```sh
   python -m venv .venv
   source .venv/bin/activate   # Linux/macOS
   .venv\Scripts\activate      # Windows

   pip install -e .            # core dependencies
   pip install -e .[ci]        # + flake8, pytest
   ```

## 3. Running the Benchmark

You can run benchmarks using the `benchmark_core.optimization_engine.run_benchmark` module.

### Running a single optimizer

```sh
    uv run -m benchmark_core.optimization_engine.run_benchmark --dataset digits --optimizer my_optimizer
```

### Comparing multiple optimizers

```sh
    uv run -m benchmark_core.optimization_engine.run_benchmark --dataset heart_disease --optimizer adam sgd cma-es
```

### Using a specific model architecture

```sh
    uv run -m benchmark_core.optimization_engine.run_benchmark --dataset digits --model mlp --optimizer adam
```

Add the `--plot` flag to generate comparison and performance plots. By default, they are saved to `reports/model_analysis`.

```sh
    uv run -m benchmark_core.optimization_engine.run_benchmark --dataset wine_quality --optimizer adam sgd cma-es --max-epochs 10 --max-gradients 100000 --plot
```

### Comparing multiple models and optimizers simultaneously

```sh
	uv run -m benchmark_core.optimization_engine.run_benchmark --dataset digits --model default mlp --optimizer adam sgd cma-es
```

### Available Arguments/Parameters

- `--dataset`: Name of a dataset registered in `src/benchmark_core/datasets.py` (required). The networks and datasets themselves are not part of this repository, so the choices are whatever has been registered -- see section 6.
- `--model`: Name of the model architecture to use (e.g., `default`, `mlp`). More than one model can be passed to test all combinations with the given optimizers (default: `['default']`).
- `--optimizer`: Name of a built-in optimizer (e.g., `adam`, `sgd`, `cma-es`) or a file path to a custom optimizer python script. More than one optimizer can be passed for comparison (required).
- `--max-gradients`: Stop condition for maximum number of gradient evaluations (default: 5000).
- `--max-db-reaches`: Stop condition for maximum database reaches (optional).
- `--max-epochs`: Stop condition for maximum number of epochs (optional).
- `--batch-size`: Batch size for data loading (default: 32).
- `--seed`: Random seed for reproducibility (default: 42).
- `--plot`: Flag to generate benchmark plots after the run.
- `--plot-dir`: Directory where plots will be saved (default: `reports/model_analysis`).

## 4. Code Structure

```text
├── README.md              <- The top-level README for developers using this project.
├── docker-compose.yml     <- The web control plane and the queue services.
├── docs/LOCAL_SETUP.md    <- Long-form setup, including the secret taxonomy.
├── downloads/             <- Run artifacts, reachable only through the artifact browser.
├── reports/               <- Generated plots and analysis artifacts.
├── scripts/               <- Environment bootstrap, lint baseline, broker definitions.
├── src/                   <- Source code for use in this project.
│   ├── benchmark_core/    <- The engine.
│   │   ├── datasets.py    <- Registry mapping a dataset name to its data and models.
│   │   ├── metrics/       <- Stop conditions, stop reasons and budget tracking.
│   │   ├── optimization_engine/  <- Evaluator, runner, optimizer protocols, optimizers.
│   │   └── plotting/      <- Plot generation and analyzer modules.
│   ├── db/                <- initdb schemas and forward-only numbered migrations.
│   ├── frontend/          <- The previous Streamlit interface.
│   ├── shared/            <- Connectors and interfaces shared by the queue services.
│   ├── task_queue/        <- Worker, poller and downloader for the compute backend.
│   └── web/               <- FastAPI control plane and the React SPA it serves.
├── tools/local_backend/   <- Development-only CPU runner and database seeder.
└── pyproject.toml         <- Project metadata, dependencies, and tool configuration.
```

The engine imports as `benchmark_core.*`, which `uv sync` puts on the path by
installing this project; outside a synced environment put `src/` on
`PYTHONPATH`.

## 5. Adding a New Optimizer

1. Create a new python script inside `src/benchmark_core/optimization_engine/optimizers/numpy/` (or `cupy/`, e.g., `my_optimizer_adapter.py`).
2. Create your optimizer class inheriting from `benchmark_core.optimization_engine.optimizer_protocols.NumpyBenchmarkOptimizer`, or from `BenchmarkOptimizer` to declare the array backend yourself.
3. Implement the `step(self, evaluator: ModelEvaluator) -> bool` method.
    - Inside `step()`, you can call `evaluator.evaluate_with_grad()` or `evaluator.evaluate()` depending on whether your optimizer needs gradients.
    - Update `self.params` and finally call `evaluator.set_params(self.params)`.
    - Return `True` if the optimizer has converged, `False` otherwise.
4. Add your new optimizer to the `BUILTIN_OPTIMIZERS` registry located in `src/benchmark_core/optimization_engine/optimizers/registry.py`. A CuPy optimizer belongs in the block that is skipped where CuPy does not load, so that a CPU-only host keeps the NumPy entries.
    - *Alternatively, you can test it directly without registering by passing the path to the file using `--optimizer path/to/my_optimizer_adapter.py`*.

## 6. Adding a New Dataset

The datasets and networks the benchmark runs against are deliberately not in
this repository, so both registries in `src/benchmark_core/datasets.py` start
empty and the deployment supplies their contents.

1. Write a callable that returns a PyTorch `Dataset`, wherever the data itself
   lives.
2. Register it under the dataset name in `DATA_SETS`:

   ```python
   from benchmark_core import datasets

   datasets.DATA_SETS["my_dataset_name"] = {"data_set": load_my_dataset}
   ```

3. Register at least a `"default"` model for it, as in section 7. The runner
   reads both registries by dataset name and raises if either has no entry.

`--dataset` offers exactly what is registered, so there is no second list to
keep in step with this one.

## 7. Configuring Model Architectures for a Dataset

To add a new model architecture to an existing dataset or hook up a completely
new dataset:

1. Create a PyTorch model inheriting from `torch.nn.Module`, next to the
   dataset it belongs to.
2. Register it in `MODELS` under the dataset name. The key is what `--model`
   selects, and `"default"` is what it selects when you do not pass one:

   ```python
   from benchmark_core import datasets

   datasets.MODELS["my_dataset_name"] = {
       "default": MyStandardModelClass,
       "experimental": MyNewModelClass,
   }
   ```

3. You can now benchmark this architecture by running
   `--dataset my_dataset_name --model experimental`.

