# Optimizer Benchmark Suite [![Flake8 Linting](https://github.com/Dnafivuq/golem_template/actions/workflows/lint.yml/badge.svg)](https://github.com/Dnafivuq/golem_template/actions/workflows/lint.yml) [![Pytest](https://github.com/Dnafivuq/golem_template/actions/workflows/test.yml/badge.svg)](https://github.com/Dnafivuq/golem_template/actions/workflows/test.yml) <a target="_blank" href="https://cookiecutter-data-science.drivendata.org/"><img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" /></a>

## 1. Overview

This repository contains a benchmark suite for testing, comparing, and analyzing various optimization algorithms (e.g., Adam, SGD, CMA-ES, Differential Evolution) on different datasets and neural network models using PyTorch. The framework tracks metrics like gradient evaluations, database reaches, and standard loss/accuracy over time, providing a comprehensive toolkit to evaluate the efficiency and convergence of both gradient-based and gradient-free optimizers. It is designed to be modular and independent dependency-wise, which yields simple addition of more problem (dataset-model) definitions and custom optimization algorithms.

## 2. Installation & Setup

To get started with this project, follow these steps:  

1. **Clone the repository**  

   ```sh
   git clone <repository-url>
   cd projekt-benchmark
   ```

2. **Set up a virtual environment**

   ```sh
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**  

   ```sh
   pip install -r requirements.txt
   ```

## 3. Running the Benchmark

You can run benchmarks using the `src.benchmark.run_benchmark` module.

### Running a single optimizer

```sh
python -m src.benchmark.run_benchmark --dataset digits --optimizer adam
```

### Comparing multiple optimizers

```sh
python -m src.benchmark.run_benchmark --dataset digits --compare adam sgd cma-es
```

### Generating plots for your run

Add the `--plot` flag to generate comparison and performance plots. By default, they are saved to `reports/model_analysis`.

```sh
python -m src.benchmark.run_benchmark --dataset digits --optimizer sgd --max-epochs 10 --max-gradients 100000 --plot
```

### Available Arguments/Parameters

- `--dataset`: Choose from `cifar10`, `heart_disease`, `wine_quality`, `digits` (required).
- `--optimizer`: Name of a built-in optimizer (e.g., `adam`, `sgd`, `cma-es`) or a file path to a custom optimizer python script.
- `--compare`: Provide a list of built-in optimizer names to compare them side-by-side.
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
├── data/                  <- Raw and processed datasets.
├── models/                <- PyTorch model definitions for each dataset.
├── outputs/               <- Benchmark run output logs and CSVs.
├── reports/               <- Generated plots and analysis artifacts.
├── src/                   <- Source code for use in this project.
│   ├── benchmark/         <- Core logic: evaluator, runner, stop conditions.
│   │   └── optimizers/    <- Optimizer adapters (Adam, SGD, CMA-ES, etc.) and registry.
│   ├── datasets/          <- Dataset loading and preprocessing scripts.
│   ├── metrics/           <- Stop metrics and evaluation tracking.
│   ├── plotting/          <- Plot generation and analyzer modules.
│   ├── trainers/          <- Internal model training loops and protocols.
│   ├── config.py          <- Global configuration and variables.
│   └── dataset.py         <- Factory mapping datasets to their PyTorch models.
├── requirements.txt       <- The requirements file for environment reproduction.
└── setup.cfg              <- Configuration file for flake8 and pytest.
```

## 5. Adding a New Optimizer

1. Create a new python script inside `src/benchmark/optimizers/` (e.g., `my_optimizer_adapter.py`).
2. Create your optimizer class inheriting from `src.benchmark.optimizer_protocol.BenchmarkOptimizer`.
3. Implement the `step(self, evaluator: ModelEvaluator) -> bool` method.
    - Inside `step()`, you can call `evaluator.evaluate_with_grad()` or `evaluator.evaluate()` depending on whether your optimizer needs gradients.
    - Update `self.params` and finally call `evaluator.set_params(self.params)`.
    - Return `True` if the optimizer has converged, `False` otherwise.
4. Add your new optimizer to the `BUILTIN_OPTIMIZERS` registry located in `src/benchmark/optimizers/registry.py`.
    - *Alternatively, you can test it directly without registering by passing the path to the file using `--optimizer path/to/my_optimizer_adapter.py`*.

## 6. Adding a New Dataset

1. Create a new python file in `src/datasets/` (e.g., `my_dataset.py`).
2. Implement a dataset class that inherits from `src.datasets.dataset.Dataset`.
3. Implement the `get(self) -> ConcatDataset` abstract method to download/load and preprocess your data, returning it as a PyTorch dataset.
4. Add your dataset name to the `ALLOWED_DATASETS` list in `src/config.py`.

## 7. Configuring a Model with a Dataset

To hook up your new dataset with a model to be evaluated:

1. Create a PyTorch model inheriting from `torch.nn.Module` in the `models/` directory (e.g., `models/my_model.py`).
2. Open `src/dataset.py` and import your new dataset class and your new model class.
3. Update `DataSetFactory.get_data_set` to support returning your new dataset.
4. Add a new entry to the `DATA_SETS` dictionary in `src/dataset.py`. Map the identifier name to the initialized dataset and the model class:

   ```python
   "my_dataset_name": {
       "data_set": lambda: DataSetFactory.get_data_set("my_dataset_name"),
       "model": MyModelClass,
   }
   ```

