set dotenv-load := false

export PYTHONPATH := "src"

dataset := "ds01"
optimizer := "adam"
epochs := "5"
model := "light"

# Default recipe: list available commands
default:
    @just --list

# Run the benchmark CLI (e.g. `just bench`, `just optimizer=sgd bench`, `just epochs=1 bench --verbose-ipc`)
bench *args="":
    uv run -m benchmark_core.optimization_engine.run_benchmark \
        --dataset {{dataset}} \
        --model {{model}} \
        --optimizer {{optimizer}} \
        --max-epochs {{epochs}} \
        {{args}}

# Compare multiple built-in optimizers and generate plots in reports/
bench-compare *args="":
    uv run -m benchmark_core.optimization_engine.run_benchmark \
        --dataset {{dataset}} \
        --model {{model}} \
        --optimizer adam sgd rmsprop lion \
        --max-epochs {{epochs}} \
        --plot \
        {{args}}

# Validate a custom optimizer file against the benchmark protocol
validate optimizer_path:
    uv run python src/benchmark_core/optimization_engine/optimizers/validation/verify_optimizer.py {{optimizer_path}}

# Options (passed to scripts/encrypt_datasets.py, see `just encrypt --help`):
#   --private DIR   private folder with csv/, prepare/, manifest.toml, key (default: data/private)
#   --gen-key       create a new key first; --force allows replacing an existing one
#   --verify        decrypt each output and compare it with the input
#   --only ID...    encrypt only these dataset ids
# Preprocess and encrypt the private datasets (e.g. `just encrypt --gen-key --verify`, `just encrypt --only ds05`)
encrypt *args="":
    uv run python scripts/encrypt_datasets.py {{args}}

# Put the encrypted datasets and their key into the package (needed to run on them and before compiling):
# checks data/private/key against every data/private/encrypted/*.bin, copies the files to
# src/benchmark_core/datasets/data/ and writes src/benchmark_core/_bundled_key.py (`just bundle --remove` undoes it)
bundle *args="":
    uv run python scripts/bundle_data.py {{args}}

# Run unit tests
test *args="":
    uv run --with pytest --with pytest-asyncio pytest {{args}}

# Lint codebase with Ruff
lint:
    uv run --with ruff ruff check src

# Format codebase with Ruff
fmt:
    uv run --with ruff ruff format src

# Start Docker Compose stack in the background
up *services="":
    docker compose up -d --build {{services}}

# Stop Docker Compose stack
down:
    docker compose down

# Tail Docker Compose logs
logs *services="":
    docker compose logs -f {{services}}

# Clean generated benchmark reports and Python cache directories
clean:
    rm -rf reports/
    find . -type d -name "__pycache__" -prune -exec rm -rf {} +
