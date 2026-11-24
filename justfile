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

# Preprocess the raw datasets in data/private/ into src/benchmark_core/datasets/data/ (e.g. `just prepare --only ds05`)
prepare *args="":
    uv run python scripts/prepare_datasets.py {{args}}

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
