"""Documentation and vocabulary served from the running code.

/docs must not drift from the contract it documents, so the examples are read
from the engine's own source files rather than copied into the page. If someone
changes example_gradient_optimizer.py, the documentation changes with it.
"""

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import PlainTextResponse

from app.services import naming
from app.settings import find_source_root, settings

router = APIRouter(prefix="/api", tags=["meta"])

REPO_ROOT = find_source_root()
PROTOCOLS = REPO_ROOT / "src/benchmark_core/optimization_engine/optimizer_protocols"

EXAMPLES = {
    "gradient": PROTOCOLS / "example_gradient_optimizer.py",
    "evolutionary": PROTOCOLS / "example_evolutionary_optimizer.py",
    "protocol": PROTOCOLS / "benchmarkable_optimizer.py",
    "base": PROTOCOLS / "numpy_benchmark_optimizer.py",
}

# The counter column is the part a participant cannot get anywhere else, and
# the part that decides whether their optimizer is spending budget it did not
# mean to spend.
EVALUATOR_API = [
    {
        "method": "evaluate()",
        "returns": "float",
        "effect": "database_reaches += batch_size",
        "note": "Forward pass. Gradient-free methods use only this.",
    },
    {
        "method": "evaluate_with_grad()",
        "returns": "(float, ndarray)",
        "effect": "database_reaches += batch_size, gradient_count += 1",
        "note": "Forward and backward pass.",
    },
    {
        "method": "grad()",
        "returns": "ndarray",
        "effect": "database_reaches += batch_size, gradient_count += 1",
        "note": "Gradient only; does not return the loss.",
    },
    {"method": "get_params()", "returns": "ndarray", "effect": "N/A", "note": "Flattened parameter vector."},
    {
        "method": "set_params(params)",
        "returns": "None",
        "effect": "N/A",
        "note": "Writes parameters back to the model.",
    },
    {
        "method": "get_predictions()",
        "returns": "(preds, targets)",
        "effect": "database_reaches += batch_size",
        "note": "Predictions and labels.",
    },
    {"method": "batch_size", "returns": "int", "effect": "N/A", "note": "Property. Current batch size."},
    {"method": "param_count", "returns": "int", "effect": "N/A", "note": "Property. Number of model parameters."},
]

SANDBOX = {
    "network": "--network none",
    "memory": "--memory 2g",
    "cpu": "--cpus 1.0",
    "filesystem": "--read-only, --tmpfs /tmp",
        "user": "unprivileged, --cap-drop ALL, no-new-privileges",
    "timeout": f"{settings.validator_timeout} s",
}


@router.get("/vocabulary")
async def vocabulary() -> dict:
    return naming.vocabulary()


@router.get("/protocol")
async def protocol() -> dict:
    available = {}
    for key, path in EXAMPLES.items():
        available[key] = {
            "filename": path.name,
            "present": path.is_file(),
            "repo_path": (str(path.relative_to(REPO_ROOT)) if path.is_file() else None),
        }
    return {
        "evaluator_api": EVALUATOR_API,
        "examples": available,
        "sandbox": SANDBOX,
        "stop_reasons": naming.STOP_REASONS,
    }


@router.get("/protocol/example/{name}", response_class=PlainTextResponse)
async def protocol_example(name: str) -> PlainTextResponse:
    path = EXAMPLES.get(name)
    if path is None or not path.is_file():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown example")
    return PlainTextResponse(
        path.read_text(encoding="utf-8"),
        media_type="text/plain; charset=utf-8",
        headers={"X-Content-Type-Options": "nosniff"},
    )


_TEMPLATE = '''"""Optimizer template for Opti-Mark (The Golem Benchmark).

You write only the optimizer mathematics. You see neither the model nor the dataset:
the network is reduced to a flat parameter vector and objective function.

The evaluator tracks budget counters, not you. Each evaluate() call costs
batch_size samples, and each evaluate_with_grad() additionally costs one gradient evaluation.
"""

import numpy as np

from benchmark_core.optimization_engine.evaluator import ModelEvaluator
from benchmark_core.optimization_engine.optimizer_protocols import NumpyBenchmarkOptimizer


class MyOptimizer(NumpyBenchmarkOptimizer):

    def __init__(self, initial_params, lr: float = 0.01, **config):
        # The first positional argument MUST be named initial_params.
        super().__init__(initial_params, **config)
        self.params = np.asarray(initial_params, dtype=np.float64)
        self.lr = lr

    def step(self, evaluator: ModelEvaluator) -> bool:
        """One optimization step.

        Return True when you consider the optimizer converged and want to stop.
        Return False to let the harness continue until the budget is exhausted.
        """
        loss, gradient = evaluator.evaluate_with_grad()

        self.params = self.params - self.lr * np.asarray(gradient)
        evaluator.set_params(self.params)

        return False
'''


@router.get("/protocol/template", response_class=PlainTextResponse)
async def protocol_template() -> PlainTextResponse:
    return PlainTextResponse(
        _TEMPLATE,
        media_type="text/plain; charset=utf-8",
        headers={
            "Content-Disposition": 'attachment; filename="optimizer_template.py"',
            "X-Content-Type-Options": "nosniff",
        },
    )
