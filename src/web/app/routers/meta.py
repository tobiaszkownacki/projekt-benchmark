"""Documentation and vocabulary served from the running code.

/docs must not drift from the contract it documents, so the examples are read
from the engine's own source files rather than copied into the page. If someone
changes example_gradient_optimizer.py, the documentation changes with it.
"""

from pathlib import Path

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import PlainTextResponse

from app.services import naming
from app.settings import settings

router = APIRouter(prefix="/api", tags=["meta"])

REPO_ROOT = Path(__file__).resolve().parents[4]
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
    {"method": "evaluate()", "returns": "float",
     "effect": "database_reaches += batch_size",
     "note": "Przejście w przód. Metody bezgradientowe używają tylko tego."},
    {"method": "evaluate_with_grad()", "returns": "(float, ndarray)",
     "effect": "database_reaches += batch_size, gradient_count += 1",
     "note": "Przejście w przód i w tył."},
    {"method": "grad()", "returns": "ndarray",
     "effect": "database_reaches += batch_size, gradient_count += 1",
     "note": "Sam gradient, bez zwracania straty."},
    {"method": "get_params()", "returns": "ndarray",
     "effect": "—", "note": "Spłaszczony wektor parametrów."},
    {"method": "set_params(params)", "returns": "None",
     "effect": "—", "note": "Zapisuje parametry z powrotem do modelu."},
    {"method": "get_predictions()", "returns": "(preds, targets)",
     "effect": "database_reaches += batch_size", "note": "Predykcje i etykiety."},
    {"method": "batch_size", "returns": "int",
     "effect": "—", "note": "Właściwość. Rozmiar bieżącej paczki."},
    {"method": "param_count", "returns": "int",
     "effect": "—", "note": "Właściwość. Liczba parametrów modelu."},
]

# Documented rather than hidden. These are limitations of the contract a
# participant is asked to write against, and finding them by trial and error
# during a competition would be worse for everyone than reading them here.
KNOWN_LIMITATIONS = [
    {"title": "step() nie wie, ile budżetu zostało",
     "detail": "Pętlę zatrzymuje harness. Optymalizator ewolucyjny, który "
               "chciałby chłodzić sigmę w funkcji pozostałego budżetu, nie ma "
               "jak. Standardowe suity (CEC, COCO) wystawiają remaining_budget."},
    {"title": "Brak evaluate_population()",
     "detail": "Metoda populacyjna z lambda=20 wykonuje w jednym step() "
               "dwadzieścia osobnych przejść w przód zamiast jednego wsadowego. "
               "Podatek płaci dokładnie ta rodzina metod, którą projekt bada."},
    {"title": "Losowość nie jest kontrolowana",
     "detail": "Przykłady używają globalnego np.random bez ziarna. Harness "
               "powinien wstrzykiwać zaziarnowany generator do konstruktora."},
    {"title": "get_output_type() bez @classmethod",
     "detail": "W klasie bazowej zadeklarowana bez self i bez dekoratora. "
               "Wywołanie na instancji rozjedzie się na liczbie argumentów."},
    {"title": "Dwie różne definicje StopReason",
     "detail": "runner.py i metrics/stop_metrics.py deklarują enum o tej samej "
               "nazwie i różnych wartościach. Baza podąża za tym, który trafia "
               "do wyniku."},
]

SANDBOX = {
    "network": "--network none",
    "memory": "--memory 2g",
    "cpu": "--cpus 1.0",
    "filesystem": "--read-only, --tmpfs /tmp",
    "user": "nieuprzywilejowany, --cap-drop ALL, no-new-privileges",
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
            "repo_path": str(path.relative_to(REPO_ROOT)) if path.is_file() else None,
        }
    return {
        "evaluator_api": EVALUATOR_API,
        "examples": available,
        "known_limitations": KNOWN_LIMITATIONS,
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


_TEMPLATE = '''"""Szablon optymalizatora dla Benchmarku Czarnej Skrzynki.

Piszesz wyłącznie matematykę optymalizatora. Nie widzisz ani modelu, ani zbioru
danych: sieć jest sprowadzona do płaskiego wektora parametrow i funkcji celu.

Liczniki budżetu prowadzi ewaluator, nie Ty. Każde wywołanie evaluate() kosztuje
batch_size próbek, każde evaluate_with_grad() dodatkowo jeden gradient.
"""

import numpy as np

from benchmark.evaluator import ModelEvaluator
from benchmark.optimizer_protocols import NumpyBenchmarkOptimizer


class MyOptimizer(NumpyBenchmarkOptimizer):

    def __init__(self, initial_params, lr: float = 0.01, **config):
        # Pierwszy argument pozycyjny MUSI nazywać się initial_params.
        super().__init__(initial_params, **config)
        self.params = np.asarray(initial_params, dtype=np.float64)
        self.lr = lr

    def step(self, evaluator: ModelEvaluator) -> bool:
        """Jeden krok optymalizacji.

        Zwróć True, jeśli uznajesz, że osiągnąłeś zbieżność i chcesz zakończyć.
        Zwróć False, żeby harness kontynuował aż do wyczerpania budżetu.
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
