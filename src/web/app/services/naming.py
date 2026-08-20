"""The vocabulary the interface uses.

§14.6 asks for the researcher's words rather than the implementation's.
"database_reaches" is a good column name and a bad label; "task_status =
running" tells a user nothing about where their job actually is.

Kept server-side so the API, the CSV export and the interface cannot drift into
describing the same state three different ways.
"""

METRIC_LABELS = {
    "gradient_count": {
        "label": "wyliczone gradienty",
        "short": "gradienty",
        "hint": "Liczba obliczeń gradientu. Rośnie o 1 przy każdym "
                "evaluate_with_grad() i grad().",
    },
    "database_reaches": {
        "label": "przetworzone próbki",
        "short": "próbki",
        "hint": "Liczba próbek pobranych ze zbioru danych. Rośnie o batch_size "
                "przy każdym przejściu w przód.",
    },
    "final_loss": {
        "label": "strata końcowa",
        "short": "strata",
        "hint": "Wartość funkcji straty w ostatniej zapisanej epoce.",
    },
    "final_accuracy": {
        "label": "dokładność końcowa",
        "short": "dokładność",
        "hint": "Odsetek poprawnych klasyfikacji w ostatniej zapisanej epoce.",
    },
    "wall_time_seconds": {
        "label": "czas zegarowy",
        "short": "czas",
        "hint": "Świadomie nieużywany do rankingu — zależy od tego, jaki sprzęt "
                "akurat przydzielił scheduler.",
    },
}

STOP_REASONS = {
    "GRADIENT_LIMIT": {
        "label": "wyczerpany limit gradientów",
        "note": "Budżet się skończył, optymalizator nie zgłosił zbieżności.",
        "converged": False,
    },
    "DATABASE_LIMIT": {
        "label": "wyczerpany limit próbek",
        "note": "Budżet się skończył, optymalizator nie zgłosił zbieżności.",
        "converged": False,
    },
    "EPOCH_LIMIT": {
        "label": "wyczerpany limit epok",
        "note": "Budżet się skończył, optymalizator nie zgłosił zbieżności.",
        "converged": False,
    },
    "MAX_STEPS": {
        "label": "wyczerpany limit kroków",
        "note": "Budżet się skończył, optymalizator nie zgłosił zbieżności.",
        "converged": False,
    },
    "OPTIMIZER_CONVERGED": {
        "label": "optymalizator zgłosił zbieżność",
        "note": "step() zwrócił True przed wyczerpaniem budżetu.",
        "converged": True,
    },
}

# §11.3: a job waiting in someone else's scheduler has more states than
# "loading" and "done", and each one owes the user a different sentence.
RUN_STATES = {
    "queued_broker": {
        "label": "w kolejce systemu",
        "detail": "Zgłoszenie czeka na wysłanie do klastra.",
        "tone": "pending",
    },
    "queued_slurm": {
        "label": "w kolejce na Atenie",
        "detail": "Zadanie ma numer w SLURM i czeka na węzeł.",
        "tone": "pending",
    },
    "running": {
        "label": "liczy się na Atenie",
        "detail": "Zadanie zajmuje węzeł obliczeniowy.",
        "tone": "active",
    },
    "downloading": {
        "label": "pobieranie wyników z klastra",
        "detail": "Obliczenia się skończyły, artefakty jeszcze się kopiują.",
        "tone": "active",
    },
    "completed": {
        "label": "zakończone",
        "detail": "Wyniki i artefakty są kompletne.",
        "tone": "success",
    },
    "completed_no_artifacts": {
        "label": "zakończone — brak artefaktów",
        "detail": "Obliczenia się powiodły, ale downloader nie znalazł plików.",
        "tone": "warning",
    },
    "failed": {
        "label": "nieudane",
        "detail": "Zadanie zakończyło się błędem.",
        "tone": "error",
    },
    "failed_no_artifacts": {
        "label": "nieudane — brak artefaktów",
        "detail": "Zadanie zawiodło i nie zostawiło plików do obejrzenia.",
        "tone": "error",
    },
    "rejected": {
        "label": "odrzucone przy walidacji",
        "detail": "Zgłoszenie nie przeszło kontroli protokołu i nie trafiło do kolejki.",
        "tone": "error",
    },
}


def derive_state(task: dict) -> str:
    """Collapse (task_status, artifact_status, executor_task_id) into one state.

    The database stores three coarse fields; the interface owes the user a
    single precise answer, and the mapping belongs in one place.
    """
    status = task.get("task_status")
    artifact = task.get("artifact_status")

    if status == "failed":
        if artifact in (None, "absent", "empty"):
            return "failed_no_artifacts"
        return "failed"
    if status == "completed":
        if artifact == "downloading":
            return "downloading"
        if artifact == "empty":
            return "completed_no_artifacts"
        return "completed"
    if status == "running":
        return "running"
    if status == "pending":
        if task.get("executor_task_id"):
            return "queued_slurm"
        return "queued_broker"
    return "queued_broker"


def vocabulary() -> dict:
    return {
        "metrics": METRIC_LABELS,
        "stop_reasons": STOP_REASONS,
        "run_states": RUN_STATES,
    }
