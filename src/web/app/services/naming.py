"""The vocabulary the interface uses.

"database_reaches" is a good column name and a bad label; "task_status =
running" tells a user nothing about where their job actually is.

Kept server-side so the API, the CSV export and the interface cannot drift into
describing the same state three different ways.
"""

METRIC_LABELS = {
    "gradient_count": {
        "label": "gradient evaluations",
        "short": "gradient evaluations",
        "hint": "Number of gradient evaluations. Increases by 1 for each evaluate_with_grad() and grad().",
    },
    "database_reaches": {
        "label": "processed samples",
        "short": "processed samples",
        "hint": "Number of samples read from the dataset. Increases by batch_size on each forward pass.",
    },
    "final_loss": {
        "label": "final loss",
        "short": "loss",
        "hint": "Loss value in the last recorded epoch.",
    },
    "final_accuracy": {
        "label": "final accuracy",
        "short": "accuracy",
        "hint": "Share of correct classifications in the last recorded epoch.",
    },
    "wall_time_seconds": {
        "label": "wall-clock time",
        "short": "time",
        "hint": "Intentionally excluded from the leaderboard because it depends on the hardware assigned by the scheduler.",
    },
}

STOP_REASONS = {
    "GRADIENT_LIMIT": {
        "label": "gradient evaluation budget exhausted",
        "note": "The budget was exhausted; the optimizer did not report convergence.",
        "converged": False,
    },
    "DATABASE_LIMIT": {
        "label": "processed sample budget exhausted",
        "note": "The budget was exhausted; the optimizer did not report convergence.",
        "converged": False,
    },
    "EPOCH_LIMIT": {
        "label": "epoch budget exhausted",
        "note": "The budget was exhausted; the optimizer did not report convergence.",
        "converged": False,
    },
    "MAX_STEPS": {
        "label": "step budget exhausted",
        "note": "The budget was exhausted; the optimizer did not report convergence.",
        "converged": False,
    },
    "OPTIMIZER_CONVERGED": {
        "label": "optimizer reported convergence",
        "note": "step() returned True before the budget was exhausted.",
        "converged": True,
    },
}

# A job waiting in someone else's scheduler has more states than "loading"
# and "done", and each one owes the user a different sentence.
RUN_STATES = {
    "queued_broker": {
        "label": "queued for publishing",
        "detail": "The submission is waiting to be sent to the cluster.",
        "tone": "pending",
    },
    "queued_slurm": {
        "label": "queued on Athena",
        "detail": "The task has an SLURM job ID and is waiting for a node.",
        "tone": "pending",
    },
    "running": {
        "label": "running on Athena",
        "detail": "The task is using a compute node.",
        "tone": "active",
    },
    "downloading": {
        "label": "downloading results from the cluster",
        "detail": "Computation has finished; artifacts are still being copied.",
        "tone": "active",
    },
    "completed": {
        "label": "completed",
        "detail": "Results and artifacts are complete.",
        "tone": "success",
    },
    "completed_no_artifacts": {
        "label": "completed without artifacts",
        "detail": "Computation succeeded, but the downloader found no files.",
        "tone": "warning",
    },
    "failed": {
        "label": "failed",
        "detail": "The task ended with an error.",
        "tone": "error",
    },
    "failed_no_artifacts": {
        "label": "failed without artifacts",
        "detail": "The task failed and left no files to inspect.",
        "tone": "error",
    },
    "rejected": {
        "label": "rejected during validation",
        "detail": "The submission failed protocol validation and was not queued.",
        "tone": "error",
    },
}


def derive_state(task: dict) -> str:
    """Collapse (task_status, artifact_status) into one state.

    The database stores two coarse fields; the interface owes the user a
    single precise answer, and the mapping belongs in one place.
    """
    status = task.get("task_status")
    artifact = task.get("artifact_status")

    if status == "FAILED":
        if artifact in (None, "absent", "empty"):
            return "failed_no_artifacts"
        return "failed"
    if status == "COMPLETED":
        if artifact == "downloading":
            return "downloading"
        if artifact == "empty":
            return "completed_no_artifacts"
        return "completed"
    if status == "RUNNING":
        return "running"
    if status == "SUBMITTED":
        return "queued_slurm"
    return "queued_broker"


def vocabulary() -> dict:
    return {
        "metrics": METRIC_LABELS,
        "stop_reasons": STOP_REASONS,
        "run_states": RUN_STATES,
    }
