"""Writes the artifact directory a finished run leaves behind.

Shape follows what BenchmarkRunner and AthenaDownloader actually produce --
reports/ with the analyzer's plots, the run log as CSV, the SLURM stdout file --
plus the two additions §12.1 recommends: metadata.json, so the directory
describes itself, and a copy of the submitted optimizer, so an artifact can be
tied to the exact code that produced it.
"""

import csv
import json
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Okabe-Ito. Chosen over the engine's current 20-colour neon palette because
# these figures end up in papers: this set stays distinguishable for the common
# forms of colour blindness, and line style carries the same information again
# so nothing is lost in greyscale print.
OKABE_ITO = [
    "#000000", "#E69F00", "#56B4E9", "#009E73",
    "#0072B2", "#D55E00", "#CC79A7", "#F0E442",
]
LINE_STYLES = ["-", "--", "-.", ":"]


def _plot(path: Path, x, y, title: str, xlabel: str, ylabel: str, colour: str) -> None:
    figure, axis = plt.subplots(figsize=(6.0, 3.6), dpi=140)
    axis.plot(x, y, color=colour, linewidth=1.6, linestyle=LINE_STYLES[0])
    axis.set_title(title, fontsize=10)
    axis.set_xlabel(xlabel, fontsize=9)
    axis.set_ylabel(ylabel, fontsize=9)
    axis.grid(True, linewidth=0.4, alpha=0.4)
    axis.tick_params(labelsize=8)
    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def write_run_artifacts(
    root: Path,
    result,
    optimizer_source: Optional[str] = None,
    slurm_job_id: Optional[str] = None,
    extra_metadata: Optional[dict] = None,
) -> tuple[int, int]:
    """Write one run's directory. Returns (file_count, total_bytes)."""
    reports = root / "reports"
    logs = root / "logs"
    data = root / "data"
    for directory in (reports, logs, data):
        directory.mkdir(parents=True, exist_ok=True)

    colour = OKABE_ITO[1]
    epochs = result.epoch_history or list(range(1, len(result.loss_history) + 1))

    # A gradient-free method never calls evaluate_with_grad(), so its gradient
    # counter stays at zero for the whole run and a "loss against gradients"
    # figure collapses to a vertical line at x=0. Drawing it anyway produces an
    # artifact that looks like a broken plot rather than like the correct answer,
    # so those two figures are omitted and metadata.json records why.
    uses_gradients = bool(result.gradient_history) and result.gradient_history[-1] > 0

    if result.loss_history:
        _plot(reports / "loss_vs_epoch.png", epochs, result.loss_history,
              "Strata wg epok", "Epoka", "Strata", colour)
        # The axis on which a gradient method and a population method can
        # actually be compared.
        _plot(reports / "loss_vs_db_reaches.png", result.database_reaches_history,
              result.loss_history, "Strata wg liczby próbek",
              "Przetworzone próbki", "Strata", colour)
        if uses_gradients:
            _plot(reports / "loss_vs_grads.png", result.gradient_history,
                  result.loss_history, "Strata wg liczby gradientów",
                  "Wyliczone gradienty", "Strata", colour)
    if result.accuracy_history:
        _plot(reports / "acc_vs_epoch.png", epochs, result.accuracy_history,
              "Dokładność wg epok", "Epoka", "Dokładność [%]", OKABE_ITO[3])
        _plot(reports / "acc_vs_db_reaches.png", result.database_reaches_history,
              result.accuracy_history, "Dokładność wg liczby próbek",
              "Przetworzone próbki", "Dokładność [%]", OKABE_ITO[3])
        if uses_gradients:
            _plot(reports / "acc_vs_grads.png", result.gradient_history,
                  result.accuracy_history, "Dokładność wg liczby gradientów",
                  "Wyliczone gradienty", "Dokładność [%]", OKABE_ITO[3])

    csv_name = f"benchmark-{result.optimizer_name}-{result.dataset_name}.csv"
    with open(data / csv_name, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "loss", "accuracy", "gradients", "samples", "seconds"])
        for index in range(len(result.loss_history)):
            writer.writerow([
                epochs[index],
                round(result.loss_history[index], 6),
                round(result.accuracy_history[index], 4),
                result.gradient_history[index],
                result.database_reaches_history[index],
                round(result.time_history[index], 4),
            ])

    job_id = slurm_job_id or "local"
    (logs / f"{job_id}.out").write_text(result.stdout, encoding="utf-8")

    metadata = {
        "optimizer": result.optimizer_name,
        "dataset": result.dataset_name,
        "model": result.model_name,
        "seed": result.seed,
        "parameter_count": result.param_count,
        "stop_reason": result.stop_reason,
        "final_loss": result.final_loss,
        "final_accuracy": result.final_accuracy,
        "gradient_count": result.gradient_count,
        "database_reaches": result.database_reaches,
        "total_steps": result.total_steps,
        "total_epochs": result.total_epochs,
        "wall_time_seconds": result.wall_time_seconds,
        "uses_gradients": uses_gradients,
        "omitted_plots": [] if uses_gradients else [
            "loss_vs_grads.png", "acc_vs_grads.png",
        ],
        "omitted_plots_reason": None if uses_gradients else (
            "Optymalizator bezgradientowy — licznik gradientów pozostaje zerowy, "
            "więc wykres w funkcji gradientów nie niesie informacji."
        ),
        **(extra_metadata or {}),
    }
    (root / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    if optimizer_source:
        (root / "optimizer.py").write_text(optimizer_source, encoding="utf-8")

    files = [p for p in root.rglob("*") if p.is_file()]
    return len(files), sum(p.stat().st_size for p in files)
