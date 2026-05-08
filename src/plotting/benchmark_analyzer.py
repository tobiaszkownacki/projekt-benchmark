import csv
from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from src.benchmark.runner import BenchmarkResult


class BenchmarkAnalyzer:
    """
    Creates plots from BenchmarkResult objects and benchmark CSV logs

    example run:
    python -m src.benchmark.run_benchmark --dataset digits --optimizer sgd --max-epochs 1 --max-gradients 100 --plot
    """

    def __init__(self, output_dir: str = "reports/model_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_results(self, results: Dict[str, BenchmarkResult]) -> Path:
        run_dir = Path("reports/runs") / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        n = len(results)
        cmap = plt.colormaps["plasma"]
        colors = [cmap(i / max(n - 1, 1)) for i in range(n)]

        series_plots = [
            ("loss_vs_epoch.png",      "Loss vs Epoch",       "Loss",         "Epoch",
             lambda r: list(range(1, len(r.loss_history) + 1)), lambda r: r.loss_history),

            ("loss_vs_db_reaches.png", "Loss vs DB Reaches",  "Loss",         "DB Reaches",
             lambda r: r.database_reaches_history,              lambda r: r.loss_history),

            ("loss_vs_grads.png",      "Loss vs Gradients",   "Loss",         "Gradients",
             lambda r: r.gradient_history,                      lambda r: r.loss_history),


            ("acc_vs_db_reaches.png",  "Accuracy vs DB Reaches", "Accuracy (%)", "DB Reaches",
             lambda r: r.database_reaches_history,              lambda r: r.accuracy_history),

            ("acc_vs_grads.png",       "Accuracy vs Gradients",  "Accuracy (%)", "Gradients",
             lambda r: r.gradient_history,                      lambda r: r.accuracy_history),

            ("acc_vs_epoch.png",       "Accuracy vs Epoch",     "Accuracy (%)", "Epoch",
             lambda r: list(range(1, len(r.accuracy_history) + 1)), lambda r: r.accuracy_history),
        ]

        # (filename, title, ylabel, value_getter)
        bar_plots = [
            ("total_gradients.png",  "Total Gradient Evaluations", "Gradients",       lambda r: r.gradient_count),
            ("total_db_reaches.png", "Total Database Reaches",     "Samples Processed", lambda r: r.database_reaches),
        ]

        with plt.style.context("dark_background"):
            for filename, title, ylabel, xlabel, x_getter, y_getter in series_plots:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                ax.set_xlabel(xlabel)
                for (name, result), color in zip(results.items(), colors):
                    x = x_getter(result)
                    y = y_getter(result)
                    if x and y:
                        ax.plot(x, y, marker="o", markersize=4, label=name, color=color)
                ax.legend()
                ax.grid(alpha=0.2)
                plt.tight_layout()
                plt.savefig(run_dir / filename, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
                plt.close(fig)

            for filename, title, ylabel, value_getter in bar_plots:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.set_title(title)
                ax.set_ylabel(ylabel)
                names = list(results.keys())
                values = [value_getter(r) for r in results.values()]
                ax.bar(names, values, color=colors)
                ax.grid(alpha=0.2, axis="y")
                plt.tight_layout()
                plt.savefig(run_dir / filename, dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
                plt.close(fig)

        print(f"Plots saved to: {run_dir}")
        return run_dir







    # def plot_single_result(self, result: BenchmarkResult) -> Path:
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     out_path = self.output_dir / (
    #         f"benchmark_curve_{result.dataset_name}_{result.optimizer_name}_{timestamp}.png"
    #     )
    #
    #     fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(14, 5))
    #     fig.suptitle(
    #         f"{result.optimizer_name} on {result.dataset_name} "
    #         f"(stop={result.stop_reason.name})"
    #     )
    #
    #     epochs = list(range(1, len(result.loss_history) + 1))
    #     if epochs:
    #         ax_loss.plot(epochs, result.loss_history, color="#1f77b4", marker="o")
    #         ax_acc.plot(epochs, result.accuracy_history, color="#2ca02c", marker="o")
    #     else:
    #         ax_loss.text(0.5, 0.5, "No epoch history", ha="center", va="center")
    #         ax_acc.text(0.5, 0.5, "No epoch history", ha="center", va="center")
    #
    #     ax_loss.set_title("Loss per Epoch")
    #     ax_loss.set_xlabel("Epoch")
    #     ax_loss.set_ylabel("Loss")
    #     ax_loss.grid(alpha=0.3)
    #
    #     ax_acc.set_title("Accuracy per Epoch")
    #     ax_acc.set_xlabel("Epoch")
    #     ax_acc.set_ylabel("Accuracy (%)")
    #     ax_acc.grid(alpha=0.3)
    #
    #     plt.tight_layout()
    #     plt.savefig(out_path, dpi=200, bbox_inches="tight")
    #     plt.close(fig)
    #     return out_path
    #
    # def plot_comparison(self, results: Dict[str, BenchmarkResult], dataset_name: str) -> Path:
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     out_path = self.output_dir / f"benchmark_comparison_{dataset_name}_{timestamp}.png"
    #
    #     names = list(results.keys())
    #     losses = [results[name].final_loss for name in names]
    #     accs = [results[name].final_accuracy for name in names]
    #     grads = [results[name].gradient_count for name in names]
    #     db_reaches = [results[name].database_reaches for name in names]
    #
    #     fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    #     fig.suptitle(f"Benchmark Comparison: {dataset_name}")
    #
    #     axes[0, 0].bar(names, losses, color="#1f77b4")
    #     axes[0, 0].set_title("Final Loss")
    #     axes[0, 0].set_ylabel("Loss")
    #
    #     axes[0, 1].bar(names, accs, color="#2ca02c")
    #     axes[0, 1].set_title("Final Accuracy")
    #     axes[0, 1].set_ylabel("Accuracy (%)")
    #
    #     axes[1, 0].bar(names, grads, color="#ff7f0e")
    #     axes[1, 0].set_title("Gradient Count")
    #     axes[1, 0].set_ylabel("Gradients")
    #
    #     axes[1, 1].bar(names, db_reaches, color="#9467bd")
    #     axes[1, 1].set_title("Database Reaches")
    #     axes[1, 1].set_ylabel("Samples Processed")
    #
    #     for ax in axes.flat:
    #         ax.tick_params(axis="x", rotation=20)
    #         ax.grid(alpha=0.2, axis="y")
    #
    #     plt.tight_layout()
    #     plt.savefig(out_path, dpi=200, bbox_inches="tight")
    #     plt.close(fig)
    #     return out_path
    #
    # def plot_log_file(self, csv_path: str) -> Path:
    #     path = Path(csv_path)
    #     if not path.exists():
    #         raise FileNotFoundError(f"Log file does not exist: {path}")
    #
    #     mini_batches = []
    #     losses = []
    #     gradients = []
    #     samples = []
    #
    #     with open(path, mode="r", newline="") as handle:
    #         reader = csv.DictReader(handle)
    #         for row in reader:
    #             mini_batches.append(int(row["mini_batches"]))
    #             losses.append(float(row["train_loss"]))
    #             gradients.append(int(row["gradients"]))
    #             samples.append(int(row["samples"]))
    #
    #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    #     out_path = self.output_dir / f"benchmark_log_plot_{path.stem}_{timestamp}.png"
    #
    #     fig, (ax_loss, ax_counters) = plt.subplots(1, 2, figsize=(14, 5))
    #
    #     ax_loss.plot(mini_batches, losses, color="#1f77b4")
    #     ax_loss.set_title("Train Loss by Mini-batch")
    #     ax_loss.set_xlabel("Mini-batch")
    #     ax_loss.set_ylabel("Train Loss")
    #     ax_loss.grid(alpha=0.3)
    #
    #     ax_counters.plot(mini_batches, samples, label="Samples", color="#2ca02c")
    #     ax_counters.plot(mini_batches, gradients, label="Gradients", color="#ff7f0e")
    #     ax_counters.set_title("Counters by Mini-batch")
    #     ax_counters.set_xlabel("Mini-batch")
    #     ax_counters.set_ylabel("Count")
    #     ax_counters.legend()
    #     ax_counters.grid(alpha=0.3)
    #
    #     plt.tight_layout()
    #     plt.savefig(out_path, dpi=200, bbox_inches="tight")
    #     plt.close(fig)
    #     return out_path
