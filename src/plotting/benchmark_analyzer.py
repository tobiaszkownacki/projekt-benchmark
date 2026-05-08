from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from src.benchmark.runner import BenchmarkResult


_DARK_BG_PALETTE = [
    "#FF4444",  # red
    "#FF8800",  # orange
    "#FFDD00",  # yellow
    "#44FF44",  # green
    "#00FFCC",  # teal
    "#00CCFF",  # sky blue
    "#CC44FF",  # violet
    "#FF44CC",  # magenta
    "#FF6B6B",  # coral
    "#FFB347",  # peach
    "#AAFF00",  # chartreuse
    "#00FF99",  # mint
    "#40E0D0",  # turquoise
    "#66BBFF",  # light blue
    "#FF91A4",  # pink
    "#FFE066",  # light yellow
    "#7FFF00",  # lime
    "#FF007F",  # rose
    "#E066FF",  # orchid
    "#00E5FF",  # cyan
]


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
        colors = _DARK_BG_PALETTE[:n]

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


