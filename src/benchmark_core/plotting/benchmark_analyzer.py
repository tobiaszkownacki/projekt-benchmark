from datetime import datetime
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt

from src.benchmark.runner import BenchmarkResult


# Okabe-Ito, the standard colour-blind-safe qualitative set.
#
# Replaces a 20-colour neon palette drawn on a black background. These figures
# end up in papers and in front of reviewers, and the previous set had two
# problems for that audience: several of the colours were indistinguishable
# under the common forms of colour blindness, and on a light page a saturated
# neon on black reads as a games console rather than as a result.
#
# Eight colours is a real constraint. Beyond eight series a line chart stops
# being readable regardless of palette, so the colours cycle and the line style
# changes with each cycle.
_OKABE_ITO_PALETTE = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

# Series are distinguished twice: by colour and by dash pattern. Colour is what
# disappears when a figure is printed in black and white, which is exactly where
# these plots are going.
_LINE_STYLES = ["-", "--", "-.", ":"]


def _series_style(index: int) -> tuple:
    """Colour and dash pattern for the n-th series."""
    colour = _OKABE_ITO_PALETTE[index % len(_OKABE_ITO_PALETTE)]
    style = _LINE_STYLES[(index // len(_OKABE_ITO_PALETTE)) % len(_LINE_STYLES)]
    return colour, style


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
        run_dir = self.output_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        styles = [_series_style(i) for i in range(len(results))]
        colors = [colour for colour, _ in styles]

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

        # Default (light) style rather than dark_background: the plots are read
        # on a light page and printed on white paper.
        for filename, title, ylabel, xlabel, x_getter, y_getter in series_plots:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel(xlabel)
            for (name, result), (color, linestyle) in zip(results.items(), styles):
                x = x_getter(result)
                y = y_getter(result)
                if x and y:
                    # Markers only when the series is short enough for them to
                    # mean anything; on a few thousand epochs they merge into a
                    # band and hide the curve.
                    marker = "o" if len(x) <= 60 else None
                    ax.plot(x, y, marker=marker, markersize=4, label=name,
                            color=color, linestyle=linestyle, linewidth=1.8)
            ax.legend(frameon=False)
            ax.grid(alpha=0.25, linewidth=0.6)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            plt.tight_layout()
            plt.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
            plt.close(fig)

        for filename, title, ylabel, value_getter in bar_plots:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            names = list(results.keys())
            values = [value_getter(r) for r in results.values()]
            ax.bar(names, values, color=colors, edgecolor="#333333", linewidth=0.6)
            ax.grid(alpha=0.25, axis="y", linewidth=0.6)
            for spine in ("top", "right"):
                ax.spines[spine].set_visible(False)
            plt.tight_layout()
            plt.savefig(run_dir / filename, dpi=200, bbox_inches="tight")
            plt.close(fig)

        print(f"Plots saved to: {run_dir}")
        return run_dir
