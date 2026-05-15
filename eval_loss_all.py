"""
Compare selected loss curves from multiple YOLO experiments in the same figure.

Usage:
    python eval_loss_all.py

Default behavior:
    - Compares only the requested three experiments
    - Plots train loss curves on one figure
    - Optionally plots train + val loss curves if INCLUDE_VAL_LOSS is set to true

Environment variables:
    PROJECT_DIR       Root directory that contains experiment folders.
    OUTPUT_NAME       Output image name. Default: loss_compare.png
    INCLUDE_VAL_LOSS  true/false, whether to include val loss curves.
    SMOOTH_SIGMA      Gaussian smoothing sigma. Default: 2.5
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


PROJECT_DIR = Path(os.environ.get("PROJECT_DIR", "runs/detect/VisDrone_Thesis_200e"))
OUTPUT_NAME = os.environ.get("OUTPUT_NAME", "loss_compare.png")
INCLUDE_VAL_LOSS = os.environ.get("INCLUDE_VAL_LOSS", "false").lower() == "true"
SMOOTH_SIGMA = float(os.environ.get("SMOOTH_SIGMA", "2.5"))

LOSS_COLUMNS = ["train/box_loss", "train/cls_loss", "train/dfl_loss"]
VAL_COLUMNS = ["val/box_loss", "val/cls_loss", "val/dfl_loss"]

DEFAULT_EXPERIMENTS = [
    ("yolo11n_baseline_scratch_200e2", "baseline"),
    ("yolo11n_p2_cbam_wiou_multi_scratch_200e", "baseline+p2+cbam+wiou"),
    ("yolo11n_p2_cbam_multi_scratch_200e", "baseline+p2+cbam"),
]


def find_experiments(project_dir: Path) -> list[tuple[str, Path]]:
    """Find the requested experiment folders under project_dir that contain results.csv."""
    experiments: list[tuple[str, Path]] = []
    for folder_name, display_name in DEFAULT_EXPERIMENTS:
        csv_path = project_dir / folder_name / "results.csv"
        if csv_path.exists():
            experiments.append((display_name, csv_path))
    return experiments


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    """Read results.csv rows."""
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def plot_curves(experiments: list[tuple[str, Path]], columns: list[str], title_prefix: str, output_path: Path) -> None:
    """Plot a set of columns from multiple experiments on shared subplots."""
    if not experiments:
        raise FileNotFoundError(f"No results.csv files found under {PROJECT_DIR.resolve()}")

    fig, axes = plt.subplots(1, len(columns), figsize=(7 * len(columns), 5.5), tight_layout=True)
    if len(columns) == 1:
        axes = [axes]

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 10,
    })

    for exp_name, csv_path in experiments:
        rows = read_csv_rows(csv_path)
        if not rows:
            continue
        epochs = [float(row["epoch"]) for row in rows]
        for ax, column in zip(axes, columns):
            if column not in rows[0]:
                continue
            values = [float(row[column]) for row in rows]
            ax.plot(epochs, values, linewidth=2.8, alpha=0.95, label=exp_name)
            if len(values) >= 5:
                ax.plot(epochs, gaussian_filter1d(values, sigma=SMOOTH_SIGMA), linestyle=":", linewidth=1.6, alpha=0.55)
            ax.set_title(f"{title_prefix}{column}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.grid(True, linestyle="--", alpha=0.25)
            ax.margins(x=0)

    axes[0].legend(fontsize=8)
    fig.savefig(output_path, dpi=320)
    plt.close(fig)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    exps = find_experiments(PROJECT_DIR)
    print(f"Found {len(exps)} requested experiments under {PROJECT_DIR}")
    if len(exps) != len(DEFAULT_EXPERIMENTS):
        missing = {name for name, _ in DEFAULT_EXPERIMENTS} - {path.parent.name for _, path in exps}
        raise FileNotFoundError(f"Missing expected experiment folders: {sorted(missing)}")

    # Train losses: the cleanest comparison for paper figures.
    train_out = PROJECT_DIR / OUTPUT_NAME
    plot_curves(exps, LOSS_COLUMNS, "", train_out)

    # Optional: include validation losses in a second figure.
    if INCLUDE_VAL_LOSS:
        val_out = PROJECT_DIR / OUTPUT_NAME.replace(".png", "_val.png")
        plot_curves(exps, VAL_COLUMNS, "", val_out)