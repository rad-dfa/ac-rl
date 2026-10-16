"""Plots run_velocity_filter_sweep.sh results: RAD vs no-RAD drone policies with and without the velocity filter.

Reads the train_drone_policy.py --log CSVs in --storage (via plot_encoder_sweep.py's loader) and writes, under --out:
  <metric>.pdf    one panel per (velocity filter, max size), RAD vs no-RAD in each; panels share the y-axis
                  (except losses, entropy and fps), so the filter's effect reads straight down a column
  summary.csv     key metrics averaged over the last --last updates, RAD and no-RAD side by side

  uv run plot_velocity_filter_sweep.py
"""
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from plot_encoder_sweep import (
    COLORS, LABELS, METHODS, SWEEP_TAG, UNSHARED, clip_rate_axis, delta_handle, draw, load_runs, summarize
)

SAMPLER, P = "RAD", "0.5"
MAX_SIZES = [5, 10]
# Rows: run-name tag of each filter setting. SWEEP_TAG is the no-filter one; the filter swaps "vel" for "vfilt".
FILTERS = {"No velocity filter": SWEEP_TAG, "Velocity filter": SWEEP_TAG.replace("_vel_", "_vfilt_")}
DELTA = 0.05


def plot_grid(by_filter, metric, path):
    fig, axes = plt.subplots(len(FILTERS), len(MAX_SIZES), figsize=(3.4 * len(MAX_SIZES), 2.6 * len(FILTERS) + 0.9),
                             sharex=True, sharey=metric not in UNSHARED, squeeze=False)
    for row, (filter_name, configs) in enumerate(by_filter.items()):
        for col, max_size in enumerate(MAX_SIZES):
            ax = axes[row, col]
            config = (SAMPLER, max_size, P)
            if config in configs:
                draw(ax, configs[config], metric, DELTA)
            else:
                ax.text(0.5, 0.5, "no runs", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{filter_name} · max size {max_size}")
            if row == len(FILTERS) - 1:
                ax.set_xlabel("Environment steps (M)")
    clip_rate_axis(axes[0, 0], metric)
    handles = [plt.Line2D([], [], color=COLORS[m], linewidth=1.5, label=METHODS[m]) for m in METHODS]
    if metric == "prob_reject":
        handles.append(delta_handle(DELTA))
    # Same title-then-legend header as plot_encoder_sweep.py's grids.
    height = fig.get_figheight()
    fig.suptitle(f"{LABELS.get(metric, metric)}: RAD vs no-RAD, {SAMPLER} p = {P}", x=0.01, y=1 - 0.1 / height,
                 ha="left", va="top", fontsize=12)
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), bbox_to_anchor=(0.005, 1 - 0.36 / height))
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.65 / height))
    fig.savefig(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot RAD vs no-RAD drone policies from run_velocity_filter_sweep.sh")
    parser.add_argument("--storage", default="storage", help="Directory with the sweep's CSV logs (default: storage)")
    parser.add_argument("--out", default="storage/plots/velocity_filter_sweep",
                        help="Output directory (default: storage/plots/velocity_filter_sweep)")
    parser.add_argument("--last", type=int, default=50, help="Updates averaged in summary.csv (default: 50)")
    parser.add_argument("--format", default="pdf", help="Figure file format (default: pdf)")
    args = parser.parse_args()

    wanted = {(SAMPLER, max_size, P) for max_size in MAX_SIZES}
    by_filter = {}
    for filter_name, tag in FILTERS.items():
        configs = {c: runs for c, runs in load_runs(args.storage, tag).items() if c in wanted}
        if not configs:
            parser.error(f"no {SAMPLER} p={P} logs in {args.storage} match tag {tag}")
        by_filter[filter_name] = configs
    first = next(iter(next(iter(by_filter.values())).values()))
    metrics = list(next(iter(first.values()))["df"].columns.get_level_values(0).unique())

    os.makedirs(args.out, exist_ok=True)
    for metric in metrics:
        plot_grid(by_filter, metric, os.path.join(args.out, f"{metric}.{args.format}"))

    summary = pd.concat(
        [summarize(configs, args.last).assign(velocity_filter=name != "No velocity filter")
         for name, configs in by_filter.items()],
        ignore_index=True,
    )
    summary = summary[["velocity_filter"] + [c for c in summary.columns if c != "velocity_filter"]]
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))
    print(f"\n{len(metrics)} metrics -> {args.out}")


if __name__ == "__main__":
    main()
