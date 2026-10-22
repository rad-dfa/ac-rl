"""Plots run_velocity_filter_sweep.sh results: RAD vs no-RAD drone policies with and without the velocity filter,
under each lambda schedule.

Reads the train_drone_policy.py --log CSVs in --storage/logs (via plot_encoder_sweep.py's loader) and writes, under --out:
  <metric>.pdf    one row per lambda schedule, one column per (velocity filter, max size), RAD vs no-RAD in each
                  panel; panels share the y-axis (except losses, entropy and fps), so schedules compare down a
                  column and filter settings along a row
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
DELTA = 0.05
# Run-name tag up to the --safe settings, without and with the filter (which swaps "vel" for "vfilt").
ENV_TAG = SWEEP_TAG.split("_safe")[0]
FILTERS = {"No velocity filter": ENV_TAG, "Velocity filter": ENV_TAG.replace("_vel_", "_vfilt_")}
# Rows: run_velocity_filter_sweep.sh's lambda schedules, as row label -> run-name suffix.
SCHEDULES = {
    "λ₀ = 0.5, lr 0.005\n(no clip)": "_safe_d0.05_lr0.005_w0_l0.5",
    "λ₀ = 0, lr 0.05\n(clip [0, 1])": "_safe_d0.05_lr0.05_w0_l0.0_c0.0_1.0",
    "λ₀ = 1, lr 0.05\n(clip [0, 1])": "_safe_d0.05_lr0.05_w0_l1.0_c0.0_1.0",
}
COLUMNS = [(filter_name, max_size) for filter_name in FILTERS for max_size in (5, 10)]


def plot_grid(data, metric, path):
    fig, axes = plt.subplots(len(SCHEDULES), len(COLUMNS),
                             figsize=(3.4 * len(COLUMNS), 2.6 * len(SCHEDULES) + 0.9),
                             sharex=True, sharey=metric not in UNSHARED, squeeze=False)
    for row, schedule in enumerate(SCHEDULES):
        for col, (filter_name, max_size) in enumerate(COLUMNS):
            ax = axes[row, col]
            runs = data[schedule, filter_name].get((SAMPLER, max_size, P))
            if runs:
                draw(ax, runs, metric, DELTA)
            else:
                ax.text(0.5, 0.5, "no runs", transform=ax.transAxes, ha="center", va="center")
            if row == 0:
                ax.set_title(f"{filter_name} · max size {max_size}")
            if col == 0:
                ax.set_ylabel(schedule)
            if row == len(SCHEDULES) - 1:
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
    parser.add_argument("--storage", default="storage", help="--save-dir of the sweep; its CSV logs are read from {storage}/logs (default: storage)")
    parser.add_argument("--out", default="storage/plots/velocity_filter_sweep",
                        help="Output directory (default: storage/plots/velocity_filter_sweep)")
    parser.add_argument("--last", type=int, default=50, help="Updates averaged in summary.csv (default: 50)")
    parser.add_argument("--format", default="pdf", help="Figure file format (default: pdf)")
    args = parser.parse_args()

    wanted = {(SAMPLER, max_size, P) for _, max_size in COLUMNS}
    data = {}
    for schedule, suffix in SCHEDULES.items():
        for filter_name, env_tag in FILTERS.items():
            runs = load_runs(args.storage, env_tag + suffix)
            data[schedule, filter_name] = {c: r for c, r in runs.items() if c in wanted}
    loaded = [configs for configs in data.values() if configs]
    if not loaded:
        parser.error(f"no {SAMPLER} p={P} logs in {args.storage} match any of the sweep's tags")
    first_run = next(iter(next(iter(loaded[0].values())).values()))
    metrics = list(first_run["df"].columns.get_level_values(0).unique())

    os.makedirs(args.out, exist_ok=True)
    for metric in metrics:
        plot_grid(data, metric, os.path.join(args.out, f"{metric}.{args.format}"))

    summary = pd.concat(
        [summarize(configs, args.last).assign(lambda_schedule=SCHEDULES[schedule].removeprefix("_safe_"),
                                              velocity_filter=filter_name == "Velocity filter")
         for (schedule, filter_name), configs in data.items() if configs],
        ignore_index=True,
    )
    first = ["lambda_schedule", "velocity_filter"]
    summary = summary[first + [c for c in summary.columns if c not in first]]
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))
    print(f"\n{len(metrics)} metrics -> {args.out}")


if __name__ == "__main__":
    main()
