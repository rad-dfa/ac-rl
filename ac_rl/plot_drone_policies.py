"""Plots RAD vs no-RAD drone policies (train_drone_policies.sh) for each sampler config.

Reads the train_drone_policy.py --log CSVs in --storage/logs whose run settings match --tag and writes, under --out:
  <metric>.pdf             one panel per config (rows: sampler; columns: max size x p), RAD vs no-RAD in each
  <config>/<metric>.pdf    the same comparison for a single config
  summary.csv              key metrics averaged over the last --last updates, RAD and no-RAD side by side
Without --tag, it plots train_drone_policies.sh's runs without and with the velocity filter into --out/vel and
--out/vfilt. A log without a checkpoint (interrupted or still running) is plotted up to where it stops and marked
incomplete; its config's summary compares both runs at the last step they share.

  uv run plot_drone_policies.py
"""
import os
import re
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from naming import DRONE_SWEEP, drone_run_name, drone_run_settings, checkpoint_path, parse_log_name, num

# Run settings of train_drone_policies.sh's runs without and with the velocity filter, by output subdirectory.
DEFAULT_TAGS = {
    mode: drone_run_settings(dict(DRONE_SWEEP, use_velocity_filter=(mode == "vfilt")))
    for mode in ("vel", "vfilt")
}

# Grid layout: rows are samplers, columns are (max size, p). Rows and columns without runs are dropped.
SAMPLERS = ["RA", "RAD", "R", "MIX"]
SAMPLER_NAMES = {"R": "Reach", "RA": "ReachAvoid", "RAD": "RAD", "MIX": "Mixture"}
COLUMNS = [(10, "0.5"), (10, "None"), (5, "0.5"), (5, "None")]

METHODS = {"rad": "RAD", "no_rad": "No RAD"}
COLORS = {"rad": "#2a78d6", "no_rad": "#eb6834"}
SURFACE, INK, INK_2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"

LABELS = {
    "prob_success": "Success rate (DFA accepted)",
    "prob_reject": "Rejection rate (DFA rejected)",
    "prob_fail": "Failure rate (rejected or timed out)",
    "lambda": "Lagrange multiplier λ",
    "disc_return_mean": "Discounted return (mean)",
    "return_mean": "Episode return (mean)",
    "return_min": "Episode return (min)",
    "return_max": "Episode return (max)",
    "return_std": "Episode return (std)",
    "ep_len_mean": "Episode length (mean)",
    "ep_len_min": "Episode length (min)",
    "ep_len_max": "Episode length (max)",
    "ep_len_std": "Episode length (std)",
    "total_loss": "Total loss",
    "value_loss": "Value loss",
    "actor_loss": "Actor loss",
    "entropy": "Policy entropy",
    "fps": "Training throughput (env steps/s)",
}
# Metrics whose scale differs a lot between configs get their own y-axis per panel.
UNSHARED = {"total_loss", "value_loss", "actor_loss", "entropy", "fps"}
SUMMARY_METRICS = ["prob_success", "prob_reject", "prob_fail", "lambda", "ep_len_mean", "disc_return_mean"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 9, "legend.frameon": False,
})


def load_runs(storage, tag):
    """{(sampler, max_size, p): {"rad"/"no_rad": {"df": per-timestep mean/std over seeds, "seeds", "incomplete"}}}"""
    grouped = {}
    for path in sorted(glob.glob(os.path.join(storage, "logs", "log_drone_*.csv"))):
        cfg = parse_log_name(path)
        if cfg is None or drone_run_settings(cfg) != tag:
            continue
        config = (cfg["sampler"], cfg["max_size"], num(cfg["p"]))
        ckpt = checkpoint_path(storage, drone_run_name(cfg))
        runs = grouped.setdefault(config, {}).setdefault("no_rad" if cfg["no_rad"] else "rad", [])
        runs.append((cfg["seed"], pd.read_csv(path), not os.path.exists(ckpt)))

    configs = {}
    for config, methods in grouped.items():
        configs[config] = {}
        for method, runs in methods.items():
            df = pd.concat([r[1] for r in runs]).groupby("timestep").agg(["mean", "std"])
            configs[config][method] = {
                "df": df,
                "seeds": sorted(r[0] for r in runs),
                "incomplete": any(r[2] for r in runs),
            }
    return configs


def config_title(config):
    sampler, max_size, p = config
    return f"{SAMPLER_NAMES[sampler]} · max size {max_size} · p = {p}"


def config_dir(config):
    # As in run names (naming.py)
    sampler, max_size, p = config
    return f"{sampler}_{max_size}_p{p}"


def incomplete_note(method, run):
    return f"{METHODS[method]} run incomplete: stops at {run['df'].index[-1] / 1e6:.1f}M steps"


def method_label(method, run):
    label = METHODS[method]
    if len(run["seeds"]) > 1:
        label += f" ({len(run['seeds'])} seeds)"
    if run["incomplete"]:
        label += f" (incomplete, {run['df'].index[-1] / 1e6:.1f}M steps)"
    return label


def delta_handle(delta):
    return plt.Line2D([], [], color=MUTED, linewidth=1, linestyle=(0, (4, 3)), label=f"δ = {delta:g} (allowed rejection rate)")


def draw(ax, runs, metric, delta):
    for method in METHODS:
        if method not in runs or metric not in runs[method]["df"]:
            continue
        run = runs[method]
        steps = run["df"].index.values / 1e6
        mean, std = run["df"][metric]["mean"].values, run["df"][metric]["std"].values
        ax.plot(steps, mean, color=COLORS[method], linewidth=1.5, label=method_label(method, run))
        if len(run["seeds"]) > 1:
            ax.fill_between(steps, mean - std, mean + std, color=COLORS[method], alpha=0.15, linewidth=0)
        if run["incomplete"]:
            # Mark where the run stops, ringed in the surface color so it reads over the other line.
            ax.plot(steps[-1], mean[-1], "o", markersize=6, color=COLORS[method], markeredgecolor=SURFACE,
                    markeredgewidth=1.5, zorder=3)
    if metric == "prob_reject" and delta is not None:
        ax.axhline(delta, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    if metric in ("prob_success", "prob_fail"):
        ax.set_ylim(-0.03, 1.03)


def clip_rate_axis(ax, metric):
    # Autoscaling pads below 0, which is meaningless for a rate; keep only a sliver so a line at 0 stays visible.
    if metric == "prob_reject":
        top = ax.get_ylim()[1]
        ax.set_ylim(-0.03 * top, top)


def plot_grid(configs, metric, delta, path):
    # Only rows and columns with at least one run, e.g. a max-size-5-only sweep gets no max-size-10 column.
    samplers = [s for s in SAMPLERS if any(c[0] == s for c in configs)]
    columns = [col for col in COLUMNS if any(c[1:] == col for c in configs)]
    fig, axes = plt.subplots(len(samplers), len(columns), figsize=(3.4 * len(columns), 2.6 * len(samplers) + 0.9),
                             sharex=True, sharey=metric not in UNSHARED, squeeze=False)
    notes = []
    for row, sampler in enumerate(samplers):
        for col, (max_size, p) in enumerate(columns):
            ax, config = axes[row, col], (sampler, max_size, p)
            title = config_title(config)
            if config in configs:
                draw(ax, configs[config], metric, delta)
                for method, run in configs[config].items():
                    if run["incomplete"]:
                        notes.append(f"{'*' * (len(notes) + 1)} {title}: {incomplete_note(method, run)} (marked •)")
                        title += "*" * len(notes)
            else:
                ax.text(0.5, 0.5, "no runs", transform=ax.transAxes, ha="center", va="center", color=MUTED)
                # No data, so no y-scale to show; tick_params only touches this panel even when y is shared.
                ax.grid(False)
                ax.spines["left"].set_visible(False)
                ax.tick_params(axis="y", left=False, labelleft=False)
            ax.set_title(title)
            if row == len(samplers) - 1:
                ax.set_xlabel("Environment steps (M)")
    clip_rate_axis(axes[0, 0], metric)
    handles = [plt.Line2D([], [], color=COLORS[m], linewidth=1.5, label=METHODS[m]) for m in METHODS]
    if metric == "prob_reject" and delta is not None:
        handles.append(delta_handle(delta))
    # Title, then the legend on its own line below it, so the two never collide however narrow the grid is.
    height = fig.get_figheight()
    fig.suptitle(f"{LABELS.get(metric, metric)}: RAD vs no-RAD embeddings", x=0.01, y=1 - 0.1 / height,
                 ha="left", va="top", fontsize=12)
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), bbox_to_anchor=(0.005, 1 - 0.36 / height))
    bottom = 0
    if notes:
        fig.text(0.01, 0.01, "\n".join(notes), fontsize=8, color=INK_2, va="bottom")
        bottom = (0.15 + 0.15 * len(notes)) / height
    fig.tight_layout(rect=(0, bottom, 1, 1 - 0.65 / height))
    fig.savefig(path)
    plt.close(fig)


def plot_single(config, runs, metric, delta, path):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    draw(ax, runs, metric, delta)
    clip_rate_axis(ax, metric)
    ax.set_title(f"{config_title(config)}: RAD vs no-RAD", loc="left")
    ax.set_xlabel("Environment steps (M)")
    ax.set_ylabel(LABELS.get(metric, metric))
    handles, _ = ax.get_legend_handles_labels()
    if metric == "prob_reject" and delta is not None:
        handles.append(delta_handle(delta))
    ax.legend(handles=handles, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def summarize(configs, last):
    rows = []
    for config in sorted(configs, key=lambda c: (SAMPLERS.index(c[0]), COLUMNS.index(c[1:]))):
        runs = configs[config]
        # Compare both methods over the same window, ending where the shorter run stops.
        end = min(run["df"].index[-1] for run in runs.values())
        row = {"sampler": config[0], "max_size": config[1], "p": config[2], "steps_M": round(end / 1e6, 2)}
        for metric in SUMMARY_METRICS:
            for method in METHODS:
                if method in runs:
                    window = runs[method]["df"].loc[:end, (metric, "mean")].tail(last)
                    row[f"{metric}_{method}"] = window.mean()
        row["incomplete"] = ",".join(m for m in METHODS if m in runs and runs[m]["incomplete"])
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Plot RAD vs no-RAD drone policies from train_drone_policies.sh")
    parser.add_argument("--storage", default="storage", help="--save-dir of the runs; their CSV logs are read from {storage}/logs (default: storage)")
    parser.add_argument("--out", default="storage/plots/drone_policies",
                        help="Output directory (default: storage/plots/drone_policies)")
    parser.add_argument("--tag", default=None,
                        help="Run settings after the rad/no_rad part of the log name (default: train_drone_policies.sh's "
                             "without and with the velocity filter, into --out/vel and --out/vfilt)")
    parser.add_argument("--last", type=int, default=50, help="Updates averaged in summary.csv (default: 50)")
    parser.add_argument("--format", default="pdf", help="Figure file format (default: pdf)")
    args = parser.parse_args()

    tags = {"": args.tag} if args.tag else DEFAULT_TAGS
    n_plotted = 0
    for subdir, tag in tags.items():
        out = os.path.join(args.out, subdir)
        configs = load_runs(args.storage, tag)
        if not configs:
            print(f"No logs in {args.storage} match --tag {tag}, skipping")
            continue
        delta = re.search(r"_safe_d([^_]+)", tag)
        delta = float(delta[1]) if delta else None
        metrics = [c for c in next(iter(next(iter(configs.values())).values()))["df"].columns.get_level_values(0).unique()]

        os.makedirs(out, exist_ok=True)
        for metric in metrics:
            plot_grid(configs, metric, delta, os.path.join(out, f"{metric}.{args.format}"))
        for config, runs in configs.items():
            out_dir = os.path.join(out, config_dir(config))
            os.makedirs(out_dir, exist_ok=True)
            for metric in metrics:
                plot_single(config, runs, metric, delta, os.path.join(out_dir, f"{metric}.{args.format}"))

        summary = summarize(configs, args.last)
        summary.to_csv(os.path.join(out, "summary.csv"), index=False)
        print(tag)
        with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.3f}".format):
            print(summary.to_string(index=False))
        print(f"\n{len(configs)} configs, {len(metrics)} metrics -> {out}\n")
        n_plotted += 1
    if not n_plotted:
        parser.error(f"no logs in {args.storage} match any of: {', '.join(tags.values())}")


if __name__ == "__main__":
    main()
