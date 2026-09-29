"""Plots run_encoder_sweep.sh results: RAD vs no-RAD drone policies for each pretrained-encoder config.

Reads the train_drone_policy.py --log CSVs in --storage whose run settings match --tag and writes, under --out:
  <metric>.pdf             one panel per config (rows: sampler; columns: max size x p), RAD vs no-RAD in each
  <config>/<metric>.pdf    the same comparison for a single config
  summary.csv              key metrics averaged over the last --last updates, RAD and no-RAD side by side
A log without a checkpoint (interrupted or still running) is plotted up to where it stops and marked incomplete;
its config's summary compares both runs at the last step they share.

  uv run plot_encoder_sweep.py
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

SWEEP_TAG = "shaped_x-1.0_1.0_y-1.0_1.0_z-1.0_1.0_speed1.0_dt0.1_vel_steps500_safe_d0.05_lr0.005_w0_l0.5"
LOG_RE = re.compile(
    r"^log_drone_seed_(?P<seed>\d+)_(?P<sampler>Reach|ReachAvoid|RAD)_(?P<max_size>\d+)_(?P<n_tokens>\d+)"
    r"(?:_p(?P<p>[^_]+))?_(?P<rad>no_rad|rad)_(?P<tag>.+)\.csv$"
)

# Grid layout, in run_encoder_sweep.sh order (Reach, no longer swept, last): rows are samplers, columns are (max size, p).
SAMPLERS = ["ReachAvoid", "RAD", "Reach"]
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
    for path in sorted(glob.glob(os.path.join(storage, "log_drone_*.csv"))):
        name = os.path.basename(path)
        m = LOG_RE.match(name)
        if m is None or m["tag"] != tag:
            continue
        config = (m["sampler"], int(m["max_size"]), m["p"] or "None")
        ckpt = os.path.join(storage, f"policy_params_drone_{name[len('log_drone_'):-len('.csv')]}.msgpack")
        runs = grouped.setdefault(config, {}).setdefault(m["rad"], [])
        runs.append((int(m["seed"]), pd.read_csv(path), not os.path.exists(ckpt)))

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
    return f"{sampler} · max size {max_size} · p = {p}"


def config_dir(config):
    sampler, max_size, p = config
    return f"{sampler}_{max_size}" + ("" if p == "None" else f"_p{p}")


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
    samplers = [s for s in SAMPLERS if any(c[0] == s for c in configs)]
    fig, axes = plt.subplots(len(samplers), len(COLUMNS), figsize=(3.4 * len(COLUMNS), 2.6 * len(samplers) + 0.9),
                             sharex=True, sharey=metric not in UNSHARED, squeeze=False)
    notes = []
    for row, sampler in enumerate(samplers):
        for col, (max_size, p) in enumerate(COLUMNS):
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
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), bbox_to_anchor=(0.995, 0.995))
    fig.suptitle(f"{LABELS.get(metric, metric)}: RAD vs no-RAD embeddings", x=0.01, ha="left", fontsize=12)
    bottom = 0
    if notes:
        fig.text(0.01, 0.01, "\n".join(notes), fontsize=8, color=INK_2, va="bottom")
        bottom = (0.15 + 0.15 * len(notes)) / fig.get_figheight()
    fig.tight_layout(rect=(0, bottom, 1, 1 - 0.3 / fig.get_figheight()))
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
    parser = argparse.ArgumentParser(description="Plot RAD vs no-RAD drone policies from run_encoder_sweep.sh")
    parser.add_argument("--storage", default="storage", help="Directory with the sweep's CSV logs (default: storage)")
    parser.add_argument("--out", default="storage/plots/encoder_sweep",
                        help="Output directory (default: storage/plots/encoder_sweep)")
    parser.add_argument("--tag", default=SWEEP_TAG,
                        help="Run settings after the rad/no_rad part of the log name (default: run_encoder_sweep.sh's)")
    parser.add_argument("--last", type=int, default=50, help="Updates averaged in summary.csv (default: 50)")
    parser.add_argument("--format", default="pdf", help="Figure file format (default: pdf)")
    args = parser.parse_args()

    configs = load_runs(args.storage, args.tag)
    if not configs:
        parser.error(f"no logs in {args.storage} match --tag {args.tag}")
    delta = re.search(r"_safe_d([\d.]+)", args.tag)
    delta = float(delta[1]) if delta else None
    metrics = [c for c in next(iter(next(iter(configs.values())).values()))["df"].columns.get_level_values(0).unique()]

    os.makedirs(args.out, exist_ok=True)
    for metric in metrics:
        plot_grid(configs, metric, delta, os.path.join(args.out, f"{metric}.{args.format}"))
    for config, runs in configs.items():
        out_dir = os.path.join(args.out, config_dir(config))
        os.makedirs(out_dir, exist_ok=True)
        for metric in metrics:
            plot_single(config, runs, metric, delta, os.path.join(out_dir, f"{metric}.{args.format}"))

    summary = summarize(configs, args.last)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))
    print(f"\n{len(configs)} configs, {len(metrics)} metrics -> {args.out}")


if __name__ == "__main__":
    main()
