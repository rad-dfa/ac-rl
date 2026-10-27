"""Plots a one-setting sweep of the drone policies: each value's training curves, RAD and no-RAD side by side.

--vary picks the setting, PBRS's beta (beta_sweep.sh) or PPO-Lagrangian's delta. It reads the train_drone_policy.py
--log CSVs in --storage/logs with MIX, max size 5, p None, the velocity filter and 1e7 steps, and otherwise
train_drone_policies.sh's settings but any value of the swept one, so train_drone_policies.sh's seed 42 run of that
config (beta 0.7, delta 0.05, from before it went to 2e7 steps) is in both sweeps. It writes, under --out (default storage/plots/{vary}_sweep):
  overview.pdf      success rate, rejection rate, lambda and episode length; rows: metric, columns: RAD, no-RAD
  <metric>.pdf      one metric, RAD and no-RAD panels
  summary.csv       key metrics averaged over each run's last --last updates
A log without a checkpoint (interrupted or still running) is plotted up to where it stops and marked incomplete.
Returns and success/rejection rates are of the unshaped reward (wrappers.py), so they compare across betas; the losses
are of the shaped one and don't.

  uv run plot_sweep.py --vary beta
  uv run plot_sweep.py --vary delta
"""
import os
import glob
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from naming import DRONE_SWEEP, DEFAULT_TOTAL_TIMESTEPS, drone_run_name, drone_run_settings, checkpoint_path, parse_log_name
from plot_drone_policies import (LABELS, UNSHARED, SUMMARY_METRICS, METHODS, SURFACE, INK, INK_2, MUTED,
                                 clip_rate_axis, delta_handle)

SAMPLER, MAX_SIZE, P = "MIX", 5, None
# The sweeps ran for the default 1e7 steps, before train_drone_policies.sh went to 2e7.
BASE = dict(DRONE_SWEEP, use_velocity_filter=True, total_timesteps=DEFAULT_TOTAL_TIMESTEPS)
OVERVIEW_METRICS = ["prob_success", "prob_reject", "lambda", "ep_len_mean"]

# Both settings are ordinal, so one blue ramp light to dark, fixed per value so that adding runs never repaints the
# others. No PBRS (beta None) would be the gray baseline.
SWEEPS = {
    "beta": dict(name="PBRS β", colors={None: MUTED, 0.25: "#86b6ef", 0.5: "#3987e5", 0.7: "#1c5cab", 0.75: "#0d366b"},
                 label=lambda b: "No PBRS" if b is None else f"β = {b:g}"),
    "delta": dict(name="PPO-Lagrangian δ", colors={0.05: "#6da7ec", 0.1: "#184f95"}, label=lambda d: f"δ = {d:g}"),
}


def load_runs(storage, vary):
    """{"rad"/"no_rad": {value of vary: {"df": per-timestep mean/std over seeds, "seeds", "incomplete"}}}"""
    grouped = {}
    for path in sorted(glob.glob(os.path.join(storage, "logs", "log_drone_*.csv"))):
        cfg = parse_log_name(path)
        if (cfg is None or (cfg["sampler"], cfg["max_size"], cfg["p"]) != (SAMPLER, MAX_SIZE, P)
                or drone_run_settings(cfg) != drone_run_settings(dict(BASE, **{vary: cfg.get(vary)}))):
            continue
        ckpt = checkpoint_path(storage, drone_run_name(cfg))
        runs = grouped.setdefault("no_rad" if cfg["no_rad"] else "rad", {}).setdefault(cfg[vary], [])
        runs.append((cfg["seed"], pd.read_csv(path), not os.path.exists(ckpt)))

    methods = {}
    for method, values in grouped.items():
        methods[method] = {}
        for value, runs in values.items():
            df = pd.concat([r[1] for r in runs]).groupby("timestep").agg(["mean", "std"])
            methods[method][value] = {
                "df": df,
                "seeds": sorted(r[0] for r in runs),
                "incomplete": any(r[2] for r in runs),
            }
    return methods


def sorted_values(values):
    return sorted(values, key=lambda v: -1 if v is None else v)


def all_values(methods):
    return sorted_values({value for values in methods.values() for value in values})


def color(sweep, value):
    return sweep["colors"].get(value, INK_2)


def draw(ax, runs, metric, vary):
    sweep = SWEEPS[vary]
    for value in sorted_values(runs):
        run = runs[value]
        if metric not in run["df"]:
            continue
        steps = run["df"].index.values / 1e6
        mean, std = run["df"][metric]["mean"].values, run["df"][metric]["std"].values
        ax.plot(steps, mean, color=color(sweep, value), linewidth=1.5)
        if len(run["seeds"]) > 1:
            ax.fill_between(steps, mean - std, mean + std, color=color(sweep, value), alpha=0.15, linewidth=0)
        if run["incomplete"]:
            # Mark where the run stops, ringed in the surface color so it reads over the other lines.
            ax.plot(steps[-1], mean[-1], "o", markersize=6, color=color(sweep, value), markeredgecolor=SURFACE,
                    markeredgewidth=1.5, zorder=3)
    if metric == "prob_reject":
        # Each run's allowed rejection rate, in its color when delta is what varies.
        if vary == "delta":
            for value in runs:
                ax.axhline(value, color=color(sweep, value), linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        else:
            ax.axhline(BASE["delta"], color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    if metric in ("prob_success", "prob_fail"):
        ax.set_ylim(-0.03, 1.03)


def value_handles(methods, vary):
    sweep = SWEEPS[vary]
    handles = []
    for value in all_values(methods):
        label = sweep["label"](value)
        n_seeds = max(len(values[value]["seeds"]) for values in methods.values() if value in values)
        if n_seeds > 1:
            label += f" ({n_seeds} seeds)"
        handles.append(plt.Line2D([], [], color=color(sweep, value), linewidth=1.5, label=label))
    return handles


def incomplete_notes(methods, vary):
    return [f"• {METHODS[method]}, {SWEEPS[vary]['label'](value)}: incomplete, stops at "
            f"{run['df'].index[-1] / 1e6:.1f}M steps"
            for method in METHODS for value, run in methods.get(method, {}).items() if run["incomplete"]]


def plot(methods, metrics, vary, title, path):
    columns = [m for m in METHODS if m in methods]
    fig, axes = plt.subplots(len(metrics), len(columns), figsize=(4.2 * len(columns), 2.5 * len(metrics) + 1.1),
                             sharex=True, squeeze=False)
    for row, metric in enumerate(metrics):
        for col, method in enumerate(columns):
            ax = axes[row, col]
            # Metrics whose scale differs a lot between the encoders get their own y-axis per panel.
            if col > 0 and metric not in UNSHARED:
                ax.sharey(axes[row, 0])
                ax.tick_params(axis="y", labelleft=False)
            draw(ax, methods[method], metric, vary)
            if row == 0:
                ax.set_title(f"{METHODS[method]} embeddings")
            if col == 0:
                ax.set_ylabel(LABELS.get(metric, metric))
            if row == len(metrics) - 1:
                ax.set_xlabel("Environment steps (M)")
        clip_rate_axis(axes[row, 0], metric)
    handles = value_handles(methods, vary)
    if "prob_reject" in metrics:
        if vary == "delta":
            handles.append(plt.Line2D([], [], color=MUTED, linewidth=1, linestyle=(0, (4, 3)),
                                      label="δ (allowed rejection rate), in its run's color"))
        else:
            handles.append(delta_handle(BASE["delta"]))
    height = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - 0.1 / height, ha="left", va="top", fontsize=12, color=INK)
    fig.legend(handles=handles, loc="upper left", ncol=len(handles), bbox_to_anchor=(0.005, 1 - 0.36 / height))
    notes = incomplete_notes(methods, vary)
    bottom = 0
    if notes:
        fig.text(0.01, 0.01, "\n".join(notes), fontsize=8, color=INK_2, va="bottom")
        bottom = (0.15 + 0.15 * len(notes)) / height
    fig.tight_layout(rect=(0, bottom, 1, 1 - 0.65 / height))
    fig.savefig(path)
    plt.close(fig)


def summarize(methods, vary, last):
    rows = []
    for method in METHODS:
        for value in sorted_values(methods.get(method, {})):
            run = methods[method][value]
            row = {"encoder": METHODS[method], vary: "None" if value is None else value,
                   "steps_M": round(run["df"].index[-1] / 1e6, 2)}
            for metric in SUMMARY_METRICS:
                row[metric] = run["df"][(metric, "mean")].tail(last).mean()
            row["incomplete"] = run["incomplete"]
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Plot a beta or delta sweep of the drone policies")
    parser.add_argument("--vary", choices=list(SWEEPS), default="beta", help="Swept setting (default: beta)")
    parser.add_argument("--storage", default="storage", help="--save-dir of the runs; their CSV logs are read from {storage}/logs (default: storage)")
    parser.add_argument("--out", default=None, help="Output directory (default: storage/plots/{vary}_sweep)")
    parser.add_argument("--last", type=int, default=50, help="Updates averaged in summary.csv (default: 50)")
    parser.add_argument("--format", default="pdf", help="Figure file format (default: pdf)")
    args = parser.parse_args()
    out = args.out or os.path.join("storage", "plots", f"{args.vary}_sweep")

    methods = load_runs(args.storage, args.vary)
    if not methods:
        parser.error(f"no logs in {args.storage}/logs match the {args.vary} sweep's settings")
    # The settings held fixed, the other of beta and delta among them
    fixed = {"beta": f"δ = {BASE['delta']:g}", "delta": f"β = {BASE['beta']:g}"}[args.vary]
    config = f"Mixture · max size {MAX_SIZE} · p = {P} · velocity filter · {fixed}"
    sweep = SWEEPS[args.vary]["name"]
    metrics = next(iter(next(iter(methods.values())).values()))["df"].columns.get_level_values(0).unique()

    os.makedirs(out, exist_ok=True)
    plot(methods, OVERVIEW_METRICS, args.vary, f"{sweep} sweep: {config}", os.path.join(out, f"overview.{args.format}"))
    for metric in metrics:
        plot(methods, [metric], args.vary, f"{LABELS.get(metric, metric)}: {sweep} sweep, {config}",
             os.path.join(out, f"{metric}.{args.format}"))

    summary = summarize(methods, args.vary, args.last)
    summary.to_csv(os.path.join(out, "summary.csv"), index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))
    print(f"\n{len(summary)} runs, {len(metrics)} metrics -> {out}")


if __name__ == "__main__":
    main()
