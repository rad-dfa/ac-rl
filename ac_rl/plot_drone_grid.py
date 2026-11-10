"""Plots train_drone_policies.sh's whole grid: every noise x arm x PBRS x encoder config, over the seeds it has so far.

Reads the train_drone_policy.py --log CSVs in --storage/logs whose settings are one of the script's configs (naming.py's
DRONE_SWEEP with DRONE_SWEEP_NOISES, DRONE_SWEEP_ARMS and DRONE_SWEEP_BETAS) and writes, under --out:
  curves_<noise>.pdf     training curves under one observation noise. Rows: success, rejection, lambda, episode length;
                         columns: the arms; RAD vs no-RAD by color, PBRS vs none by solid vs dashed
  metrics/<metric>.pdf   one metric's training curves, rows: the observation noises, columns: the arms
  compare_rows.pdf       every config's final success and rejection rates, one row per config, one marker per noise
  compare_scatter.pdf    the same as success vs rejection, one panel per noise
  summary.csv            each config's final rates, i.e. the mean of each run's last --last updates, mean and std over
                         its seeds, and whether its rejection rate is within PPO-Lagrangian's delta
A log without a checkpoint (interrupted or still running) is plotted up to where it stops and marked incomplete.

The rates are of the unshaped reward (wrappers.py), so they compare across arms and betas. Only the PPO-Lagrangian
(--safe) runs log the rejection rate. For the other shaped runs it is derived as prob_success - return_mean, which is
exact: their unshaped episode return is its last reward, +1 (accepted), -1 (rejected) or 0 (timed out or left the box),
and both are over the same episodes. Binary-reward runs (the script's binary arm until 2026-11-10) have a 1/0 unshaped
reward instead, so a rejection looks like a timeout and their rejection rate isn't in the logs (test_drone_policies.py
measures it); it is n/a here.

  uv run plot_drone_grid.py
"""
import os
import glob
import math
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
from naming import (DRONE_SWEEP, DRONE_SWEEP_NOISES, DRONE_SWEEP_ARMS, DRONE_SWEEP_BETAS, drone_run_name,
                    drone_run_settings, checkpoint_path, parse_log_name, num)
from plot_drone_policies import (LABELS, UNSHARED, METHODS, COLORS, SURFACE, INK, INK_2, MUTED, GRID,
                                 clip_rate_axis, delta_handle)

SAMPLER, MAX_SIZE, P = "MIX", 5, None
BASE = dict(DRONE_SWEEP, use_velocity_filter=True)
DELTA = BASE["delta"]
CONFIG = f"Mixture · max size {MAX_SIZE} · p = {P} · velocity filter · {BASE['total_timesteps'] / 1e6:g}M steps"

# Names of the script's arms, past ones included (binary and safe_l1 until 2026-11-10), so re-adding one to
# DRONE_SWEEP_ARMS plots it again.
ARM_NAMES = {"shaped": "Shaped PPO", "binary": "Binary PPO", "safe_l0": "PPO-Lagrangian, λ from 0",
             "safe_l1": "PPO-Lagrangian, λ from 1"}
ARMS = {arm: ARM_NAMES[arm] for arm in DRONE_SWEEP_ARMS}
BINARY_ARMS = [arm for arm in ARMS if DRONE_SWEEP_ARMS[arm]["binary_reward"]]
CURVE_METRICS = ["prob_success", "prob_reject", "lambda", "ep_len_mean"]
SHORT_LABELS = {"prob_success": "Success rate", "prob_reject": "Rejection rate", "lambda": "Lagrange multiplier λ",
                "ep_len_mean": "Episode length (mean)"}
# The final metrics and their summary.csv names
FINAL_METRICS = {"prob_success": "success", "prob_reject": "reject", "lambda": "lambda", "ep_len_mean": "ep_len"}
# PBRS by line style (solid with, dashed without, as in plot_paper.py), in DRONE_SWEEP_BETAS' order
LINESTYLES = ["-", (0, (4, 2)), (0, (1, 1.5))]
# The observation noises in compare_rows.pdf, in DRONE_SWEEP_NOISES' order: two hues other than the encoders' blue and
# orange (the dataviz reference palette's aqua and violet), with the marker shape as a second cue.
NOISE_STYLES = [dict(color="#1baf7a", marker="o", markersize=7), dict(color="#6250d6", marker="D", markersize=6)]
# The arms in compare_scatter.pdf, by marker shape; color is the encoder there, as in every other plot.
ARM_MARKERS = {"shaped": "o", "binary": "D", "safe_l0": "^", "safe_l1": "s"}
NA_BINARY = "n/a: the binary reward pays 0 on rejection as on a timeout, so the logs don't have its rejection rate"
DERIVED = "Shaped PPO's rejection rate is derived from its logs as success rate − mean unshaped return"


def linestyle(beta):
    return LINESTYLES[DRONE_SWEEP_BETAS.index(beta)]


def noise_name(noise):
    return "no obs noise" if noise is None else f"obs noise σ {noise[0]:g} / {noise[1]:g}"


def noise_tag(noise):
    # As in run names (naming.py)
    return "none" if noise is None else f"noise{num(noise[0])}_{num(noise[1])}"


def pbrs_name(beta):
    return "no PBRS" if beta is None else f"PBRS β = {beta:g}"


def config_name(key):
    noise, arm, beta, encoder = key
    return f"{ARMS[arm]}, {METHODS[encoder]}, {pbrs_name(beta)}, {noise_name(noise)}"


def all_keys():
    """Every config of the grid as (noise, arm, beta, encoder), in plotting order."""
    return [(noise, arm, beta, encoder) for noise in DRONE_SWEEP_NOISES for arm in ARMS
            for encoder in METHODS for beta in DRONE_SWEEP_BETAS]


def load_runs(storage, last):
    """{(noise, arm, beta, encoder): {"df": per-timestep mean/std over seeds, "seeds", "incomplete": [(seed, last
    step)], "steps": the last step all seeds reached, "final": {metric: (mean, std) over seeds}}}"""
    # The grid's configs by their run settings (everything but the seed, sampler, sizes, p and encoder)
    configs = {
        drone_run_settings(dict(BASE, obs_noise_std=noise, beta=beta, **DRONE_SWEEP_ARMS[arm])): (noise, arm, beta)
        for noise in DRONE_SWEEP_NOISES for arm in ARMS for beta in DRONE_SWEEP_BETAS
    }
    grouped = {}
    for path in sorted(glob.glob(os.path.join(storage, "logs", "log_drone_*.csv"))):
        cfg = parse_log_name(path)
        if cfg is None or (cfg["sampler"], cfg["max_size"], cfg["p"]) != (SAMPLER, MAX_SIZE, P):
            continue
        config = configs.get(drone_run_settings(cfg))
        if config is None:
            continue
        try:
            df = pd.read_csv(path)
        except pd.errors.EmptyDataError:  # a run that was just started
            continue
        if df.empty:
            continue
        if "prob_reject" not in df and not cfg["binary_reward"]:
            df["prob_reject"] = df["prob_success"] - df["return_mean"]
        incomplete = not os.path.exists(checkpoint_path(storage, drone_run_name(cfg)))
        grouped.setdefault((*config, "no_rad" if cfg["no_rad"] else "rad"), []).append((cfg["seed"], df, incomplete))

    runs = {}
    for key, seed_runs in grouped.items():
        finals = pd.DataFrame([{m: df[m].tail(last).mean() for m in FINAL_METRICS if m in df}
                               for _, df, _ in seed_runs])
        runs[key] = {
            "df": pd.concat([df for _, df, _ in seed_runs]).groupby("timestep").agg(["mean", "std"]),
            "seeds": sorted(seed for seed, _, _ in seed_runs),
            "incomplete": sorted((seed, df["timestep"].iloc[-1]) for seed, df, incomplete in seed_runs if incomplete),
            "steps": min(df["timestep"].iloc[-1] for _, df, _ in seed_runs),
            # The std over a single seed is NaN, so nothing draws error bars for it.
            "final": {m: (finals[m].mean(), finals[m].std()) for m in finals},
        }
    return runs


def final(run, metric):
    return run["final"].get(metric, (np.nan, np.nan))


def seeds_note(runs):
    seeds = sorted({seed for run in runs.values() for seed in run["seeds"]})
    counts = sorted({len(run["seeds"]) for run in runs.values()})
    if counts == [1]:
        return f"seed {seeds[0]}" if len(seeds) == 1 else "1 seed per config"
    n = f"{counts[0]}" if len(counts) == 1 else f"{counts[0]}–{counts[-1]}"
    return f"mean ± std over {n} seeds"


def incomplete_notes(runs, keys):
    return [f"• {config_name(key)}: seed {seed} is incomplete, it stops at {step / 1e6:.1f}M steps"
            for key in keys if key in runs for seed, step in runs[key]["incomplete"]]


def best_within_delta(runs):
    """{noise: the complete config with the highest final success rate among those whose rejection rate is <= DELTA}"""
    best = {}
    for key, run in runs.items():
        success, reject = final(run, "prob_success")[0], final(run, "prob_reject")[0]
        if run["incomplete"] or not reject <= DELTA:  # NaN (binary) is never <= DELTA
            continue
        if key[0] not in best or success > final(runs[best[key[0]]], "prob_success")[0]:
            best[key[0]] = key
    return best


def finish(fig, title, subtitle, handles, notes, path, legend_rows=1):
    """Title and subtitle at the top left, the legend on its own lines below them, notes at the bottom."""
    height = fig.get_figheight()
    fig.suptitle(title, x=0.01, y=1 - 0.1 / height, ha="left", va="top", fontsize=12, color=INK)
    fig.text(0.01, 1 - 0.38 / height, subtitle, ha="left", va="top", fontsize=9, color=INK_2)
    fig.legend(handles=handles, loc="upper left", ncol=math.ceil(len(handles) / legend_rows),
               bbox_to_anchor=(0.005, 1 - 0.58 / height))
    bottom = 0
    if notes:
        fig.text(0.01, 0.01, "\n".join(notes), fontsize=8, color=INK_2, va="bottom")
        bottom = (0.15 + 0.15 * len(notes)) / height
    fig.tight_layout(rect=(0, bottom, 1, 1 - (0.7 + 0.22 * legend_rows) / height))
    fig.savefig(path)
    plt.close(fig)


# Training curves


def placeholder(ax, text):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", color=MUTED, fontsize=8)
    # No data, so no y-scale to show
    ax.grid(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", left=False, labelleft=False)


def missing_text(runs, noise, arm, metric):
    if metric == "prob_reject" and DRONE_SWEEP_ARMS[arm]["binary_reward"]:
        return "not logged:\nthe binary reward pays 0 on rejection"
    if metric == "lambda" and not DRONE_SWEEP_ARMS[arm]["safe"]:
        return "no λ: unconstrained PPO"
    if not any(key[:2] == (noise, arm) for key in runs):
        return "not run yet"
    return "not logged"


def draw_curves(ax, runs, noise, arm, metric):
    """An arm's encoder x PBRS curves under one noise. False, with a placeholder instead, if there are none."""
    drawn = False
    for encoder in METHODS:
        for beta in DRONE_SWEEP_BETAS:
            run = runs.get((noise, arm, beta, encoder))
            if run is None or metric not in run["df"]:
                continue
            df = run["df"][metric]
            steps = df.index.values / 1e6
            ax.plot(steps, df["mean"].values, color=COLORS[encoder], linestyle=linestyle(beta), linewidth=1.4)
            if len(run["seeds"]) > 1:
                ax.fill_between(steps, (df["mean"] - df["std"]).values, (df["mean"] + df["std"]).values,
                                color=COLORS[encoder], alpha=0.12, linewidth=0)
            for _, step in run["incomplete"]:
                # Where an incomplete seed stops, ringed in the surface color so it reads over the other lines
                ax.plot(step / 1e6, df["mean"][step], "o", markersize=6, color=COLORS[encoder],
                        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
            drawn = True
    if not drawn:
        placeholder(ax, missing_text(runs, noise, arm, metric))
    elif metric == "prob_reject":
        ax.axhline(DELTA, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    return drawn


def share_y(axes, metric):
    """One y-scale for the panels of a metric that have data, unless its scale differs a lot between them."""
    if axes and metric not in UNSHARED:
        if metric in ("prob_success", "prob_fail"):
            low, high = -0.03, 1.03
        else:
            lows, highs = zip(*(ax.get_ylim() for ax in axes))
            low, high = min(lows), max(highs)
        for ax in axes:
            ax.set_ylim(low, high)
    for ax in axes:
        clip_rate_axis(ax, metric)


def curve_handles(runs, metrics):
    handles = [plt.Line2D([], [], color=COLORS[encoder], linestyle=linestyle(beta), linewidth=1.4,
                          label=f"{METHODS[encoder]}, {pbrs_name(beta)}")
               for encoder in METHODS for beta in DRONE_SWEEP_BETAS]
    if "prob_reject" in metrics:
        handles.append(delta_handle(DELTA))
    if any(run["incomplete"] and any(m in run["df"] for m in metrics) for run in runs.values()):
        handles.append(plt.Line2D([], [], color=INK_2, marker="o", markersize=6, markeredgecolor=SURFACE,
                                  linestyle="none", label="incomplete run stops"))
    return handles


def plot_curves(runs, rows, title, path):
    """rows: [(y label, noise, metric)], columns: the arms."""
    # At least wide enough for the legend's single row
    fig, axes = plt.subplots(len(rows), len(ARMS), figsize=(max(3.5 * len(ARMS), 10), 2.3 * len(rows) + 1.5),
                             sharex=True, squeeze=False)
    drawn = {}  # the panels with data, by metric
    for r, (ylabel, noise, metric) in enumerate(rows):
        first = True
        for c, arm in enumerate(ARMS):
            ax = axes[r, c]
            if draw_curves(ax, runs, noise, arm, metric):
                drawn.setdefault(metric, []).append(ax)
                # Shared y-scales are labeled on each row's first panel with data only.
                if not first and metric not in UNSHARED:
                    ax.tick_params(axis="y", labelleft=False)
                first = False
            if r == 0:
                ax.set_title(ARMS[arm])
            if r == len(rows) - 1:
                ax.set_xlabel("Environment steps (M)")
        axes[r, 0].set_ylabel(ylabel)
    for metric, metric_axes in drawn.items():
        share_y(metric_axes, metric)
    # Shared by all panels (sharex)
    axes[0, 0].xaxis.set_major_locator(MultipleLocator(5))
    noises = {noise for _, noise, _ in rows}
    shown = {key: run for key, run in runs.items() if key[0] in noises}
    metrics = {metric for _, _, metric in rows}
    notes = incomplete_notes(runs, [key for key in all_keys() if key[0] in noises])
    if "prob_reject" in metrics:
        notes.append(f"• {DERIVED}")
    finish(fig, title, f"{CONFIG} · {seeds_note(shown)}", curve_handles(shown, metrics), notes, path)


# Final rates


def plot_compare_rows(runs, last, path):
    # Rows from the top: each arm's header, then its encoder x PBRS configs
    ticks = []  # (y, label, whether it's an arm's header)
    rows = []  # (y, arm, beta, encoder)
    group_ends = []
    y = 0.0
    for arm in ARMS:
        ticks.append((y, ARMS[arm], True))
        y += 1
        for encoder in METHODS:
            for beta in DRONE_SWEEP_BETAS:
                ticks.append((y, f"{METHODS[encoder]} · {pbrs_name(beta)}", False))
                rows.append((y, arm, beta, encoder))
                y += 1
        group_ends.append(y - 0.3)
        y += 0.4
    offsets = np.linspace(-0.2, 0.2, len(DRONE_SWEEP_NOISES))
    best = best_within_delta(runs)
    rejects = [final(run, "prob_reject") for run in runs.values()]
    reject_top = 1.15 * max([m + np.nan_to_num(s) for m, s in rejects if not np.isnan(m)] + [DELTA])

    fig, (success_ax, reject_ax) = plt.subplots(1, 2, figsize=(11, 0.3 * y + 2.6), sharey=True,
                                                gridspec_kw=dict(width_ratios=[1.3, 1]))
    for y_row, arm, beta, encoder in rows:
        for ax in (success_ax, reject_ax):
            ax.axhline(y_row, color=GRID, linewidth=0.6, zorder=0)
        if DRONE_SWEEP_ARMS[arm]["binary_reward"]:
            reject_ax.text(0.01, y_row, "n/a (binary reward)", transform=reject_ax.get_yaxis_transform(), fontsize=7,
                           color=MUTED, va="center")
        for i, noise in enumerate(DRONE_SWEEP_NOISES):
            key, style, y_mark = (noise, arm, beta, encoder), NOISE_STYLES[i], y_row + offsets[i]
            if key not in runs:
                success_ax.text(0.01, y_mark, f"{noise_name(noise)}: not run yet", fontsize=6.5, color=MUTED,
                                transform=success_ax.get_yaxis_transform(), va="center")
                continue
            run = runs[key]
            for ax, metric in ((success_ax, "prob_success"), (reject_ax, "prob_reject")):
                mean, std = final(run, metric)
                if np.isnan(mean):
                    continue
                if not np.isnan(std):
                    ax.plot([mean - std, mean + std], [y_mark, y_mark], color=style["color"], linewidth=1.2,
                            solid_capstyle="butt", zorder=2)
                # Hollow if incomplete; else filled, ringed in the surface color where markers overlap
                hollow = bool(run["incomplete"])
                ax.plot(mean, y_mark, linestyle="none", marker=style["marker"], markersize=style["markersize"],
                        color=style["color"], markerfacecolor=SURFACE if hollow else style["color"],
                        markeredgecolor=style["color"] if hollow else SURFACE, markeredgewidth=1.4 if hollow else 0.8,
                        zorder=3)
            if best.get(noise) == key:
                mean, std = final(run, "prob_success")
                success_ax.plot(mean + np.nan_to_num(std) + 0.03, y_mark, linestyle="none", marker="*", markersize=9,
                                color=INK, zorder=3)

    success_ax.set_xlim(0, 1.06)
    success_ax.set_xlabel("Success rate (DFA accepted)")
    reject_ax.set_xlim(0, reject_top)
    reject_ax.set_xlabel("Rejection rate (DFA rejected)")
    reject_ax.axvline(DELTA, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
    reject_ax.axvspan(DELTA, reject_top, color=MUTED, alpha=0.07, linewidth=0, zorder=0)
    reject_ax.text(DELTA, 0.995, " above δ", transform=reject_ax.get_xaxis_transform(), fontsize=7, color=MUTED,
                   ha="left", va="top")
    for ax in (success_ax, reject_ax):
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
        ax.set_ylim(y - 0.4, -0.7)
        for end in group_ends[:-1]:
            ax.axhline(end + 0.5, color=MUTED, linewidth=0.6, zorder=0)
    success_ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks])
    for label, (_, _, header) in zip(success_ax.get_yticklabels(), ticks):
        label.set_fontweight("bold" if header else "normal")
        label.set_color(INK if header else INK_2)

    handles = [plt.Line2D([], [], linestyle="none", marker=s["marker"], markersize=s["markersize"], color=s["color"],
                          markeredgecolor=SURFACE, label=noise_name(noise))
               for noise, s in zip(DRONE_SWEEP_NOISES, NOISE_STYLES)]
    if any(len(run["seeds"]) > 1 for run in runs.values()):
        handles.append(plt.Line2D([], [], color=INK_2, linewidth=1.2, label="± 1 std over seeds"))
    if any(run["incomplete"] for run in runs.values()):
        handles.append(plt.Line2D([], [], linestyle="none", marker="o", markersize=7, color=INK_2,
                                  markerfacecolor=SURFACE, markeredgewidth=1.4, label="incomplete run (hollow)"))
    handles.append(delta_handle(DELTA))
    handles.append(plt.Line2D([], [], linestyle="none", marker="*", markersize=9, color=INK,
                              label="highest success with rejection ≤ δ, per noise (complete runs)"))
    missing = [key for key in all_keys() if key not in runs]
    notes = incomplete_notes(runs, all_keys())
    if BINARY_ARMS:
        notes.append(f"• {NA_BINARY}")
    notes.append(f"• {DERIVED}")
    if missing:
        notes.append(f"• {len(missing)} of the {len(all_keys())} configs are not run yet")
    finish(fig, "Every config's final success and rejection rates",
           f"{CONFIG} · mean of each run's last {last} updates · {seeds_note(runs)}", handles, notes, path,
           legend_rows=2)


def plot_compare_scatter(runs, last, path):
    noises = DRONE_SWEEP_NOISES
    binary = BINARY_ARMS
    # Per noise, a main panel and, if there are binary arms, a narrow strip for them, as their rejection rate is
    # unknown; a spacer between the noises
    widths = ([5, 1.1, 0.35] if binary else [5, 0.35]) * len(noises)
    per_noise = len(widths) // len(noises)
    fig, axes = plt.subplots(1, len(widths) - 1, figsize=(12 if binary else 10.5, 6.4), sharey=True,
                             gridspec_kw=dict(width_ratios=widths[:-1]))
    for spacer in axes[per_noise - 1::per_noise]:
        spacer.axis("off")
    rejects = [final(run, "prob_reject") for run in runs.values()]
    reject_top = 1.15 * max([m + np.nan_to_num(s) for m, s in rejects if not np.isnan(m)] + [DELTA])
    successes = [final(run, "prob_success") for run in runs.values()]
    success_low = max(0.0, math.floor(10 * (min(m - np.nan_to_num(s) for m, s in successes) - 0.05)) / 10)
    # Where each encoder x PBRS config of a binary arm sits across its strip
    strip_x = {(arm, encoder, beta): (i + 0.5) / (len(binary) * len(METHODS) * len(DRONE_SWEEP_BETAS))
               for i, (arm, encoder, beta) in enumerate(
                   (a, e, b) for a in binary for e in METHODS for b in DRONE_SWEEP_BETAS)}

    def point(ax, x, y, x_std, y_std, arm, beta, encoder, incomplete):
        color = COLORS[encoder]
        if not np.isnan(y_std) or not np.isnan(x_std):
            ax.errorbar(x, y, xerr=None if np.isnan(x_std) else x_std, yerr=None if np.isnan(y_std) else y_std,
                        color=color, linewidth=1, capsize=0, zorder=2, linestyle="none")
        # Filled with PBRS (ringed in the surface color where markers overlap), hollow without
        filled = beta is not None
        ax.plot(x, y, linestyle="none", marker=ARM_MARKERS[arm], markersize=9 if filled else 8, color=color,
                markerfacecolor=color if filled else SURFACE, markeredgecolor=SURFACE if filled else color,
                markeredgewidth=0.6 if filled else 1.5, zorder=3)
        if incomplete:
            ax.annotate("†", (x, y), xytext=(5, 3), textcoords="offset points", fontsize=8, color=INK_2)

    for n, noise in enumerate(noises):
        main = axes[per_noise * n]
        strip = axes[per_noise * n + 1] if binary else None
        for (noise_, arm, beta, encoder), run in runs.items():
            if noise_ != noise:
                continue
            success, success_std = final(run, "prob_success")
            incomplete = bool(run["incomplete"])
            if arm in binary:
                point(strip, strip_x[arm, encoder, beta], success, np.nan, success_std, arm, beta, encoder,
                      incomplete)
            else:
                reject, reject_std = final(run, "prob_reject")
                point(main, reject, success, reject_std, success_std, arm, beta, encoder, incomplete)
        main.set_xlim(0, reject_top)
        main.axvline(DELTA, color=MUTED, linewidth=1, linestyle=(0, (4, 3)), zorder=1)
        main.axvspan(DELTA, reject_top, color=MUTED, alpha=0.07, linewidth=0, zorder=0)
        main.set_xlabel("Rejection rate (DFA rejected)")
        main.set_title(noise_name(noise).capitalize())
        if not any(key[0] == noise for key in runs):
            main.text(0.5, 0.5, "not run yet", transform=main.transAxes, ha="center", va="center", color=MUTED)
        if strip is not None:
            strip.set_xlim(0, 1)
            strip.set_xticks([])
            strip.grid(axis="x", visible=False)
            strip.set_title(f"{' / '.join(ARMS[arm] for arm in binary)}\nrejection n/a", fontsize=8, color=INK_2)
    axes[0].set_ylim(success_low, 1.02)
    axes[0].set_ylabel("Success rate (DFA accepted)")

    # Color swatches rather than markers, since every marker shape stands for an arm
    handles = [Patch(color=COLORS[encoder], label=METHODS[encoder]) for encoder in METHODS]
    handles += [plt.Line2D([], [], linestyle="none", marker=ARM_MARKERS[arm], markersize=8, color=INK_2,
                           label=ARMS[arm]) for arm in ARMS]
    handles += [plt.Line2D([], [], linestyle="none", marker="o", markersize=8, color=INK_2,
                           markerfacecolor=INK_2 if beta is not None else SURFACE, markeredgewidth=1.5,
                           label=f"{pbrs_name(beta)} ({'filled' if beta is not None else 'hollow'})")
                for beta in DRONE_SWEEP_BETAS]
    handles.append(delta_handle(DELTA))
    if any(run["incomplete"] for run in runs.values()):
        handles.append(plt.Line2D([], [], linestyle="none", marker="$†$", markersize=7, color=INK_2,
                                  label="incomplete run"))
    notes = incomplete_notes(runs, all_keys())
    if binary:
        notes.append(f"• {NA_BINARY}; its configs are placed by success alone")
    notes.append(f"• {DERIVED}")
    finish(fig, "Every config's final success vs rejection rate",
           f"{CONFIG} · mean of each run's last {last} updates · {seeds_note(runs)}", handles, notes, path,
           legend_rows=2)


def summarize(runs):
    rows = []
    for key in all_keys():
        if key not in runs:
            continue
        noise, arm, beta, encoder = key
        run = runs[key]
        row = {"noise": "none" if noise is None else f"{noise[0]:g},{noise[1]:g}", "arm": arm,
               "beta": "None" if beta is None else beta, "encoder": encoder,
               "seeds": " ".join(map(str, run["seeds"])), "steps_M": round(run["steps"] / 1e6, 2),
               "incomplete": " ".join(str(seed) for seed, _ in run["incomplete"])}
        for metric, name in FINAL_METRICS.items():
            row[f"{name}_mean"], row[f"{name}_std"] = final(run, metric)
        row["within_delta"] = None if np.isnan(row["reject_mean"]) else bool(row["reject_mean"] <= DELTA)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Plot every config of train_drone_policies.sh's grid")
    parser.add_argument("--storage", default="storage",
                        help="--save-dir of the runs; their CSV logs are read from {storage}/logs (default: storage)")
    parser.add_argument("--out", default="storage/plots/drone_grid",
                        help="Output directory (default: storage/plots/drone_grid)")
    parser.add_argument("--last", type=int, default=50,
                        help="Updates at the end of each run averaged into its final rates (default: 50)")
    parser.add_argument("--format", default="pdf", help="Figure file format (default: pdf)")
    args = parser.parse_args()

    runs = load_runs(args.storage, args.last)
    if not runs:
        parser.error(f"no logs in {args.storage}/logs match train_drone_policies.sh's configs")
    noises = [noise for noise in DRONE_SWEEP_NOISES if any(key[0] == noise for key in runs)]
    metrics = list(dict.fromkeys(m for run in runs.values() for m in run["df"].columns.get_level_values(0)))

    os.makedirs(os.path.join(args.out, "metrics"), exist_ok=True)
    for noise in noises:
        plot_curves(runs, [(SHORT_LABELS[m], noise, m) for m in CURVE_METRICS],
                    f"Training curves, {noise_name(noise)}",
                    os.path.join(args.out, f"curves_{noise_tag(noise)}.{args.format}"))
    for metric in metrics:
        plot_curves(runs, [(noise_name(noise).capitalize(), noise, metric) for noise in noises],
                    LABELS.get(metric, metric), os.path.join(args.out, "metrics", f"{metric}.{args.format}"))
    plot_compare_rows(runs, args.last, os.path.join(args.out, f"compare_rows.{args.format}"))
    plot_compare_scatter(runs, args.last, os.path.join(args.out, f"compare_scatter.{args.format}"))

    summary = summarize(runs)
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    columns = ["arm", "beta", "encoder", "seeds", "steps_M", "incomplete", "success_mean", "success_std",
               "reject_mean", "reject_std", "lambda_mean", "within_delta"]
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        for noise, table in summary.groupby("noise", sort=False):
            print(f"noise {noise}, by final success rate:")
            print(table.sort_values("success_mean", ascending=False)[columns].to_string(index=False), "\n")
    missing = [key for key in all_keys() if key not in runs]
    if missing:
        print(f"Not run yet ({len(missing)}):\n" + "\n".join(f"  {config_name(key)}" for key in missing) + "\n")
    print(f"{len(runs)} of {len(all_keys())} configs ({sum(len(run['seeds']) for run in runs.values())} runs), "
          f"{len(metrics)} metrics -> {args.out}")


if __name__ == "__main__":
    main()
