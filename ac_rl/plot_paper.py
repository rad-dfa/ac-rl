"""Paper figure: success and rejection rates of train_drone_policies.sh's 2e7-step runs, its arms side by side.

The runs are the current sweep's (naming.py's DRONE_SWEEP): with the tracking controller and observation noise since
2026-11-11; the figure of the earlier sweep without them is in storage/plots/drone_policies_20M/paper.

Each arm in --arms (default: both, PPO on the shaped reward and PPO-Lagrangian) gets a success and a rejection panel.
Each panel has four curves, mean ± std over seeds: RAD vs no-RAD embeddings × PBRS β 0.7 vs no PBRS, one color each.
The colors are the dataviz reference palette's first four categorical slots, in order. The pairs of curves that
overlap in the success panels (RAD's two, no-RAD's two) are neighbors in that order, which is what its color-vision
checks cover. The rejection panels mark PPO-Lagrangian's δ, the allowed rejection rate (labeled α, the paper's name
for it). Both rates are of the unshaped reward (wrappers.py), so they compare across arms and with and without PBRS;
PPO's rejection rate is derived from its logs (see plot_drone_grid.py).

Writes --out/success_reject.{pdf,png}, and --out/success_reject.csv with each curve's final rates (the mean of each
run's last --last updates, mean and std over seeds), since the aqua and yellow curves are faint on white.

  uv run plot_paper.py
  uv run plot_paper.py --arms safe_l0      # PPO-Lagrangian only
"""
import os
import argparse
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from naming import DRONE_SWEEP, parse_timesteps
from plot_drone_grid import BASE, load_runs

INK, INK_2, MUTED, GRID, AXIS = "#0b0b0b", "#52514e", "#898781", "#e6e5df", "#b5b4a9"
# One color per curve, by (encoder, PBRS beta): the dataviz reference palette's categorical slots 1-4 (blue, orange,
# aqua, yellow), in this order.
CURVES = {
    ("rad", 0.7): dict(label="RAD, PBRS", color="#2a78d6"),
    ("rad", None): dict(label="RAD, no PBRS", color="#eb6834"),
    ("no_rad", 0.7): dict(label="No RAD, PBRS", color="#1baf7a"),
    ("no_rad", None): dict(label="No RAD, no PBRS", color="#eda100"),
}
ARMS = {"shaped": "PPO", "safe_l0": "PPO-Lagrangian"}
PANELS = {"prob_success": "Success rate", "prob_reject": "Rejection rate"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "text.color": INK,
    "xtick.color": INK_2, "ytick.color": INK_2, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
    "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 8, "axes.labelsize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 8, "legend.frameon": False,
    # TrueType fonts in the PDF: venues reject Type 3.
    "pdf.fonttype": 42, "ps.fonttype": 42,
})


def main():
    parser = argparse.ArgumentParser(description="Paper figure: success and rejection rates, RAD × PBRS, per arm")
    parser.add_argument("--storage", default="storage", help="--save-dir of the runs (default: storage)")
    parser.add_argument("--out", default="storage/plots/drone_policies_20M_tracking/paper",
                        help="Output directory (default: storage/plots/drone_policies_20M_tracking/paper)")
    parser.add_argument("--arms", nargs="+", choices=list(ARMS), default=list(ARMS),
                        help="Arms to plot, a success and a rejection panel each (default: shaped safe_l0)")
    parser.add_argument("--total-timesteps", type=parse_timesteps, default=20_000_000,
                        help="Training steps of the runs (default: 2e7, train_drone_policies.sh's)")
    parser.add_argument("--last", type=int, default=50,
                        help="Updates at the end of each run averaged into success_reject.csv's final rates (default: 50)")
    parser.add_argument("--width", type=float, default=6.75, help="Figure width in inches (default: 6.75)")
    parser.add_argument("--height", type=float, default=2.3, help="Figure height in inches (default: 2.3)")
    args = parser.parse_args()

    # The sweep's runs at --total-timesteps, by (arm, encoder, beta)
    noise = DRONE_SWEEP["obs_noise_std"]
    loaded = load_runs(args.storage, args.last, dict(BASE, total_timesteps=args.total_timesteps))
    keys = [(arm, encoder, beta) for arm in args.arms for encoder, beta in CURVES]
    missing = [key for key in keys if (noise, key[0], key[2], key[1]) not in loaded]
    if missing:
        parser.error(f"no runs for {missing} in {args.storage}/logs")
    runs = {key: loaded[noise, key[0], key[2], key[1]] for key in keys}

    fig = plt.figure(figsize=(args.width, args.height), layout="constrained")
    subfigs = fig.subfigures(1, len(args.arms), squeeze=False, wspace=0.03)[0]
    rate_axes = {metric: [] for metric in PANELS}
    for subfig, arm in zip(subfigs, args.arms):
        subfig.suptitle(ARMS[arm], fontsize=9, fontweight="semibold")
        subfig.supxlabel("Environment steps (M)", fontsize=8, color=INK_2)
        for ax, (metric, title) in zip(subfig.subplots(1, len(PANELS), sharex=True), PANELS.items()):
            for (encoder, beta), curve in CURVES.items():
                df = runs[arm, encoder, beta]["df"][metric]
                steps = df.index.values / 1e6
                ax.plot(steps, df["mean"].values, color=curve["color"], linewidth=1.3)
                ax.fill_between(steps, (df["mean"] - df["std"]).values, (df["mean"] + df["std"]).values,
                                color=curve["color"], alpha=0.12, linewidth=0)
            ax.set_title(title)
            ax.set_xlim(0, args.total_timesteps / 1e6)
            ax.xaxis.set_major_locator(MultipleLocator(10))
            rate_axes[metric].append(ax)

    # One y-scale per rate across the arms, so they compare
    for ax in rate_axes["prob_success"]:
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_locator(MultipleLocator(0.2))
    # Up to the highest mean, the first update's; the seed bands of the first few updates run past it.
    top = max(run["df"]["prob_reject"]["mean"].max() for run in runs.values())
    delta = BASE["delta"]
    for ax in rate_axes["prob_reject"]:
        ax.set_ylim(0, 1.05 * top)
        ax.yaxis.set_major_locator(MultipleLocator(0.05))
        # δ, the rejection rate PPO-Lagrangian allows
        ax.axhline(delta, color=INK_2, linewidth=1, linestyle=(0, (1, 1.5)), zorder=1.5)

    handles = [plt.Line2D([], [], color=curve["color"], linewidth=1.6, label=curve["label"])
               for curve in CURVES.values()]
    handles.append(plt.Line2D([], [], color=INK_2, linewidth=1, linestyle=(0, (1, 1.5)), label=rf"$\alpha$ = {delta:g}"))
    fig.legend(handles=handles, loc="outside upper center", ncol=len(handles), handlelength=2.0, columnspacing=1.4)

    os.makedirs(args.out, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(args.out, f"success_reject.{ext}"), dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    # The table view: each curve's final rates
    rows = []
    for (arm, encoder, beta), run in runs.items():
        success, reject = run["final"]["prob_success"], run["final"]["prob_reject"]
        rows.append({"arm": arm, "curve": CURVES[encoder, beta]["label"], "seeds": " ".join(map(str, run["seeds"])),
                     "success_mean": success[0], "success_std": success[1],
                     "reject_mean": reject[0], "reject_std": reject[1]})
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(args.out, "success_reject.csv"), index=False)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(table.to_string(index=False))
    incomplete = [f"{arm} {CURVES[encoder, beta]['label']} seed {seed}" for (arm, encoder, beta), run in runs.items()
                  for seed, _ in run["incomplete"]]
    if incomplete:
        print("Incomplete (no checkpoint yet): " + ", ".join(incomplete))
    print(f"-> {args.out}/success_reject.{{pdf,png,csv}}")


if __name__ == "__main__":
    main()
