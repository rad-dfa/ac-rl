import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 18,
    "axes.titleweight": "bold",
})

tasks = {
    "RAD Embd; 5 Events": "log_seed_*_RAD_5_5_rad_n_events_5.csv",
    "RAD Embd; 10 Events": "log_seed_*_RAD_5_5_rad_n_events_10.csv",
    "RAD Embd; 20 Events": "log_seed_*_RAD_5_5_rad_n_events_20.csv",
    "No RAD Embd; 5 Events": "log_seed_*_RAD_5_5_no_rad_n_events_5.csv",
    "No RAD Embd; 10 Events": "log_seed_*_RAD_5_5_no_rad_n_events_10.csv",
    "No RAD Embd; 20 Events": "log_seed_*_RAD_5_5_no_rad_n_events_20.csv",
}

plot_cols = [
    ("prob_success", "Success Probability", (0.0, 1.0)),
    ("disc_return_mean", "Discounted Return", (0.0, 1.2)),
]

env_name = "TokenEnv"
TARGET_SIDE_MARGIN = 0.03


def fit_content_horizontally(fig, axes, leg_ax, target_margin=TARGET_SIDE_MARGIN):
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bboxes = [
        ax.get_tightbbox(renderer).transformed(fig.transFigure.inverted())
        for ax in axes
    ]
    legend_bbox = leg_ax.get_legend().get_window_extent(renderer).transformed(
        fig.transFigure.inverted()
    )
    content_x0 = min(bb.x0 for bb in bboxes + [legend_bbox])
    content_x1 = max(bb.x1 for bb in bboxes + [legend_bbox])
    content_width = content_x1 - content_x0
    available_width = 1.0 - 2 * target_margin
    scale = available_width / content_width

    def rescale_x(x):
        return target_margin + (x - content_x0) * scale

    for artist in [*axes, leg_ax]:
        pos = artist.get_position()
        artist.set_position([
            rescale_x(pos.x0),
            pos.y0,
            pos.width * scale,
            pos.height,
        ])


def load_results(storage_dir):
    results = {}
    for task, pattern in tasks.items():
        log_files = glob.glob(os.path.join(storage_dir, pattern))
        if not log_files:
            print(f"Warning: No files for {task} in {storage_dir}")
            continue

        dfs = [pd.read_csv(f) for f in log_files]

        for i in range(len(dfs)):
            if dfs[i]["timestep"].duplicated().any():
                dfs[i] = dfs[i].groupby("timestep", as_index=False).mean()

        all_timesteps = sorted(set().union(*[df["timestep"].values for df in dfs]))

        reindexed_dfs = []
        for df in dfs:
            df = df.set_index("timestep").reindex(all_timesteps)
            reindexed_dfs.append(df.reset_index().rename(columns={"index": "timestep"}))

        timesteps = np.array(all_timesteps)
        base_columns = [c for c in dfs[0].columns if c != "timestep"]

        data_mean, data_std = {}, {}
        for col in base_columns:
            values = np.stack([df[col].values for df in reindexed_dfs], axis=1)
            data_mean[col] = np.nanmean(values, axis=1)
            data_std[col] = np.nanstd(values, axis=1)

        results[task] = {
            "timesteps": timesteps,
            "mean": data_mean,
            "std": data_std,
        }
    return results


def main():
    parser = argparse.ArgumentParser(description="Plot TokenEnv training curves across policies and seeds")
    parser.add_argument(
        "--save-dir",
        type=str,
        default="storage",
        help="Directory containing log CSV files (default: storage)",
    )
    args = parser.parse_args()

    results = load_results(args.save_dir)
    if not results:
        print(f"No log files found in {args.save_dir}")
        return

    plot_dir = os.path.join(args.save_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    fig = plt.figure(figsize=(17, 6.5))
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1, 0.20],
        hspace=0.28,
        wspace=0.15,
        top=0.92,
        bottom=0.05,
        left=TARGET_SIDE_MARGIN,
        right=1.0 - TARGET_SIDE_MARGIN,
    )
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    leg_ax = fig.add_subplot(gs[1, :])
    leg_ax.axis("off")

    for ax, (col, ylabel, ylim) in zip(axes, plot_cols):
        for task in tasks:
            if task not in results:
                continue
            timesteps = results[task]["timesteps"]
            mean = results[task]["mean"][col]
            std = results[task]["std"][col]

            ax.plot(timesteps, mean, label=task, linewidth=2.5)
            ax.fill_between(timesteps, mean - std, mean + std, alpha=0.3)

        ax.set_xlabel("Timestep")
        ax.set_ylabel(ylabel)
        ax.set_title(env_name)
        ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bb_left = axes[0].get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    bb_right = axes[1].get_tightbbox(renderer).transformed(fig.transFigure.inverted())
    leg_pos = leg_ax.get_position()
    leg_ax.set_position([
        bb_left.x0,
        leg_pos.y0,
        bb_right.x1 - bb_left.x0,
        leg_pos.height,
    ])

    leg_ax.legend(
        handles,
        labels,
        loc="center",
        ncol=3,
        frameon=True,
        fancybox=False,
        facecolor="none",
        edgecolor="#c8c8c8",
        borderaxespad=0.8,
    )
    fit_content_horizontally(fig, axes, leg_ax)

    pdf_path = os.path.join(plot_dir, "combined.pdf")
    plt.savefig(pdf_path)
    plt.close()
    print(f"Saved {pdf_path}")
    print(f"Plots saved in {plot_dir}")


if __name__ == "__main__":
    main()

