import glob
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

parser = argparse.ArgumentParser(description="Plot AC-RL training curves from saved CSV logs.")
parser.add_argument("exp_name", nargs="?", help="Experiment name prefix (storage/{exp_name}_reach/, storage/{exp_name}_reach_avoid/)")
parser.add_argument("--drone-policy", action="store_true", help="Plot drone-policy RAD vs No-RAD sweep results from storage/log_drone_*.csv, one folder per sampler")
args = parser.parse_args()


def aggregate(groups):
    # Dictionary to hold mean/std for each group
    results = {}

    for label, log_files in groups.items():
        if not log_files:
            print(f"Warning: No files found for {label}")
            continue

        dfs = [pd.read_csv(f) for f in log_files]

        # ensure unique timesteps per df (take mean if duplicates exist)
        for i in range(len(dfs)):
            if dfs[i]["timestep"].duplicated().any():
                dfs[i] = dfs[i].groupby("timestep", as_index=False).mean()

        # union of all timesteps
        all_timesteps = sorted(set().union(*[df["timestep"].values for df in dfs]))

        # reindex each df to have the full timestep range
        reindexed_dfs = []
        for df in dfs:
            df = df.set_index("timestep").reindex(all_timesteps)
            reindexed_dfs.append(df.reset_index().rename(columns={"index": "timestep"}))

        timesteps = np.array(all_timesteps)
        base_columns = [c for c in dfs[0].columns if c != "timestep"]

        data_mean, data_std = {}, {}
        for col in base_columns:
            values = np.stack([df[col].values for df in reindexed_dfs], axis=1)  # (T, num_seeds)
            data_mean[col] = np.nanmean(values, axis=1)
            data_std[col] = np.nanstd(values, axis=1)

        results[label] = {
            "timesteps": timesteps,
            "mean": data_mean,
            "std": data_std,
            "columns": base_columns,
        }

    return results


def plot(results, groups, colors, out_dir, title):
    if not results:
        print(f"Warning: Nothing to plot for {title}")
        return

    # Create folder to save plots
    os.makedirs(out_dir, exist_ok=True)

    for col in next(iter(results.values()))["columns"]:
        plt.figure(figsize=(8, 5))
        for label in groups.keys():
            if label not in results:
                continue
            timesteps = results[label]["timesteps"]
            mean = results[label]["mean"][col]
            std = results[label]["std"][col]

            plt.plot(timesteps, mean, label=label, color=colors[label])
            plt.fill_between(timesteps, mean - std, mean + std, alpha=0.2, color=colors[label])

        plt.xlabel("timestep")
        plt.ylabel(col)
        plt.title(f"{title} -- {col}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        plt.savefig(os.path.join(out_dir, f"{col}.pdf"))
        plt.close()

    print(f"✅ Plots saved in {out_dir}")


if args.drone_policy:
    drone_csvs = glob.glob("storage/log_drone_*.csv")
    colors = {"RAD": "tab:blue", "No RAD": "tab:orange"}
    # Output folder name -> sampler name as it appears in the log file names.
    # Matched case-sensitively so the "RAD" sampler is not confused with the "rad"/"no_rad" embedding tag.
    samplers = {"R": "Reach", "RA": "ReachAvoid", "RAD": "RAD"}
    for short, full in samplers.items():
        files = [f for f in drone_csvs if f"_{full}_" in os.path.basename(f)]
        groups = {
            "RAD": [f for f in files if "_no_rad_" not in f],
            "No RAD": [f for f in files if "_no_rad_" in f],
        }
        plot(aggregate(groups), groups, colors,
             out_dir=os.path.join("storage", "plots", "drone_policy", short),
             title=f"Drone Policy ({full}): RAD vs No-RAD")
else:
    if not args.exp_name:
        parser.error("exp_name is required unless --drone-policy is given")
    groups = {
        "Reach": glob.glob(f"storage/{args.exp_name}_reach/log_*.csv"),
        "ReachAvoid": glob.glob(f"storage/{args.exp_name}_reach_avoid/log_*.csv"),
    }
    colors = {"Reach": "tab:blue", "ReachAvoid": "tab:orange"}
    plot(aggregate(groups), groups, colors,
         out_dir=os.path.join("storage", "plots", args.exp_name),
         title=args.exp_name)
