"""Tests many drone policies (train_drone_policy.py checkpoints) at once and summarizes each config over its seeds.

Every setting comes from the checkpoint's name (naming.py), so nothing has to be passed per run: each checkpoint is
built by test_drone_policy.setup exactly as test_drone_policy.py would build it from matching arguments, and runs
--n episodes on its training task distribution. All policies get the same episode keys, so with the same sampler
they face the same DFAs. Writes, under --out:
  per_seed.csv   each checkpoint's success / reject / timeout rates, mean episode length and discounted return
  summary.csv    the same per config (the run name without its seed): mean and std over seeds

  uv run test_drone_policies.py storage/models/policy_params_drone_seed_*_total20000000_*.msgpack
"""
import os
import re
import time
import argparse
import numpy as np
import pandas as pd
import jax
from naming import parse_checkpoint_name
from test_drone_policy import setup, run_episodes

METRICS = ["success", "reject", "timeout", "ep_len", "disc_return"]


def test(path, n, batch_size, deterministic):
    cfg = parse_checkpoint_name(path)
    if cfg is None:
        raise ValueError(f"Unrecognized checkpoint filename: {os.path.basename(path)}")
    # setup reads test_drone_policy.py's arguments, which are named like the run settings.
    args = argparse.Namespace(**cfg, model_path=path, eval_sampler=None, deterministic=deterministic)
    _, env, network, params, run_tag, _ = setup(args)
    successes, fails, ep_lens, _, disc_returns, _, _ = run_episodes(
        env, network, params, jax.random.split(jax.random.PRNGKey(0), n), deterministic,
        cfg["max_steps_in_episode"], batch_size, gamma=cfg["gamma"]
    )
    successes, fails = np.asarray(successes), np.asarray(fails)
    return {
        # The run name with its seed left out, shared by a config's seeds
        "config": re.sub(r"^drone_seed_\d+_", "drone_", run_tag),
        "seed": cfg["seed"],
        "success": successes.mean(),
        "reject": fails.mean(),
        "timeout": (~(successes | fails)).mean(),
        "ep_len": float(np.mean(ep_lens)),
        "disc_return": float(np.mean(disc_returns)),
    }


def main():
    parser = argparse.ArgumentParser(description="Test drone policies and summarize each config over its seeds")
    parser.add_argument("ckpts", nargs="+", help="train_drone_policy.py checkpoints (.msgpack)")
    parser.add_argument("--n", type=int, default=1000, help="Test episodes per checkpoint (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=-1, help="Episodes vmapped at once (default: n)")
    parser.add_argument("--deterministic", action="store_true", help="Act with the policy mean instead of sampling")
    parser.add_argument("--out", default="storage/test/drone_policies",
                        help="Output directory (default: storage/test/drone_policies)")
    args = parser.parse_args()
    for path in args.ckpts:
        if parse_checkpoint_name(path) is None:  # fail on an unrecognized name before testing any
            parser.error(f"unrecognized checkpoint filename: {os.path.basename(path)}")
    batch_size = args.n if args.batch_size == -1 else args.batch_size
    if args.deterministic:
        args.out += "_det"
    os.makedirs(args.out, exist_ok=True)

    rows = []
    for i, path in enumerate(args.ckpts):
        t0 = time.time()
        rows.append(test(path, args.n, batch_size, args.deterministic))
        print(f"[{i + 1}/{len(args.ckpts)}] seed {rows[-1]['seed']} {rows[-1]['config']}\n    "
              + ", ".join(f"{m} {rows[-1][m]:.3f}" for m in METRICS) + f" ({time.time() - t0:.0f}s)", flush=True)
        # Rewritten after every checkpoint, so an interrupted run keeps what it finished.
        per_seed = pd.DataFrame(rows)
        per_seed.to_csv(os.path.join(args.out, "per_seed.csv"), index=False)

    summary = per_seed.groupby("config").agg(
        seeds=("seed", lambda s: " ".join(map(str, sorted(s)))),
        **{f"{m}_{stat}": (m, stat) for m in METRICS for stat in ("mean", "std")},
    ).reset_index()
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.max_colwidth", None,
                           "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))
    print(f"\n{len(rows)} checkpoints, {len(summary)} configs, {args.n} episodes each -> {args.out}")


if __name__ == "__main__":
    main()
