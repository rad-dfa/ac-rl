"""Paper table: tests train_drone_policies.sh's 2e7-step policies on their training tasks and on larger ones.

Tests every config of the sweep (naming.py's DRONE_SWEEP with DRONE_SWEEP_ARMS × DRONE_SWEEP_BETAS × RAD/no-RAD:
PPO or PPO-Lagrangian, PBRS β 0.7 or none, pretrained RAD embeddings or not), seeds --seeds, on two task sets:
  in   the training tasks: Mixture DFAs of up to 5 states, p None
  out  the same, but DFAs of up to --ood-max-size (default 10) states
Each checkpoint runs --n (default 1000) episodes per task set, with the training observation noise and the unshaped
reward, actions sampled unless --deterministic. All checkpoints get the same episode keys, so on a task set they face
the same DFAs from the same start states. On the larger tasks every encoder keeps its params (RAD's is the pretrained
one it was trained with) and runs a round of message passing per DFA state, as on every training DFA
(test_drone_policy.py's --eval-max-size).

The checkpoints that share a network (arm × encoder; PBRS only changes the training reward) roll out together, vmapped
over checkpoints and episodes: one compile per network and task set (8), each running (betas × seeds) × --n episodes
at once. --batch-size splits the episodes if that doesn't fit in memory.

Writes, under --out (default storage/test/paper_table, + _det with --deterministic):
  per_seed.csv  each checkpoint's rates per task set: success, reject, other (timed out or left the box), mean episode
                length and discounted return; rewritten after every network and task set
  summary.csv   the same per config and task set: mean and std over seeds
  table.tex     success and rejection rates per config (rows) and task set (columns), mean ± std over seeds, the best
                success per task set in bold; needs booktabs and amssymb

  uv run test_paper_table.py
  uv run test_paper_table.py --deterministic
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from ppo import batchify
from naming import DRONE_SWEEP, DRONE_SWEEP_ARMS, DRONE_SWEEP_BETAS, drone_run_name, checkpoint_path
from test_drone_policy import setup

# train_drone_policies.sh's task and env settings; DRONE_SWEEP has the rest
BASE = dict(DRONE_SWEEP, sampler="MIX", max_size=5, n_tokens=10, p=None, use_velocity_filter=True)
ALGORITHMS = {"shaped": "PPO", "safe_l0": "PPO-Lagrangian"}
ENCODERS = {"rad": False, "no_rad": True}  # by no_rad
METRICS = ["success", "reject", "other", "ep_len", "disc_return"]
KEYS = ["tasks", "max_size", "algorithm", "encoder", "pbrs"]


def make_rollout(env, network, deterministic, max_steps, gamma):
    """Jitted: one episode per (checkpoint, key), from params stacked over checkpoints and a batch of keys -> success,
    reject, ep_len and disc_return, each (checkpoints, keys). Steps like test_drone_policy.run_episodes, without
    keeping the trajectory."""

    def episode(params, key):
        key, subkey = jax.random.split(key)
        obs, state = env.reset(subkey)
        carry = {
            "key": key,
            "obs": obs,
            "state": state,
            "done": jnp.array(False),
            "success": jnp.array(False),
            "reject": jnp.array(False),
            "ep_len": jnp.array(0),
            "disc_return": jnp.array(0.0),
        }

        def step(carry, _):
            key, act_key, step_key = jax.random.split(carry["key"], 3)
            out, _ = network.apply(params, batchify(carry["obs"], env.agents))
            actions = out if deterministic else out.sample(seed=act_key)
            actions = {agent: actions[i] for i, agent in enumerate(env.agents)}
            # step_env rather than step: step auto-resets on done.
            obs, state, rewards, dones, _ = env.step_env(step_key, carry["state"], actions)
            done = dones["__all__"]
            reward = jnp.mean(jnp.array([rewards[a] for a in env.agents]))
            # Rejected: the DFA minimized to a rejecting sink
            rejected = jnp.any(jnp.array([
                (state.dfas[a].n_states <= 1) & (state.dfas[a].reward(binary=False) < 0) for a in env.agents
            ]))
            new = {
                "key": key,
                "obs": obs,
                "state": state,
                "done": done,
                "success": done & (reward > 0),
                "reject": done & rejected,
                "ep_len": carry["ep_len"] + 1,
                "disc_return": carry["disc_return"] + reward * gamma ** carry["ep_len"],
            }
            # Freeze the carry once the episode has ended.
            return jax.tree_util.tree_map(lambda old, nw: jnp.where(carry["done"], old, nw), carry, new), None

        carry, _ = jax.lax.scan(step, carry, None, length=max_steps)
        return carry["success"], carry["reject"], carry["ep_len"], carry["disc_return"]

    return jax.jit(jax.vmap(jax.vmap(episode, (None, 0)), (0, None)))


def sweep(seeds, storage):
    """{(arm, encoder): [(beta, seed, cfg, checkpoint path)]}: the sweep's runs, grouped by network."""
    groups = {}
    for arm, arm_cfg in DRONE_SWEEP_ARMS.items():
        for encoder, no_rad in ENCODERS.items():
            runs = []
            for beta in DRONE_SWEEP_BETAS:
                for seed in seeds:
                    cfg = dict(BASE, **arm_cfg, beta=beta, seed=seed, no_rad=no_rad)
                    runs.append((beta, seed, cfg, checkpoint_path(storage, drone_run_name(cfg))))
            groups[arm, encoder] = runs
    return groups


def test_group(runs, max_size, keys, batch_size, deterministic):
    """Rolls out every run of a group (same network) on the tasks of up to max_size states, one episode per key ->
    success, reject, ep_len, disc_return as numpy arrays of shape (runs, keys)."""
    loaded = []
    for _, _, cfg, path in runs:
        args = argparse.Namespace(**cfg, model_path=path, eval_sampler=None, eval_max_size=max_size,
                                  deterministic=deterministic)
        _, env, network, params, _, _ = setup(args)
        loaded.append((env, network, params))
    env, network, _ = loaded[0]
    if not runs[0][2]["no_rad"]:
        # The frozen RAD encoder is a constant of the rollout, so every run must have loaded the same one.
        for _, other, _ in loaded[1:]:
            same = jax.tree_util.tree_map(np.array_equal, network.encoder.encoder_params, other.encoder.encoder_params)
            if not jax.tree_util.tree_all(same):
                raise ValueError("The runs of a group loaded different pretrained RAD encoders; test them apart")
    params = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[run_params for _, _, run_params in loaded])
    rollout = make_rollout(env, network, deterministic, BASE["max_steps_in_episode"], BASE["gamma"])
    chunks = [jax.device_get(rollout(params, keys[start:start + batch_size]))
              for start in range(0, keys.shape[0], batch_size)]
    return [np.concatenate([np.asarray(chunk[i]) for chunk in chunks], axis=1) for i in range(4)]


def rate(mean, std, bold=False):
    mean_str = rf"\mathbf{{{mean:.3f}}}" if bold else f"{mean:.3f}"
    return f"${mean_str}$" if np.isnan(std) else rf"${mean_str} \pm {std:.3f}$"


def latex_table(summary, task_sets):
    """booktabs/amssymb table: rows algorithm × RAD × PBRS, columns success and rejection per task set."""
    rows = {(r.tasks, r.algorithm, r.encoder, bool(r.pbrs)): r for r in summary.itertuples(index=False)}
    tasks = list(task_sets.items())
    best = {name: max(round(r.success_mean, 3) for key, r in rows.items() if key[0] == name) for name, _ in tasks}
    lines = [
        r"\begin{tabular}{lcc" + "cc" * len(tasks) + "}",
        r"\toprule",
        " & & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{'In' if name == 'in' else 'Out of'} distribution "
                               rf"($\le {size}$ states)}}" for name, size in tasks) + r" \\",
        " ".join(rf"\cmidrule(lr){{{4 + 2 * i}-{5 + 2 * i}}}" for i in range(len(tasks))),
        "Algorithm & RAD & PBRS & " + " & ".join("Success & Rejection" for _ in tasks) + r" \\",
    ]
    for arm, algorithm in ALGORITHMS.items():
        if arm not in DRONE_SWEEP_ARMS:
            continue
        lines.append(r"\midrule")
        first = True
        for encoder in ENCODERS:
            for beta in DRONE_SWEEP_BETAS:
                cells = [algorithm if first else "", r"\checkmark" if encoder == "rad" else "--",
                         r"\checkmark" if beta is not None else "--"]
                first = False
                for name, size in tasks:
                    row = rows[name, algorithm, encoder, beta is not None]
                    cells.append(rate(row.success_mean, row.success_std,
                                      bold=round(row.success_mean, 3) == best[name]))
                    cells.append(rate(row.reject_mean, row.reject_std))
                lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description="Paper table: drone policies on in- and out-of-distribution tasks")
    parser.add_argument("--storage", default="storage", help="--save-dir of the runs (default: storage)")
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5], help="Seeds (default: 1 2 3 4 5)")
    parser.add_argument("--ood-max-size", type=int, default=10,
                        help="Max DFA size of the out-of-distribution tasks (default: 10)")
    parser.add_argument("--n", type=int, default=1000, help="Episodes per checkpoint and task set (default: 1000)")
    parser.add_argument("--batch-size", type=int, default=-1,
                        help="Episodes per checkpoint vmapped at once; lower it if a rollout runs out of memory "
                             "(default: n)")
    parser.add_argument("--deterministic", action="store_true", help="Act with the policy mean instead of sampling")
    parser.add_argument("--key", type=int, default=0, help="Seed of the episode keys (default: 0)")
    parser.add_argument("--out", default="storage/test/paper_table",
                        help="Output directory (default: storage/test/paper_table, + _det with --deterministic)")
    args = parser.parse_args()
    batch_size = args.n if args.batch_size == -1 else args.batch_size
    if args.deterministic:
        args.out += "_det"

    groups = sweep(args.seeds, args.storage)
    missing = [path for runs in groups.values() for *_, path in runs if not os.path.exists(path)]
    if missing:
        parser.error(f"{len(missing)} missing checkpoints:\n  " + "\n  ".join(missing))
    task_sets = {"in": BASE["max_size"], "out": args.ood_max_size}
    os.makedirs(args.out, exist_ok=True)
    keys = jax.random.split(jax.random.PRNGKey(args.key), args.n)
    n_runs = sum(len(runs) for runs in groups.values())
    print(f"JAX devices: {jax.devices()}")
    print(f"{n_runs} checkpoints × {len(task_sets)} task sets × {args.n} episodes, "
          f"{'deterministic' if args.deterministic else 'sampled'} actions -> {args.out}", flush=True)

    rows = []
    start = time.time()
    for i, ((arm, encoder), runs) in enumerate(groups.items()):
        for name, max_size in task_sets.items():
            t0 = time.time()
            success, reject, ep_len, disc_return = test_group(runs, max_size, keys, batch_size, args.deterministic)
            for j, (beta, seed, _, path) in enumerate(runs):
                rows.append({
                    "tasks": name, "max_size": max_size, "algorithm": ALGORITHMS.get(arm, arm), "encoder": encoder,
                    "pbrs": beta is not None, "beta": beta, "seed": seed,
                    "success": success[j].mean(), "reject": reject[j].mean(),
                    "other": (~(success[j] | reject[j])).mean(),
                    "ep_len": ep_len[j].mean(), "disc_return": disc_return[j].mean(),
                    "checkpoint": os.path.basename(path),
                })
            done = pd.DataFrame(rows[-len(runs):]).groupby("pbrs", sort=False)[["success", "reject"]].mean()
            print(f"[{i + 1}/{len(groups)}] {ALGORITHMS.get(arm, arm)}, {encoder}, tasks {name} (≤ {max_size} states), "
                  f"{len(runs)} checkpoints in {time.time() - t0:.0f}s: "
                  + "; ".join(f"{'PBRS' if pbrs else 'no PBRS'} success {r.success:.3f} reject {r.reject:.3f}"
                              for pbrs, r in done.iterrows()), flush=True)
            # Rewritten after every group, so an interrupted run keeps what it finished.
            per_seed = pd.DataFrame(rows)
            per_seed.to_csv(os.path.join(args.out, "per_seed.csv"), index=False)

    summary = per_seed.groupby(KEYS, sort=False).agg(
        seeds=("seed", lambda s: " ".join(map(str, sorted(s)))),
        **{f"{m}_{stat}": (m, stat) for m in METRICS for stat in ("mean", "std")},
    ).reset_index()
    summary.to_csv(os.path.join(args.out, "summary.csv"), index=False)
    with open(os.path.join(args.out, "table.tex"), "w") as f:
        f.write(latex_table(summary, task_sets))
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(summary.drop(columns=["ep_len_std", "disc_return_std"]).to_string(index=False))
    print(f"\nDone in {(time.time() - start) / 60:.1f} min -> {args.out}/{{per_seed.csv,summary.csv,table.tex}}")


if __name__ == "__main__":
    main()
