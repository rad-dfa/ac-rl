"""Packages the drone policies in POLICIES for use outside this repo, with example DFA tasks to evaluate them on.

  uv run export_drone_bundle.py [--dfas drone_dfas.json] [--out storage/share/drone_rad_max_size5] [--n 100]

The tasks are defined by hand in --dfas, a JSON list of DFAs over DroneEnv's tokens 0-9, each like

  {"name": "dfa_0", "note": "reach 8; avoid 0", "start": 0, "accepting": [2], "transitions": {"0": {"0": 1, "8": 2}}}

with states 0-4 (the policies' max DFA size is 5). transitions[state][token] is the next state; any (state, token)
not listed stays in its state, so a non-accepting state without transitions is a rejecting (failure) state.
Accepting states can't have transitions: the task ends when one is reached. "note" is optional. The default
drone_dfas.json holds the bundle's examples, drawn from the policies' training sampler (RAD, max size 5, p 0.5):
the first PRNG keys whose DFA has a rejecting state, one DFA with 3 states, two with 4 and two with 5.

Writes under --out, replacing the dfas/ and reference/ files of an earlier run:
  README.md, run_example.py     copied from drone_bundle/, where they are written by hand (README.md's result tables
                                are filled in by hand from reference/rollouts.csv and test_drone_policy.py)
  policies/<name>.onnx          export_onnx.py's model of each policy, checked against the JAX network
  policies.json                 each policy's checkpoint and whether it acts through the velocity filter
  env.json                      DroneEnv's box, start state, step sizes, velocity-filter parameters and token regions
  dfas/<name>.json              each task, with the policy's DFA inputs precomputed for every DFA state
  reference/rollouts.csv        each policy on each DFA in DroneEnv: the outcome of the deterministic (mean action)
                                episode, and success/reject/timeout rates over --n episodes sampling actions
  reference/<policy>_<dfa>.gif  the deterministic episodes
"""
import os
import re
import csv
import glob
import json
import shutil
import argparse
import jax
import jax.numpy as jnp
import numpy as np
from dfax import DFAx
from dfa_gym import animate_drone_trace
from export_onnx import export
from test_drone_policy import run_episodes
from naming import DRONE_SWEEP, DEFAULT_TOTAL_TIMESTEPS, drone_run_name, checkpoint_path

# train_drone_policies.sh's runs (lambda_0 = 1, clip [0, 1], gamma 0.999, PBRS beta 0.7) with seed 42, the MIX
# sampler, max size 5, p 0.5 and the pretrained RAD encoder, from before it went to 2e7 steps.
POLICIES = {
    name: checkpoint_path("storage", drone_run_name(dict(
        DRONE_SWEEP, seed=42, sampler="MIX", max_size=5, n_tokens=10, p=0.5, no_rad=False, use_velocity_filter=vfilt,
        total_timesteps=DEFAULT_TOTAL_TIMESTEPS,
    )))
    for name, vfilt in (("rad_no_filter", False), ("rad_velocity_filter", True))
}
# The bundle's hand-written README.md and run_example.py.
BUNDLE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "drone_bundle")


def load_dfas(path, max_size, n_tokens):
    """--dfas' tasks as {name: (DFAx padded to max_size states, spec)}; see the module docstring for the format."""
    def no_duplicate_keys(pairs):
        # json.load would otherwise keep the last of a repeated state or token and silently drop the rest.
        keys = [k for k, _ in pairs]
        duplicates = sorted({k for k in keys if keys.count(k) > 1})
        if duplicates:
            raise ValueError(f"{path}: repeated key(s) {duplicates} in {dict(pairs)}")
        return dict(pairs)

    with open(path) as f:
        specs = json.load(f, object_pairs_hook=no_duplicate_keys)
    dfas = {}
    for spec in specs:
        name = spec.get("name")

        def check(ok, msg):
            if not ok:
                raise ValueError(f"{path}, DFA {name!r}: {msg}")

        check(isinstance(name, str) and re.fullmatch(r"[\w.-]+", name), "needs a file-name-safe \"name\"")
        check(name not in dfas, "duplicate name")
        accepting = spec.get("accepting", [])
        check(0 <= spec.get("start", -1) < max_size, f"\"start\" must be a state 0-{max_size - 1}")
        check(all(0 <= q < max_size for q in accepting), f"accepting states must be 0-{max_size - 1}")
        transitions = np.tile(np.arange(max_size, dtype=np.int32)[:, None], (1, n_tokens))
        for state, row in spec.get("transitions", {}).items():
            for token, nxt in row.items():
                q, a = int(state), int(token)
                check(0 <= q < max_size and 0 <= nxt < max_size, f"states must be 0-{max_size - 1}")
                check(0 <= a < n_tokens, f"tokens must be 0-{n_tokens - 1}")
                check(q not in accepting or nxt == q,
                      f"accepting state {q} has a transition; the task ends when an accepting state is reached")
                transitions[q, a] = nxt
        labels = np.zeros(max_size, dtype=bool)
        labels[accepting] = True
        dfa = DFAx.create(start=spec["start"], transitions=jnp.asarray(transitions), labels=jnp.asarray(labels))
        check(int(dfa.minimize().n_states) > 1, "the task is already decided at the start state")
        dfas[name] = dfa, spec
    return dfas


def same_graph(g, h):
    return all(np.array_equal(np.asarray(g[k]), np.asarray(h[k])) for k in g)


def dfa_table(dfa):
    """(transitions, status, graphs) of a DFAx's states 0..k-1, k - 1 being the highest state reachable from the
    start; state ids are kept as given.

    graphs[q] is DFAx.to_graph() of the DFA as DFAWrapper holds it once the task has reached state q: started at q
    and minimized. DFAWrapper doesn't look states up; after every step it advances the DFA it holds on the step's
    label and minimizes the result. This checks that doing so always lands on the graph of the next state in
    `transitions`, so stepping through the table gives the policy exactly the inputs it saw in training.
    """
    is_reach = np.asarray(dfa.is_reach)
    k = int(np.flatnonzero(is_reach).max()) + 1
    T = np.asarray(dfa.transitions)[:k]
    residuals = [DFAx.create(start=q, transitions=dfa.transitions, labels=dfa.labels).minimize() for q in range(k)]
    graphs = [r.to_graph() for r in residuals]
    # DFAWrapper ends the episode once the DFA minimizes to a single state, accepting or rejecting. That includes a
    # state with transitions from which no accepting state can be reached.
    status = ["running" if int(r.n_states) > 1 else "accept" if bool(r.labels[r.start]) else "reject"
              for r in residuals]

    assert same_graph(dfa.minimize().to_graph(), graphs[int(dfa.start)])
    for q, residual in enumerate(residuals):
        if not is_reach[q]:
            continue
        for sym in range(-1, dfa.n_tokens):  # -1: no token
            nxt = q if sym < 0 else T[q, sym]
            assert same_graph(residual.advance(sym).minimize().to_graph(), graphs[nxt]), (q, sym)
    return T, status, graphs


def describe(T, status):
    lines = []
    for q, row in enumerate(T):
        if status[q] != "running":
            lines.append(f"state {q}: {status[q]} (episode ends)")
            continue
        moves = {}
        for token, nxt in enumerate(row):
            if nxt != q:
                moves.setdefault(int(nxt), []).append(token)
        parts = [
            f"token{'s' if len(tokens) > 1 else ''} {', '.join(map(str, tokens))} -> state {nxt}"
            + (f" ({status[nxt]})" if status[nxt] != "running" else "")
            for nxt, tokens in moves.items()
        ]
        lines.append(f"state {q}: " + "; ".join(parts) + "; any other token, or none: stay")
    return lines


def dumps(obj):
    # Indented JSON, but with every innermost list of numbers on one line.
    s = json.dumps(obj, indent=2)
    return re.sub(r"\[\s+([-\d.,\se]+?)\s+\]", lambda m: "[" + re.sub(r"\s+", " ", m.group(1)) + "]", s) + "\n"


def env_spec(drone_env):
    regions = []
    for token, kind, params in drone_env.label_regions():
        params = [float(x) for x in params]
        if kind == "circle":
            cx, cy, r, z_lo, z_hi = params
            regions.append({"token": token, "kind": kind, "center_xy": [cx, cy], "radius": r, "z": [z_lo, z_hi]})
        else:
            x_lo, x_hi, y_lo, y_hi, z_lo, z_hi = params
            regions.append({"token": token, "kind": kind, "x": [x_lo, x_hi], "y": [y_lo, y_hi], "z": [z_lo, z_hi]})
    return {
        "box": {"low": np.asarray(drone_env.low).tolist(), "high": np.asarray(drone_env.high).tolist()},
        "start": {"position": [0.0, 0.0, 0.0], "velocity": [0.0, 0.0, 0.0]},
        "dt": drone_env.dt,
        "max_speed": drone_env.max_speed,
        "max_steps_in_episode": drone_env.max_steps_in_episode,
        "velocity_filter": {
            "tau": drone_env.velocity_filter_tau,
            "max_accel": drone_env.max_filter_accel,
            "filter_dt": drone_env.filter_dt,
        },
        "n_tokens": drone_env.n_tokens,
        "regions": regions,
    }


def main():
    parser = argparse.ArgumentParser(description="Package drone policies as ONNX with example DFA tasks")
    parser.add_argument("--dfas", default="drone_dfas.json",
                        help="Hand-defined DFA tasks, see the module docstring (default: drone_dfas.json)")
    parser.add_argument("--out", default="storage/share/drone_rad_max_size5",
                        help="Output directory (default: storage/share/drone_rad_max_size5)")
    parser.add_argument("--n", type=int, default=100,
                        help="Episodes per policy and DFA for the sampled-action rates (default: 100)")
    args = parser.parse_args()
    for sub in ("policies", "dfas", "reference"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)
    # Drop an earlier run's tasks and results, so no file is left from a DFA that's no longer in --dfas.
    for path in glob.glob(os.path.join(args.out, "dfas", "*.json")) + \
            glob.glob(os.path.join(args.out, "reference", "*.gif")) + \
            glob.glob(os.path.join(args.out, "reference", "rollouts.csv")):
        os.remove(path)
    for name in ("README.md", "run_example.py"):
        shutil.copy(os.path.join(BUNDLE_DIR, name), args.out)

    policies = {}
    for name, ckpt_path in POLICIES.items():
        out_path = os.path.join(args.out, "policies", f"{name}.onnx")
        cfg, env, network, params, d_action, d_value = export(ckpt_path, out_path)
        print(f"{name}: {out_path} (max |Δ action_mean| {d_action:.1e}, max |Δ value| {d_value:.1e})")
        assert max(d_action, d_value) < 1e-4
        # export() wraps the DFAWrapper in a LogWrapper; run_episodes needs the DFAWrapper itself.
        policies[name] = dict(cfg=cfg, env=env._env, network=network, params=params)

    first = next(iter(policies.values()))
    drone_env, sampler = first["env"].env, first["env"].sampler
    for p in policies.values():
        assert (p["env"].sampler == sampler) and (p["env"].env.n_tokens == drone_env.n_tokens)

    with open(os.path.join(args.out, "policies.json"), "w") as f:
        f.write(dumps({
            name: {
                "file": f"policies/{name}.onnx",
                "checkpoint": os.path.basename(POLICIES[name]),
                "velocity_filter": p["cfg"]["use_velocity_filter"],
                "max_action": float(p["network"].max_action),
            }
            for name, p in policies.items()
        }))
    with open(os.path.join(args.out, "env.json"), "w") as f:
        f.write(dumps(env_spec(drone_env)))

    dfas = load_dfas(args.dfas, sampler.max_size, sampler.n_tokens)
    for name, (dfa, spec) in dfas.items():
        T, status, graphs = dfa_table(dfa)
        with open(os.path.join(args.out, "dfas", f"{name}.json"), "w") as f:
            f.write(dumps({
                "name": name,
                **({"note": spec["note"]} if "note" in spec else {}),
                "n_states": len(T),
                "description": describe(T, status),
                "start": int(dfa.start),
                "status": status,
                "transitions": T.tolist(),
                "policy_inputs": [
                    {
                        "node_features": np.asarray(g["node_features"]).tolist(),
                        "edge_features": np.asarray(g["edge_features"]).tolist(),
                        "current_state": np.asarray(g["current_state"]).tolist(),
                        "n_states": np.asarray(g["n_states"]).tolist(),
                    }
                    for g in graphs
                ],
            }))
        print(f"{name} ({len(T)} states):\n  " + "\n  ".join(describe(T, status)))

    rows = []
    key = jax.random.PRNGKey(0)
    for name, p in policies.items():
        env, network, params = p["env"], p["network"], p["params"]
        max_steps = p["cfg"]["max_steps_in_episode"]
        for dfa_name, (dfa, _) in dfas.items():
            # As DFAWrapper.reset hands out the sampler's DFAs: minimized.
            task = dfa.minimize()
            key, det_key, sample_key = jax.random.split(key, 3)
            # The env and the mean action are deterministic, so one episode is the deterministic outcome.
            success, fail, ep_len, _, _, traj, _ = run_episodes(
                env, network, params, det_key[None], True, max_steps, 1, dfa=task
            )
            outcome = "accept" if success[0] else "reject" if fail[0] else "timeout"
            successes, fails, ep_lens, *_ = run_episodes(
                env, network.clone(deterministic=False), params, jax.random.split(sample_key, args.n), False,
                max_steps, args.n, dfa=task
            )
            timeouts = ~(successes | fails)
            rows.append({
                "policy": name, "dfa": dfa_name,
                "deterministic_outcome": outcome, "deterministic_steps": int(ep_len[0]),
                "sampled_success": f"{float(successes.mean()):.2f}", "sampled_reject": f"{float(fails.mean()):.2f}",
                "sampled_timeout": f"{float(timeouts.mean()):.2f}", "sampled_mean_steps": f"{float(ep_lens.mean()):.1f}",
            })
            print(", ".join(f"{k} {v}" for k, v in rows[-1].items()))

            trace = [jax.tree_util.tree_map(lambda x: x[0, t], traj) for t in range(int(ep_len[0]) + 1)]
            animate_drone_trace(env.env, trace, save_path=os.path.join(args.out, "reference", f"{name}_{dfa_name}.gif"),
                                dfa=dfa)

    with open(os.path.join(args.out, "reference", "rollouts.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
