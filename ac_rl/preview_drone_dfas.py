"""Quick look at export_drone_bundle.py's policies on the hand-defined DFAs in drone_dfas.json, for iterating on
the DFAs: each policy runs one deterministic episode (mean actions, as the ONNX models act) per DFA, drawn as a
gif. No ONNX export and no sampled-action rates; run export_drone_bundle.py for those once the DFAs are settled.

  uv run preview_drone_dfas.py [--only dfa_2 dfa_3] [--policies rad_velocity_filter] [--watch]

Writes under --out (default storage/gifs/dfa_preview):
  <dfa>_<policy>.gif  the episode, in real time: a top-down view (regions colored by token; the drone marker is
                      filled while the drone is at region height, where tokens count, and hollow above or below),
                      the altitude, and the DFA with its current state highlighted
  outcomes.csv        each episode's outcome and length

--watch keeps the policies loaded and, whenever --dfas is saved, redraws the gifs of the DFAs that changed.
"""
import os
import csv
import time
import glob
import argparse
import numpy as np
import jax
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from PIL import Image
from dfa_gym.utils import DRONE_LABEL_COLORS, _dfa_layout, _draw_dfa
from ppo import batchify
from export_onnx import build_network_and_params
from export_drone_bundle import POLICIES, load_dfas, dfa_table

DPI = 80
FRAME_MS = 100  # one frame per policy step of dt = 0.1 s
LAST_FRAME_MS = 2000


def make_rollout(env, network):
    """Jitted deterministic episode on a (minimized) DFA, run by DFAWrapper as in training. The DFA is an argument,
    so all DFAs share one compilation. Returns (positions, tokens, dones) of steps 1..max_steps; steps after the
    first done are padding."""
    agent = env.agents[0]
    max_steps = env.env.max_steps_in_episode

    @jax.jit
    def rollout(params, dfa):
        _, state = env.reset(jax.random.PRNGKey(0))  # always the origin at rest; only the DFA is replaced
        state = state.replace(dfas={agent: dfa}, init_dfas={agent: dfa})
        obs = env.get_obs(state=state)

        def step(carry, _):
            obs, state = carry
            action, _ = network.apply(params, batchify(obs, env.agents))
            obs, state, _, dones, _ = env.step_env(jax.random.PRNGKey(0), state, {agent: action[0]})
            return (obs, state), (state.env_state.positions[0], env.env.label_f(state.env_state)[agent],
                                  dones["__all__"])

        _, out = jax.lax.scan(step, (obs, state), None, length=max_steps)
        return out

    return rollout


def run_episode(rollout, params, dfa):
    """(positions, tokens, states, outcome) of the episode on `dfa` as written (its own state ids), steps 0..end."""
    T, status, _ = dfa_table(dfa)
    positions, tokens, dones = (np.asarray(x) for x in rollout(params, dfa.minimize()))
    end = int(np.argmax(dones)) + 1 if dones.any() else len(dones)
    states = [int(dfa.start)]
    for token in tokens[:end]:
        states.append(int(T[states[-1], token]) if token >= 0 else states[-1])
    outcome = status[states[-1]] if status[states[-1]] != "running" else "timeout"
    # DFAWrapper ends the episode exactly when the table reaches a final state (or on timeout).
    assert all(status[q] == "running" for q in states[:-1]) and (outcome != "timeout" or end == len(dones))
    positions = np.concatenate([np.zeros((1, 3)), positions[:end]])
    return positions, [-1] + tokens[:end].tolist(), states, outcome


def dfa_panels(dfa, states, size):
    """The DFA drawing for each state in `states`, pre-rendered as RGB arrays (drawing it every frame is slow)."""
    G, pos, edges = _dfa_layout(dfa)
    panels = {}
    for q in sorted(set(states)):
        fig = plt.figure(figsize=size, dpi=DPI)
        ax = fig.add_axes([0, 0, 1, 1])
        _draw_dfa(ax, dfa, G, pos, edges, q)
        fig.canvas.draw()
        panels[q] = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        plt.close(fig)
    return panels


def draw_gif(path, drone_env, dfa, positions, tokens, states, outcome, title, note):
    regions = drone_env.label_regions()
    z_lo, z_hi = (float(v) for v in regions[0][2][-2:])  # every region spans the same heights
    low, high = np.asarray(drone_env.low), np.asarray(drone_env.high)
    n = len(positions) - 1

    fig = plt.figure(figsize=(15, 5.4), dpi=DPI)
    ax_map = fig.add_axes([0.045, 0.1, 0.29, 0.78])
    ax_z = fig.add_axes([0.39, 0.1, 0.27, 0.78])
    ax_dfa = fig.add_axes([0.68, 0.05, 0.31, 0.86])

    for token, kind, params in reversed(regions):  # earlier regions win overlaps, so draw them on top
        params = [float(v) for v in params]
        color = DRONE_LABEL_COLORS[token]
        if kind == "circle":
            cx, cy, r = params[:3]
            ax_map.add_patch(patches.Circle((cx, cy), r, color=color, alpha=0.6))
        else:
            x0, x1, y0, y1 = params[:4]
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            ax_map.add_patch(patches.Rectangle((x0, y0), x1 - x0, y1 - y0, color=color, alpha=0.6))
        ax_map.text(cx, cy, str(token), ha="center", va="center", fontsize=11, weight="bold")
    ax_map.plot([0], [0], "x", color="black")
    ax_map.set_xlim(low[0], high[0])
    ax_map.set_ylim(low[1], high[1])
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("x")
    ax_map.set_ylabel("y")
    ax_map.set_title("top-down")

    ax_z.axhspan(z_lo, z_hi, color="gray", alpha=0.25, label="region height (tokens count)")
    ax_z.set_xlim(0, max(n, 10))
    ax_z.set_ylim(low[2], high[2])
    ax_z.set_xlabel("step")
    ax_z.set_ylabel("z")
    ax_z.set_title("altitude")
    ax_z.legend(loc="upper right", fontsize=9)

    ax_dfa.axis("off")
    panels = dfa_panels(dfa, states, (ax_dfa.get_position().width * 15, ax_dfa.get_position().height * 5.4))
    fig.text(0.01, 0.01, note, fontsize=10, color="dimgray")

    path_line, = ax_map.plot([], [], color="crimson", lw=1.5, animated=True)
    drone, = ax_map.plot([], [], "o", ms=10, mec="crimson", mew=2, animated=True)
    z_line, = ax_z.plot([], [], color="crimson", lw=1.5, animated=True)
    z_dot, = ax_z.plot([], [], "o", ms=6, color="crimson", animated=True)
    dfa_image = ax_dfa.imshow(panels[states[0]], animated=True)
    header = fig.text(0.01, 0.95, "", fontsize=13, weight="bold", animated=True)
    animated = [path_line, drone, z_line, z_dot, dfa_image, header]

    # Draw the static parts once, then each frame only the moving parts on top (blitting).
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)
    frames = []
    for t in range(n + 1):
        fig.canvas.restore_region(background)
        x, y, z = positions[t]
        path_line.set_data(positions[:t + 1, 0], positions[:t + 1, 1])
        drone.set_data([x], [y])
        drone.set_markerfacecolor("crimson" if z_lo <= z <= z_hi else "white")
        z_line.set_data(np.arange(t + 1), positions[:t + 1, 2])
        z_dot.set_data([t], [z])
        dfa_image.set_data(panels[states[t]])
        token = tokens[t] if tokens[t] >= 0 else "none"
        header.set_text(f"{title}   step {t}/{n}   token {token}   DFA state {states[t]}"
                        + (f"   -> {outcome.upper()}" if t == n else ""))
        for artist in animated:
            fig.draw_artist(artist)
        frames.append(Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()))
    plt.close(fig)

    frames = [f.quantize(colors=64, method=Image.Quantize.FASTOCTREE) for f in frames]
    frames[0].save(path, save_all=True, append_images=frames[1:], loop=0,
                   duration=[FRAME_MS] * n + [LAST_FRAME_MS])


def main():
    parser = argparse.ArgumentParser(description="Preview the drone policies on hand-defined DFAs as gifs")
    parser.add_argument("--dfas", default="drone_dfas.json", help="DFA tasks (default: drone_dfas.json)")
    parser.add_argument("--only", nargs="+", metavar="DFA", help="Only these DFAs (default: all)")
    parser.add_argument("--policies", nargs="+", choices=list(POLICIES), default=list(POLICIES),
                        help="Policies to run (default: all)")
    parser.add_argument("--out", default="storage/gifs/dfa_preview",
                        help="Output directory (default: storage/gifs/dfa_preview)")
    parser.add_argument("--watch", action="store_true", help="Keep running and redraw changed DFAs when --dfas is saved")
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    t0 = time.time()
    policies = {}
    for name in args.policies:
        cfg, env, network, params, _ = build_network_and_params(POLICIES[name])
        env = env._env  # export_onnx wraps the DFAWrapper in a LogWrapper
        policies[name] = dict(env=env, params=params, rollout=make_rollout(env, network))
    first = next(iter(policies.values()))["env"]
    print(f"Loaded {', '.join(policies)} in {time.time() - t0:.0f}s")

    drawn = {}  # DFA name -> spec its gifs were drawn from
    # Keep earlier runs' rows (e.g. of DFAs left out by --only); (dfa, policy) -> row.
    outcomes_path = os.path.join(args.out, "outcomes.csv")
    outcomes = {}
    if os.path.exists(outcomes_path):
        with open(outcomes_path) as f:
            outcomes = {(r["dfa"], r["policy"]): list(r.values()) for r in csv.DictReader(f)}
    while True:
        mtime = os.path.getmtime(args.dfas)
        try:
            all_dfas = load_dfas(args.dfas, first.sampler.max_size, first.sampler.n_tokens)
            unknown = set(args.only or []) - set(all_dfas)
            if unknown:
                raise ValueError(f"{args.dfas} has no DFA named {sorted(unknown)}")
        except (ValueError, KeyError, TypeError) as e:  # json.JSONDecodeError is a ValueError
            if not args.watch:
                raise
            print(f"Not loaded: {e}")
            all_dfas = None

        if all_dfas is not None:
            # Drop the gifs and rows of DFAs that were removed from (or renamed in) --dfas.
            for path in glob.glob(os.path.join(args.out, "*.gif")):
                stem = os.path.basename(path)[:-len(".gif")]
                dfa_name = next((stem[:-len(p) - 1] for p in POLICIES if stem.endswith(f"_{p}")), None)
                if dfa_name is not None and dfa_name not in all_dfas:
                    os.remove(path)
            outcomes = {key: row for key, row in outcomes.items() if key[0] in all_dfas}

            dfas = {name: all_dfas[name] for name in args.only} if args.only else all_dfas
            for dfa_name, (dfa, spec) in dfas.items():
                if drawn.get(dfa_name) == spec:
                    continue
                for policy_name, p in policies.items():
                    t0 = time.time()
                    positions, tokens, states, outcome = run_episode(p["rollout"], p["params"], dfa)
                    path = os.path.join(args.out, f"{dfa_name}_{policy_name}.gif")
                    draw_gif(path, p["env"].env, dfa, positions, tokens, states, outcome,
                             f"{dfa_name} / {policy_name}", spec.get("note", ""))
                    outcomes[dfa_name, policy_name] = [dfa_name, policy_name, outcome, len(positions) - 1]
                    print(f"{dfa_name:12} {policy_name:20} {outcome:7} after {len(positions) - 1:3} steps "
                          f"-> {path} ({time.time() - t0:.1f}s)")
                drawn[dfa_name] = spec

            with open(outcomes_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["dfa", "policy", "outcome", "steps"])
                writer.writerows(row for _, row in sorted(outcomes.items()))

        if not args.watch:
            break
        print(f"Watching {args.dfas} (Ctrl-C to stop)")
        while os.path.getmtime(args.dfas) == mtime:
            time.sleep(0.5)


if __name__ == "__main__":
    main()
