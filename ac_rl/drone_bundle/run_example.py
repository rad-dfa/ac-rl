"""Runs the policies on the example DFA tasks in a numpy copy of the training simulator, with only numpy and
onnxruntime. Use it as a template for driving a policy (see README.md); its outcomes should match the
deterministic ones in reference/rollouts.csv.

  python run_example.py                                    # every policy on every DFA
  python run_example.py --policy rad_velocity_filter --dfa dfas/dfa_3.json --trace trace.csv
"""
import os
import csv
import glob
import json
import math
import argparse
import numpy as np
import onnxruntime as ort

HERE = os.path.dirname(os.path.abspath(__file__))
f32 = np.float32  # The simulator ran in float32.


def load_json(path):
    with open(path) as f:
        return json.load(f)


class Sim:
    """The training simulator (dfa_gym's DroneEnv) for one drone."""

    def __init__(self, env, velocity_filter):
        self.low = np.array(env["box"]["low"], dtype=f32)
        self.high = np.array(env["box"]["high"], dtype=f32)
        self.dt = env["dt"]
        self.max_speed = env["max_speed"]
        self.velocity_filter = velocity_filter
        self.tau = env["velocity_filter"]["tau"]
        self.max_accel = env["velocity_filter"]["max_accel"]
        self.n_substeps = math.ceil(self.dt / env["velocity_filter"]["filter_dt"])
        self.h = self.dt / self.n_substeps
        self.regions = env["regions"]
        self.p = np.array(env["start"]["position"], dtype=f32)
        self.v = np.array(env["start"]["velocity"], dtype=f32)
        self.a = np.zeros(3, dtype=f32)  # velocity filter state; not observed by the policy
        self.prev_p = self.p  # position one step earlier, so a token only fires on entering its region

    def obs(self):
        return np.concatenate([self.p, self.v])

    def step(self, action):
        self.prev_p = self.p
        if not self.velocity_filter:
            # The action is a velocity, held for dt and stopped by the box; the observed velocity is the achieved one.
            p = np.clip(self.p + action * f32(self.dt), self.low, self.high)
            self.v = (p - self.p) / f32(self.dt)
            self.p = p
            return
        # The action, scaled by max_speed, is the target velocity of a critically damped second-order filter, held
        # for dt and integrated in substeps of h, with the acceleration's norm limited to max_accel.
        target = action * f32(self.max_speed)
        p, v, a = self.p, self.v, self.a
        for _ in range(self.n_substeps):
            jerk = (target - v) / f32(self.tau ** 2) - f32(2.0) * a / f32(self.tau)
            a_next = a + f32(self.h) * jerk
            norm = np.linalg.norm(a_next)
            if norm > self.max_accel:
                a_next = a_next * (f32(self.max_accel) / norm)
            v_next = v + f32(self.h) * a_next
            p_next = p + f32(0.5 * self.h) * (v + v_next)
            # An axis at (or pushed past) a wall while moving into it is stopped there.
            pinned = ((p_next <= self.low) & (v_next < 0)) | ((p_next >= self.high) & (v_next > 0))
            p = np.clip(p_next, self.low, self.high)
            v = np.where(pinned, f32(0), v_next)
            a = np.where(pinned, f32(0), a_next)
        self.p, self.v, self.a = p, v, a

    def label(self):
        """Token of the region the last step entered, or -1. A token fires only on the step the drone moves into its
        region (from no region or from another token's region), not on every step it stays inside."""
        token = self.region_token(self.p)
        return token if token != self.region_token(self.prev_p) else -1

    def region_token(self, p):
        """Token of the region p is in (the first listed region wins an overlap), or -1 if none."""
        x, y, z = p
        for r in self.regions:
            if not f32(r["z"][0]) <= z <= f32(r["z"][1]):
                continue
            if r["kind"] == "circle":
                cx, cy = f32(r["center_xy"][0]), f32(r["center_xy"][1])
                inside = (x - cx) ** 2 + (y - cy) ** 2 <= f32(r["radius"]) ** 2
            else:
                inside = f32(r["x"][0]) <= x <= f32(r["x"][1]) and f32(r["y"][0]) <= y <= f32(r["y"][1])
            if inside:
                return r["token"]
        return -1


class Policy:
    def __init__(self, path, max_action):
        self.session = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        self.max_action = max_action

    def __call__(self, obs, dfa_inputs):
        (action,) = self.session.run(["action_mean"], {"obs": obs.astype(f32), **dfa_inputs})
        return np.clip(action[0], -self.max_action, self.max_action)  # already in range; the simulator clips too


def policy_inputs(dfa):
    """The policy's DFA inputs for each DFA state, as numpy arrays of the ONNX models' input types."""
    return [
        {
            "node_features": np.array(x["node_features"], dtype=f32),
            "edge_features": np.array(x["edge_features"], dtype=f32),
            "current_state": np.array(x["current_state"], dtype=np.int64),
            "n_states": np.array(x["n_states"], dtype=np.int64),
        }
        for x in dfa["policy_inputs"]
    ]


def run_episode(policy, sim, dfa, max_steps, trace=None):
    """Returns (outcome, steps): outcome "accept", "reject" or "timeout"."""
    inputs = policy_inputs(dfa)
    q = dfa["start"]
    for t in range(1, max_steps + 1):
        action = policy(sim.obs(), inputs[q])
        sim.step(action)
        token = sim.label()
        if token >= 0:
            q = dfa["transitions"][q][token]
        if trace is not None:
            trace.append([t, *sim.p.tolist(), *sim.v.tolist(), *action.tolist(), token, q])
        if dfa["status"][q] != "running":
            return dfa["status"][q], t
    return "timeout", max_steps


def main():
    parser = argparse.ArgumentParser(description="Run the ONNX drone policies on the example DFA tasks")
    parser.add_argument("--policy", help="Policy name from policies.json (default: all)")
    parser.add_argument("--dfa", help="DFA file (default: all of dfas/)")
    parser.add_argument("--trace", help="Write the episode's per-step trace to this CSV (one policy and DFA only)")
    args = parser.parse_args()

    env = load_json(os.path.join(HERE, "env.json"))
    policies = load_json(os.path.join(HERE, "policies.json"))
    names = [args.policy] if args.policy else list(policies)
    dfa_paths = [args.dfa] if args.dfa else sorted(glob.glob(os.path.join(HERE, "dfas", "*.json")))
    if args.trace and len(names) * len(dfa_paths) != 1:
        parser.error("--trace needs --policy and --dfa")

    reference = {}
    reference_path = os.path.join(HERE, "reference", "rollouts.csv")
    if os.path.exists(reference_path):
        with open(reference_path) as f:
            reference = {(r["policy"], r["dfa"]): r for r in csv.DictReader(f)}

    for name in names:
        spec = policies[name]
        policy = Policy(os.path.join(HERE, spec["file"]), spec["max_action"])
        for dfa_path in dfa_paths:
            dfa = load_json(dfa_path)
            trace = [] if args.trace else None
            sim = Sim(env, spec["velocity_filter"])
            outcome, steps = run_episode(policy, sim, dfa, env["max_steps_in_episode"], trace)
            line = f"{name:20} {dfa['name']}: {outcome:7} after {steps:3} steps"
            ref = reference.get((name, dfa["name"]))
            if ref:
                line += f"  (reference: {ref['deterministic_outcome']} after {ref['deterministic_steps']} steps)"
            print(line)
            if trace is not None:
                with open(args.trace, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["t", "x", "y", "z", "vx", "vy", "vz", "action_x", "action_y", "action_z",
                                     "token", "dfa_state"])
                    writer.writerows(trace)


if __name__ == "__main__":
    main()
