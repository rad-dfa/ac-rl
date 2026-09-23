"""Worked example: closed-loop rollout of one of the ONNX policies.

This script only needs numpy + onnxruntime (see requirements.txt) -- no JAX,
no dfa-gym, no dfax. The physics in `simulate_step()` below is a stand-in for
your real drone: replace `simulate_step` with your actual state estimator /
low-level velocity controller, and replace `sense_label()` with however your
drone perceives which labeled region (if any) it's currently in. Everything
else -- building the goal graph, calling the model, stepping the DFA, and the
success/failure check -- is exactly what you'd run against real hardware.

Run: python example_infer.py
"""
import numpy as np
import onnxruntime as ort

import dfa_graph as dg

MODEL_PATH = "models/seed25715_rad.onnx"

# Must match the geofence/speed/dt the chosen model was trained with -- see
# README.md's "Training configuration" table. All 10 shipped models share
# these same values.
LOW = np.array([-1.0, -1.0, -1.0], dtype=np.float32)
HIGH = np.array([1.0, 1.0, 1.0], dtype=np.float32)
MAX_ACTION = 1.0   # max_speed, since these are velocity-command ("vel") models
DT = 0.1           # run your real control loop at this cadence (10 Hz)
MAX_STEPS = 500


def simulate_step(positions, velocities, action):
    """Stand-in for the real drone. Same kinematics as training
    (dfa_gym.DroneEnv.step_env): clip the action, integrate one dt of
    displacement, clip to the geofence, and report *achieved* velocity
    (zeroed along any axis the geofence clipped). Replace this with your
    real state estimator + velocity controller.
    """
    action = np.clip(action, -MAX_ACTION, MAX_ACTION)
    delta = action * DT
    new_positions = np.clip(positions + delta, LOW, HIGH)
    new_velocities = (new_positions - positions) / DT
    return new_positions, new_velocities


def sense_label(positions):
    """Stand-in for your perception stack: which labeled region (if any) is
    the drone in right now? Returns a token 0..4, or -1 for "none".
    Replace with real sensing if your regions aren't purely geometric.
    """
    return dg.label_from_position(positions)


def run(sequence, hazards, verbose=True):
    """Fly the goal 'visit `sequence` in order, never touching a token in
    `hazards`' using the policy at MODEL_PATH."""
    sess = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])

    transitions, labels, start = dg.reach_avoid_chain(sequence, hazards)
    graph = dg.build_graph(transitions, labels, start)
    current_state = start

    positions = np.zeros(3, dtype=np.float32)   # replace with your real starting position
    velocities = np.zeros(3, dtype=np.float32)  # replace with your real starting velocity

    for t in range(MAX_STEPS):
        obs = np.concatenate([positions, velocities]).astype(np.float32)
        onnx_inputs = {
            "obs": obs,
            "node_features": graph["node_features"],
            "edge_features": graph["edge_features"],
            "current_state": np.array([current_state], dtype=np.int64),
            "n_states": graph["n_states"],
        }
        action_mean, value = sess.run(None, onnx_inputs)
        action = action_mean[0]  # velocity command, shape (3,), already in [-MAX_ACTION, MAX_ACTION]

        # --- send `action` to your real velocity controller here ---
        positions, velocities = simulate_step(positions, velocities, action)
        # --- read back your real position/velocity estimate here ---

        token = sense_label(positions)
        current_state = dg.advance_state(graph["edge_features"], current_state, token)

        is_accept, is_reject = dg.is_terminal(graph["node_features"], current_state)
        if verbose:
            print(f"t={t:3d} pos={positions.round(2)} token={token:2d} state={current_state} value={float(value[0]):+.2f}")
        if is_accept:
            return "SUCCESS", t
        if is_reject:
            return "FAIL", t

    return "TIMEOUT", MAX_STEPS


if __name__ == "__main__":
    # Example goal: fly to region 1 (a corner pad), then region 3 (a side
    # corridor), never entering region 4 (the other side corridors). See
    # README.md for what these token numbers mean physically.
    outcome, steps = run(sequence=[1, 3], hazards=[4])
    print(f"\noutcome={outcome} after {steps} steps")
