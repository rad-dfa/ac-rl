# AC-RL drone policies (ONNX)

10 trained policies for a simulated point-mass drone that flies toward a goal
specified as a small automaton (DFA) over labeled 3D regions -- "visit region
A, then region B, while never entering region C," etc. Exported from JAX/Flax
PPO checkpoints so they can be run with just **numpy + onnxruntime**, no
JAX/TensorFlow/dfa-gym/dfax required.

## Contents

```
models/            10 .onnx policies, seed{N}_{rad,no_rad}.onnx
dfa_graph.py        Pure-numpy helpers: build a goal, step it, sense labels
example_infer.py     Worked closed-loop example (run this first)
requirements.txt     numpy, onnxruntime
```

## Quickstart

```
pip install -r requirements.txt
python example_infer.py
```

This flies a synthetic "visit region 1, then region 3, avoid region 4" goal
against `models/seed25715_rad.onnx` using a stand-in point-mass simulator, and
prints each step's position/token/DFA-state/value until success or failure.
Read the comments in `example_infer.py` and swap `simulate_step()` /
`sense_label()` for your real state estimator, velocity controller, and
perception stack -- everything else (building the goal graph, calling the
model, stepping the DFA, checking for success/failure) is exactly what you'd
run against real hardware.

## Which model to use

All 10 came from the same training setup (see "Training configuration"
below); they differ only in **random seed** (5 independent training runs) and
**rad vs no_rad**:

- **`rad`** -- the DFA goal is embedded with a separately pretrained graph
  encoder (frozen during policy training). This is the paper's main method.
- **`no_rad`** -- the DFA encoder is trained from scratch, jointly with the
  policy. This is the ablation.

Final training success rate (fraction of episodes that reached the goal,
averaged over the last ~20 logged windows of training; higher is better,
`mean_ep_len` is in environment steps at 10 Hz):

| seed  | rad success | rad ep_len | no_rad success | no_rad ep_len |
|-------|:-----------:|:----------:|:---------------:|:-------------:|
| 6054  | 0.979 | 12.8 | 0.973 | 12.9 |
| 13996 | 0.968 | 14.0 | 0.921 | 34.4 |
| 25715 | 0.978 | 12.2 | 0.978 | 10.5 |
| 28254 | 0.980 | 12.2 | 0.980 | 10.8 |
| 31010 | 0.960 | 14.7 | 0.967 | 12.0 |

**Recommendation: start with `seed25715_rad.onnx` or `seed28254_rad.onnx`**
(highest, most consistent success rate). Avoid `seed13996_no_rad` -- it's a
clear outlier (91% success, 2-3x longer episodes than the others), suggesting
that seed/variant didn't converge as well.

These numbers are from simulated training rollouts on the *same* fixed-map
task distribution the policy was trained on -- treat them as a ranking signal
for picking a checkpoint, not as an estimate of real-world success rate.

## Training configuration

All 10 models were trained with identical environment settings (baked into
each ONNX graph, not something you can change without retraining):

| setting | value |
|---|---|
| geofence (x, y, z) | all `[-1.0, 1.0]` |
| max speed | `1.0` (normalized units / second) |
| control timestep `dt` | `0.1` s (10 Hz) |
| action mode | **velocity** command (not position delta) |
| reward | binary (task success/fail only) |
| goal family | `ReachAvoid` DFAs, up to 5 states, 5-token alphabet |
| episode horizon | 500 steps |

Positions, velocities and actions are all in this **normalized `[-1, 1]`
frame**, not meters. Before flying a real drone you must define your own
affine mapping from real-world coordinates to this cube (e.g. if your safe
flight volume is a 4m-wide cube centered at your takeoff point:
`normalized = (real_position - center) / 2.0`), and run your control loop at
the same **10 Hz** cadence the policy was trained at -- a different rate or a
different physical volume moves you out of the training distribution.

## Model input/output

Every model takes 5 inputs and returns 2 outputs, all batch-less (no leading
batch dimension except in the outputs, which carry an artifact batch-of-1
axis):

| name | shape | dtype | meaning |
|---|---|---|---|
| `obs` | `[6]` | float32 | `[x, y, z, vx, vy, vz]`, normalized frame |
| `node_features` | `[5, 4]` | float32 | goal DFA's per-state features |
| `edge_features` | `[25, 13]` | float32 | goal DFA's per-transition features |
| `current_state` | `[1]` | int64 | index of the DFA's current state |
| `n_states` | `[5]` | int64 | true state count, same value repeated 5x |
| **`action_mean`** (output) | `[1, 3]` | float32 | velocity command, already clipped to `[-1, 1]` |
| **`value`** (output) | `[1]` | float32 | critic's value estimate (diagnostic only) |

`action_mean` is the policy's deterministic (mean) action -- these models
were exported without their exploration-noise head, so every call is
reproducible and there's nothing to sample.

**You don't build `edge_index` yourself** -- for a 5-state DFA it's always
the same fixed dense 5x5 state-pair meshgrid, so it's baked into the model as
a constant.

## The goal DFA: what it is and how to build one

The policy is conditioned on a small automaton over 5 "token" labels (0-4),
each corresponding to a labeled 3D region of the geofence (see next
section). At each instant the drone is inside at most one labeled region
(or none); the automaton advances one step whenever a new label is observed,
and the policy is trained to steer the drone so the automaton reaches its
accepting state.

`dfa_graph.py` has everything you need, with no dependency beyond numpy:

- **`reach_avoid_chain(sequence, hazards)`** -- convenience builder for "visit
  these tokens in order, never touching any of these hazard tokens." Returns
  `(transitions, labels, start)`.
- **`build_graph(transitions, labels, start)`** -- turns any DFA (up to 5
  states, 5-token alphabet; write `transitions`/`labels` by hand for anything
  fancier than a plain reach-avoid chain) into the 4 graph tensors above.
- **`advance_state(edge_features, current_state, token)`** -- steps
  `current_state` given a newly observed token (or a negative token for "no
  label right now" -- the automaton just stays put). Reads the transition
  straight out of the `edge_features` you're already feeding the model, so
  there's no separate transition table to keep in sync at runtime.
- **`is_terminal(node_features, current_state)`** -- `(is_accept, is_reject)`.
  Once either is `True` the goal is permanently decided (succeeded or
  failed) -- stop commanding the policy and hover/land.

`n_states` is a property of the *goal* (how many of its states are actually
reachable), fixed once you build the graph -- it does **not** change as
`current_state` advances within an episode; only rebuild the graph (and
`n_states` with it) when you switch to a new goal.

All of the above (`build_graph`, `advance_state`, and the region-label logic
below) were checked to match the original JAX training code (`dfax.DFAx`,
`dfa_gym.DroneEnv`) bit-for-bit on hundreds of random cases before this
handoff -- see "Provenance" below.

## Token -> physical region mapping

`label_from_position(x, y, z)` in `dfa_graph.py` tells you which token (if
any) a position falls in, for the `[-1, 1]^3` geofence above. Concretely,
for that geofence:

| token | region | where |
|:---:|---|---|
| 0 | center landing pad | disc of radius 0.4 centered at `(0, 0)`, `z in [0.25, 0.5]` |
| 1 | 4 corner pads | radius-0.2 discs near each of the 4 corners, `z in [-0.75, 0.25]` |
| 2 | 4 edge-midpoint pads | radius-0.2 discs at the midpoint of each of the 4 sides, `z in [-0.75, 0.25]` |
| 3 | 4 vertical-edge corridors | thin rectangles along the left/right edges, between corners and edge midpoints |
| 4 | 4 horizontal-edge corridors | thin rectangles along the top/bottom edges, between corners and edge midpoints |

Regions can overlap; where they do, the lower-numbered token wins (region 0
takes priority over 1, etc. -- see `label_regions()`'s docstring). A position
outside every region reports token `-1` ("no label"), which the automaton
treats as a stutter (no state change).

If your real drone's perception can't directly evaluate this geometry (e.g.
you're using markers/AprilTags/a motion-capture zone map instead), you only
need to replicate *which token index is active when*, at each control step --
the geometry above is a reference implementation, not a hard requirement.

## Safety notes before flying real hardware

- **These are sim-trained policies with no sim-to-real transfer validation.**
  Expect a reality gap. Test extensively (tethered, in a net, with a safety
  pilot on a kill switch) before any autonomous flight.
- **The exported model has no awareness of physical safety limits.** Training
  clipped positions to the geofence every step; the ONNX graph doesn't know
  about your real hardware's limits at all. Your flight controller must
  independently enforce hard position/velocity bounds and an emergency stop
  -- don't blindly integrate `action_mean` without your own clipping.
  This is why the example simulator explicitly clips before applying an
  action, mirroring `dfa_gym.DroneEnv.step_env`.
- Stop commanding the policy and hover/land once `is_terminal()` reports
  either `is_accept` or `is_reject` -- there's no other built-in stop signal.
- The success-rate table above is from simulation on the training task
  distribution, not a guarantee for novel goals or real-world conditions.

## Provenance

These ONNX files were exported from JAX/Flax PPO checkpoints trained in the
`ac-rl` repo (`ac_rl/train_drone_policy.py`), using a hand-built ONNX graph
constructor (`ac_rl/export_onnx.py` in that repo -- TensorFlow's `jax2tf`
path wasn't usable on the export machine). Each exported model was verified
against the real JAX forward pass on 8 random inputs; worst-case max-abs
difference across all 10 models was `7.9e-7` (float32 precision). The
`dfa_graph.py` helpers here were separately checked against the real
`dfax`/`dfa_gym` implementations (200-500 random trials each, exact match)
before this handoff.

| file here | original checkpoint |
|---|---|
| `seed{N}_rad.onnx` / `seed{N}_no_rad.onnx` | `policy_params_drone_seed_{N}_ReachAvoid_5_5_{rad,no_rad}_binary_x-1.0_1.0_y-1.0_1.0_z-1.0_1.0_speed1.0_dt0.1_vel_steps500.msgpack` |

N in `{6054, 13996, 25715, 28254, 31010}`.
