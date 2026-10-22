# RAD drone policies for DFA tasks (up to 5 DFA states)

This bundle has two drone policies. Each flies a point-mass drone in a 3D box to complete a task given as a DFA (deterministic finite automaton) over 10 region tokens. The DFA reaches its goal when the drone visits regions in the right order, and it fails if the drone enters a forbidden region.

Each policy reads the DFA through a pretrained, frozen RAD graph encoder, and the encoder is included in the ONNX model. The two policies differ only in how an action moved the drone in training:

| Policy | An action in training was |
|---|---|
| `policies/rad_no_filter.onnx` | a velocity, applied instantly for one step |
| `policies/rad_velocity_filter.onnx` | a target velocity tracked by the deployment velocity filter |

**Changed from the previous bundle:** a region's token now fires only on the step the drone enters the region, not on every step it stays inside. See step 5 of [Running a policy](#running-a-policy). The policies were retrained with this rule.

Quick start (needs only `numpy` and `onnxruntime`):

```
pip install numpy onnxruntime
python run_example.py
```

This runs both policies on the example DFAs in a numpy copy of the training simulator. Each outcome is printed next to the reference outcome from the training code.

## Contents

| Path | |
|---|---|
| `policies/*.onnx` | the two policies |
| `policies.json` | each policy's file, whether it uses the velocity filter, and its training checkpoint |
| `env.json` | the training world: box, start state, step sizes, velocity-filter parameters, token regions |
| `dfas/dfa_<i>.json` | the example tasks, with the policy's DFA inputs precomputed for every DFA state |
| `reference/rollouts.csv` | how each policy does on each example DFA in the training simulator |
| `reference/<policy>_<dfa>.gif` | the deterministic episodes behind `rollouts.csv`: the trajectory and the DFA state |
| `run_example.py` | reference implementation of the control loop below (numpy + onnxruntime) |

## Running a policy

Repeat one control step every `dt` = 0.1 s:

1. **Observation:** `obs` = float32[6], the position (x, y, z) followed by the velocity (vx, vy, vz).
2. **DFA inputs:** take `policy_inputs[q]` from the task's DFA file, where `q` is the current DFA state. Start at `q = 0`.
3. **Run the model.** Its output `action_mean` is float32[1, 3], with each component in [-1, 1].
4. **Apply the action** for `dt`. See [Actions](#actions).
5. **Advance the DFA.** A token fires only on the step the drone enters its region. Using the regions in `env.json`, find the token of the region the drone is in now and the token of the region it was in one step earlier. Either token is -1 if the drone is in no region. If the two tokens differ and the current one is ≥ 0, set `q = transitions[q][token]`. Otherwise no token fires and `q` stays the same.
6. **Stop** when `status[q]` is `"accept"` (task done) or `"reject"` (task failed). In training an episode also ended after 500 steps (timeout).

ONNX inputs and outputs:

| Name | Type | |
|---|---|---|
| `obs` | float32[6] | position, velocity |
| `node_features` | float32[5, 4] | `policy_inputs[q]["node_features"]` |
| `edge_features` | float32[25, 18] | `policy_inputs[q]["edge_features"]` |
| `current_state` | int64[1] | `policy_inputs[q]["current_state"]` |
| `n_states` | int64[5] | `policy_inputs[q]["n_states"]` |
| output `action_mean` | float32[1, 3] | the action: the mean of the trained Gaussian policy |
| output `value` | float32[1] | critic: expected discounted probability of accepting |
| output `cost_value` | float32[1] | cost critic: expected discounted probability of a rejection |

The DFA inputs are a graph encoding of the task that remains in state `q` (the DFA restarted at `q` and minimized). Pass them exactly as given, zero padding rows included. They are precomputed for every DFA state, so you don't need the training code to run these tasks. The value outputs are not needed to act. They were trained with discount 0.999 and are not calibrated, but `cost_value` can serve as a rough risk signal.

### Actions

- **`rad_no_filter`**: the action is a velocity in box units per second, at most 1 per axis. The new position is `position + action * dt`, clipped to the box. The observed velocity is the one achieved: `(new position - old position) / dt`.
- **`rad_velocity_filter`**: the action is a normalized target velocity. The target velocity is `max_speed * action`, with `max_speed` = 1.0 (deployment `VELOCITY_SCALE`). It is held for `dt` = 0.1 s (`POLICY_DT`) and tracked by the critically damped second-order filter:
  - τ = 0.1 s (`dfa_velocity_filter_tau_s`)
  - acceleration norm at most 3.0 (`dfa_max_filter_acceleration_mps2`)
  - integrated in 20 substeps of 0.005 s (`FILTER_DT`)

  The observed velocity is the filter's velocity. The filter's acceleration is not observed.

`Sim.step` in `run_example.py` has both updates exactly. In both modes, motion into a wall stops at the wall.

## World

All training episodes start at the origin with zero velocity, inside the box [-1, 1]³. The policies have never seen anything else, so map the real arena into this frame.

There are 10 tokens. Each region is a prism over z ∈ [-0.75, 0.25], so the drone is in no region when it is above z = 0.25 or below z = -0.75. It can fly over regions.

A token fires once, on the step the drone enters its region, either from no region or from a region with another token. Staying inside a region fires nothing, so the drone has to leave the region and come back to fire its token again. The start position is in no region.

Top-down view, with each cell 0.4 × 0.4:

```
 y = 1  +-----+-----+-----+-----+-----+
        |  0  |  5  |  1  |  9  |  0  |
        +-----+-----+-----+-----+-----+
        |  4  |                 |  8  |
        +-----+                 +-----+
        |  1  |      start      |  1  |
        +-----+   (0, 0, 0)     +-----+
        |  2  |                 |  6  |
        +-----+-----+-----+-----+-----+
        |  0  |  3  |  1  |  7  |  0  |
 y = -1 +-----+-----+-----+-----+-----+
     x = -1                           x = 1
```

- Token 0 is the four corners and token 1 is the four edge midpoints. Both are circles of radius 0.2, inscribed in their cells.
- Tokens 2–9 are the rectangular cells between them.

`env.json` lists the exact regions in priority order. A point on a shared edge gets the earlier region's token.

## Example DFAs

<!-- TODO: describe the tasks in ac_rl/drone_dfas.json, which were hand-edited after the previous bundle and no longer match the table below. -->
TODO: say where the example DFAs come from. "Avoid" holds only until the DFA leaves that state. Each file's `description` gives the full transitions.

| DFA | States | Task |
|---|---|---|
| `dfa_0` | TODO | TODO |
| `dfa_1` | TODO | TODO |
| `dfa_2` | TODO | TODO |
| `dfa_3` | TODO | TODO |
| `dfa_4` | TODO | TODO |
| `dfa_5` | TODO | TODO |

Outcomes in the training simulator (`reference/rollouts.csv`):
- **Deterministic:** one episode acting with `action_mean`, which is what the ONNX models do. The simulator is deterministic, so a single episode gives the outcome.
- **Sampled:** 100 episodes acting with the Gaussian policy's random actions, as in training. This is a rough measure of robustness to action noise.

<!-- TODO (after retraining): fill in this table from reference/rollouts.csv. -->
| DFA | `rad_no_filter` deterministic | `rad_no_filter` sampled (success / reject / timeout) | `rad_velocity_filter` deterministic | `rad_velocity_filter` sampled (success / reject / timeout) |
|---|---|---|---|---|
| `dfa_0` | TODO | TODO | TODO | TODO |
| `dfa_1` | TODO | TODO | TODO | TODO |
| `dfa_2` | TODO | TODO | TODO | TODO |
| `dfa_3` | TODO | TODO | TODO | TODO |
| `dfa_4` | TODO | TODO | TODO | TODO |
| `dfa_5` | TODO | TODO | TODO | TODO |

TODO (after retraining): describe the examples' failures, if any (see the deterministic episodes' gifs in `reference/`).

`run_example.py` reproduces the deterministic column with the ONNX models, step for step.

## Training and overall performance

Both policies were trained with PPO for 10M steps in the simulator above, seed 42. Tasks were random DFAs with up to 5 states over the 10 tokens, from the same distribution as the examples. The policies are constrained (PPO-Lagrangian): they maximize the probability of accepting subject to P(reject) ≤ 0.05. The Lagrange multiplier started at 1 and was kept in [0, 1]. Rewards and costs were discounted by 0.999.

Over 1000 random tasks from the training distribution:

<!-- TODO (after retraining): fill in with test_drone_policy.py --n 1000 (with and without --deterministic). -->
| Policy | Actions | Success | Reject | Timeout | Mean steps |
|---|---|---|---|---|---|
| `rad_no_filter` | deterministic (ONNX) | TODO | TODO | TODO | TODO |
| `rad_no_filter` | sampled | TODO | TODO | TODO | TODO |
| `rad_velocity_filter` | deterministic (ONNX) | TODO | TODO | TODO | TODO |
| `rad_velocity_filter` | sampled | TODO | TODO | TODO | TODO |

## Caveats

- The policies are not safe by construction. They still reject (enter a forbidden region) on about TODO (after retraining)% of tasks in simulation.
- The training simulator has no noise or disturbances, and the policy sees the exact state. `rad_no_filter` also assumes the commanded velocity is reached instantly. On a real drone, `rad_velocity_filter` is the better match.
- The policies only handle DFAs with at most 5 states over these 10 tokens. A new DFA's `policy_inputs` have to be computed with the training code (`dfax`). Send us the DFA and we can generate its file.
