# RAD drone policy for DFA tasks (up to 5 DFA states)

This bundle has one drone policy. It flies a point-mass drone in a 3D box to complete a task given as a DFA (deterministic finite automaton) over 10 region tokens. The DFA reaches its goal when the drone visits regions in the right order, and it fails if the drone enters a forbidden region.

The policy reads the DFA through a pretrained, frozen RAD graph encoder, and the encoder is included in the ONNX model. Its actions are target velocities tracked by the deployment velocity filter (see [Actions](#actions)).

**Changed from the previous bundle:**
- One policy, `rad_velocity_filter`, retrained for twice as long (20M steps) on tasks of every size up to 5 DFA states. The no-filter policy is gone.
- The model has a new output, `action_std`, so the policy can be run **deterministically** (act with `action_mean`, as before) or **stochastically** (sample actions as in training). See step 3 of [Running a policy](#running-a-policy).
- New example DFAs.

As in the previous bundle, a region's token fires only on the step the drone enters the region.

Quick start (needs only `numpy` and `onnxruntime`):

```
pip install numpy onnxruntime
python run_example.py                 # deterministic: one episode per example DFA
python run_example.py --stochastic    # stochastic: 100 sampled episodes per example DFA
```

This runs the policy on the example DFAs in a numpy copy of the training simulator. Each result is printed next to the reference result from the training code.

## Contents

| Path | |
|---|---|
| `policies/rad_velocity_filter.onnx` | the policy |
| `policies.json` | the policy's file, that it uses the velocity filter, its max action and action std, and its training checkpoint |
| `env.json` | the training world: box, start state, step sizes, velocity-filter parameters, token regions |
| `dfas/dfa_<i>.json` | the example tasks, with the policy's DFA inputs precomputed for every DFA state |
| `reference/rollouts.csv` | how the policy does on each example DFA in the training simulator, in both modes |
| `reference/<policy>_<dfa>.gif` | the deterministic episodes behind `rollouts.csv`: the trajectory and the DFA state |
| `run_example.py` | reference implementation of the control loop below (numpy + onnxruntime), in both modes |

## Running a policy

Repeat one control step every `dt` = 0.1 s:

1. **Observation:** `obs` = float32[6], the position (x, y, z) followed by the velocity (vx, vy, vz).
2. **DFA inputs:** take `policy_inputs[q]` from the task's DFA file, where `q` is the current DFA state. Start at `q = 0`.
3. **Run the model and pick the action** (float32[3]):
   - **Deterministic:** `action = action_mean`. Each component is already in [-1, 1].
   - **Stochastic:** `action = clip(action_mean + action_std * eps, -1, 1)`, with `eps` three independent standard normal samples, drawn fresh every step. This is how the policy acted in training. The clip to [-1, 1] (`max_action` in `policies.json`) is what the training simulator did to every action.
4. **Apply the action** for `dt`. See [Actions](#actions).
5. **Advance the DFA.** A token fires only on the step the drone enters its region. Using the regions in `env.json`, find the token of the region the drone is in now and the token of the region it was in one step earlier. Either token is -1 if the drone is in no region. If the two tokens differ and the current one is ≥ 0, set `q = transitions[q][token]`. Otherwise no token fires and `q` stays the same.
6. **Stop** when `status[q]` is `"accept"` (task done) or `"reject"` (task failed). In training an episode also ended after 500 steps (timeout).

`Policy` in `run_example.py` implements both modes of step 3.

ONNX inputs and outputs:

| Name | Type | |
|---|---|---|
| `obs` | float32[6] | position, velocity |
| `node_features` | float32[5, 4] | `policy_inputs[q]["node_features"]` |
| `edge_features` | float32[25, 18] | `policy_inputs[q]["edge_features"]` |
| `current_state` | int64[1] | `policy_inputs[q]["current_state"]` |
| `n_states` | int64[5] | `policy_inputs[q]["n_states"]` |
| output `action_mean` | float32[1, 3] | the mean of the trained Gaussian policy: the deterministic action |
| output `value` | float32[1] | critic: expected discounted probability of accepting |
| output `cost_value` | float32[1] | cost critic: expected discounted probability of a rejection |
| output `action_std` | float32[1, 3] | the Gaussian policy's standard deviation per axis, for the stochastic action. It doesn't depend on the inputs: always [0.517, 0.495, 0.449] (also in `policies.json`) |

The DFA inputs are a graph encoding of the task that remains in state `q` (the DFA restarted at `q` and minimized). Pass them exactly as given, zero padding rows included. They are precomputed for every DFA state, so you don't need the training code to run these tasks. The value outputs are not needed to act. They were trained with discount 0.999 and are not calibrated, but `cost_value` can serve as a rough risk signal.

### Actions

The action is a normalized target velocity. The target velocity is `max_speed * action`, with `max_speed` = 1.0 (deployment `VELOCITY_SCALE`). It is held for `dt` = 0.1 s (`POLICY_DT`) and tracked by the critically damped second-order filter:
- τ = 0.1 s (`dfa_velocity_filter_tau_s`)
- acceleration norm at most 3.0 (`dfa_max_filter_acceleration_mps2`)
- integrated in 20 substeps of 0.005 s (`FILTER_DT`)

The observed velocity is the filter's velocity. The filter's acceleration is not observed. `Sim.step` in `run_example.py` has the update exactly. Motion into a wall stops at the wall.

## World

All training episodes start at the origin with zero velocity, inside the box [-1, 1]³. The policy has never seen anything else, so map the real arena into this frame.

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

The examples are written by hand. "Avoid" holds only until the DFA leaves that state. Each file's `description` gives the full transitions.

| DFA | States | Task |
|---|---|---|
| `dfa_0` | 3 | reach 0, then 2 |
| `dfa_1` | 5 | reach 2, then 4, then 8, then 6 |
| `dfa_2` | 3 | reach 0, avoiding 8 and 9 |
| `dfa_3` | 4 | reach 0 avoiding 4 and 5, then 6 |
| `dfa_4` | 5 | reach 1 avoiding 3, then 0 avoiding 7, then 1 avoiding 6 |
| `dfa_5` | 4 | reach 0, avoiding 1; entering 2 or 3 first means visiting 1 before 0 counts again, avoiding 4 and 5 meanwhile (the FMCAD'22 example) |

Outcomes in the training simulator (`reference/rollouts.csv`):
- **Deterministic:** one episode acting with `action_mean`. The simulator is deterministic, so a single episode gives the outcome.
- **Stochastic:** 100 episodes sampling actions as in step 3, as in training.

| DFA | Deterministic | Stochastic: success / reject / timeout | Stochastic: mean steps |
|---|---|---|---|
| `dfa_0` | accept after 31 steps | 1.00 / 0.00 / 0.00 | 41 |
| `dfa_1` | accept after 124 steps | 1.00 / 0.00 / 0.00 | 96 |
| `dfa_2` | accept after 25 steps | 0.99 / 0.01 / 0.00 | 25 |
| `dfa_3` | accept after 30 steps | 1.00 / 0.00 / 0.00 | 64 |
| `dfa_4` | accept after 84 steps | 0.98 / 0.00 / 0.02 | 187 |
| `dfa_5` | accept after 11 steps | 0.99 / 0.01 / 0.00 | 15 |

The policy solves every example deterministically. Sampling, it fails only rarely: one rejection each on `dfa_2` and `dfa_5`, and two timeouts on `dfa_4`, the longest task.

`python run_example.py` reproduces the deterministic column with the ONNX model, step for step. `python run_example.py --stochastic` reproduces the stochastic rates up to sampling noise (about ±0.1 over 100 episodes), since its random numbers differ from the training code's.

## Training and overall performance

The policy was trained with PPO for 20M steps in the simulator above, seed 1. Tasks were random DFAs with up to 5 states over the 10 tokens, every size equally likely: an equal mix of reach (visit tokens in order), reach-avoid (the same, with a forbidden token at each step) and randomly mutated reach-avoid tasks. The policy is constrained (PPO-Lagrangian): it maximizes the probability of accepting subject to P(reject) ≤ 0.05. The Lagrange multiplier started at 1 and was kept in [0, 1]. Rewards and costs were discounted by 0.999, and the reward was shaped by the DFA's distance to acceptance (potential-based, so it doesn't change the optimal policy).

Over 100 random tasks from the training distribution:

| Actions | Success | Reject | Timeout | Mean steps |
|---|---|---|---|---|
| deterministic (`action_mean`) | 0.90 | 0.06 | 0.04 | 58 |
| stochastic (sampled) | 0.91 | 0.09 | 0.00 | 43 |

## Caveats

- The policy is not safe by construction. It still rejects (enters a forbidden region) on about 6% of random tasks in simulation acting deterministically, and about 9% sampling actions, above the 5% it was trained for.
- Stochastic mode is noisy: `action_std` is about half the action range on every axis. It reaches the goal faster on average but rejects more often. Deterministic mode is the safer choice on the real drone; stochastic mode is useful for seeing how robust the policy is to action noise.
- The training simulator has no noise or disturbances, and the policy sees the exact state.
- The policy only handles DFAs with at most 5 states over these 10 tokens. A new DFA's `policy_inputs` have to be computed with the training code (`dfax`). Send us the DFA and we can generate its file.
