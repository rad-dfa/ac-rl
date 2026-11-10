# RAD drone policies for DFA tasks (up to 5 DFA states)

This bundle has two drone policies. Each flies a point-mass drone in a 3D box to complete a task given as a DFA (deterministic finite automaton) over 10 region tokens. The DFA reaches its goal when the drone visits regions in the right order, and it fails if the drone enters a forbidden region.

The policies read the DFA through a pretrained, frozen RAD graph encoder, and the encoder is included in the ONNX models. Their actions are target velocities tracked by the deployment velocity filter (see [Actions](#actions)).

The two policies differ only in what they observed during training:

| Policy | Trained with |
|---|---|
| `rad` | exact observations |
| `rad_obs_noise` | noisy observations: Gaussian noise with std 0.02 on the position and 0.1 on the velocity, fresh every step (see [Observation noise](#observation-noise)) |

**Changed from the previous bundle:**
- Two new policies, `rad` and `rad_obs_noise`, replace `rad_velocity_filter`. They were trained with plain PPO on a shaped DFA reward instead of with a rejection constraint, so the models no longer have a `cost_value` output.
- **No geofence**, as on the real drone. Nothing stops the drone at the box walls; in training, leaving the box ended the episode. This is a new outcome, "out of bounds", next to accept, reject and timeout.
- **The regions moved.** They are smaller, sit 0.25 inside the box walls, and have 0.1 gaps between neighbors (see [World](#world)). The token layout is unchanged.
- Every result is reported with and without observation noise, for both policies, and `run_example.py --obs-noise` simulates that noise.

As in the previous bundle, a region's token fires only on the step the drone enters the region.

Quick start (needs only `numpy` and `onnxruntime`):

```
pip install numpy onnxruntime
python run_example.py                          # deterministic: one episode per policy and example DFA
python run_example.py --stochastic             # stochastic: 100 sampled episodes per policy and example DFA
python run_example.py --obs-noise [--stochastic]   # the same with observation noise
```

This runs the policies on the example DFAs in a numpy copy of the training simulator. Each result is printed next to the reference result from the training code.

## Contents

| Path | |
|---|---|
| `policies/rad.onnx`, `policies/rad_obs_noise.onnx` | the policies |
| `policies.json` | each policy's file, training settings, that it uses the velocity filter, its max action and action std, and its training checkpoint |
| `env.json` | the training world: box, start state, step sizes, velocity-filter parameters, token regions, observation noise |
| `dfas/dfa_<i>.json` | the example tasks, with the policies' DFA inputs precomputed for every DFA state |
| `reference/rollouts.csv` | how each policy does on each example DFA in the training simulator: with and without observation noise, deterministic and stochastic |
| `run_example.py` | reference implementation of the control loop below (numpy + onnxruntime), in every mode |

## Running a policy

Repeat one control step every `dt` = 0.1 s:

1. **Observation:** `obs` = float32[6], the position (x, y, z) followed by the velocity (vx, vy, vz).
2. **DFA inputs:** take `policy_inputs[q]` from the task's DFA file, where `q` is the current DFA state. Start at `q = 0`.
3. **Run the model and pick the action** (float32[3]):
   - **Deterministic:** `action = action_mean`. Each component is already in [-1, 1].
   - **Stochastic:** `action = clip(action_mean + action_std * eps, -1, 1)`, with `eps` three independent standard normal samples, drawn fresh every step. This is how the policies acted in training. The clip to [-1, 1] (`max_action` in `policies.json`) is what the training simulator did to every action.
4. **Apply the action** for `dt`. See [Actions](#actions).
5. **Advance the DFA.** A token fires only on the step the drone enters its region. Using the regions in `env.json`, find the token of the region the drone is in now and the token of the region it was in one step earlier. Either token is -1 if the drone is in no region. If the two tokens differ and the current one is ≥ 0, set `q = transitions[q][token]`. Otherwise no token fires and `q` stays the same.
6. **Stop** when `status[q]` is `"accept"` (task done) or `"reject"` (task failed). In training an episode also ended when the drone left the box (out of bounds) and after 500 steps (timeout).

`Policy` in `run_example.py` implements both modes of step 3, and `run_episode` the whole loop.

ONNX inputs and outputs:

| Name | Type | |
|---|---|---|
| `obs` | float32[6] | position, velocity |
| `node_features` | float32[5, 4] | `policy_inputs[q]["node_features"]` |
| `edge_features` | float32[25, 18] | `policy_inputs[q]["edge_features"]` |
| `current_state` | int64[1] | `policy_inputs[q]["current_state"]` |
| `n_states` | int64[5] | `policy_inputs[q]["n_states"]` |
| output `action_mean` | float32[1, 3] | the mean of the trained Gaussian policy: the deterministic action |
| output `value` | float32[1] | critic of the shaped training reward; not needed to act |
| output `action_std` | float32[1, 3] | the Gaussian policy's standard deviation per axis, for the stochastic action. It doesn't depend on the inputs: [0.71, 0.65, 1.08] for `rad` and [0.70, 0.66, 0.86] for `rad_obs_noise` (also in `policies.json`) |

The DFA inputs are a graph encoding of the task that remains in state `q` (the DFA restarted at `q` and minimized). Pass them exactly as given, zero padding rows included. They are precomputed for every DFA state, so you don't need the training code to run these tasks.

### Observation noise

`rad_obs_noise` was trained on observations with noise added: every observation, at the start and after every step, got fresh Gaussian noise with std 0.02 on each position axis and 0.1 on each velocity axis, truncated to ±3 std (`obs_noise` in `env.json`). Only the observation was noisy: the dynamics and the regions used the true state. `rad` saw exact observations.

On the drone, feed either policy the state estimate as it is; don't add noise. The noise only stands in for estimation error in simulation. `run_example.py --obs-noise` adds it, and the results below show both policies with and without it.

### Actions

The action is a normalized target velocity. The target velocity is `max_speed * action`, with `max_speed` = 1.0 (deployment `VELOCITY_SCALE`). It is held for `dt` = 0.1 s (`POLICY_DT`) and tracked by the critically damped second-order filter:
- τ = 0.1 s (`dfa_velocity_filter_tau_s`)
- acceleration norm at most 3.0 (`dfa_max_filter_acceleration_mps2`)
- integrated in 20 substeps of 0.005 s (`FILTER_DT`)

The observed velocity is the filter's velocity. The filter's acceleration is not observed. `Sim.step` in `run_example.py` has the update exactly. There are no walls: the simulator doesn't stop or clip the drone anywhere.

## World

All training episodes start at the origin with zero velocity, inside the box [-1, 1]³. The policies have never seen anything else, so map the real arena into this frame. An episode ended as soon as the drone left the box, so the policies have learned to stay inside it but have no experience near or past its walls.

There are 10 tokens. Each region is a prism over z ∈ [-0.75, 0.25], so the drone is in no region when it is above z = 0.25 or below z = -0.75. It can fly over regions.

A token fires once, on the step the drone enters its region, either from no region or from a region with another token. Staying inside a region fires nothing, so the drone has to leave the region and come back to fire its token again. The start position is in no region.

The regions form a ring of 0.3 × 0.3 cells 0.25 inside the walls, around an empty center. Top-down view:

```
 y = 1     +-----------------------------------------+
           |                                         |
 y = 0.75  |     +-----+-----+-----+-----+-----+     |
           |     |  0  |  5  |  1  |  9  |  0  |     |
           |     +-----+-----+-----+-----+-----+     |
           |     |  4  |                 |  8  |     |
           |     +-----+                 +-----+     |
           |     |  1  |      start      |  1  |     |
           |     +-----+    (0, 0, 0)    +-----+     |
           |     |  2  |                 |  6  |     |
           |     +-----+-----+-----+-----+-----+     |
           |     |  0  |  3  |  1  |  7  |  0  |     |
 y = -0.75 |     +-----+-----+-----+-----+-----+     |
           |                                         |
 y = -1    +-----------------------------------------+
        x = -1  -0.75                     0.75     x = 1
```

- The cells are centered at x, y ∈ {-0.6, -0.3, 0, 0.3, 0.6}. Each region is centered in its cell, and neighboring regions are 0.1 apart.
- Token 0 is the four corners and token 1 is the four edge midpoints. Both are circles of radius 0.1.
- Tokens 2–9 are 0.2 × 0.2 squares between them.
- The band between the ring and the walls (|x| or |y| > 0.75) and the center (|x|, |y| < 0.45) have no regions.

`env.json` lists the exact regions in priority order. A point on a shared edge gets the earlier region's token.

## Example DFAs

The examples are written by hand and are the same as in the previous bundle. "Avoid" holds only until the DFA leaves that state. Each file's `description` gives the full transitions.

| DFA | States | Task |
|---|---|---|
| `dfa_0` | 3 | reach 0, then 2 |
| `dfa_1` | 5 | reach 2, then 4, then 8, then 6 |
| `dfa_2` | 3 | reach 0, avoiding 8 and 9 |
| `dfa_3` | 4 | reach 0 avoiding 4 and 5, then 6 |
| `dfa_4` | 5 | reach 1 avoiding 3, then 0 avoiding 7, then 1 avoiding 6 |
| `dfa_5` | 4 | reach 0, avoiding 1; entering 2 or 3 first means visiting 1 before 0 counts again, avoiding 4 and 5 meanwhile (the FMCAD'22 example) |

Outcomes in the training simulator (`reference/rollouts.csv`), for each policy with exact and with noisy observations:
- **Deterministic without noise:** one episode acting with `action_mean`. The simulator is deterministic, so a single episode gives the outcome.
- **Every other column:** 100 episodes. The success rate, followed by any nonzero rate of rejection (rej), leaving the box (oob) or timeout.

`rad`:

| DFA | exact obs, deterministic | exact obs, stochastic | noisy obs, deterministic | noisy obs, stochastic |
|---|---|---|---|---|
| `dfa_0` | accept, 18 steps | 1.00 | 1.00 | 1.00 |
| `dfa_1` | accept, 57 steps | 1.00 | 1.00 | 1.00 |
| `dfa_2` | accept, 10 steps | 1.00 | 1.00 | 1.00 |
| `dfa_3` | timeout, 500 steps | 1.00 | 0.32 (timeout 0.68) | 1.00 |
| `dfa_4` | accept, 71 steps | 0.98 (rej 0.01, oob 0.01) | 0.89 (timeout 0.11) | 0.99 (oob 0.01) |
| `dfa_5` | accept, 12 steps | 1.00 | 1.00 | 0.99 (rej 0.01) |

`rad_obs_noise`:

| DFA | exact obs, deterministic | exact obs, stochastic | noisy obs, deterministic | noisy obs, stochastic |
|---|---|---|---|---|
| `dfa_0` | accept, 27 steps | 1.00 | 0.80 (timeout 0.20) | 1.00 |
| `dfa_1` | accept, 50 steps | 1.00 | 1.00 | 1.00 |
| `dfa_2` | accept, 11 steps | 1.00 | 0.91 (timeout 0.09) | 1.00 |
| `dfa_3` | accept, 35 steps | 1.00 | 1.00 | 1.00 |
| `dfa_4` | timeout, 500 steps | 1.00 | 0.83 (timeout 0.17) | 1.00 |
| `dfa_5` | accept, 12 steps | 1.00 | 0.99 (timeout 0.01) | 0.99 (rej 0.01) |

What the tables show:
- **Stochastic:** both policies solve every example in 98–100% of episodes, with or without observation noise. A rejection or leaving the box happens in at most 1 of 100 episodes per task.
- **Deterministic:** each policy stalls on one example even with exact observations, running out of time without failing: `rad` on `dfa_3` and `rad_obs_noise` on `dfa_4`. With noisy observations both stall more often: `rad` mostly on `dfa_3` (68% of episodes), `rad_obs_noise` on four of the six tasks (up to 20%). The deterministic failures are all timeouts, never rejections.

`python run_example.py` reproduces the deterministic exact-observation column with the ONNX models, step for step. `python run_example.py --stochastic` and/or `--obs-noise` reproduce the other columns up to sampling noise (about ±0.1 over 100 episodes), since their random numbers differ from the training code's.

## Training and overall performance

Both policies were trained with PPO for 20M steps in the simulator above, seed 1. Tasks were random DFAs with up to 5 states over the 10 tokens, every size equally likely: an equal mix of reach (visit tokens in order), reach-avoid (the same, with a forbidden token at each step) and randomly mutated reach-avoid tasks. The reward was +1 when the DFA accepts, -1 when it rejects and 0 otherwise, including on leaving the box or timing out. It was discounted by 0.999 and shaped by the DFA's distance to acceptance (potential-based, β = 0.7, so it doesn't change the optimal policy). Unlike the previous bundle's policy, these have no explicit constraint on the rejection rate.

At the end of training (the last 50 PPO updates, about 0.4M steps, sampling actions), on random tasks from that distribution:

| Policy | Observations | Success | Reject | Out of bounds or timeout |
|---|---|---|---|---|
| `rad` | exact | 0.95 | 0.02 | 0.03 |
| `rad_obs_noise` | noisy | 0.96 | 0.02 | 0.02 |

## Caveats

- The policies are not safe by construction. Training penalized a rejection (reward −1) but did not constrain the rejection rate, and they reject on about 2% of random tasks in simulation.
- **Deterministic or stochastic.** Acting deterministically, the policies can stall on a task until they time out, more often with noisy observations (see [Example DFAs](#example-dfas)). Sampling actions avoided this on every example, unlike with the previous bundle's policy, where deterministic was the safer choice. But stochastic mode sends noisy targets: `action_std` is 0.65–1.08 per axis on a [-1, 1] range, which the velocity filter smooths. Check that the real drone tolerates that before flying it stochastically.
- Leaving the box ended every training episode, so the policies have no experience at or beyond the walls. Keep a safety margin or geofence of your own around the mapped box.
- The training simulator has no disturbances, and the only noise is `rad_obs_noise`'s observation noise.
- The policies only handle DFAs with at most 5 states over these 10 tokens. A new DFA's `policy_inputs` have to be computed with the training code (`dfax`). Send us the DFA and we can generate its file.
- Each policy is one training run (seed 1). Other seeds are still training.
