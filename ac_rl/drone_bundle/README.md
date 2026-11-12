# RAD drone policy for DFA tasks (up to 5 DFA states), with controller tracking

This bundle has one drone policy, `rad_tracking`. It flies a point-mass drone in a 3D box to complete a task given as a DFA (deterministic finite automaton) over 10 region tokens. The DFA reaches its goal when the drone visits regions in the right order, and it fails if the drone enters a forbidden region.

The policy reads the DFA through a pretrained, frozen RAD graph encoder, and the encoder is included in the ONNX model. Its actions are target velocities for the deployment velocity filter. The drone follows the filter through the stack's controller and motor delay (see [Actions](#actions)). It was trained on noisy observations (see [Observation noise](#observation-noise)).

**Changed from the previous bundle:**
- **Controller tracking.** The training simulator now uses your point-mass model of the stack. The velocity filter's position, velocity and acceleration are a reference, and the drone tracks it through the controller (gains KP, KV) and a first-order motor delay (TAU). Before, the drone followed the filter exactly. The drone now lags the filter and can overshoot its velocity. The policy observes the drone's own state, not the filter's.
- **One policy**, `rad_tracking`, replaces `rad` and `rad_obs_noise`. It was trained like `rad_obs_noise`, with the same observation noise, but in the simulator with controller tracking.
- `env.json` has the controller's gains under `tracking_controller`. `policies.json` marks the policy with `"tracking_controller": true`. `run_example.py` simulates the tracking.

The world, the regions, the example DFAs, and the ONNX inputs and outputs are unchanged. As before, a region's token fires only on the step the drone enters the region.

Quick start (needs only `numpy` and `onnxruntime`):

```
pip install numpy onnxruntime
python run_example.py                          # deterministic: one episode per example DFA
python run_example.py --stochastic             # stochastic: 100 sampled episodes per example DFA
python run_example.py --obs-noise [--stochastic]   # the same with observation noise
```

This runs the policy on the example DFAs in a numpy copy of the training simulator. Each result is printed next to the reference result from the training code.

## Contents

| Path | |
|---|---|
| `policies/rad_tracking.onnx` | the policy |
| `policies.json` | the policy's file, training settings, that it uses the velocity filter and the tracking controller, its max action and action std, and its training checkpoint |
| `env.json` | the training world: box, start state, step sizes, velocity-filter and tracking-controller parameters, token regions, observation noise |
| `dfas/dfa_<i>.json` | the example tasks, with the policy's DFA inputs precomputed for every DFA state |
| `reference/rollouts.csv` | how the policy does on each example DFA in the training simulator: with and without observation noise, deterministic and stochastic |
| `run_example.py` | reference implementation of the control loop below (numpy + onnxruntime), in every mode |

## Running the policy

Repeat one control step every `dt` = 0.1 s:

1. **Observation:** `obs` = float32[6], the drone's position (x, y, z) followed by its velocity (vx, vy, vz).
2. **DFA inputs:** take `policy_inputs[q]` from the task's DFA file, where `q` is the current DFA state. Start at `q = 0`.
3. **Run the model and pick the action** (float32[3]):
   - **Deterministic:** `action = action_mean`. Each component is already in [-1, 1].
   - **Stochastic:** `action = clip(action_mean + action_std * eps, -1, 1)`, with `eps` three independent standard normal samples, drawn fresh every step. This is how the policy acted in training. The clip to [-1, 1] (`max_action` in `policies.json`) is what the training simulator did to every action.
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
| output `action_std` | float32[1, 3] | the Gaussian policy's standard deviation per axis, for the stochastic action. It doesn't depend on the inputs: [0.69, 0.68, 0.92] (also in `policies.json`) |

The DFA inputs are a graph encoding of the task that remains in state `q` (the DFA restarted at `q` and minimized). Pass them exactly as given, zero padding rows included. They are precomputed for every DFA state, so you don't need the training code to run these tasks.

### Observation noise

The policy was trained on observations with noise added: every observation, at the start and after every step, got fresh Gaussian noise with std 0.02 on each position axis and 0.1 on each velocity axis, truncated to ±3 std (`obs_noise` in `env.json`). Only the observation was noisy: the dynamics and the regions used the true state.

On the drone, feed the policy the state estimate as it is; don't add noise. The noise only stands in for estimation error in simulation. `run_example.py --obs-noise` adds it, and the results below show the policy with and without it.

### Actions

The action is a normalized target velocity. The target velocity is `max_speed * action`, with `max_speed` = 1.0 (deployment `VELOCITY_SCALE`). It is held for `dt` = 0.1 s (`POLICY_DT`) and fed to the critically damped second-order velocity filter:
- τ = 0.1 s (`dfa_velocity_filter_tau_s`)
- acceleration norm at most 3.0 (`dfa_max_filter_acceleration_mps2`)
- integrated in 20 substeps of h = 0.005 s (`FILTER_DT`)

The filter's position `ref_p`, velocity `ref_v` and acceleration `ref_a` are the reference the drone tracks. In each substep, after the filter advances, the controller requests an acceleration, the motors move the drone's acceleration `a` toward it with a first-order delay, and the drone integrates it. Per axis:

```
request = ref_a + KP * (ref_p - p) + KV * (ref_v - v)
a       = a + (1 - exp(-h / TAU)) * (request - a)
v_new   = v + h * a
p       = p + h * (v + v_new) / 2
```

| Axis | KP | KV | TAU (s) |
|---|---|---|---|
| x | 7.1 | 2.4 | 0.080019 |
| y | 7.1 | 2.4 | 0.080019 |
| z | 11.9 | 6.7 | 0.058938 |

These are the stack's position gain, velocity gain and motor delay, as in your point-mass model. The reference starts at the drone (the origin, at rest). The policy observes the drone's position and velocity. The filter's state and both accelerations are not observed. Because the drone lags the filter, its velocity can exceed `max_speed`.

`Sim.step` in `run_example.py` has the update exactly. There are no walls: the simulator doesn't stop or clip the drone anywhere.

## World

All training episodes start at the origin with zero velocity, inside the box [-1, 1]³. The policy has never seen anything else, so map the real arena into this frame. An episode ended as soon as the drone left the box, so the policy has learned to stay inside it but has no experience near or past its walls.

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

Outcomes in the training simulator (`reference/rollouts.csv`), with exact and with noisy observations:
- **Deterministic without noise:** one episode acting with `action_mean`. The simulator is deterministic, so a single episode gives the outcome.
- **Every other column:** 100 episodes. The success rate, followed by any nonzero rate of rejection (rej), leaving the box (oob) or timeout.

| DFA | exact obs, deterministic | exact obs, stochastic | noisy obs, deterministic | noisy obs, stochastic |
|---|---|---|---|---|
| `dfa_0` | accept, 40 steps | 1.00 | 1.00 | 1.00 |
| `dfa_1` | accept, 57 steps | 1.00 | 1.00 | 1.00 |
| `dfa_2` | accept, 9 steps | 1.00 | 1.00 | 1.00 |
| `dfa_3` | accept, 28 steps | 1.00 | 1.00 | 1.00 |
| `dfa_4` | accept, 90 steps | 0.95 (rej 0.04, timeout 0.01) | 1.00 | 0.97 (rej 0.03) |
| `dfa_5` | timeout, 500 steps | 0.96 (rej 0.03, timeout 0.01) | 0.97 (timeout 0.03) | 0.99 (rej 0.01) |

What the table shows:
- **`dfa_0`–`dfa_3`** are solved in every episode, in every mode. The drone never left the box on any example.
- **Deterministic:** the policy never rejects. With exact observations it stalls on `dfa_5` until it times out. With noisy observations it stalls on 3% of `dfa_5` episodes and solves everything else.
- **Stochastic:** it rejects on `dfa_4` and `dfa_5` in 1–4% of episodes.
- **Compared with the previous bundle's `rad_obs_noise`:** deterministic mode stalls less. `rad_obs_noise` timed out on four of the six tasks with noisy observations, up to 20% of the time. Stochastic mode rejects slightly more often, up to 4% instead of at most 1%.

`python run_example.py` reproduces the deterministic exact-observation column with the ONNX model, step for step. `python run_example.py --stochastic` and/or `--obs-noise` reproduce the other columns up to sampling noise (about ±0.1 over 100 episodes), since their random numbers differ from the training code's.

## Training and overall performance

The policy was trained with PPO for 20M steps in the simulator above, with observation noise, seed 1. Tasks were random DFAs with up to 5 states over the 10 tokens, every size equally likely: an equal mix of reach (visit tokens in order), reach-avoid (the same, with a forbidden token at each step) and randomly mutated reach-avoid tasks. The reward was +1 when the DFA accepts, -1 when it rejects and 0 otherwise, including on leaving the box or timing out. It was discounted by 0.999 and shaped by the DFA's distance to acceptance (potential-based, β = 0.7, so it doesn't change the optimal policy). It has no explicit constraint on the rejection rate.

At the end of training (the last 50 PPO updates, about 0.4M steps, sampling actions, noisy observations), on random tasks from that distribution:

| Policy | Success | Reject | Out of bounds or timeout |
|---|---|---|---|
| `rad_tracking` | 0.95 | 0.03 | 0.02 |
| `rad_obs_noise` (previous bundle, no tracking) | 0.96 | 0.02 | 0.02 |

## Caveats

- The policy is not safe by construction. Training penalized a rejection (reward −1) but did not constrain the rejection rate, and it rejects on about 3% of random tasks in simulation.
- **Deterministic or stochastic.** Acting deterministically, the policy never rejected an example, but it can stall until it times out (`dfa_5`). Sampling actions avoided most stalls but rejected in up to 4% of episodes on `dfa_4` and `dfa_5`. Stochastic mode also sends noisy targets: `action_std` is 0.68–0.92 per axis on a [-1, 1] range, which the velocity filter and the tracking smooth. Check that the real drone tolerates that before flying it stochastically.
- Leaving the box ended every training episode, so the policy has no experience at or beyond the walls. Keep a safety margin or geofence of your own around the mapped box.
- The training simulator has no disturbances. The only noise is the observation noise, and the only model of the real drone's response is the controller tracking above, with fixed gains.
- The policy only handles DFAs with at most 5 states over these 10 tokens. A new DFA's `policy_inputs` have to be computed with the training code (`dfax`). Send us the DFA and we can generate its file.
- The policy is one training run (seed 1). Other seeds are still training.
