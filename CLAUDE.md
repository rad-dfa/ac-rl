# ac-rl

## What this is

`ac-rl` is a **JAX/Flax implementation of Automata-Conditioned Reinforcement Learning (AC-RL)**: PPO training of a policy that is conditioned on a goal specified as a DFA (automaton), in a grid-world token-collection environment. It's the RL/policy-training half of a small multi-repo research project (the "rad-dfa" ecosystem, see below); the environment, the DFA formalism, and the pretrained automaton encoder all live in sibling repos and are pulled in as PyPI dependencies, not vendored.

Two papers are cited in `README.md` (Yalcinkaya, Lauffer, Vazquez-Chanlatte, Seshia): "Compositional Automata Embeddings for Goal-Conditioned RL" (NeurIPS 2024) and "Provably Correct Automata Embeddings for Optimal Automata-Conditioned RL" (NeuS 2025). The embedding theory those papers describe is implemented in `../rad-embeddings`, not here — this repo trains/evaluates policies *using* those embeddings (or an untrained from-scratch version of the same architecture, as an ablation).

**Naming quirk**: `pyproject.toml`'s `[project].name` is `"rad-marl"`, not `"ac-rl"` — an apparent copy-paste leftover from the sibling `../rad-marl` repo (whose own `pyproject.toml` also declares `name = "rad-marl"`), never renamed here. Harmless (the importable package is `ac_rl`), just confusing if you go looking for a `rad-marl` package.

## Repo layout

| Path | Purpose |
|---|---|
| `ac_rl/train.py` | Entry point. Defines the `ActorCritic` flax Module and a CLI that builds the env/sampler/encoder, runs `ppo.make_train`, and saves a checkpoint. |
| `ac_rl/ppo.py` | `make_train(config, env, network)` — the actual JAX PPO algorithm (PureJaxRL/JaxMARL-style: fully `jax.jit`/`vmap`/`lax.scan`). |
| `ac_rl/test.py` | Evaluation CLI: loads saved checkpoint(s), runs vmapped episodes, reports success rate / length / reward / discounted return. Doesn't support `--dynamic-alphabet` checkpoints (Known issues #1). |
| `ac_rl/wrappers.py` | `LogWrapper` — JaxMARL-style episode-return/length bookkeeping wrapper around a `dfa_gym.MultiAgentEnv`. |
| `ac_rl/plot.py` | Standalone matplotlib script over saved CSV logs; expects a `storage/{exp_name}_reach/` + `storage/{exp_name}_reach_avoid/` directory convention that `train.py` does not itself produce (see Known issues #4) — treat as a sweep-companion script, not a drop-in visualizer for a single `train.py` run. |
| `ac_rl/train_drone_policies.sh` | The one training script for drone policies: 16 PPO-Lagrangian runs of `train_drone_policy.py` on the MIX sampler with PBRS beta 0.7 (max size 5/10 × p 0.5/None × velocity filter on/off × RAD/no-RAD), resumable. Plot with `plot_drone_policies.py`, package the handoff bundle with `export_drone_bundle.py`. Older sweep scripts were removed 2026-10-22 (in git history). |
| `ac_rl/naming.py` | The only place run names are built and parsed (checkpoints `policy_params_{run}.msgpack`, logs `logs/log_{run}.csv`, gifs, ONNX, plot tags, bundle policies). Every setting that changes training is in the name; samplers go by `rad_embeddings`' short names R/RA/RAD/MIX. `DRONE_SWEEP` mirrors `train_drone_policies.sh`'s fixed settings — keep them in sync. |
| `ac_rl/storage/` | Checkpoint (`.msgpack`) output dir; `--log` CSVs go to `storage/logs/`. Started fresh on 2026-10-22, when DroneEnv labels switched to firing only on region entry (dfa-gym `40fe01b`). Everything trained under the old every-step ("sticky") labels is archived in the gitignored `ac_rl/storage_sticky_labels.zip`. |
| `ac_rl/drone_bundle/` | Hand-written `README.md` + `run_example.py` (numpy/onnxruntime reference control loop) of the drone handoff bundle; `export_drone_bundle.py` copies them into `storage/share/drone_rad_max_size5/`. |

No tests, no CI, no Makefile/Dockerfile — a pure research script repo. Dependency/lock management via `uv` (`pyproject.toml` + `uv.lock`), Python `>=3.10`.

## The rad-dfa ecosystem and dependency versions

`ac-rl` depends on three sibling packages, all developed in adjacent directories under `../` but **installed from PyPI** per `uv.lock` (not local editable installs) — the local checkouts are ahead of what's actually pinned/resolved:

- **`dfa-gym`** (`../dfa-gym`) — supplies `TokenEnv`, `DFAWrapper`, `MultiAgentEnv`/`State`. Pinned `>=0.2.14`, resolved/locked to **0.2.14**. Local checkout is on version 0.2.18 (branch `drone_env`), but the commits between 0.2.14 and 0.2.18 only touch the bisimulation-game envs (`DFABisimEnv`), not `TokenEnv`/`DFAWrapper` — the code paths `ac-rl` actually uses are unaffected by the drift. `dfa-gym` has its own `CLAUDE.md` at its repo root with much more detail (env mechanics, `DFAx` integration, WIP state) — read that before making changes that touch the env/wrapper boundary.
- **`dfax`** (`../dfax`) — supplies `DFAx` (the DFA representation), the samplers, and `list2batch`/`batch2graph`. Pinned `>=0.1.13`, resolved to **0.1.13**. Local checkout is at 0.1.19 (6 patches ahead) — features like `mutate_reject_lang`, `shrink`/`expand`, or `data2sampler` may differ from what's actually installed; don't assume the local source is byte-for-byte what runs here.
- **`rad-embeddings`** (`../rad-embeddings`) — supplies `Encoder`/`EncoderModule` (the pretrained automaton GNN encoder). Pinned `>=0.2.3`, resolved to **0.2.3**. Local checkout is at 0.2.10 (7 patches ahead) — `EncoderModule.train` as a classmethod, unbounded-DFA-size support, and the binary-reward/gamma-sweep bundled checkpoints all postdate 0.2.3 and may not exist in the installed version.

**Practical implication**: when in doubt about an API detail, check `uv.lock` (or `uv pip show <pkg>` in the actual env) for the resolved version before trusting the newer local `../dfa-gym`/`../dfax`/`../rad-embeddings` source as ground truth for what `ac-rl` runs against.

Other repos under `../` are part of the same research group but are **not** dependencies of `ac-rl` — useful context, not code to read for understanding this repo: `acc-marl` (multi-agent cooperative follow-on), `rad-vs-instruct` (compares RAD embeddings against natural-language instruction embeddings, consumes `dfa2instruct`'s datasets), `dfa2instruct` (automata↔instruction dataset generator), `rad-marl` (an earlier project — the likely source of the copy-pasted `pyproject.toml` name), `papers/` (PDF sources for the cited papers, plus `icml26.pdf`/`neus26.pdf` preprints suggesting follow-up work), `flann26` (unrelated/empty scaffold).

## Task setup: env, DFA goals, samplers

- **`TokenEnv`** (`dfa_gym`): a grid-world token-collection environment (torus grid, discrete 5-action space — up/down/left/right/noop), single- or multi-agent. `ac-rl` always uses `n_agents=1`. Per-agent observation is a `(C, H, W)` image. `fixed_map_seed` (always passed, `=seed`) makes every episode reset to the *same* map/token layout rather than a fresh random one per episode — this is a fixed-map training/eval regime.
- **`DFAWrapper`** (`dfa_gym`) wraps `TokenEnv` with a per-agent goal DFA (`dfax.DFAx`), sampled fresh each episode. `--sampler` (train.py/test.py) selects among:
  - `Reach`/`R` → `ReachSampler`: a linear-chain DFA — visit a fixed sequence of token types in order, ignore (stutter on) everything else.
  - `ReachAvoid`/`RA` → `ReachAvoidSampler`: same chain, plus one designated "hazard" token per state that jumps straight to an absorbing fail state.
  - `ReachAvoidDerived`/`RAD` (**default** in both `train.py` and `test.py`) → `RADSampler`: builds a Reach/ReachAvoid-style chain, then applies up to `max_mutations` random structural mutations. This is the same sampler family used in `rad-embeddings` to pretrain the encoder via its bisimulation-game — i.e. the RL task distribution here and the embedding-pretraining distribution are drawn from the same family.
  - `Mixture`/`MIX` → `MixtureSampler` (dfax ≥ 0.1.22): each DFA drawn from R, RA or RAD with equal probability.
  - The pretrained encoder is always the one `rad-embeddings` trained on the same sampler, p and reward type (`train.py` uses p = None).
  - `max_size` = max DFA states (padded capacity), `n_tokens` = alphabet size = number of distinct grid token types.
- `--binary-reward` (default off): if off, the DFA gives a shaped/graded terminal reward; if on, a strict ±1.
- `--dynamic-alphabet` (default off, `train.py` only): each episode remaps DFA alphabet symbols onto a random subset of env tokens, exposed to the policy as `obs["tkn"]`, forcing the encoder to read the live mapping rather than memorize fixed symbol↔token identity. Uses a much larger PPO config (64 envs × 1024 steps × 1e7 total timesteps, vs. the default 16×128×1e6) — apparently needs substantially more samples. **`test.py` does not support this mode** — see Known issues #1.
- The DFA goal is observed as a dense, fixed-shape **batched graph** (`obs["dfa"]`): `DFAWrapper` calls `DFAx.to_graph()` per agent, and `ac_rl/train.py` merges the env-batch dimension via `dfax.batch2graph` (`train.py:49`) before feeding it to the encoder. Graph dict keys: `node_features (…, 4)`, `edge_features (…, n_tokens+8)`, `edge_index` (dense meshgrid), `current_state`, `n_states`.

## `ActorCritic` network (`ac_rl/train.py`)

Three feature streams, concatenated:
1. **CNN** over the `(C,H,W)→(H,W,C)` grid image — 3× `Conv`+ReLU (`padding` field, default `"VALID"`), flattened.
2. **Automaton embedding** — `encoder(dfa_graph)`, where `encoder` is either:
   - `rad_embeddings.Encoder` (pretrained, frozen via `stop_gradient` internally, loaded from a bundled `.msgpack` checkpoint keyed by `max_size`/`n_tokens`/`seed`/`binary_reward`/`gamma`; the default checkpoint handles ≤10-state DFAs over a 10-token alphabet), or
   - `rad_embeddings.EncoderModule` (same GATv2-based architecture, randomly initialized, trained end-to-end with the policy) — selected via `--no-rad`.
   This is the "with vs. without pretrained RAD embeddings" ablation the top-level `README.md` refers to.
3. If `dynamic_alphabet`: an extra small MLP over `[dfa_feat, token-embedding-of-obs["tkn"]]`, replacing the raw `dfa_feat` in the final concatenation.

Separate 2-layer value head and 3-layer policy head (`distrax.Categorical` over the 5 grid actions) on top of the concatenated feature, orthogonal init throughout (standard PPO init). `deterministic=True` (argmax action instead of a distribution) is a defined-but-unused field — no CLI flag sets it currently.

## Training loop (`ac_rl/ppo.py::make_train`)

Textbook vectorized PPO (PureJaxRL/JaxMARL style): fully `jax.jit`/`lax.scan`/`vmap`, `NUM_ENVS` parallel envs, GAE advantages, clipped surrogate + clipped value loss + entropy bonus, Adam with optional linear LR annealing and global-norm clipping.

- **`batchify(obss, agents)` just returns `obss[agents[0]]`** — despite using `dfa_gym`'s `MultiAgentEnv` API and per-agent dicts throughout, this training loop only ever drives **one** policy-controlled agent (`n_agents=1` is hardcoded at the `TokenEnv(...)` call site in both `train.py` and `test.py`). It is not actually multi-agent today.
- Three logging backends, switched on by config flags, all wired through `jax.experimental.io_callback`/`jax.debug.callback` so they can run inside the jitted scan: `LOG` → append CSV, `WANDB` → `wandb.log` (hardcoded entity `beyazit-y-berkeley-eecs`, project `rad-rl-jax` — not configurable via CLI), `DEBUG` → `jax.debug.print` to stdout. All three duplicate nearly identical windowed-metric bookkeeping (`deque(maxlen=steps_per_update)` buffers for ep_len/return/disc_return, `prob_success`/`prob_fail` from `returned_episode_last_returns`) — copy-pasted 3×, a known simplification target if this file is ever touched again.

## CLI usage

```
# Train
python ac_rl/train.py --seed 42 --sampler RAD --max-size 5 --n-tokens 10 \
    [--wandb] [--debug] [--log] [--no-rad] [--binary-reward] [--dynamic-alphabet]
# → storage/policy_params_token_seed_42_RAD_5_10_pNone_rad_shaped.msgpack
#   (+ storage/logs/log_token_seed_42_RAD_5_10_pNone_rad_shaped.csv if --log); see ac_rl/naming.py

# Test
python ac_rl/test.py --seeds 42 43 44 --max-size 5 --n-tokens 10 --sampler RAD \
    --model-dir storage --n 10 [--batch-size B] [--no-rad] [--binary-reward] [--csv]

# Plot (expects a sweep-script directory layout, see Repo layout table)
python ac_rl/plot.py <exp_name>
```

## Known issues / gotchas (read before touching this code)

1. `test.py` doesn't support `--dynamic-alphabet` checkpoints: its `ActorCritic(...)`/`DFAWrapper(...)` calls don't thread `dynamic_alphabet` through, and it only loads non-dynamic-alphabet run names.
2. `pyproject.toml`'s package name is `"rad-marl"`, not `"ac-rl"` (see "What this is" above) — cosmetic, but don't be misled by it.
3. `dfax.data2sampler` is imported in `test.py` (`test.py:16`) but never called — dead import. (Separately, `dfax`'s own `data2sampler` has a broken `str`-path branch — an undefined `data_path` variable — unrelated to `ac-rl` since it's never exercised here.)
4. `list2batch` is imported in `train.py` (`train.py:11`) but never called — only `batch2graph` is used, since the batched `obs["dfa"]` graph already arrives pre-batched from `DFAWrapper`'s vmapped rollout.
5. Dependency version drift: see the ecosystem section above — all three sibling deps are pinned older than what's checked out locally under `../`.
6. `ac_rl/storage/` ships committed binary checkpoints and CSV logs in git — these are the README's reference example artifacts (`seed=42, RAD, max-size=5, n-tokens=10`), not a place to accumulate every experiment's output.

## Git branches / project state (as of this writing)

- `main` and the current branch `drone_policy` are identical (both at commit `d0de1b5`, "Add option for dynamic alphabet").
- `origin/dynamic-alphabet` is 2 commits ahead of `main` (`5aceb09` "new architecture and sweeps", `d0d1b03` "figs generated") — has newer, unmerged work.
- `origin/dev` diverged much earlier into a long separate history of encoder-architecture experiments (MLP vs. attention for the CNN branch — "Try attention again" ×3, "Revert to MLP" ×2, etc.) that never merged back to `main`. Treat it as an abandoned experimentation branch, not a source of newer stable code.
- The current branch name `drone_policy`, together with `../dfa-gym` currently being checked out on its own `drone_env` branch (adding a continuous 3D `DroneEnv`, not yet released to PyPI or used here), suggests the likely next step for this repo is porting the AC-RL setup from the current grid-world `TokenEnv` to that new drone environment. Check `../dfa-gym`'s `drone_env` branch before starting that work — the base env this repo builds on is mid-migration.
