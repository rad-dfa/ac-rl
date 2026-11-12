#!/usr/bin/env bash
# Trains the drone policies over seeds 1-5 on DroneEnv without a geofence (episodes end on leaving the box, dfa-gym
# da64b02), with the tracking controller (dfa-gym b835818: the drone tracks DroneEnv's velocity filter through the
# deployed stack's controller and motor delay, --use-velocity-filter --use-tracking-controller) and observation noise
# (--obs-noise-std 0.02 0.1, position and velocity std): MIX sampler, max DFA size 5, p None, 2e7 steps and gamma 0.999
# for every run, varying
#
#   arm                shaped:  PPO on the shaped +1/-1/0 DFA reward
#                      safe_l0: PPO-Lagrangian (--safe) with delta 0.05, lambda learned from 0 (lr 0.05, no warm-up,
#                               unclipped, i.e. in [0, inf))
#   PBRS               beta 0.7, or none
#   encoder            pretrained RAD, or trained from scratch (--no-rad)
#
# for 5 x 2 x 2 x 2 = 40 runs, seeds outermost, so a partial sweep has every config for its first seeds.
# Until 2026-11-11 it ran without the tracking controller (velocity filter only), both without observation noise and
# with --obs-noise-std 0.02 0.1, for 80 runs; they are kept in storage under their _vfilt_ names (the new runs' names
# have _vtrack_). Until 2026-11-10 it also had a binary arm (PPO on the binary 1/0 DFA reward, --binary-reward) and a
# safe_l1 arm (safe_l0 with lambda learned from 1), for 160 runs; they were dropped after seed 1, whose runs of them
# are kept.
# naming.py's DRONE_SWEEP lists the settings shared by all runs, with the reference arm's values (safe_l0, beta 0.7)
# for the varied ones, and DRONE_SWEEP_NOISES, DRONE_SWEEP_ARMS and DRONE_SWEEP_BETAS the noises, arms and
# betas below; keep them in sync. Arguments are passed on to every train_drone_policy.py call, e.g.
#
#   ./train_drone_policies.sh --wandb
#   ./train_drone_policies.sh --save-dir storage/final
#
# SHARD=k/N runs only the sweep's runs k, k + N, k + 2N, ... (1 <= k <= N), to split it over N machines or GPUs:
#
#   SHARD=1/4 ./train_drone_policies.sh    # ... through SHARD=4/4
#
# Shards never share a run. Running more than one shard on a GPU needs XLA_PYTHON_CLIENT_PREALLOCATE=false, as JAX
# otherwise gives most of the GPU's memory to the first one. DRY_RUN=1 prints the train_drone_policy.py calls of the
# shard (by default the whole sweep) without running them.
#
# Checkpoints go to the save dir's (default storage) models/ subdir and CSV logs to its logs/ subdir. Runs that
# already have a checkpoint are skipped (train_drone_policy.py refuses to overwrite them), so re-running the script
# after an interruption resumes training. A run interrupted mid-training leaves a log without a checkpoint; it is reported
# as failed until that log is deleted.
#
# Plot the training logs of every config, finished or not, into storage/plots/drone_grid_tracking with
#
#   uv run plot_drone_grid.py
#
# Then test the policies (each under the observation noise it was trained with) with
#
#   uv run test_drone_policies.py storage/models/*_vtrack_*.msgpack --out storage/test/drone_policies_tracking

set -o pipefail
cd "$(dirname "$0")"

seeds=(1 2 3 4 5)
noises=("0.02 0.1")  # None for none
arms=(shaped safe_l0)
betas=(0.7 None)
rad_flags=("" --no-rad)

# The reward and PPO-Lagrangian flags of an arm
arm_flags() {
    case "$1" in
        shaped) ;;
        safe_l0) echo --safe --delta 0.05 --lambda-lr 0.05 --lambda-warmup 0 --lambda-init 0 ;;
        *) echo "Unknown arm: $1" >&2; return 1 ;;
    esac
}
for arm in "${arms[@]}"; do
    arm_flags "$arm" > /dev/null || exit 2
done

shard=${SHARD:-1/1}
if ! [[ "$shard" =~ ^[0-9]+/[0-9]+$ ]]; then
    echo "SHARD must be k/N, got $shard" >&2
    exit 2
fi
shard_k=$((10#${shard%/*}))
shard_n=$((10#${shard#*/}))
if (( shard_k < 1 || shard_k > shard_n )); then
    echo "SHARD=k/N needs 1 <= k <= N, got $shard" >&2
    exit 2
fi
dry_run=${DRY_RUN:-0}

total=$(( ${#seeds[@]} * ${#noises[@]} * ${#arms[@]} * ${#betas[@]} * ${#rad_flags[@]} ))
shard_total=$(( total >= shard_k ? (total - shard_k) / shard_n + 1 : 0 ))
n_trained=0
n_skipped=0
failed=()

out=$(mktemp)
trap 'rm -f "$out"' EXIT
# Without this, Ctrl-C would only stop the current run and the sweep would move on to the next one.
trap 'echo; echo "Interrupted."; exit 130' INT

i=0  # the run's index in the sweep
j=0  # and in this shard
for seed in "${seeds[@]}"; do
    for noise in "${noises[@]}"; do
        for arm in "${arms[@]}"; do
            for beta in "${betas[@]}"; do
                for rad_flag in "${rad_flags[@]}"; do
                    i=$((i + 1))
                    if (( (i - 1) % shard_n != shard_k - 1 )); then
                        continue
                    fi
                    j=$((j + 1))
                    run="seed=$seed noise=${noise// /,} arm=$arm beta=$beta ${rad_flag:-rad}"
                    echo "=== [$j/$shard_total] (sweep run $i/$total) $run"

                    args=(--seed "$seed" --sampler MIX --max-size 5 --p None --use-velocity-filter
                          --use-tracking-controller --gamma 0.999 --total-timesteps 2e7 --log)
                    if [ "$noise" != None ]; then
                        read -ra noise_stds <<< "$noise"
                        args+=(--obs-noise-std "${noise_stds[@]}")
                    fi
                    read -ra flags <<< "$(arm_flags "$arm")"
                    args+=("${flags[@]}")
                    if [ "$beta" != None ]; then
                        args+=(--beta "$beta")
                    fi
                    if [ -n "$rad_flag" ]; then
                        args+=("$rad_flag")
                    fi

                    if [ "$dry_run" != 0 ]; then
                        echo uv run train_drone_policy.py "${args[@]}" "$@"
                    elif uv run train_drone_policy.py "${args[@]}" "$@" 2>&1 | tee "$out"; then
                        n_trained=$((n_trained + 1))
                    elif grep -q "already trained" "$out"; then
                        echo "--- already trained, skipping"
                        n_skipped=$((n_skipped + 1))
                    else
                        failed+=("$run")
                    fi
                done
            done
        done
    done
done

echo
if [ "$dry_run" != 0 ]; then
    echo "Dry run: $j of the sweep's $total runs in shard $shard_k/$shard_n"
    exit 0
fi
echo "Trained: $n_trained, skipped (already trained): $n_skipped, failed: ${#failed[@]}"
for run in "${failed[@]}"; do
    echo "  FAILED: $run"
done
[ ${#failed[@]} -eq 0 ]
