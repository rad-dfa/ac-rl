#!/usr/bin/env bash
# Sweeps PBRS's beta for the drone policies: MIX sampler with max DFA size 5 and p None, with DroneEnv's velocity
# filter (--use-velocity-filter), each beta with and without --no-rad (3 x 2 = 6 runs per seed). Everything else is
# as in train_drone_policies.sh: PPO-Lagrangian (--safe) with delta 0.05 and lambda starting at 1, clipped to [0, 1],
# lr 0.05, no warm-up, and gamma 0.999 for both rewards and costs. Add seeds to `seeds` below for more runs per config.
# Arguments are passed on to every train_drone_policy.py call, e.g.
#
#   ./beta_sweep.sh --wandb
#   ./beta_sweep.sh --save-dir storage/beta_sweep
#
# Checkpoints go to the save dir's (default storage) models/ subdir and CSV logs to its logs/ subdir. Runs that
# already have a checkpoint are skipped (train_drone_policy.py refuses to overwrite them), so re-running the script
# after an interruption resumes training. A run interrupted mid-training leaves a log without a checkpoint; it is reported
# as failed until that log is deleted.

set -o pipefail
cd "$(dirname "$0")"

sampler=MIX
p=None
max_size=5
seeds=(42)
betas=(0.25 0.5 0.75)
rad_flags=("" --no-rad)

total=$(( ${#seeds[@]} * ${#betas[@]} * ${#rad_flags[@]} ))
n_trained=0
n_skipped=0
failed=()

out=$(mktemp)
trap 'rm -f "$out"' EXIT
# Without this, Ctrl-C would only stop the current run and the sweep would move on to the next one.
trap 'echo; echo "Interrupted."; exit 130' INT

i=0
for seed in "${seeds[@]}"; do
    for beta in "${betas[@]}"; do
        for rad_flag in "${rad_flags[@]}"; do
            i=$((i + 1))
            run="seed=$seed sampler=$sampler max_size=$max_size p=$p beta=$beta ${rad_flag:-rad}"
            echo "=== [$i/$total] $run"

            args=(--seed "$seed" --sampler "$sampler" --max-size "$max_size" --p "$p" --use-velocity-filter
                  --gamma 0.999 --beta "$beta" --safe --delta 0.05 --lambda-init 1 --lambda-lr 0.05 --lambda-clip 0 1
                  --lambda-warmup 0 --log)
            if [ -n "$rad_flag" ]; then
                args+=("$rad_flag")
            fi

            if uv run train_drone_policy.py "${args[@]}" "$@" 2>&1 | tee "$out"; then
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

echo
echo "Trained: $n_trained, skipped (already trained): $n_skipped, failed: ${#failed[@]}"
for run in "${failed[@]}"; do
    echo "  FAILED: $run"
done
[ ${#failed[@]} -eq 0 ]
