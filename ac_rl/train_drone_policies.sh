#!/usr/bin/env bash
# Trains the RAD vs no-RAD drone policies over seeds 1-5: MIX sampler, max DFA size 5, p None, with DroneEnv's
# velocity filter (--use-velocity-filter), with the reward shaped by PBRS with beta 0.7 and unshaped (no --beta),
# with and without --no-rad (5 x 2 x 2 = 20 runs). Every run trains for 2e7 steps with PPO-Lagrangian (--safe) with
# delta 0.05 and lambda starting at 1, clipped to [0, 1], lr 0.05, no warm-up, and gamma 0.999 for both rewards and
# costs. Add values to the lists below for more seeds, samplers (R, RA, RAD, MIX), max sizes, ps, betas (None: no
# PBRS) or filter settings.
# naming.py's DRONE_SWEEP lists the fixed settings for plot_drone_policies.py, with beta 0.7; keep the two in sync.
# Arguments are passed on to every train_drone_policy.py call, e.g.
#
#   ./train_drone_policies.sh --wandb
#   ./train_drone_policies.sh --save-dir storage/final
#
# Checkpoints go to the save dir's (default storage) models/ subdir and CSV logs to its logs/ subdir. Runs that
# already have a checkpoint are skipped (train_drone_policy.py refuses to overwrite them), so re-running the script
# after an interruption resumes training. A run interrupted mid-training leaves a log without a checkpoint; it is reported
# as failed until that log is deleted.
#
# Then plot RAD vs no-RAD (mean and std over seeds) with
#
#   uv run plot_drone_policies.py
#
# export_drone_bundle.py packages the script's earlier 1e7-step, seed 42, max size 5, p 0.5 policies for handoff.

set -o pipefail
cd "$(dirname "$0")"

samplers=(MIX)
ps=(None)
seeds=(1 2 3 4 5)
max_sizes=(5)
filter_flags=(--use-velocity-filter)
betas=(0.7 None)
rad_flags=("" --no-rad)

total=$(( ${#samplers[@]} * ${#seeds[@]} * ${#max_sizes[@]} * ${#ps[@]} * ${#filter_flags[@]} * ${#betas[@]}
          * ${#rad_flags[@]} ))
n_trained=0
n_skipped=0
failed=()

out=$(mktemp)
trap 'rm -f "$out"' EXIT
# Without this, Ctrl-C would only stop the current run and the sweep would move on to the next one.
trap 'echo; echo "Interrupted."; exit 130' INT

i=0
for sampler in "${samplers[@]}"; do
    for p in "${ps[@]}"; do
        for seed in "${seeds[@]}"; do
            for max_size in "${max_sizes[@]}"; do
                for filter_flag in "${filter_flags[@]}"; do
                    for beta in "${betas[@]}"; do
                        for rad_flag in "${rad_flags[@]}"; do
                            i=$((i + 1))
                            run="seed=$seed sampler=$sampler max_size=$max_size p=$p ${filter_flag:-no-filter} beta=$beta ${rad_flag:-rad}"
                            echo "=== [$i/$total] $run"

                            args=(--seed "$seed" --sampler "$sampler" --max-size "$max_size" --p "$p" --gamma 0.999
                                  --total-timesteps 2e7 --safe --delta 0.05 --lambda-init 1 --lambda-lr 0.05 --lambda-clip 0 1
                                  --lambda-warmup 0 --log)
                            if [ "$beta" != None ]; then
                                args+=(--beta "$beta")
                            fi
                            if [ -n "$filter_flag" ]; then
                                args+=("$filter_flag")
                            fi
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
            done
        done
    done
done

echo
echo "Trained: $n_trained, skipped (already trained): $n_skipped, failed: ${#failed[@]}"
for run in "${failed[@]}"; do
    echo "  FAILED: $run"
done
[ ${#failed[@]} -eq 0 ]
