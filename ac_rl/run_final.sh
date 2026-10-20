#!/usr/bin/env bash
# Final RAD vs no-RAD drone policies: RAD sampler with each max DFA size and p, with and without DroneEnv's velocity
# filter (--use-velocity-filter), with and without --no-rad (2 x 2 x 2 x 2 = 16 runs per seed). Every run is
# PPO-Lagrangian (--safe) with delta 0.05 and lambda starting at 1, clipped to [0, 1], lr 0.05 (as in
# run_velocity_filter_sweep.sh's clipped schedules), no warm-up, and gamma 0.999 for both rewards and costs.
# Add seeds to `seeds` below for more runs per config.
# Arguments are passed on to every train_drone_policy.py call, e.g.
#
#   ./run_final.sh --wandb
#   ./run_final.sh --save-dir storage/final
#
# Runs that already have a checkpoint are skipped (train_drone_policy.py refuses to overwrite them), so
# re-running the script after an interruption resumes the sweep. A run interrupted mid-training leaves a log
# without a checkpoint; it is reported as failed until that log is deleted.
#
# Plot each filter setting with plot_encoder_sweep.py (vel = no filter, vfilt = filter):
#
#   uv run plot_encoder_sweep.py --out storage/plots/final/vfilt \
#       --tag shaped_x-1.0_1.0_y-1.0_1.0_z-1.0_1.0_speed1.0_dt0.1_vfilt_steps500_g0.999_safe_d0.05_lr0.05_w0_l1.0_c0.0_1.0

set -o pipefail
cd "$(dirname "$0")"

ps=(0.5 None)
seeds=(42)
max_sizes=(5 10)
filter_flags=("" --use-velocity-filter)
rad_flags=("" --no-rad)

total=$(( ${#seeds[@]} * ${#max_sizes[@]} * ${#ps[@]} * ${#filter_flags[@]} * ${#rad_flags[@]} ))
n_trained=0
n_skipped=0
failed=()

out=$(mktemp)
trap 'rm -f "$out"' EXIT
# Without this, Ctrl-C would only stop the current run and the sweep would move on to the next one.
trap 'echo; echo "Interrupted."; exit 130' INT

i=0
for p in "${ps[@]}"; do
    for seed in "${seeds[@]}"; do
        for max_size in "${max_sizes[@]}"; do
            for filter_flag in "${filter_flags[@]}"; do
                for rad_flag in "${rad_flags[@]}"; do
                    i=$((i + 1))
                    run="seed=$seed sampler=RAD max_size=$max_size p=$p ${filter_flag:-no-filter} ${rad_flag:-rad}"
                    echo "=== [$i/$total] $run"

                    args=(--seed "$seed" --sampler RAD --max-size "$max_size" --p "$p" --gamma 0.999
                          --safe --delta 0.05 --lambda-init 1 --lambda-lr 0.05 --lambda-clip 0 1 --lambda-warmup 0
                          --log)
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

echo
echo "Trained: $n_trained, skipped (already trained): $n_skipped, failed: ${#failed[@]}"
for run in "${failed[@]}"; do
    echo "  FAILED: $run"
done
[ ${#failed[@]} -eq 0 ]
