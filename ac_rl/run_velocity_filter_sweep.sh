#!/usr/bin/env bash
# RAD vs no-RAD drone policies with and without DroneEnv's velocity filter (--use-velocity-filter, the deployed
# drone's critically-damped velocity tracking): RAD sampler with p 0.5, each max DFA size, with and without --no-rad,
# under three lambda schedules (3 x 2 x 2 x 2 = 24 runs), seed 42. Every run is PPO-Lagrangian (--safe) with
# delta 0.05 and no warm-up. The first schedule is run_encoder_sweep.sh's first one (lambda starting at 0.5, lr
# 0.005), so its runs without the filter are the same as that sweep's RAD p 0.5 runs. The other two start lambda
# at 0 and at 1, clipped to [0, 1], with lr 0.05: lambda falls by at most lr * delta per update (P(reject) >= 0),
# so this lr lets it cross [0, 1] downwards in ~400 of the ~1220 updates, where 0.005 would need ~4000.
# Arguments are passed on to every train_drone_policy.py call, e.g.
#
#   ./run_velocity_filter_sweep.sh --wandb
#   ./run_velocity_filter_sweep.sh --save-dir storage/velocity_filter_sweep
#
# Runs that already have a checkpoint are skipped (train_drone_policy.py refuses to overwrite them), so
# re-running the script after an interruption resumes the sweep. A run interrupted mid-training leaves a log
# without a checkpoint; it is reported as failed until that log is deleted.
#
# Plot each filter setting and lambda schedule with plot_encoder_sweep.py (vel = no filter, vfilt = filter; the
# schedule is the run-name suffix after "_safe", e.g. _d0.05_lr0.005_w0_l0.5 or _d0.05_lr0.05_w0_l0.0_c0.0_1.0):
#
#   uv run plot_encoder_sweep.py --out storage/plots/velocity_filter_sweep/vfilt_l0.0_c0.0_1.0 \
#       --tag shaped_x-1.0_1.0_y-1.0_1.0_z-1.0_1.0_speed1.0_dt0.1_vfilt_steps500_safe_d0.05_lr0.05_w0_l0.0_c0.0_1.0

set -o pipefail
cd "$(dirname "$0")"

lambda_schedules=(
    "--lambda-init 0.5 --lambda-lr 0.005"
    "--lambda-init 0 --lambda-lr 0.05 --lambda-clip 0 1"
    "--lambda-init 1 --lambda-lr 0.05 --lambda-clip 0 1"
)
max_sizes=(5 10)
filter_flags=("" --use-velocity-filter)
rad_flags=("" --no-rad)

total=$(( ${#lambda_schedules[@]} * ${#max_sizes[@]} * ${#filter_flags[@]} * ${#rad_flags[@]} ))
n_trained=0
n_skipped=0
failed=()

out=$(mktemp)
trap 'rm -f "$out"' EXIT
# Without this, Ctrl-C would only stop the current run and the sweep would move on to the next one.
trap 'echo; echo "Interrupted."; exit 130' INT

i=0
for schedule in "${lambda_schedules[@]}"; do
    read -ra lambda_args <<< "$schedule"
    for max_size in "${max_sizes[@]}"; do
        for filter_flag in "${filter_flags[@]}"; do
            for rad_flag in "${rad_flags[@]}"; do
                i=$((i + 1))
                run="$schedule sampler=RAD p=0.5 max_size=$max_size ${filter_flag:-no-filter} ${rad_flag:-rad}"
                echo "=== [$i/$total] $run"

                args=(--seed 42 --sampler RAD --max-size "$max_size" --p 0.5
                      --safe "${lambda_args[@]}" --lambda-warmup 0 --delta 0.05 --log)
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

echo
echo "Trained: $n_trained, skipped (already trained): $n_skipped, failed: ${#failed[@]}"
for run in "${failed[@]}"; do
    echo "  FAILED: $run"
done
[ ${#failed[@]} -eq 0 ]
