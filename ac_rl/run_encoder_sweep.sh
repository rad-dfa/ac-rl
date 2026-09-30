#!/usr/bin/env bash
# RAD vs no-RAD drone policies for the 10-token, shaped-reward pretrained encoders in rad-embeddings of the RAD and
# RA samplers: each sampler, max DFA size and p (2 x 2 x 2 = 8 encoders), with and without --no-rad, under two
# lambda schedules (32 runs), seed 42. The policy's task distribution (--sampler, --max-size, --p) is the one its
# encoder was pretrained on; every run is PPO-Lagrangian (--safe) with delta 0.05 and lambda lr 0.005.
# Arguments are passed on to every train_drone_policy.py call, e.g.
#
#   ./run_encoder_sweep.sh --wandb
#   ./run_encoder_sweep.sh --save-dir storage/encoder_sweep
#
# Runs that already have a checkpoint are skipped (train_drone_policy.py refuses to overwrite them), so
# re-running the script after an interruption resumes the sweep. A run interrupted mid-training leaves a log
# without a checkpoint; it is reported as failed until that log is deleted.

set -o pipefail
cd "$(dirname "$0")"

# "--lambda-init --lambda-warmup" pairs: lambda starting at 0.5, and lambda held at 0 for the first 2M env steps
# so reaching is learned before rejections are penalized.
lambda_schedules=("0.5 0" "0 2e6")
samplers=(RA RAD)
max_sizes=(5)
ps=(0.5 None)
rad_flags=("" --no-rad)

total=$(( ${#lambda_schedules[@]} * ${#samplers[@]} * ${#max_sizes[@]} * ${#ps[@]} * ${#rad_flags[@]} ))
n_trained=0
n_skipped=0
failed=()

out=$(mktemp)
trap 'rm -f "$out"' EXIT
# Without this, Ctrl-C would only stop the current run and the sweep would move on to the next one.
trap 'echo; echo "Interrupted."; exit 130' INT

i=0
for schedule in "${lambda_schedules[@]}"; do
    read -r lambda_init lambda_warmup <<< "$schedule"
    for sampler in "${samplers[@]}"; do
        for max_size in "${max_sizes[@]}"; do
            for p in "${ps[@]}"; do
                for rad_flag in "${rad_flags[@]}"; do
                    i=$((i + 1))
                    run="lambda_init=$lambda_init lambda_warmup=$lambda_warmup sampler=$sampler max_size=$max_size p=$p ${rad_flag:-rad}"
                    echo "=== [$i/$total] $run"

                    args=(--seed 42 --sampler "$sampler" --max-size "$max_size" --p "$p"
                          --safe --lambda-init "$lambda_init" --lambda-warmup "$lambda_warmup"
                          --delta 0.05 --lambda-lr 0.005 --log)
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
