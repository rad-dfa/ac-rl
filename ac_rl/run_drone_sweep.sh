#!/usr/bin/env bash
# Runs train_drone_policy.py over 5 random seeds, each max DFA size in MAX_SIZES,
# each sampler, with and without --no-rad, and with and without --binary-reward
# (60 runs per max size), keeping --log fixed.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

MAX_SIZES=(5)

NUM_SEEDS=5
SEEDS=()
for _ in $(seq 1 "$NUM_SEEDS"); do
    SEEDS+=("$RANDOM")
done

for seed in "${SEEDS[@]}"; do
    for max_size in "${MAX_SIZES[@]}"; do
        for sampler in R RA RAD; do
            for reward_flag in "--binary-reward" ""; do
                for rad_flag in "" "--no-rad"; do
                    cmd=(uv run train_drone_policy.py
                        --seed "${seed}"
                        --max-size "${max_size}"
                        --sampler "${sampler}"
                        --log)
                    if [[ -n "${reward_flag}" ]]; then
                        cmd+=("${reward_flag}")
                    fi
                    if [[ -n "${rad_flag}" ]]; then
                        cmd+=("${rad_flag}")
                    fi
                    echo "=== $(printf '%q ' "${cmd[@]}")==="
                    "${cmd[@]}"
                done
            done
        done
    done
done
