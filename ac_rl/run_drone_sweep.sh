#!/usr/bin/env bash
# Runs train_drone_policy.py over 5 random seeds, each sampler, and once with
# and once without --no-rad (30 runs total), keeping --binary-reward --log fixed.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

NUM_SEEDS=5
SEEDS=()
for _ in $(seq 1 "$NUM_SEEDS"); do
    SEEDS+=("$RANDOM")
done
echo "Seeds: ${SEEDS[*]}"

for seed in "${SEEDS[@]}"; do
    for sampler in R RA RAD; do
        for rad_flag in "" "--no-rad"; do
            echo "=== seed=${seed} sampler=${sampler} ${rad_flag:-rad} ==="
            uv run train_drone_policy.py \
                --seed "${seed}" \
                --sampler "${sampler}" \
                --binary-reward \
                --log \
                ${rad_flag}
        done
    done
done
