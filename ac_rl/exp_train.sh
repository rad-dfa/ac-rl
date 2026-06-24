#!/bin/bash

mkdir -p storage

for no_rad_flag in "" "--no-rad"; do
  for n_events in 5 10 20; do
    for seed in {0..4}; do
      rad_label=$([ -n "$no_rad_flag" ] && echo "no_rad" || echo "rad")
      echo "===== ${rad_label} n_events=${n_events} seed=${seed} ====="

      uv run python train.py \
        --seed "$seed" \
	--n-tokens "$n_events" \
	--n-symbols 5 \
	--dynamic-alphabet \
	--log \
	$no_rad_flag
    done
  done
done
