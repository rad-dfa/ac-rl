#!/bin/bash

if [ $# -lt 2 ]; then
  echo "Usage: $0 <N_samples> <csv_file>"
  echo "Example: ./exp_test.sh 100 results.csv"
  exit 1
fi

N=$1
CSV=$2

echo "RAD, N_Events, Sampler, OOD, Success Probability, Episode Length, Episode Reward, Episode Discounted Reward" > "$CSV"

for ood_flag in "" "--ood"; do
  for sampler in R RA RAD; do
    for no_rad_flag in "" "--no-rad"; do
      for n_events in 5 10 20; do
        ood_label=$([ -n "$ood_flag" ] && echo "ood" || echo "in_dist")
        rad_label=$([ -n "$no_rad_flag" ] && echo "no_rad" || echo "rad")
        echo "===== ood=${ood_label} sampler=${sampler} ${rad_label} n_events=${n_events} ====="
        uv run python test.py \
          --csv \
          --n "$N" \
          --batch-size 64 \
          --seeds 0 1 2 3 4 \
          --sampler "$sampler" \
          --n-tokens "$n_events" \
          --n-symbols 5 \
          --dynamic-alphabet \
          $ood_flag \
          $no_rad_flag >> "$CSV"
      done
    done
  done
done
