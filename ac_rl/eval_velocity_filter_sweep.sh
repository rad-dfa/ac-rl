#!/usr/bin/env bash
# Evaluates every run_velocity_filter_sweep.sh checkpoint with test_drone_policy.py, acting both by sampling from the
# policy (as in training, so the rejection rate compares with the training logs' prob_reject) and with the policy mean
# (--deterministic, as deployed): 3 lambda schedules x 2 max sizes x 2 filter settings x RAD/no-RAD x 2 action modes
# = 48 evaluations of 1000 episodes each. Prints one line per evaluation and collects them in
# storage/eval_velocity_filter_sweep.csv (rates as test_drone_policy.py prints them, to 2 decimals; its "Fail" is a
# DFA rejection, timeouts are counted separately). Arguments are passed on to every test_drone_policy.py call and
# override the defaults here, e.g.
#
#   ./eval_velocity_filter_sweep.sh --n 200
#
# Nothing else is written: test_drone_policy.py only saves files when asked for a gif.

set -o pipefail
cd "$(dirname "$0")"

env_tag=x-1.0_1.0_y-1.0_1.0_z-1.0_1.0_speed1.0_dt0.1
# run_velocity_filter_sweep.sh's lambda schedules as "flags|run-name suffix". Evaluation doesn't use lambda; the
# flags only make test_drone_policy.py's file-name check match the checkpoint.
lambda_schedules=(
    "--lambda-init 0.5 --lambda-lr 0.005|lr0.005_w0_l0.5"
    "--lambda-init 0 --lambda-lr 0.05 --lambda-clip 0 1|lr0.05_w0_l0.0_c0.0_1.0"
    "--lambda-init 1 --lambda-lr 0.05 --lambda-clip 0 1|lr0.05_w0_l1.0_c0.0_1.0"
)
max_sizes=(5 10)
filter_flags=("" --use-velocity-filter)
rad_flags=("" --no-rad)
action_flags=("" --deterministic)

csv=storage/eval_velocity_filter_sweep.csv
echo "lambda_schedule,max_size,velocity_filter,rad,actions,success,reject,timeout,ep_len" > "$csv"

total=$(( ${#lambda_schedules[@]} * ${#max_sizes[@]} * ${#filter_flags[@]} * ${#rad_flags[@]} * ${#action_flags[@]} ))
failed=()

out=$(mktemp)
trap 'rm -f "$out"' EXIT
# Without this, Ctrl-C would only stop the current evaluation and the script would move on to the next one.
trap 'echo; echo "Interrupted."; exit 130' INT

i=0
for schedule in "${lambda_schedules[@]}"; do
    IFS='|' read -r flags suffix <<< "$schedule"
    read -ra lambda_args <<< "$flags"
    for max_size in "${max_sizes[@]}"; do
        for filter_flag in "${filter_flags[@]}"; do
            mode=$([ -n "$filter_flag" ] && echo vfilt || echo vel)
            for rad_flag in "${rad_flags[@]}"; do
                rad=$([ -n "$rad_flag" ] && echo no_rad || echo rad)
                ckpt="storage/policy_params_drone_seed_42_RAD_${max_size}_10_p0.5_${rad}_shaped_${env_tag}_${mode}_steps500_safe_d0.05_${suffix}.msgpack"
                for action_flag in "${action_flags[@]}"; do
                    i=$((i + 1))
                    actions=$([ -n "$action_flag" ] && echo deterministic || echo sampled)
                    run="$suffix max_size=$max_size $mode $rad $actions"
                    if [ ! -f "$ckpt" ]; then
                        echo "=== [$i/$total] $run: no checkpoint $ckpt"
                        failed+=("$run")
                        continue
                    fi

                    args=(--model-path "$ckpt" --seed 42 --sampler RAD --max-size "$max_size" --p 0.5
                          --safe --delta 0.05 "${lambda_args[@]}" --n 1000 --batch-size 250)
                    for flag in "$filter_flag" "$rad_flag" "$action_flag"; do
                        if [ -n "$flag" ]; then
                            args+=("$flag")
                        fi
                    done

                    if uv run test_drone_policy.py "${args[@]}" "$@" > "$out" 2>&1; then
                        # "Success rate: 0.85 +/- 0.36", ..., "Average episode length: 58.93 +/- 40.12"
                        read -r success reject timeout ep_len < <(awk '
                            $1 == "Success" && $2 == "rate:" { s = $3 }
                            $1 == "Fail" && $2 == "rate:" { f = $3 }
                            $1 == "Timeout" && $2 == "rate:" { t = $3 }
                            $1 == "Average" && $3 == "length:" { l = $4 }
                            END { print s, f, t, l }' "$out")
                        echo "=== [$i/$total] $run: success $success, reject $reject, timeout $timeout, ep_len $ep_len"
                        echo "$suffix,$max_size,$mode,$rad,$actions,$success,$reject,$timeout,$ep_len" >> "$csv"
                    else
                        echo "=== [$i/$total] $run: FAILED"
                        tail -n 15 "$out" | sed 's/^/    /'
                        failed+=("$run")
                    fi
                done
            done
        done
    done
done

echo
echo "Evaluated: $((total - ${#failed[@]}))/$total -> $csv"
for run in "${failed[@]}"; do
    echo "  FAILED: $run"
done
[ ${#failed[@]} -eq 0 ]
