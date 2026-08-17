#!/usr/bin/env bash
set -u

ROOT="/home/alex/Development"
STATUS="$ROOT/multi_finetune_evaluation_20k_1708_1_evaluation_status.tsv"
LOG="$ROOT/multi_finetune_evaluation_20k_1708_1.log"
REPORT="$ROOT/groot_multi_1708_1_report_1030.txt"
CSV="$ROOT/groot_multi_1708_1_completed_evaluations_1030.csv"
RAW="$(mktemp)"
trap 'rm -f "$RAW"' EXIT

if [[ -f "$STATUS" ]]; then
    while IFS=$'\t' read -r stage model status exit_code; do
        [[ "$stage" == "evaluation" && "$status" == "PASS" ]] || continue
        metrics="$model/evaluation_exec_hor_8/metrics_by_checkpoint.csv"
        [[ -f "$metrics" ]] || continue
        awk -F, -v model="$(basename "$model")" '
            $1 == "validation" && ($2 + 0) >= max_step {
                max_step = $2 + 0
                mae = $5
                mse = $6
            }
            END {
                if (max_step > 0) {
                    printf "%s,%d,%.15g,%.15g\n", model, max_step, mae, mse
                }
            }
        ' "$metrics" >> "$RAW"
    done < "$STATUS"
fi

{
    echo "model,checkpoint_step,mae,mae_score_best_100,mse,mse_score_best_100"
    if [[ -s "$RAW" ]]; then
        awk -F, '
            NR == 1 || $3 < best_mae { best_mae = $3 }
            NR == 1 || $4 < best_mse { best_mse = $4 }
            { rows[NR] = $0 }
            END {
                for (i = 1; i <= NR; i++) {
                    split(rows[i], row, ",")
                    printf "%s,%s,%.15g,%.2f,%.15g,%.2f\n", \
                        row[1], row[2], row[3], best_mae / row[3] * 100, \
                        row[4], best_mse / row[4] * 100
                }
            }
        ' "$RAW"
    fi
} > "$CSV"

{
    date '+timestamp=%Y-%m-%d %H:%M:%S %Z (%z)'
    echo
    echo "SERVICE"
    systemctl --user show groot-multi-1708-1.service \
        --property=ActiveState,SubState,MainPID,ExecMainStatus,ExecMainStartTimestamp || true
    echo
    echo "GPU"
    nvidia-smi \
        --query-gpu=timestamp,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
        --format=csv,noheader,nounits || true
    echo
    echo "RAM"
    free -h || true
    echo
    echo "CPU"
    mpstat 1 1 || true
    echo
    echo "PIPELINE STATUS"
    [[ -f "$STATUS" ]] && cat "$STATUS" || echo "status file missing"
    echo
    echo "COMPLETED EVALUATIONS CSV"
    cat "$CSV"
    echo
    echo "RECENT LOG"
    [[ -f "$LOG" ]] && tail -n 1000 "$LOG" || echo "log file missing"
} > "$REPORT" 2>&1
