#!/usr/bin/env bash
set -euo pipefail

CURRENT_UNIT="groot-partial-multi-1608-1.service"
PARTIAL_SCRIPT="/home/alex/Development/scripts/partial_multi_finetune_evaluation.sh"
LOG_ROOT="/home/alex/Development/logs/groot/training"
mkdir -p "$LOG_ROOT"
LOG_FILE="$LOG_ROOT/partial_multi_finetune_evaluation_20k_1608_1.log"
STATUS_FILE="$LOG_ROOT/partial_multi_finetune_evaluation_20k_1608_1_status.tsv"
POLL_SECONDS=20

exec >> "$LOG_FILE" 2>&1

echo "[$(date --iso-8601=seconds)] Queued behind $CURRENT_UNIT"
while systemctl --user is-active --quiet "$CURRENT_UNIT"; do
    sleep "$POLL_SECONDS"
done

echo "[$(date --iso-8601=seconds)] $CURRENT_UNIT is inactive; waiting for a free GPU"
while true; do
    gpu_processes="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
    if [[ -z "${gpu_processes//[[:space:]]/}" ]]; then
        break
    fi
    sleep "$POLL_SECONDS"
done

echo "[$(date --iso-8601=seconds)] GPU is free; starting partial training and evaluations"
exec env \
    STATUS_FILE="$STATUS_FILE" \
    RUN_SUFFIX="1608_1" \
    REEVAL_TAG="current_evaluator_1608_1" \
    "$PARTIAL_SCRIPT"
