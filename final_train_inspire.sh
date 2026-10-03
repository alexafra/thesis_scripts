#!/usr/bin/env bash
# Opt-in final fit: train + validation together, with no automatic evaluation.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MODE="${1:-}"
if [[ $# -ne 1 || ! "$MODE" =~ ^(prepare|check|run)$ ]]; then
    echo "Usage: FINAL_DATASET_SOURCES=/base:/new FINAL_VIEW_ROOT=/new/view [MAX_STEPS=35000] $0 prepare|check|run" >&2
    exit 2
fi
if [[ -z "${FINAL_DATASET_SOURCES:-}" || -z "${FINAL_VIEW_ROOT:-}" ]]; then
    echo "Error: set FINAL_DATASET_SOURCES (colon-separated split roots) and FINAL_VIEW_ROOT." >&2
    exit 1
fi
for selector in DATASET_SOURCES VIRTUAL_DATASET_ROOT DATASET_ROOT TRAIN_DATASET VALIDATION_DATASET TEST_DATASET; do
    if [[ -n "${!selector:-}" ]]; then
        echo "Error: $selector conflicts with final-training dataset selection." >&2
        exit 1
    fi
done
if [[ "$FINAL_DATASET_SOURCES" == :* || "$FINAL_DATASET_SOURCES" == *: || "$FINAL_DATASET_SOURCES" == *::* ]]; then
    echo "Error: FINAL_DATASET_SOURCES contains an empty path." >&2
    exit 1
fi
if [[ "$MODE" != "prepare" ]]; then
    if [[ ! "${MAX_STEPS:-}" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: explicitly choose a positive MAX_STEPS before check/run." >&2
        exit 1
    fi
    if [[ ! "${SAVE_STEPS:-5000}" =~ ^[1-9][0-9]*$ ]] || (( MAX_STEPS % ${SAVE_STEPS:-5000} != 0 )); then
        echo "Error: SAVE_STEPS must be positive and divide MAX_STEPS." >&2
        exit 1
    fi
fi
if [[ "${EVALUATION_SPLIT:-none}" != "none" || "${DRY_RUN:-0}" != "0" ]]; then
    echo "Error: final training requires EVALUATION_SPLIT=none and DRY_RUN=0." >&2
    exit 1
fi

GROOT_ROOT="${GROOT_ROOT:-/home/alex/Development/Isaac-GR00T}"
IFS=: read -r -a FINAL_SOURCES <<< "$FINAL_DATASET_SOURCES"
for index in "${!FINAL_SOURCES[@]}"; do
    FINAL_SOURCES[$index]="$(realpath -e -- "${FINAL_SOURCES[$index]}")"
done
FINAL_VIEW_ROOT="$(realpath -m -- "$FINAL_VIEW_ROOT")"
cd "$GROOT_ROOT"

# Check train/validation provenance before fold-in; never require or build test.
# Preserve physical-append ordering: all train episodes, then all validation episodes.
.venv/bin/python -m gr00t.data.virtual_dataset \
    --sources "${FINAL_SOURCES[@]}" \
    --output "$FINAL_VIEW_ROOT/combined" --reuse --splits train validation
.venv/bin/python -m gr00t.data.virtual_dataset \
    --sources "$FINAL_VIEW_ROOT/combined/train" "$FINAL_VIEW_ROOT/combined/validation" \
    --output "$FINAL_VIEW_ROOT/trainval" --reuse

echo "Final training data: $FINAL_VIEW_ROOT/trainval"
echo "Automatic evaluation: disabled (validation is folded into training; test is deferred)."
echo "Source media remain in their existing folders; keep them immutable and available."
if [[ "$MODE" == "prepare" ]]; then
    echo "Prepared only; no training/evaluation started. Choose MAX_STEPS before check/run."
    exit 0
fi

export DATASET_ROOT="$FINAL_VIEW_ROOT/combined"
export TRAIN_DATASET="$FINAL_VIEW_ROOT/trainval"
export EVALUATION_SPLIT=none
export EXPERIMENTS="${EXPERIMENTS:-rgb,rgbd_turbo_late_fusion_pre_adapter}"
export RUN_SUFFIX="${RUN_SUFFIX:-$(basename "$FINAL_VIEW_ROOT")_trainval_$(date -u +%Y%m%d)}"
export MAX_STEPS
if [[ "$MODE" == "check" ]]; then
    export PRECHECK_ONLY=1
else
    export PRECHECK_ONLY=0
fi
exec bash "$SCRIPT_DIR/multi_finetune_evaluation.sh"
