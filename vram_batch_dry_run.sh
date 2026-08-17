#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Development/Isaac-GR00T"

DATASET="/home/alex/Development/Datasets/lerobot2/atomic_combined_09_08_And_10_08/train"
BASE_MODEL_PATH="$HOME/Development/Models/GR00T-N1.7-3B"
MODALITY_CONFIG="examples/UnitreeG1/g1_dex3_headonly_config.py"
RESULTS_ROOT="${RESULTS_ROOT:-$HOME/Development/Models}"
TEST_CONFIGS="${TEST_CONFIGS:-adafactor:32:1:0:1:1 adafactor:32:1:0:0:1 adafactor:16:2:0:1:1 adafactor:16:2:0:0:1 adafactor:32:1:1:1:1 adafactor:32:1:1:0:1 adafactor:16:2:1:1:1 adafactor:16:2:1:0:1 adamw_torch:32:1:0:1:1 adamw_torch:32:1:0:0:1 adamw_torch:32:1:1:1:1 adamw_torch:32:1:1:0:1 adamw_torch:8:4:1:1:1 adamw_torch:32:1:1:1:0 adamw_torch:16:2:1:1:0 adamw_torch:8:4:1:1:0 adamw_torch:8:4:1:0:1}"
MAX_STEPS="${MAX_STEPS:-2}"
CASE_TIMEOUT="${CASE_TIMEOUT:-20m}"
LOAD_BF16="${LOAD_BF16:-1}"

case "$LOAD_BF16" in
    1)
        BACKBONE_STORAGE_ARGS=(--load-bf16)
        BACKBONE_STORAGE_LABEL="bf16"
        ;;
    0)
        BACKBONE_STORAGE_ARGS=()
        BACKBONE_STORAGE_LABEL="fp32"
        ;;
    *)
        echo "Error: LOAD_BF16 must be 0 or 1, got: $LOAD_BF16" >&2
        exit 1
        ;;
esac

for required_path in \
    "$DATASET/meta/info.json" \
    "$BASE_MODEL_PATH/config.json" \
    "$MODALITY_CONFIG"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Error: required path is missing: $required_path" >&2
        exit 1
    fi
done

if ! GPU_PROCESSES="$(nvidia-smi \
    --query-compute-apps=pid,process_name,used_gpu_memory \
    --format=csv,noheader,nounits)"; then
    echo "Error: could not query GPU processes with nvidia-smi; refusing to start." >&2
    exit 1
fi
if [[ -n "$GPU_PROCESSES" ]]; then
    echo "Refusing to start because the GPU is currently in use:" >&2
    echo "$GPU_PROCESSES" >&2
    exit 1
fi

SUMMARY="$RESULTS_ROOT/c_only_${BACKBONE_STORAGE_LABEL}_vram_test_summary.tsv"
if [[ -e "$SUMMARY" ]]; then
    echo "Error: test summary already exists; move or remove it deliberately: $SUMMARY" >&2
    exit 1
fi
printf 'optimizer\tbatch_size\tgradient_accumulation\teffective_batch\tgradient_checkpointing\tvision_patch_embedding\tmultimodal_projector\tstatus\tpeak_cuda_memory\tlog\n' > "$SUMMARY"

read -r -a TEST_CONFIG_LIST <<< "$TEST_CONFIGS"

for TEST_CONFIG in "${TEST_CONFIG_LIST[@]}"; do
    if [[ ! "$TEST_CONFIG" =~ ^([a-z0-9_]+):([1-9][0-9]*):([1-9][0-9]*):([01]):([01]):([01])$ ]]; then
        echo "Error: invalid test configuration: $TEST_CONFIG (expected OPTIMIZER:BATCH:ACCUMULATION:CHECKPOINTING:PATCH:PROJECTOR)" >&2
        exit 1
    fi
    OPTIMIZER="${BASH_REMATCH[1]}"
    BATCH_SIZE="${BASH_REMATCH[2]}"
    GRADIENT_ACCUMULATION="${BASH_REMATCH[3]}"
    GRADIENT_CHECKPOINTING="${BASH_REMATCH[4]}"
    TUNE_VISION_PATCH_EMBED="${BASH_REMATCH[5]}"
    TUNE_PROJECTOR="${BASH_REMATCH[6]}"
    EFFECTIVE_BATCH=$((BATCH_SIZE * GRADIENT_ACCUMULATION))

    CHECKPOINTING_ARGS=()
    CHECKPOINTING_LABEL="no_gc"
    if [[ "$GRADIENT_CHECKPOINTING" == "1" ]]; then
        CHECKPOINTING_ARGS=(--gradient-checkpointing)
        CHECKPOINTING_LABEL="gc"
    fi

    PATCH_ARGS=(--no-tune-vision-patch-embed)
    PATCH_LABEL="patch_frozen"
    if [[ "$TUNE_VISION_PATCH_EMBED" == "1" ]]; then
        PATCH_ARGS=(--tune-vision-patch-embed)
        PATCH_LABEL="patch_tuned"
    fi

    PROJECTOR_ARGS=(--no-tune-projector)
    PROJECTOR_LABEL="projector_frozen"
    if [[ "$TUNE_PROJECTOR" == "1" ]]; then
        PROJECTOR_ARGS=(--tune-projector)
        PROJECTOR_LABEL="projector_tuned"
    fi

    CASE_NAME="c_only_${PATCH_LABEL}_${PROJECTOR_LABEL}_${BACKBONE_STORAGE_LABEL}_${OPTIMIZER}_batch_${BATCH_SIZE}_acc_${GRADIENT_ACCUMULATION}_${CHECKPOINTING_LABEL}_test"
    OUTPUT_DIR="$RESULTS_ROOT/$CASE_NAME"
    LOG_FILE="$RESULTS_ROOT/$CASE_NAME.log"
    if [[ -e "$OUTPUT_DIR" || -e "$LOG_FILE" ]]; then
        echo "Error: test output already exists; move or remove it deliberately:" >&2
        echo "  $OUTPUT_DIR" >&2
        echo "  $LOG_FILE" >&2
        exit 1
    fi

    echo "============================================================"
    echo "VRAM dry run: batch $BATCH_SIZE x accumulation $GRADIENT_ACCUMULATION"
    echo "Optimizer:                      $OPTIMIZER"
    echo "Effective optimizer batch:      $EFFECTIVE_BATCH"
    echo "Train vision patch projection: $TUNE_VISION_PATCH_EMBED"
    echo "Train multimodal projector:     $TUNE_PROJECTOR"
    echo "Frozen backbone storage:        ${BACKBONE_STORAGE_LABEL^^}"
    echo "Gradient checkpointing:         $GRADIENT_CHECKPOINTING"
    echo "Output:                         $OUTPUT_DIR"
    echo "============================================================"

    set +e
    CUDA_VISIBLE_DEVICES=0 \
        PYTORCH_ALLOC_CONF=expandable_segments:True \
        NO_ALBUMENTATIONS_UPDATE=1 \
        timeout --signal=TERM --kill-after=30s "$CASE_TIMEOUT" \
        uv run --no-sync python -m gr00t.experiment.launch_finetune \
            --base-model-path "$BASE_MODEL_PATH" \
            --dataset-path "$DATASET" \
            --embodiment-tag NEW_EMBODIMENT \
            --modality-config-path "$MODALITY_CONFIG" \
            --no-tune-llm \
            --no-tune-visual \
            "${PATCH_ARGS[@]}" \
            "${PROJECTOR_ARGS[@]}" \
            "${BACKBONE_STORAGE_ARGS[@]}" \
            "${CHECKPOINTING_ARGS[@]}" \
            --num-gpus 1 \
            --output-dir "$OUTPUT_DIR" \
            --global-batch-size "$BATCH_SIZE" \
            --gradient-accumulation-steps "$GRADIENT_ACCUMULATION" \
            --dataloader-num-workers 4 \
            --episode-sampling-rate 0.1 \
            --optim "$OPTIMIZER" \
            --learning-rate 1e-4 \
            --max-steps "$MAX_STEPS" \
            --save-steps 1000 \
            --save-total-limit 1 \
            --skip-final-model-save \
            --color-jitter-params \
                brightness 0.20 \
                contrast 0.15 \
                saturation 0.10 \
                hue 0.0 \
            2>&1 | tee "$LOG_FILE"
    CASE_EXIT_CODE="${PIPESTATUS[0]}"
    set -e

    STATUS="PASS"
    if rg -q -i 'CUDA out of memory|OutOfMemoryError|CUBLAS_STATUS_ALLOC_FAILED' "$LOG_FILE"; then
        STATUS="OOM"
    elif [[ "$CASE_EXIT_CODE" -eq 124 || "$CASE_EXIT_CODE" -eq 137 ]]; then
        STATUS="TIMEOUT"
    elif [[ "$CASE_EXIT_CODE" -ne 0 ]]; then
        STATUS="FAIL"
    fi

    PEAK_MEMORY="$(rg 'Peak CUDA memory:' "$LOG_FILE" | tail -n 1 || true)"
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
        "$OPTIMIZER" \
        "$BATCH_SIZE" \
        "$GRADIENT_ACCUMULATION" \
        "$EFFECTIVE_BATCH" \
        "$GRADIENT_CHECKPOINTING" \
        "$TUNE_VISION_PATCH_EMBED" \
        "$TUNE_PROJECTOR" \
        "$STATUS" \
        "$PEAK_MEMORY" \
        "$LOG_FILE" >> "$SUMMARY"

    if [[ "$STATUS" != "PASS" ]]; then
        echo "Batch $BATCH_SIZE x accumulation $GRADIENT_ACCUMULATION ended with $STATUS. See $LOG_FILE" >&2
        echo "Continuing to the next isolated test configuration." >&2
    fi
done

echo "Dry-run summary: $SUMMARY"
