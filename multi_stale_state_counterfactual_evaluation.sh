#!/usr/bin/env bash
set -euo pipefail

# Offline only: this script never imports the robot controller or opens DDS.
cd "$HOME/Development/Isaac-GR00T"

DATASET_ROOT="${DATASET_ROOT:-$HOME/Development/Datasets/lerobot2/atomic_combined_09_08_And_10_08_plus_pick_three_cups_right_only_1408_plus_stack_cups_09_08}"
VALIDATION_DATASET="${VALIDATION_DATASET:-$DATASET_ROOT/validation}"
TEST_DATASET="${TEST_DATASET:-$DATASET_ROOT/test}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-$HOME/Development/Models/GR00T-N1.7-3B}"
RUN_SUFFIX="${RUN_SUFFIX:-rightonly_1408_stack_0908_2026-09-09T021219+0800}"
ANALYSIS_SUFFIX="${ANALYSIS_SUFFIX:-$(date -u +%Y%m%dT%H%M%SZ)}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-8}"
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
EVENT_FILTER="${EVENT_FILTER:-all}"
DATASET_SCOPE="${DATASET_SCOPE:-validation}"
TASK="${TASK:-stack the three red cups.}"
MINIMUM_PLATEAU_FRAMES="${MINIMUM_PLATEAU_FRAMES:-25}"
STRONG_ACTION_RANGE="${STRONG_ACTION_RANGE:-0.1}"
NOISE_REPEATS="${NOISE_REPEATS:-1}"
DRY_RUN="${DRY_RUN:-0}"
LOG_ROOT="${LOG_ROOT:-$HOME/Development/logs/groot/evaluation}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$LOG_ROOT/matplotlib-cache}"
mkdir -p "$LOG_ROOT" "$MPLCONFIGDIR"

if [[ "$DRY_RUN" == "1" ]]; then
    CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-30000}"
    DECISION_OFFSETS="${DECISION_OFFSETS:-8}"
    MAX_EVENTS_PER_SPLIT="${MAX_EVENTS_PER_SPLIT:-1}"
    BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-50}"
    ANALYSIS_LABEL="dry"
elif [[ "$DRY_RUN" == "0" ]]; then
    # "all" evaluates every physical 5k checkpoint. RGB also includes the local base model.
    CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-all}"
    DECISION_OFFSETS="${DECISION_OFFSETS:-1 8 16}"
    MAX_EVENTS_PER_SPLIT="${MAX_EVENTS_PER_SPLIT:-0}"
    BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-1000}"
    ANALYSIS_LABEL="full"
else
    echo "Error: DRY_RUN must be 0 or 1." >&2
    exit 1
fi

MODEL_DIRS=(
    "$HOME/Development/Models/c_rgb_patch_tuned_bf16_batch_32_acc_1_30k_${RUN_SUFFIX}"
    "$HOME/Development/Models/c_d1_4ch_early_fusion_patch_tuned_depth_init_rgb_mean_bf16_batch_32_acc_1_30k_${RUN_SUFFIX}"
    "$HOME/Development/Models/c_normals_6ch_early_fusion_patch_tuned_normals_init_rgb_mean_bf16_batch_32_acc_1_30k_${RUN_SUFFIX}"
)
INCLUDE_BASE_MODEL=(1 0 0)

if [[ ${#MODEL_DIRS[@]} -ne ${#INCLUDE_BASE_MODEL[@]} ]]; then
    echo "Error: model arrays must have the same length." >&2
    exit 1
fi

case "$DATASET_SCOPE" in
    validation)
        DATASET_ARGS=(--dataset "validation=$VALIDATION_DATASET")
        DATASET_PATHS=("$VALIDATION_DATASET")
        ;;
    test)
        if [[ "$CHECKPOINT_STEPS" == "all" ]]; then
            echo "Error: test scope requires one preselected CHECKPOINT_STEPS value." >&2
            echo "Run validation first, then use DATASET_SCOPE=test CHECKPOINT_STEPS=30000." >&2
            exit 1
        fi
        read -r -a TEST_CHECKPOINTS <<< "$CHECKPOINT_STEPS"
        if [[ ${#TEST_CHECKPOINTS[@]} -ne 1 ]]; then
            echo "Error: test scope requires exactly one preselected checkpoint." >&2
            exit 1
        fi
        DATASET_ARGS=(--dataset "test=$TEST_DATASET")
        DATASET_PATHS=("$TEST_DATASET")
        ;;
    *)
        echo "Error: DATASET_SCOPE must be validation or test." >&2
        exit 1
        ;;
esac

for dataset in "${DATASET_PATHS[@]}"; do
    if [[ ! -f "$dataset/meta/info.json" ]]; then
        echo "Error: held-out dataset is not ready: $dataset/meta/info.json" >&2
        exit 1
    fi
    if [[ ! -f "$dataset/../provenance/merge_manifest.json" ]]; then
        echo "Error: merge provenance is missing: $dataset/../provenance/merge_manifest.json" >&2
        exit 1
    fi
done

if pgrep -af '[l]aunch_finetune.py|[g]r00t.experiment.launch_finetune|[t]orchrun.*gr00t' >/dev/null; then
    echo "Error: training is still active; refusing to compete for the GPU." >&2
    exit 1
fi

if [[ ! -d "$BASE_MODEL_PATH" ]]; then
    echo "Error: local base model is missing: $BASE_MODEL_PATH" >&2
    exit 1
fi

read -r -a DECISION_OFFSET_ARGS <<< "$DECISION_OFFSETS"
CHECKPOINT_ARGS=()
if [[ "$CHECKPOINT_STEPS" != "all" ]]; then
    read -r -a CHECKPOINT_STEP_ARGS <<< "$CHECKPOINT_STEPS"
    CHECKPOINT_ARGS=(--checkpoint-steps "${CHECKPOINT_STEP_ARGS[@]}")
fi

STATUS_FILE="${STATUS_FILE:-$LOG_ROOT/multi_stale_state_counterfactual_${ANALYSIS_LABEL}_${ANALYSIS_SUFFIX}.tsv}"
printf 'model\tstatus\texit_code\toutput_dir\n' > "$STATUS_FILE"
FAILURES=()

for i in "${!MODEL_DIRS[@]}"; do
    MODEL_DIR="${MODEL_DIRS[$i]}"
    if [[ ! -d "$MODEL_DIR" ]]; then
        echo "Error: trained model directory does not exist: $MODEL_DIR" >&2
        exit 1
    fi
    if [[ ! -f "$MODEL_DIR/checkpoint-30000/config.json" ]]; then
        echo "Error: final checkpoint is incomplete or missing: $MODEL_DIR/checkpoint-30000" >&2
        exit 1
    fi
    OUTPUT_DIR="$MODEL_DIR/stale_state_counterfactual_${DATASET_SCOPE}_exec_hor_${EXECUTION_HORIZON}_${ANALYSIS_SUFFIX}"
    if [[ -e "$OUTPUT_DIR" ]]; then
        echo "Error: analysis output already exists: $OUTPUT_DIR" >&2
        exit 1
    fi

    BASE_ARGS=()
    if [[ "${INCLUDE_BASE_MODEL[$i]}" == "1" ]]; then
        BASE_ARGS=(--base-model-path "$BASE_MODEL_PATH")
    fi

    echo "============================================================"
    echo "Offline stale-state comparison $((i + 1))/${#MODEL_DIRS[@]}"
    echo "Model:             $MODEL_DIR"
    echo "Dataset scope:     $DATASET_SCOPE"
    echo "Task:              $TASK"
    echo "Checkpoints:       $CHECKPOINT_STEPS"
    echo "Decision offsets:  $DECISION_OFFSETS frame(s)"
    echo "Execution horizon: $EXECUTION_HORIZON"
    echo "Event filter:      $EVENT_FILTER"
    echo "Output:             $OUTPUT_DIR"
    echo "============================================================"

    if CUDA_VISIBLE_DEVICES=0 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       NO_ALBUMENTATIONS_UPDATE=1 \
       MPLCONFIGDIR="$MPLCONFIGDIR" \
       HF_HUB_OFFLINE=1 \
       TRANSFORMERS_OFFLINE=1 \
       uv run --no-sync python \
           "$HOME/Development/scripts/stale_state_counterfactual_evaluation.py" \
           --run-dir "$MODEL_DIR" \
           "${BASE_ARGS[@]}" \
           "${CHECKPOINT_ARGS[@]}" \
           "${DATASET_ARGS[@]}" \
           --output-dir "$OUTPUT_DIR" \
           --embodiment-tag NEW_EMBODIMENT \
           --task "$TASK" \
           --execution-horizon "$EXECUTION_HORIZON" \
           --decision-offsets "${DECISION_OFFSET_ARGS[@]}" \
           --minimum-plateau-frames "$MINIMUM_PLATEAU_FRAMES" \
           --strong-action-range "$STRONG_ACTION_RANGE" \
           --event-filter "$EVENT_FILTER" \
           --inference-batch-size "$INFERENCE_BATCH_SIZE" \
           --denoising-steps 4 \
           --inference-seed 42 \
           --noise-repeats "$NOISE_REPEATS" \
           --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
           --max-events-per-split "$MAX_EVENTS_PER_SPLIT" \
           --device cuda:0; then
        printf '%s\tPASS\t0\t%s\n' "$MODEL_DIR" "$OUTPUT_DIR" >> "$STATUS_FILE"
    else
        exit_code=$?
        printf '%s\tFAIL\t%d\t%s\n' \
            "$MODEL_DIR" "$exit_code" "$OUTPUT_DIR" >> "$STATUS_FILE"
        FAILURES+=("$MODEL_DIR (exit $exit_code)")
        echo "WARNING: analysis failed for $MODEL_DIR; continuing." >&2
    fi
done

echo "Status: $STATUS_FILE"
if [[ ${#FAILURES[@]} -gt 0 ]]; then
    printf '  %s\n' "${FAILURES[@]}" >&2
    exit 1
fi
echo "All offline stale-state comparisons finished successfully."
