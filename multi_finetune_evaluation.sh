#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Development/Isaac-GR00T"

DATASET_ROOT="${DATASET_ROOT:-/home/alex/Development/Datasets/lerobot2/atomic_combined_09_08_And_10_08_plus_pick_three_cups_right_only_1408_plus_stack_cups_09_08}"
TRAIN_DATASET="${TRAIN_DATASET:-$DATASET_ROOT/train}"
VALIDATION_DATASET="${VALIDATION_DATASET:-$DATASET_ROOT/validation}"
BASE_MODEL_PATH="$HOME/Development/Models/GR00T-N1.7-3B"
EXECUTION_HORIZON=8
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
DRY_RUN="${DRY_RUN:-0}"
RUN_SUFFIX="${RUN_SUFFIX:-three_cups_rightonly_1408_stack_0908_$(date +%Y-%m-%dT%H%M%S%z)}"
LOG_ROOT="${LOG_ROOT:-$HOME/Development/logs/groot/training}"
mkdir -p "$LOG_ROOT"

if [[ "$DRY_RUN" == "1" ]]; then
    MAX_STEPS=1
    SAVE_STEPS=1
    RUN_LABEL="dry1"
    EVAL_STEPS=8
    EVAL_SELECTION_ARGS=(--traj-ids 0 --train-traj-ids 0 --trajectory-plot-episodes 0)
elif [[ "$DRY_RUN" == "0" ]]; then
    MAX_STEPS=30000
    SAVE_STEPS=5000
    RUN_LABEL="30k"
    EVAL_STEPS=0
    EVAL_SELECTION_ARGS=(--train-probe-episodes 3)
else
    echo "Error: DRY_RUN must be 0 or 1." >&2
    exit 1
fi

EVALUATION_STATUS_FILE="${EVALUATION_STATUS_FILE:-$LOG_ROOT/multi_finetune_evaluation_${RUN_LABEL}_${RUN_SUFFIX}_evaluation_status.tsv}"
printf 'stage\tmodel\tstatus\texit_code\n' > "$EVALUATION_STATUS_FILE"
TRAINING_FAILURES=()
EVALUATION_FAILURES=()

MODALITY_CONFIGS=(
    "examples/UnitreeG1/g1_dex3_headonly_config.py"
    "examples/UnitreeG1/g1_dex3_head_4_channel_gray_depth_fusion_config.py"
    "examples/UnitreeG1/g1_dex3_head_6_channel_surface_normals_fusion_config.py"
)

MODEL_DIRS=(
    "$HOME/Development/Models/c_rgb_patch_tuned_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}"
    "$HOME/Development/Models/c_d1_4ch_early_fusion_patch_tuned_depth_init_rgb_mean_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}"
    "$HOME/Development/Models/c_normals_6ch_early_fusion_patch_tuned_normals_init_rgb_mean_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}"
)

PATCH_EMBED_FLAGS=(
    "--tune-vision-patch-embed"
    "--tune-vision-patch-embed"
    "--tune-vision-patch-embed"
)

LOAD_BF16_FLAGS=(1 1 1)
BATCH_SIZES=(32 32 32)
ACCUMULATION_STEPS=(1 1 1)
PATCH_INIT_MODES=("" "rgb_mean" "rgb_mean")
INCLUDE_BASE_MODEL=(1 0 0)

if [[ ${#MODALITY_CONFIGS[@]} -ne ${#MODEL_DIRS[@]} ||
      ${#MODALITY_CONFIGS[@]} -ne ${#PATCH_EMBED_FLAGS[@]} ||
      ${#MODALITY_CONFIGS[@]} -ne ${#LOAD_BF16_FLAGS[@]} ||
      ${#MODALITY_CONFIGS[@]} -ne ${#BATCH_SIZES[@]} ||
      ${#MODALITY_CONFIGS[@]} -ne ${#ACCUMULATION_STEPS[@]} ||
      ${#MODALITY_CONFIGS[@]} -ne ${#PATCH_INIT_MODES[@]} ||
      ${#MODALITY_CONFIGS[@]} -ne ${#INCLUDE_BASE_MODEL[@]} ]]; then
    echo "Error: experiment arrays must have the same number of entries." >&2
    exit 1
fi

for dataset in "$TRAIN_DATASET" "$VALIDATION_DATASET"; do
    if [[ ! -f "$dataset/meta/info.json" ]]; then
        echo "Error: dataset is not ready: $dataset/meta/info.json is missing." >&2
        exit 1
    fi
    for feature in \
        observation.images.ego_view \
        observation.images.depth_gray_view \
        observation.images.surface_normals_view; do
        if ! grep -Fq "\"$feature\"" "$dataset/meta/info.json"; then
            echo "Error: $dataset is missing required feature $feature." >&2
            exit 1
        fi
    done
done

for path in "${MODALITY_CONFIGS[@]}"; do
    if [[ ! -f "$path" ]]; then
        echo "Error: modality config does not exist: $path" >&2
        exit 1
    fi
done

for path in "${MODEL_DIRS[@]}"; do
    if [[ -e "$path" ]]; then
        echo "Error: output already exists; choose a new RUN_SUFFIX or remove it deliberately: $path" >&2
        exit 1
    fi
done

uv run --no-sync python -m gr00t.data.stats \
    --dataset-path "$TRAIN_DATASET" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "${MODALITY_CONFIGS[0]}"

for i in "${!MODALITY_CONFIGS[@]}"; do
    MODALITY_CONFIG_PATH="${MODALITY_CONFIGS[$i]}"
    MODEL_DIR="${MODEL_DIRS[$i]}"
    PATCH_EMBED_FLAG="${PATCH_EMBED_FLAGS[$i]}"
    LOAD_BF16="${LOAD_BF16_FLAGS[$i]}"
    BATCH_SIZE="${BATCH_SIZES[$i]}"
    ACCUMULATION_STEP="${ACCUMULATION_STEPS[$i]}"
    PATCH_INIT_MODE="${PATCH_INIT_MODES[$i]}"
    PATCH_INIT_ARGS=()
    if [[ -n "$PATCH_INIT_MODE" ]]; then
        PATCH_INIT_ARGS=(--vision-patch-embed-init "$PATCH_INIT_MODE")
    fi
    BACKBONE_STORAGE_ARGS=()
    BACKBONE_STORAGE_LABEL="FP32"
    if [[ "$LOAD_BF16" == "1" ]]; then
        BACKBONE_STORAGE_ARGS=(--load-bf16)
        BACKBONE_STORAGE_LABEL="BF16"
    elif [[ "$LOAD_BF16" != "0" ]]; then
        echo "Error: LOAD_BF16_FLAGS entries must be 0 or 1, got: $LOAD_BF16" >&2
        exit 1
    fi

    echo "============================================================"
    echo "Training experiment $((i + 1))/${#MODALITY_CONFIGS[@]}"
    echo "Dataset:         $TRAIN_DATASET"
    echo "Base model:      $BASE_MODEL_PATH"
    echo "Modality config: $MODALITY_CONFIG_PATH"
    echo "Backbone storage: $BACKBONE_STORAGE_LABEL"
    echo "Patch embedding: $PATCH_EMBED_FLAG"
    echo "Patch init:      ${PATCH_INIT_MODE:-default}"
    echo "Batch:           $BATCH_SIZE x accumulation $ACCUMULATION_STEP"
    echo "Output:          $MODEL_DIR"
    echo "============================================================"

    if CUDA_VISIBLE_DEVICES=0 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       NO_ALBUMENTATIONS_UPDATE=1 \
       uv run --no-sync python -m gr00t.experiment.launch_finetune \
           --base-model-path "$BASE_MODEL_PATH" \
           --dataset-path "$TRAIN_DATASET" \
           --embodiment-tag NEW_EMBODIMENT \
           --modality-config-path "$MODALITY_CONFIG_PATH" \
           --no-tune-llm \
           --no-tune-visual \
           "$PATCH_EMBED_FLAG" \
           "${PATCH_INIT_ARGS[@]}" \
           --tune-projector \
           "${BACKBONE_STORAGE_ARGS[@]}" \
           --num-gpus 1 \
           --output-dir "$MODEL_DIR" \
           --global-batch-size "$BATCH_SIZE" \
           --gradient-accumulation-steps "$ACCUMULATION_STEP" \
           --dataloader-num-workers 4 \
           --episode-sampling-rate 0.1 \
           --optim adafactor \
           --learning-rate 1e-4 \
           --warmup-ratio 0.05 \
           --max-steps "$MAX_STEPS" \
           --save-steps "$SAVE_STEPS" \
           --save-total-limit 8 \
           --color-jitter-params \
               brightness 0.20 \
               contrast 0.15 \
               saturation 0.10 \
               hue 0.0; then
        printf 'training\t%s\tPASS\t0\n' "$MODEL_DIR" >> "$EVALUATION_STATUS_FILE"
    else
        training_exit_code=$?
        printf 'training\t%s\tFAIL\t%d\n' \
            "$MODEL_DIR" "$training_exit_code" >> "$EVALUATION_STATUS_FILE"
        printf 'evaluation\t%s\tSKIP\tNA\n' "$MODEL_DIR" >> "$EVALUATION_STATUS_FILE"
        TRAINING_FAILURES+=("$MODEL_DIR (exit $training_exit_code)")
        echo "WARNING: training failed for $MODEL_DIR; skipping its evaluation and continuing." >&2
        continue
    fi

    if ! uv run --no-sync python scripts/analysis_tools/plot_training_history.py \
        --run-dir "$MODEL_DIR" \
        --smooth-window 20; then
        echo "WARNING: training-history plotting failed for $MODEL_DIR; continuing." >&2
    fi

    echo "Finished training: $MODEL_DIR"

    BASE_MODEL_ARGS=()
    if [[ "${INCLUDE_BASE_MODEL[$i]}" == "1" ]]; then
        BASE_MODEL_ARGS=(--base-model-path "$BASE_MODEL_PATH")
    fi

    echo "============================================================"
    echo "Evaluating experiment $((i + 1))/${#MODEL_DIRS[@]}"
    echo "Model: $MODEL_DIR"
    echo "============================================================"

    if CUDA_VISIBLE_DEVICES=0 \
       NO_ALBUMENTATIONS_UPDATE=1 \
       uv run --no-sync python -m scripts.analysis_tools.evaluate_checkpoints \
           --run-dir "$MODEL_DIR" \
           "${BASE_MODEL_ARGS[@]}" \
           --dataset-path "$VALIDATION_DATASET" \
           --train-dataset-path "$TRAIN_DATASET" \
           "${EVAL_SELECTION_ARGS[@]}" \
           --output-dir "$MODEL_DIR/evaluation_exec_hor_${EXECUTION_HORIZON}" \
           --steps "$EVAL_STEPS" \
           --execution-horizon "$EXECUTION_HORIZON" \
           --inference-batch-size "$INFERENCE_BATCH_SIZE" \
           --denoising-steps 4 \
           --inference-seed 42 \
           --modality-keys left_arm right_arm left_hand right_hand \
           --train-probe-seed 42; then
        printf 'evaluation\t%s\tPASS\t0\n' "$MODEL_DIR" >> "$EVALUATION_STATUS_FILE"
        echo "Finished evaluation: $MODEL_DIR"
    else
        evaluation_exit_code=$?
        printf 'evaluation\t%s\tFAIL\t%d\n' \
            "$MODEL_DIR" "$evaluation_exit_code" >> "$EVALUATION_STATUS_FILE"
        EVALUATION_FAILURES+=("$MODEL_DIR (exit $evaluation_exit_code)")
        echo "WARNING: evaluation failed for $MODEL_DIR; continuing to the next training run." >&2
    fi
done

echo "Evaluation status: $EVALUATION_STATUS_FILE"
if [[ ${#TRAINING_FAILURES[@]} -gt 0 ]]; then
    echo "${#TRAINING_FAILURES[@]} training run(s) failed; all remaining experiments were still attempted:" >&2
    printf '  %s\n' "${TRAINING_FAILURES[@]}" >&2
fi
if [[ ${#EVALUATION_FAILURES[@]} -gt 0 ]]; then
    echo "${#EVALUATION_FAILURES[@]} evaluation(s) failed; all remaining experiments were still attempted:" >&2
    printf '  %s\n' "${EVALUATION_FAILURES[@]}" >&2
fi
if [[ ${#TRAINING_FAILURES[@]} -gt 0 || ${#EVALUATION_FAILURES[@]} -gt 0 ]]; then
    exit 1
fi

echo "All training and evaluation experiments finished successfully."
