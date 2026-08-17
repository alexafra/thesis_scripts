#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Development/Isaac-GR00T"

TRAIN_DATASET="/home/alex/Development/Datasets/lerobot2/atomic_combined_09_08_And_10_08/train"
VALIDATION_DATASET="/home/alex/Development/Datasets/lerobot2/atomic_combined_09_08_And_10_08/validation"
BASE_MODEL_PATH="$HOME/Development/Models/GR00T-N1.7-3B"
EXECUTION_HORIZON=8
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"

MODALITY_CONFIGS=(
    # "examples/UnitreeG1/g1_dex3_head_3_channel_gray_depth_config.py"
    "examples/UnitreeG1/g1_dex3_headonly_config.py"
)

MODEL_DIRS=(
    # "$HOME/Development/Models/c_d1_batch_32_acc_1_1008_1"
    "$HOME/Development/Models/c_only_batch_32_acc_1_chunk_16_1008"
)

if [[ ${#MODALITY_CONFIGS[@]} -ne ${#MODEL_DIRS[@]} ]]; then
    echo "Error: MODALITY_CONFIGS and MODEL_DIRS must have the same number of entries." >&2
    exit 1
fi

for i in "${!MODALITY_CONFIGS[@]}"; do
    MODALITY_CONFIG_PATH="${MODALITY_CONFIGS[$i]}"
    MODEL_DIR="${MODEL_DIRS[$i]}"

    echo "============================================================"
    echo "Starting experiment $((i + 1))/${#MODALITY_CONFIGS[@]}"
    echo "Dataset:        $TRAIN_DATASET"
    echo "Base model:     $BASE_MODEL_PATH"
    echo "Modality config: $MODALITY_CONFIG_PATH"
    echo "Output:         $MODEL_DIR"
    echo "============================================================"

    CUDA_VISIBLE_DEVICES=0 \
    NO_ALBUMENTATIONS_UPDATE=1 \
    uv run --no-sync python -m scripts.analysis_tools.evaluate_checkpoints \
        --run-dir "$MODEL_DIR" \
        --base-model-path "$BASE_MODEL_PATH" \
        --dataset-path "$VALIDATION_DATASET" \
        --train-dataset-path "$TRAIN_DATASET" \
        --output-dir "$MODEL_DIR/evaluation_exec_hor_${EXECUTION_HORIZON}" \
        --steps 0 \
        --execution-horizon "$EXECUTION_HORIZON" \
        --inference-batch-size "$INFERENCE_BATCH_SIZE" \
        --denoising-steps 4 \
        --inference-seed 42 \
        --modality-keys left_arm right_arm left_hand right_hand \
        --train-probe-episodes 3 \
        --train-probe-seed 42 




    echo "Finished experiment: $MODEL_DIR"
done

echo "All training experiments finished."
