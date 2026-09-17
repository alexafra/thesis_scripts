#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Development/Isaac-GR00T"

TRAIN_DATASET="/home/alex/Development/Datasets/lerobot2/atomic_combined_09_08_And_10_08/train"
VALIDATION_DATASET="/home/alex/Development/Datasets/lerobot2/atomic_combined_09_08_And_10_08/validation"
BASE_MODEL_PATH="$HOME/Development/Models/GR00T-N1.7-3B"

MODALITY_CONFIGS=(
    "examples/UnitreeG1/g1_dex3_head_3_channel_gray_depth_config.py"
    "examples/UnitreeG1/g1_dex3_headonly_config.py"
)

MODEL_DIRS=(
    "$HOME/Development/Models/dex3/c_d1_batch_32_acc_1_1008_1_test"
    "$HOME/Development/Models/dex3/c_only_batch_32_acc_1_1008_1_test"
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

    uv run --no-sync python -m gr00t.data.stats \
        --dataset-path "$TRAIN_DATASET" \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path "$MODALITY_CONFIG_PATH"

    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    NO_ALBUMENTATIONS_UPDATE=1 \
    uv run --no-sync python -m gr00t.experiment.launch_finetune \
        --base-model-path "$BASE_MODEL_PATH" \
        --dataset-path "$TRAIN_DATASET" \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path "$MODALITY_CONFIG_PATH" \
        --tune_visual \
        --num-gpus 1 \
        --output-dir "$MODEL_DIR" \
        --global-batch-size 4 \
        --gradient-accumulation-steps 8 \
        --dataloader-num-workers 4 \
        --episode-sampling-rate 0.1 \
        --optim adafactor \
        --learning-rate 1e-4 \
        --max-steps 50 \
        --save-steps 25 \
        --save-total-limit 2 \
        --skip-final-model-save \
        --color-jitter-params \
            brightness 0.20 \
            contrast 0.15 \
            saturation 0.10 \
            hue 0.0





    echo "Finished experiment: $MODEL_DIR"
done

echo "All training experiments finished."
