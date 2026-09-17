#!/usr/bin/env bash
set -euo pipefail

cd "$HOME/Development/Isaac-GR00T"

TRAIN_DATASET="/home/alex/Development/Datasets/lerobot2/combined_09_08_1/train"

MODALITY_CONFIGS=(
    "examples/UnitreeG1/g1_dex3_head_3_channel_gray_depth_config.py"
    "examples/UnitreeG1/g1_dex3_headonly_config.py"
)

TRAIN_DIRS=(
    "$HOME/Development/Models/dex3/combined_gray_depth_batch_32_acc_1_0908_1_test"
    "$HOME/Development/Models/dex3/combined_colour_only_batch_32_acc_1_0908_1_test"
)

if [[ ${#MODALITY_CONFIGS[@]} -ne ${#TRAIN_DIRS[@]} ]]; then
    echo "Error: MODALITY_CONFIGS and TRAIN_DIRS must have the same number of entries." >&2
    exit 1
fi

for i in "${!MODALITY_CONFIGS[@]}"; do
    MODALITY_CONFIG_PATH="${MODALITY_CONFIGS[$i]}"
    TRAIN_DIR="${TRAIN_DIRS[$i]}"

    echo "============================================================"
    echo "Starting experiment $((i + 1))/${#MODALITY_CONFIGS[@]}"
    echo "Dataset:        $TRAIN_DATASET"
    echo "Modality config: $MODALITY_CONFIG_PATH"
    echo "Output:         $TRAIN_DIR"
    echo "============================================================"

    uv run --no-sync python -m gr00t.data.stats \
        --dataset-path "$TRAIN_DATASET" \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path "$MODALITY_CONFIG_PATH"

    CUDA_VISIBLE_DEVICES=0 \
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    NO_ALBUMENTATIONS_UPDATE=1 \
    uv run --no-sync python -m gr00t.experiment.launch_finetune \
        --base-model-path "$HOME/Development/Models/GR00T-N1.7-3B" \
        --dataset-path "$TRAIN_DATASET" \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path "$MODALITY_CONFIG_PATH" \
        --num-gpus 1 \
        --output-dir "$TRAIN_DIR" \
        --global-batch-size 32 \
        --gradient-accumulation-steps 1 \
        --dataloader-num-workers 4 \
        --episode-sampling-rate 0.1 \
        --optim adafactor \
        --learning-rate 1e-4 \
        --max-steps 5 \
        --save-steps 5 \
        --save-total-limit 5 \
        --skip-final-model-save \
        --color-jitter-params \
            brightness 0.20 \
            contrast 0.15 \
            saturation 0.10 \
            hue 0.0

    echo "Finished experiment: $TRAIN_DIR"
done

echo "All training experiments finished."
