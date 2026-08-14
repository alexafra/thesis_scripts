set -euo pipefail

cd "$HOME/Development/Isaac-GR00T"

TRAIN_DATASET="/home/alex/Development/Datasets/lerobot2/combined_09_08_1/train"
TRAIN_DIR="$HOME/Development/Models/combined_gray_scale_1_batch_32_acc_1_0908_1_test"

echo "$TRAIN_DATASET"
echo "$TRAIN_DIR"

uv run --no-sync python -m gr00t.data.stats \
    --dataset-path "$TRAIN_DATASET" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/UnitreeG1/g1_dex3_head_3_channel_gray_depth_config.py

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
NO_ALBUMENTATIONS_UPDATE=1 \
uv run --no-sync python -m gr00t.experiment.launch_finetune \
    --base-model-path "$HOME/Development/Models/GR00T-N1.7-3B" \
    --dataset-path "$TRAIN_DATASET" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/UnitreeG1/g1_dex3_head_3_channel_gray_depth_config.py \
    --num-gpus 1 \
    --output-dir "$TRAIN_DIR" \
    --global-batch-size 32 \
    --gradient-accumulation-steps 1 \
    --dataloader-num-workers 4 \
    --episode-sampling-rate 0.1 \
    --optim adafactor \
    --learning-rate 1e-4 \
    --max-steps 5000 \
    --save-steps 1000 \
    --save-total-limit 5 \
    --color-jitter-params \
        brightness 0.20 \
        contrast 0.15 \
        saturation 0.10 \
        hue 0.0

cd "$HOME/Development/Isaac-GR00T"

TRAIN_DATASET="/home/alex/Development/Datasets/lerobot2/combined_09_08_1/train"
TRAIN_DIR="$HOME/Development/Models/combined_colour_only_scale_1_batch_32_acc_1_0908_1"

echo "$TRAIN_DATASET"
echo "$TRAIN_DIR"

uv run --no-sync python -m gr00t.data.stats \
    --dataset-path "$TRAIN_DATASET" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/UnitreeG1/g1_dex3_headonly_config.py

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
NO_ALBUMENTATIONS_UPDATE=1 \
uv run --no-sync python -m gr00t.experiment.launch_finetune \
    --base-model-path "$HOME/Development/Models/GR00T-N1.7-3B" \
    --dataset-path "$TRAIN_DATASET" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/UnitreeG1/g1_dex3_headonly_config.py \
    --num-gpus 1 \
    --output-dir "$TRAIN_DIR" \
    --global-batch-size 32 \
    --gradient-accumulation-steps 1 \
    --dataloader-num-workers 4 \
    --episode-sampling-rate 0.1 \
    --optim adafactor \
    --learning-rate 1e-4 \
    --max-steps 5000 \
    --save-steps 1000 \
    --save-total-limit 5 \
    --color-jitter-params \
        brightness 0.20 \
        contrast 0.15 \
        saturation 0.10 \
        hue 0.0