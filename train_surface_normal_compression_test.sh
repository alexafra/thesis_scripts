#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/alex/Development/Isaac-GR00T"
DATASET_PATH="/home/alex/Development/Datasets/lerobot2/atomic_combined_09_08_And_10_08_testing_surface_normal_compression/train_plain_lz4"
BASE_MODEL_PATH="/home/alex/Development/Models/GR00T-N1.7-3B"
OUTPUT_DIR="/home/alex/Development/Models/dex3/c_normals_separate_batch_8_acc_4_chunk_32_400_surface_normal_compression_test"
MODALITY_CONFIG="examples/UnitreeG1/g1_dex3_head_3_channel_surface_normals_config.py"

if [[ ! -f "$DATASET_PATH/meta/info.json" ]]; then
    echo "Missing LZ4 training view: $DATASET_PATH" >&2
    exit 1
fi
if [[ -e "$OUTPUT_DIR" ]]; then
    echo "Refusing to overwrite existing output: $OUTPUT_DIR" >&2
    exit 1
fi

cd "$PROJECT_ROOT"

echo "Dataset:          $DATASET_PATH"
echo "Output:           $OUTPUT_DIR"
echo "Modality:         $MODALITY_CONFIG"
echo "Microbatch:       8"
echo "Accumulation:     4"
echo "Optimizer steps:  400"
echo "Normal storage:   plain LZ4, 32-frame independent chunks"

CUDA_VISIBLE_DEVICES=0 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
NO_ALBUMENTATIONS_UPDATE=1 \
uv run --no-sync python -m gr00t.experiment.launch_finetune \
    --base-model-path "$BASE_MODEL_PATH" \
    --dataset-path "$DATASET_PATH" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$MODALITY_CONFIG" \
    --no-tune-llm \
    --no-tune-visual \
    --tune-vision-patch-embed \
    --tune-projector \
    --num-gpus 1 \
    --output-dir "$OUTPUT_DIR" \
    --global-batch-size 8 \
    --gradient-accumulation-steps 4 \
    --dataloader-num-workers 4 \
    --episode-sampling-rate 0.1 \
    --optim adafactor \
    --learning-rate 1e-4 \
    --warmup-ratio 0.05 \
    --max-steps 400 \
    --save-steps 400 \
    --save-total-limit 8 \
    --skip-final-model-save \
    --color-jitter-params \
        brightness 0.20 \
        contrast 0.15 \
        saturation 0.10 \
        hue 0.0

uv run --no-sync python scripts/analysis_tools/plot_training_history.py \
    --run-dir "$OUTPUT_DIR" \
    --smooth-window 20

echo "Finished: $OUTPUT_DIR"
