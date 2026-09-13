#!/usr/bin/env bash
set -uo pipefail

cd /home/alex/Development/Isaac-GR00T

LOG_ROOT=/home/alex/Development/logs/groot/smoke
mkdir -p "$LOG_ROOT"
LOG=$LOG_ROOT/surface_normals_separate_patch_frozen_smoke_1608_1.log
STATUS=$LOG_ROOT/surface_normals_separate_patch_frozen_smoke_1608_1_status.tsv
CONFIG=examples/UnitreeG1/g1_dex3_head_3_channel_surface_normals_config.py

OUTPUTS=(
    /home/alex/Development/Models/c_normals_separate_patch_frozen_bf16_batch_32_acc_1_chunk_32_1608_1_test
    /home/alex/Development/Models/c_normals_separate_patch_frozen_bf16_batch_16_acc_2_chunk_32_1608_1_test
)
BATCH_SIZES=(32 16)
ACCUMULATION_STEPS=(1 2)

exec >>"$LOG" 2>&1
printf 'model\tbatch_size\tgradient_accumulation\tstatus\texit_code\n' > "$STATUS"
echo "[$(date --iso-8601=seconds)] Starting patch-frozen separate-normal smoke tests"

failures=0
for i in "${!OUTPUTS[@]}"; do
    output="${OUTPUTS[$i]}"
    batch_size="${BATCH_SIZES[$i]}"
    accumulation="${ACCUMULATION_STEPS[$i]}"

    echo "[$(date --iso-8601=seconds)] Starting batch $batch_size x accumulation $accumulation: $output"
    if CUDA_VISIBLE_DEVICES=0 \
       PYTORCH_ALLOC_CONF=expandable_segments:True \
       NO_ALBUMENTATIONS_UPDATE=1 \
       uv run --no-sync python -m gr00t.experiment.launch_finetune \
           --base-model-path /home/alex/Development/Models/GR00T-N1.7-3B \
           --dataset-path /home/alex/Development/Datasets/lerobot2/atomic_combined_09_08_And_10_08/train \
           --embodiment-tag NEW_EMBODIMENT \
           --modality-config-path "$CONFIG" \
           --no-tune-llm \
           --no-tune-visual \
           --no-tune-vision-patch-embed \
           --tune-projector \
           --load-bf16 \
           --num-gpus 1 \
           --output-dir "$output" \
           --global-batch-size "$batch_size" \
           --gradient-accumulation-steps "$accumulation" \
           --dataloader-num-workers 4 \
           --episode-sampling-rate 0.1 \
           --optim adafactor \
           --learning-rate 1e-4 \
           --warmup-ratio 0.05 \
           --max-steps 1 \
           --save-steps 1 \
           --save-total-limit 1 \
           --color-jitter-params \
               brightness 0.20 \
               contrast 0.15 \
               saturation 0.10 \
               hue 0.0; then
        printf '%s\t%s\t%s\tPASS\t0\n' \
            "$output" "$batch_size" "$accumulation" >> "$STATUS"
        echo "[$(date --iso-8601=seconds)] PASS: $output"
    else
        exit_code=$?
        printf '%s\t%s\t%s\tFAIL\t%d\n' \
            "$output" "$batch_size" "$accumulation" "$exit_code" >> "$STATUS"
        echo "[$(date --iso-8601=seconds)] FAIL ($exit_code): $output"
        failures=$((failures + 1))
    fi
done

echo "[$(date --iso-8601=seconds)] COMPLETE: $failures failure(s)"
exit "$failures"
