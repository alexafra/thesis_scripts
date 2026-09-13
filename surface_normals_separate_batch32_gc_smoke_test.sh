#!/usr/bin/env bash
set -uo pipefail

cd /home/alex/Development/Isaac-GR00T

OUTPUT=/home/alex/Development/Models/c_normals_separate_patch_tuned_bf16_batch_32_acc_1_chunk_32_gc_1608_1_test
LOG_ROOT=/home/alex/Development/logs/groot/smoke
mkdir -p "$LOG_ROOT"
LOG=$LOG_ROOT/surface_normals_separate_batch32_gc_smoke_1608_1.log
STATUS=$LOG_ROOT/surface_normals_separate_batch32_gc_smoke_1608_1_status.tsv

exec >>"$LOG" 2>&1
printf 'model\tstatus\texit_code\n' > "$STATUS"
echo "[$(date --iso-8601=seconds)] Starting separate-normal batch-32 checkpointing smoke test"

if CUDA_VISIBLE_DEVICES=0 \
   PYTORCH_ALLOC_CONF=expandable_segments:True \
   NO_ALBUMENTATIONS_UPDATE=1 \
   uv run --no-sync python -m gr00t.experiment.launch_finetune \
       --base-model-path /home/alex/Development/Models/GR00T-N1.7-3B \
       --dataset-path /home/alex/Development/Datasets/lerobot2/atomic_combined_09_08_And_10_08/train \
       --embodiment-tag NEW_EMBODIMENT \
       --modality-config-path examples/UnitreeG1/g1_dex3_head_3_channel_surface_normals_config.py \
       --no-tune-llm \
       --no-tune-visual \
       --tune-vision-patch-embed \
       --tune-projector \
       --load-bf16 \
       --gradient-checkpointing \
       --num-gpus 1 \
       --output-dir "$OUTPUT" \
       --global-batch-size 32 \
       --gradient-accumulation-steps 1 \
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
    printf '%s\tPASS\t0\n' "$OUTPUT" >> "$STATUS"
    echo "[$(date --iso-8601=seconds)] PASS"
    exit 0
else
    exit_code=$?
    printf '%s\tFAIL\t%d\n' "$OUTPUT" "$exit_code" >> "$STATUS"
    echo "[$(date --iso-8601=seconds)] FAIL ($exit_code)"
    exit "$exit_code"
fi
