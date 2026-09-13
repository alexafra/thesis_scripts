#!/usr/bin/env bash
set -uo pipefail

cd /home/alex/Development/Isaac-GR00T

DATASET=/home/alex/Development/Datasets/lerobot2/atomic_combined_09_08_And_10_08/train
BASE_MODEL=/home/alex/Development/Models/GR00T-N1.7-3B
LOG_ROOT=/home/alex/Development/logs/groot/smoke
mkdir -p "$LOG_ROOT"
LOG=$LOG_ROOT/surface_normals_6ch_fusion_bf16_batch32_zero_smoke_1708_2.log
STATUS=$LOG_ROOT/surface_normals_6ch_fusion_bf16_batch32_zero_smoke_1708_2_status.tsv

exec >>"$LOG" 2>&1

printf 'model\tstatus\texit_code\n' > "$STATUS"
echo "[$(date --iso-8601=seconds)] Starting zero-initialized 6-channel surface-normal fusion BF16 batch-32 smoke test"

CONFIGS=(
    examples/UnitreeG1/g1_dex3_head_6_channel_surface_normals_fusion_config.py
)

OUTPUTS=(
    /home/alex/Development/Models/c_normals_6ch_early_fusion_patch_tuned_normals_init_zero_bf16_batch_32_acc_1_smoke_1708_2_test
)

if ! uv run --no-sync python -m gr00t.data.stats \
    --dataset-path "$DATASET" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path examples/UnitreeG1/g1_dex3_headonly_config.py; then
    echo "[$(date --iso-8601=seconds)] Dataset statistics check failed"
    exit 2
fi

failures=0
for i in "${!CONFIGS[@]}"; do
    config="${CONFIGS[$i]}"
    output="${OUTPUTS[$i]}"

    echo "[$(date --iso-8601=seconds)] Starting: $output"
    if CUDA_VISIBLE_DEVICES=0 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       NO_ALBUMENTATIONS_UPDATE=1 \
       uv run --no-sync python -m gr00t.experiment.launch_finetune \
           --base-model-path "$BASE_MODEL" \
           --dataset-path "$DATASET" \
           --embodiment-tag NEW_EMBODIMENT \
           --modality-config-path "$config" \
           --no-tune-llm \
           --no-tune-visual \
           --tune-vision-patch-embed \
           --tune-projector \
           --load-bf16 \
           --num-gpus 1 \
           --output-dir "$output" \
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
        printf '%s\tPASS\t0\n' "$output" >> "$STATUS"
        echo "[$(date --iso-8601=seconds)] PASS: $output"
    else
        exit_code=$?
        printf '%s\tFAIL\t%d\n' "$output" "$exit_code" >> "$STATUS"
        echo "[$(date --iso-8601=seconds)] FAIL ($exit_code): $output"
        failures=$((failures + 1))
    fi
done

echo "[$(date --iso-8601=seconds)] COMPLETE: $failures failure(s)"
exit "$failures"
