#!/usr/bin/env bash
set -uo pipefail

cd "$HOME/Development/Isaac-GR00T"

TRAIN_DATASET="/home/alex/Development/Datasets/lerobot2/atomic_combined_09_08_And_10_08/train"
VALIDATION_DATASET="/home/alex/Development/Datasets/lerobot2/atomic_combined_09_08_And_10_08/validation"
EXECUTION_HORIZON=8
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
RUN_SUFFIX="${RUN_SUFFIX:-1808_1_missed_normals}"
STATUS_FILE="${EVALUATION_STATUS_FILE:-$HOME/Development/partial_multi_finetune_evaluation_${RUN_SUFFIX}_evaluation_status.tsv}"

# Both models completed training in the main 1808_1 pipeline. Their first
# evaluations stopped before inference because the policy contract rejected
# the supported six-channel rgb_mean initialization.
MODEL_DIRS=(
    "$HOME/Development/Models/c_normals_6ch_early_fusion_patch_tuned_normals_init_rgb_mean_fp32_batch_8_acc_4_20k_1808_1"
    "$HOME/Development/Models/c_normals_6ch_early_fusion_patch_tuned_normals_init_rgb_mean_bf16_batch_32_acc_1_20k_1808_1"
)

printf 'stage\tmodel\tstatus\texit_code\n' > "$STATUS_FILE"
FAILURES=()

for model_dir in "${MODEL_DIRS[@]}"; do
    if [[ ! -f "$model_dir/checkpoint-20000/trainer_state.json" ]]; then
        printf 'evaluation\t%s\tSKIP\tNA\n' "$model_dir" >> "$STATUS_FILE"
        FAILURES+=("$model_dir (missing checkpoint-20000)")
        echo "WARNING: completed checkpoint is missing; skipping $model_dir" >&2
        continue
    fi

    echo "============================================================"
    echo "Evaluating missed model: $model_dir"
    echo "============================================================"

    if CUDA_VISIBLE_DEVICES=0 \
       NO_ALBUMENTATIONS_UPDATE=1 \
       uv run --no-sync python -m scripts.analysis_tools.evaluate_checkpoints \
           --run-dir "$model_dir" \
           --dataset-path "$VALIDATION_DATASET" \
           --train-dataset-path "$TRAIN_DATASET" \
           --train-probe-episodes 3 \
           --output-dir "$model_dir/evaluation_exec_hor_${EXECUTION_HORIZON}" \
           --steps 0 \
           --execution-horizon "$EXECUTION_HORIZON" \
           --inference-batch-size "$INFERENCE_BATCH_SIZE" \
           --denoising-steps 4 \
           --inference-seed 42 \
           --modality-keys left_arm right_arm left_hand right_hand \
           --train-probe-seed 42; then
        printf 'evaluation\t%s\tPASS\t0\n' "$model_dir" >> "$STATUS_FILE"
        echo "Finished evaluation: $model_dir"
    else
        evaluation_exit_code=$?
        printf 'evaluation\t%s\tFAIL\t%d\n' \
            "$model_dir" "$evaluation_exit_code" >> "$STATUS_FILE"
        FAILURES+=("$model_dir (exit $evaluation_exit_code)")
        echo "WARNING: evaluation failed for $model_dir; continuing." >&2
    fi
done

echo "Evaluation status: $STATUS_FILE"
if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo "${#FAILURES[@]} missed evaluation(s) did not complete:" >&2
    printf '  %s\n' "${FAILURES[@]}" >&2
    exit 1
fi

echo "Both missed normals evaluations finished successfully."
