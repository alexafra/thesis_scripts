#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME/Development/Isaac-GR00T"
source "$SCRIPT_DIR/resolve_dataset_view.sh" || exit 1

TRAIN_DATASET="${TRAIN_DATASET:-/home/alex/Development/Datasets/lerobot2/atomic_combined_09_08_And_10_08/train}"
VALIDATION_DATASET="${VALIDATION_DATASET:-/home/alex/Development/Datasets/lerobot2/atomic_combined_09_08_And_10_08/validation}"
EXECUTION_HORIZON=8
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
RUN_SUFFIX="${RUN_SUFFIX:-1808_1_missed_normals}"
RUN_LABEL="${RUN_LABEL:-25k}"
EXPERIMENTS="${EXPERIMENTS:-missed_normals}"
MODALITY_DROPOUT="${MODALITY_DROPOUT:-}"
if [[ -z "$MODALITY_DROPOUT" ]]; then
    MODALITY_DROPOUT=1
    # This fixed historical selection predates modality dropout. Named recipes
    # otherwise follow the current training runner's enabled-by-default recipe.
    if [[ "${EXPERIMENTS:-missed_normals}" == "missed_normals" ]]; then
        MODALITY_DROPOUT=0
    fi
fi
MODALITY_DROPOUT_STATE_POLICY="${MODALITY_DROPOUT_STATE_POLICY:-independent}"
if [[ "$MODALITY_DROPOUT" != "0" && "$MODALITY_DROPOUT" != "1" ]]; then
    echo "Error: MODALITY_DROPOUT must be 0 or 1." >&2
    exit 1
fi
if [[ "$MODALITY_DROPOUT_STATE_POLICY" != "state_first" &&
      "$MODALITY_DROPOUT_STATE_POLICY" != "independent" ]]; then
    echo "Error: MODALITY_DROPOUT_STATE_POLICY must be state_first or independent." >&2
    exit 1
fi
MODALITY_DROPOUT_LABEL=""
if [[ "$MODALITY_DROPOUT" == "1" ]]; then
    MODALITY_DROPOUT_LABEL="_moddrop05each_${MODALITY_DROPOUT_STATE_POLICY}"
fi
LOG_ROOT="${LOG_ROOT:-$HOME/Development/logs/groot/training}"
mkdir -p "$LOG_ROOT"
STATUS_FILE="${EVALUATION_STATUS_FILE:-$LOG_ROOT/partial_multi_finetune_evaluation_${RUN_SUFFIX}${MODALITY_DROPOUT_LABEL}_evaluation_status.tsv}"

TRAIN_INFO="$TRAIN_DATASET/meta/info.json"
if [[ ! -f "$TRAIN_INFO" ]]; then
    echo "Error: dataset is not ready: $TRAIN_INFO is missing." >&2
    exit 1
fi
DATASET_ROBOT_TYPE="$(
    .venv/bin/python -c \
        'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["robot_type"])' \
        "$TRAIN_INFO"
)"
case "$DATASET_ROBOT_TYPE" in
    Unitree_G1_Inspire_HeadOnly)
        DEFAULT_MODEL_PREFIX="inspire_"
        DEFAULT_MODEL_ROOT="$HOME/Development/Models/inspire"
        ;;
    Unitree_G1_Dex3_HeadOnly)
        DEFAULT_MODEL_PREFIX=""
        DEFAULT_MODEL_ROOT="$HOME/Development/Models/dex3"
        ;;
    *)
        echo "Error: unsupported dataset robot_type: $DATASET_ROBOT_TYPE" >&2
        exit 1
        ;;
esac
MODEL_ROOT="${MODEL_ROOT:-$DEFAULT_MODEL_ROOT}"
MODEL_PREFIX="${MODEL_PREFIX:-$DEFAULT_MODEL_PREFIX}"
[[ "$MODEL_PREFIX" =~ ^[A-Za-z0-9._-]*$ ]] || {
    echo "Error: invalid MODEL_PREFIX: $MODEL_PREFIX" >&2
    exit 1
}

# The default preserves the original missed-normal evaluations. Opt-in late-fusion
# names match multi_finetune_evaluation.sh exactly, so this evaluator can resume a
# completed training run without a dedicated one-off launcher.
MODEL_DIRS=()
REQUIRED_CHECKPOINT_STEPS=()
SELECTED_EXPERIMENTS=()
declare -A SELECTED_EXPERIMENT_SET=()
IFS=',' read -r -a REQUESTED_EXPERIMENTS <<< "$EXPERIMENTS"
for experiment in "${REQUESTED_EXPERIMENTS[@]}"; do
    if [[ -n "${SELECTED_EXPERIMENT_SET[$experiment]:-}" ]]; then
        echo "Error: duplicate EXPERIMENTS entry: $experiment" >&2
        exit 1
    fi
    SELECTED_EXPERIMENT_SET[$experiment]=1
    SELECTED_EXPERIMENTS+=("$experiment")
    case "$experiment" in
        rgb)
            MODEL_DIRS+=("$MODEL_ROOT/${MODEL_PREFIX}c_rgb_patch_tuned_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}")
            REQUIRED_CHECKPOINT_STEPS+=("${FINAL_CHECKPOINT_STEP:-25000}")
            ;;
        depth)
            MODEL_DIRS+=("$MODEL_ROOT/${MODEL_PREFIX}c_d1_4ch_early_fusion_patch_tuned_depth_init_rgb_mean_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}")
            REQUIRED_CHECKPOINT_STEPS+=("${FINAL_CHECKPOINT_STEP:-25000}")
            ;;
        normals)
            MODEL_DIRS+=("$MODEL_ROOT/${MODEL_PREFIX}c_normals_6ch_early_fusion_patch_tuned_normals_init_rgb_mean_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}")
            REQUIRED_CHECKPOINT_STEPS+=("${FINAL_CHECKPOINT_STEP:-25000}")
            ;;
        missed_normals)
            if [[ "$MODALITY_DROPOUT" == "1" ]]; then
                echo "Error: missed_normals names fixed legacy runs without modality dropout; select normals or a named geometry recipe instead." >&2
                exit 1
            fi
            MODEL_DIRS+=(
                "$MODEL_ROOT/${MODEL_PREFIX}c_normals_6ch_early_fusion_patch_tuned_normals_init_rgb_mean_fp32_batch_8_acc_4_20k_1808_1"
                "$MODEL_ROOT/${MODEL_PREFIX}c_normals_6ch_early_fusion_patch_tuned_normals_init_rgb_mean_bf16_batch_32_acc_1_20k_1808_1"
            )
            REQUIRED_CHECKPOINT_STEPS+=(20000 20000)
            ;;
        rgbd_late_fusion_pre_adapter)
            MODEL_DIRS+=("$MODEL_ROOT/${MODEL_PREFIX}c_rgbd_late_fusion_pre_adapter_4x_linear_rgb50_geo50_patch_frozen_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}")
            REQUIRED_CHECKPOINT_STEPS+=("${FINAL_CHECKPOINT_STEP:-25000}")
            ;;
        rgbd_turbo_late_fusion_pre_adapter)
            MODEL_DIRS+=("$MODEL_ROOT/${MODEL_PREFIX}c_rgbd_turbo_late_fusion_pre_adapter_4x_linear_rgb50_geo50_patch_frozen_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}")
            REQUIRED_CHECKPOINT_STEPS+=("${FINAL_CHECKPOINT_STEP:-25000}")
            ;;
        rgbd_late_fusion_post_adapter)
            MODEL_DIRS+=("$MODEL_ROOT/${MODEL_PREFIX}c_rgbd_late_fusion_post_adapter_4x_linear_rgb50_geo50_patch_frozen_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}")
            REQUIRED_CHECKPOINT_STEPS+=("${FINAL_CHECKPOINT_STEP:-25000}")
            ;;
        normals_late_fusion_pre_adapter)
            MODEL_DIRS+=("$MODEL_ROOT/${MODEL_PREFIX}c_rgb_surface_normals_late_fusion_pre_adapter_4x_linear_rgb50_geo50_patch_frozen_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}")
            REQUIRED_CHECKPOINT_STEPS+=("${FINAL_CHECKPOINT_STEP:-25000}")
            ;;
        normals_late_fusion_post_adapter)
            MODEL_DIRS+=("$MODEL_ROOT/${MODEL_PREFIX}c_rgb_surface_normals_late_fusion_post_adapter_4x_linear_rgb50_geo50_patch_frozen_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}")
            REQUIRED_CHECKPOINT_STEPS+=("${FINAL_CHECKPOINT_STEP:-25000}")
            ;;
        rgbd_turbo_separate_views)
            MODEL_DIRS+=("$MODEL_ROOT/${MODEL_PREFIX}c_rgbd_turbo_separate_views_patch_frozen_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}")
            REQUIRED_CHECKPOINT_STEPS+=("${FINAL_CHECKPOINT_STEP:-25000}")
            ;;
        normals_separate_views)
            MODEL_DIRS+=("$MODEL_ROOT/${MODEL_PREFIX}c_rgb_surface_normals_separate_views_patch_frozen_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}")
            REQUIRED_CHECKPOINT_STEPS+=("${FINAL_CHECKPOINT_STEP:-25000}")
            ;;
        rgbd_gray_separate_views)
            MODEL_DIRS+=("$MODEL_ROOT/${MODEL_PREFIX}c_rgbd_gray_separate_views_patch_frozen_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}")
            REQUIRED_CHECKPOINT_STEPS+=("${FINAL_CHECKPOINT_STEP:-25000}")
            ;;
        *)
            echo "Error: unsupported EXPERIMENTS entry: $experiment" >&2
            echo "Supported: missed_normals rgb depth normals rgbd_late_fusion_pre_adapter rgbd_turbo_late_fusion_pre_adapter rgbd_late_fusion_post_adapter normals_late_fusion_pre_adapter normals_late_fusion_post_adapter rgbd_turbo_separate_views normals_separate_views rgbd_gray_separate_views" >&2
            exit 1
            ;;
    esac
    # Directory selection only. The evaluator itself always uses eval mode and
    # receives no training-dropout overrides; RGB-only paths remain unchanged.
    if [[ "$MODALITY_DROPOUT" == "1" && "$experiment" != "rgb" ]]; then
        last_model_index=$((${#MODEL_DIRS[@]} - 1))
        MODEL_DIRS[$last_model_index]+="$MODALITY_DROPOUT_LABEL"
    fi
done
if [[ ${#MODEL_DIRS[@]} -eq 0 ]]; then
    echo "Error: EXPERIMENTS must select at least one experiment." >&2
    exit 1
fi
if [[ ${#MODEL_DIRS[@]} -ne ${#REQUIRED_CHECKPOINT_STEPS[@]} ]]; then
    echo "Error: partial-evaluation arrays must have the same number of entries." >&2
    exit 1
fi
if [[ "$DATASET_ROBOT_TYPE" != "Unitree_G1_Inspire_HeadOnly" ]]; then
    for experiment in "${SELECTED_EXPERIMENTS[@]}"; do
        if [[ "$experiment" == *_late_fusion_* || "$experiment" == "rgbd_turbo_separate_views" ]]; then
            echo "Error: $experiment currently has an Inspire-only modality config." >&2
            exit 1
        fi
    done
fi

echo "Training recipe directory suffix: ${MODALITY_DROPOUT_LABEL:-none}; evaluation modality dropout is disabled."
printf 'stage\tmodel\tstatus\texit_code\n' > "$STATUS_FILE"
FAILURES=()

for i in "${!MODEL_DIRS[@]}"; do
    model_dir="${MODEL_DIRS[$i]}"
    required_checkpoint_step="${REQUIRED_CHECKPOINT_STEPS[$i]}"
    if [[ ! -f "$model_dir/checkpoint-$required_checkpoint_step/trainer_state.json" ]]; then
        printf 'evaluation\t%s\tSKIP\tNA\n' "$model_dir" >> "$STATUS_FILE"
        FAILURES+=("$model_dir (missing checkpoint-$required_checkpoint_step)")
        echo "WARNING: completed checkpoint-$required_checkpoint_step is missing; skipping $model_dir" >&2
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
           --train-probe-seed 42 &&
       # Consume the complete saved frame-level predictions without rewriting them.
       uv run --no-sync python -m scripts.analysis_tools.normalized_action_metrics \
           --run-dir "$model_dir" \
           --evaluation-dir "$model_dir/evaluation_exec_hor_${EXECUTION_HORIZON}" \
           --output-dir "$model_dir/normalized_action_metrics_exec_hor_${EXECUTION_HORIZON}" \
           --statistics-path "$model_dir/experiment_cfg/dataset_statistics.json" \
           --embodiment-tag NEW_EMBODIMENT \
           --execution-horizon "$EXECUTION_HORIZON"; then
        printf 'evaluation\t%s\tPASS\t0\n' "$model_dir" >> "$STATUS_FILE"
        echo "Finished evaluation and normalized action metrics: $model_dir"
    else
        evaluation_exit_code=$?
        printf 'evaluation\t%s\tFAIL\t%d\n' \
            "$model_dir" "$evaluation_exit_code" >> "$STATUS_FILE"
        FAILURES+=("$model_dir (exit $evaluation_exit_code)")
        echo "WARNING: evaluation or normalized metrics failed for $model_dir; continuing." >&2
    fi
done

echo "Evaluation status: $STATUS_FILE"
if [[ ${#FAILURES[@]} -gt 0 ]]; then
    echo "${#FAILURES[@]} missed evaluation(s) did not complete:" >&2
    printf '  %s\n' "${FAILURES[@]}" >&2
    exit 1
fi

echo "All selected partial evaluations finished successfully."
