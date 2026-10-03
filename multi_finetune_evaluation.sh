#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$HOME/Development/Isaac-GR00T"
# Test-set evaluation is deliberately disabled without a separate explicit opt-in.
# Check before any dataset-view preparation so a deferred test split is not read.
EVALUATION_SPLIT="${EVALUATION_SPLIT:-validation}"
case "$EVALUATION_SPLIT" in
    validation|none) ;;
    test)
        if [[ "${ALLOW_TEST_EVALUATION:-0}" != "1" ]]; then
            echo "Error: test evaluation is deferred; explicitly set ALLOW_TEST_EVALUATION=1 only when ready." >&2
            exit 1
        fi
        ;;
    *)
        echo "Error: EVALUATION_SPLIT must be validation, none, or explicitly enabled test." >&2
        exit 1
        ;;
esac
source "$SCRIPT_DIR/resolve_dataset_view.sh" || exit 1

DATASETS_ROOT="${DATASETS_ROOT:-/home/alex/Development/Datasets}"
DEFAULT_DEX3_DATASET_NAME="atomic_combined_09_08_And_10_08_plus_pick_three_cups_right_only_1408_plus_stack_cups_09_08"
NESTED_DEX3_DATASET="$DATASETS_ROOT/lerobot2/dex3/$DEFAULT_DEX3_DATASET_NAME"
LEGACY_DEX3_DATASET="$DATASETS_ROOT/lerobot2/$DEFAULT_DEX3_DATASET_NAME"
if [[ -d "$NESTED_DEX3_DATASET" ]]; then
    DEFAULT_DATASET_ROOT="$NESTED_DEX3_DATASET"
else
    DEFAULT_DATASET_ROOT="$LEGACY_DEX3_DATASET"
fi
DATASET_ROOT="${DATASET_ROOT:-$DEFAULT_DATASET_ROOT}"
TRAIN_DATASET="${TRAIN_DATASET:-$DATASET_ROOT/train}"
VALIDATION_DATASET="${VALIDATION_DATASET:-$DATASET_ROOT/validation}"
TEST_DATASET="${TEST_DATASET:-$DATASET_ROOT/test}"
BASE_MODEL_PATH="$HOME/Development/Models/GR00T-N1.7-3B"
EXECUTION_HORIZON=8
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"
DRY_RUN="${DRY_RUN:-0}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
EXPERIMENTS="${EXPERIMENTS:-rgb,normals,depth}"
# Explicit opt-in only: never silently resume an existing output directory.
RESUME_EXPERIMENTS="${RESUME_EXPERIMENTS:-}"
MODALITY_DROPOUT="${MODALITY_DROPOUT:-1}"
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

# The base checkpoint refers to its Cosmos processor by Hugging Face repo ID.
# Use the already-qualified local cache so an overnight run never depends on
# network access or an interactive gated-repository login.
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export GROOT_HF_LOCAL_FIRST=1
export GROOT_PATCH_MISTRAL=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

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
        CONFIG_PREFIX="g1_inspire"
        DEFAULT_RUN_SUFFIX="$(basename "$DATASET_ROOT")_$(date -u +%Y%m%d)"
        ;;
    Unitree_G1_Dex3_HeadOnly)
        DEFAULT_MODEL_PREFIX=""
        DEFAULT_MODEL_ROOT="$HOME/Development/Models/dex3"
        CONFIG_PREFIX="g1_dex3"
        DEFAULT_RUN_SUFFIX="three_cups_rightonly_1408_stack_0908_$(date -u +%Y%m%d)"
        ;;
    *)
        echo "Error: unsupported dataset robot_type: $DATASET_ROBOT_TYPE" >&2
        exit 1
        ;;
esac

RUN_SUFFIX="${RUN_SUFFIX:-$DEFAULT_RUN_SUFFIX}"
MODEL_ROOT="${MODEL_ROOT:-$DEFAULT_MODEL_ROOT}"
MODEL_PREFIX="${MODEL_PREFIX:-$DEFAULT_MODEL_PREFIX}"
MODEL_DATE_SUFFIX="${MODEL_DATE_SUFFIX:-}"
[[ "$MODEL_PREFIX" =~ ^[A-Za-z0-9._-]*$ ]] || {
    echo "Error: invalid MODEL_PREFIX: $MODEL_PREFIX" >&2
    exit 1
}
if [[ -n "$MODEL_DATE_SUFFIX" && ! "$MODEL_DATE_SUFFIX" =~ ^[0-9]{2}_[0-9]{2}_[0-9]{2}$ ]]; then
    echo "Error: MODEL_DATE_SUFFIX must use MM_DD_HH format." >&2
    exit 1
fi
if [[ "$PRECHECK_ONLY" != "0" && "$PRECHECK_ONLY" != "1" ]]; then
    echo "Error: PRECHECK_ONLY must be 0 or 1." >&2
    exit 1
fi

if [[ "$DRY_RUN" == "1" ]]; then
    MAX_STEPS=1
    SAVE_STEPS=1
    RUN_LABEL="dry1"
    EVAL_STEPS=8
    EVAL_SELECTION_ARGS=(--traj-ids 0 --train-traj-ids 0 --trajectory-plot-episodes 0)
elif [[ "$DRY_RUN" == "0" ]]; then
    MAX_STEPS="${MAX_STEPS:-25000}"
    SAVE_STEPS="${SAVE_STEPS:-5000}"
    if [[ ! "$MAX_STEPS" =~ ^[1-9][0-9]*$ || ! "$SAVE_STEPS" =~ ^[1-9][0-9]*$ ]]; then
        echo "Error: MAX_STEPS and SAVE_STEPS must be positive integers without leading zeros." >&2
        exit 1
    fi
    if (( MAX_STEPS % 1000 == 0 )); then
        RUN_LABEL="$((MAX_STEPS / 1000))k"
    else
        RUN_LABEL="${MAX_STEPS}steps"
    fi
    EVAL_STEPS=0
    EVAL_SELECTION_ARGS=(--train-probe-episodes-per-goal 5)
else
    echo "Error: DRY_RUN must be 0 or 1." >&2
    exit 1
fi
if (( MAX_STEPS % SAVE_STEPS != 0 )); then
    echo "Error: MAX_STEPS must be divisible by SAVE_STEPS so the final weights exist as a checkpoint." >&2
    exit 1
fi

# Final train+validation fits skip evaluation entirely while test is deferred.
# Validation mode retains the historical all-checkpoint comparison unchanged.
EVALUATION_DATASET=""
EVALUATION_DIR_NAME=""
NORMALIZED_DIR_NAME=""
EVAL_CHECKPOINT_ARGS=()
REQUIRED_DATASETS=("$TRAIN_DATASET")
case "$EVALUATION_SPLIT" in
    validation)
        EVALUATION_DATASET="$VALIDATION_DATASET"
        REQUIRED_DATASETS+=("$VALIDATION_DATASET")
        EVALUATION_DIR_NAME="evaluation_exec_hor_${EXECUTION_HORIZON}"
        NORMALIZED_DIR_NAME="normalized_action_metrics_exec_hor_${EXECUTION_HORIZON}"
        ;;
    test)
        EVALUATION_DATASET="$TEST_DATASET"
        REQUIRED_DATASETS+=("$TEST_DATASET")
        EVALUATION_DIR_NAME="test_evaluation_exec_hor_${EXECUTION_HORIZON}"
        NORMALIZED_DIR_NAME="test_normalized_action_metrics_exec_hor_${EXECUTION_HORIZON}"
        EVAL_CHECKPOINT_ARGS=(--checkpoint-steps "$MAX_STEPS")
        ;;
    none) ;;
esac

EVALUATION_STATUS_FILE="${EVALUATION_STATUS_FILE:-$LOG_ROOT/multi_finetune_evaluation_${RUN_LABEL}_${RUN_SUFFIX}${MODALITY_DROPOUT_LABEL}_evaluation_status.tsv}"
TRAINING_FAILURES=()
EVALUATION_FAILURES=()

MODALITY_CONFIGS=(
    "examples/UnitreeG1/${CONFIG_PREFIX}_headonly_config.py"
    "examples/UnitreeG1/${CONFIG_PREFIX}_head_4_channel_gray_depth_fusion_config.py"
    "examples/UnitreeG1/${CONFIG_PREFIX}_head_6_channel_surface_normals_fusion_config.py"
    "examples/UnitreeG1/g1_inspire_head_rgbd_late_fusion_pre_adapter_config.py"
    "examples/UnitreeG1/g1_inspire_head_rgbd_turbo_late_fusion_pre_adapter_config.py"
    "examples/UnitreeG1/g1_inspire_head_rgbd_late_fusion_post_adapter_config.py"
    "examples/UnitreeG1/g1_inspire_head_rgb_surface_normals_late_fusion_pre_adapter_config.py"
    "examples/UnitreeG1/g1_inspire_head_rgb_surface_normals_late_fusion_post_adapter_config.py"
    "examples/UnitreeG1/g1_inspire_head_3_channel_turbo_depth_config.py"
    "examples/UnitreeG1/${CONFIG_PREFIX}_head_3_channel_surface_normals_config.py"
    "examples/UnitreeG1/${CONFIG_PREFIX}_head_3_channel_gray_depth_config.py"
    "examples/UnitreeG1/g1_inspire_head_6_channel_turbo_depth_fusion_config.py"
)

MODEL_DIRS=(
    "$MODEL_ROOT/${MODEL_PREFIX}c_rgb_patch_tuned_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}"
    "$MODEL_ROOT/${MODEL_PREFIX}c_d1_4ch_early_fusion_patch_tuned_depth_init_rgb_mean_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}"
    "$MODEL_ROOT/${MODEL_PREFIX}c_normals_6ch_early_fusion_patch_tuned_normals_init_rgb_mean_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}"
    "$MODEL_ROOT/${MODEL_PREFIX}c_rgbd_late_fusion_pre_adapter_4x_linear_rgb50_geo50_patch_frozen_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}"
    "$MODEL_ROOT/${MODEL_PREFIX}c_rgbd_turbo_late_fusion_pre_adapter_4x_linear_rgb50_geo50_patch_frozen_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}"
    "$MODEL_ROOT/${MODEL_PREFIX}c_rgbd_late_fusion_post_adapter_4x_linear_rgb50_geo50_patch_frozen_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}"
    "$MODEL_ROOT/${MODEL_PREFIX}c_rgb_surface_normals_late_fusion_pre_adapter_4x_linear_rgb50_geo50_patch_frozen_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}"
    "$MODEL_ROOT/${MODEL_PREFIX}c_rgb_surface_normals_late_fusion_post_adapter_4x_linear_rgb50_geo50_patch_frozen_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}"
    "$MODEL_ROOT/${MODEL_PREFIX}c_rgbd_turbo_separate_views_patch_frozen_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}"
    "$MODEL_ROOT/${MODEL_PREFIX}c_rgb_surface_normals_separate_views_patch_frozen_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}"
    "$MODEL_ROOT/${MODEL_PREFIX}c_rgbd_gray_separate_views_patch_frozen_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}"
    "$MODEL_ROOT/${MODEL_PREFIX}c_rgbd_turbo_6ch_early_fusion_patch_tuned_depth_init_rgb_mean_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}"
)

PATCH_EMBED_FLAGS=(
    "--tune-vision-patch-embed"
    "--tune-vision-patch-embed"
    "--tune-vision-patch-embed"
    "--no-tune-vision-patch-embed"
    "--no-tune-vision-patch-embed"
    "--no-tune-vision-patch-embed"
    "--no-tune-vision-patch-embed"
    "--no-tune-vision-patch-embed"
    "--no-tune-vision-patch-embed"
    "--no-tune-vision-patch-embed"
    "--no-tune-vision-patch-embed"
)

# Entry 11 is the fixed-Turbo six-channel early-fusion recipe.
PATCH_EMBED_FLAGS+=("--tune-vision-patch-embed")
LOAD_BF16_FLAGS=(1 1 1 1 1 1 1 1 1 1 1 1)
BATCH_SIZES=(32 32 32 32 32 32 32 32 32 32 32 32)
ACCUMULATION_STEPS=(1 1 1 1 1 1 1 1 1 1 1 1)
PATCH_INIT_MODES=("" "rgb_mean" "rgb_mean" "" "" "" "" "" "" "" "" "rgb_mean")
INCLUDE_BASE_MODEL=(1 0 0 0 0 0 0 0 0 0 0 0)
EXPERIMENT_NAMES=(
    rgb
    depth
    normals
    rgbd_late_fusion_pre_adapter
    rgbd_turbo_late_fusion_pre_adapter
    rgbd_late_fusion_post_adapter
    normals_late_fusion_pre_adapter
    normals_late_fusion_post_adapter
    rgbd_turbo_separate_views
    normals_separate_views
    rgbd_gray_separate_views
    rgbd_turbo_early_fusion
)

SELECTED_INDICES=()
SELECTED_EXPERIMENTS=()
declare -A SELECTED_EXPERIMENT_SET=()
IFS=',' read -r -a REQUESTED_EXPERIMENTS <<< "$EXPERIMENTS"
for experiment in "${REQUESTED_EXPERIMENTS[@]}"; do
    case "$experiment" in
        rgb) index=0 ;;
        depth) index=1 ;;
        normals) index=2 ;;
        rgbd_late_fusion_pre_adapter) index=3 ;;
        rgbd_turbo_late_fusion_pre_adapter) index=4 ;;
        rgbd_late_fusion_post_adapter) index=5 ;;
        normals_late_fusion_pre_adapter) index=6 ;;
        normals_late_fusion_post_adapter) index=7 ;;
        rgbd_turbo_separate_views) index=8 ;;
        normals_separate_views) index=9 ;;
        rgbd_gray_separate_views) index=10 ;;
        rgbd_turbo_early_fusion) index=11 ;;
        *)
            echo "Error: unsupported EXPERIMENTS entry: $experiment" >&2
            echo "Supported: ${EXPERIMENT_NAMES[*]}" >&2
            exit 1
            ;;
    esac
    if [[ -n "${SELECTED_EXPERIMENT_SET[$experiment]:-}" ]]; then
        echo "Error: duplicate EXPERIMENTS entry: $experiment" >&2
        exit 1
    fi
    SELECTED_EXPERIMENT_SET[$experiment]=1
    SELECTED_INDICES+=("$index")
    SELECTED_EXPERIMENTS+=("$experiment")
done
if [[ ${#SELECTED_INDICES[@]} -eq 0 ]]; then
    echo "Error: EXPERIMENTS must select at least one experiment." >&2
    exit 1
fi

declare -A RESUME_EXPERIMENT_SET=()
if [[ -n "$RESUME_EXPERIMENTS" ]]; then
    if [[ "$RESUME_EXPERIMENTS" == ,* || "$RESUME_EXPERIMENTS" == *, || "$RESUME_EXPERIMENTS" == *,,* ]]; then
        echo "Error: RESUME_EXPERIMENTS cannot contain empty entries." >&2
        exit 1
    fi
    IFS=',' read -r -a REQUESTED_RESUME_EXPERIMENTS <<< "$RESUME_EXPERIMENTS"
    for experiment in "${REQUESTED_RESUME_EXPERIMENTS[@]}"; do
        if [[ -z "${SELECTED_EXPERIMENT_SET[$experiment]:-}" || -n "${RESUME_EXPERIMENT_SET[$experiment]:-}" ]]; then
            echo "Error: each RESUME_EXPERIMENTS entry must be unique and selected in EXPERIMENTS: $experiment" >&2
            exit 1
        fi
        RESUME_EXPERIMENT_SET[$experiment]=1
    done
    for stats_name in stats.json relative_stats.json; do
        if [[ ! -s "$TRAIN_DATASET/meta/$stats_name" ]]; then
            echo "Error: resume requires existing training statistics; refusing to regenerate: $TRAIN_DATASET/meta/$stats_name" >&2
            exit 1
        fi
    done
fi

# Keep RGB-only output names and training arguments unchanged. Geometry runs
# get an explicit recipe suffix so they cannot collide with older no-drop runs.
for i in "${SELECTED_INDICES[@]}"; do
    if [[ "$MODALITY_DROPOUT" == "1" && "${EXPERIMENT_NAMES[$i]}" != "rgb" ]]; then
        MODEL_DIRS[$i]+="$MODALITY_DROPOUT_LABEL"
    fi
    if [[ -n "$MODEL_DATE_SUFFIX" ]]; then
        MODEL_DIRS[$i]+="_$MODEL_DATE_SUFFIX"
    fi
done

if [[ "$DATASET_ROBOT_TYPE" != "Unitree_G1_Inspire_HeadOnly" ]]; then
    for experiment in "${SELECTED_EXPERIMENTS[@]}"; do
        if [[ "$experiment" == *_late_fusion_* || "$experiment" == rgbd_turbo_* ]]; then
            echo "Error: $experiment currently has an Inspire-only modality config." >&2
            exit 1
        fi
    done
fi

if [[ ${#MODALITY_CONFIGS[@]} -ne ${#MODEL_DIRS[@]} ||
      ${#MODALITY_CONFIGS[@]} -ne ${#PATCH_EMBED_FLAGS[@]} ||
      ${#MODALITY_CONFIGS[@]} -ne ${#LOAD_BF16_FLAGS[@]} ||
      ${#MODALITY_CONFIGS[@]} -ne ${#BATCH_SIZES[@]} ||
      ${#MODALITY_CONFIGS[@]} -ne ${#ACCUMULATION_STEPS[@]} ||
      ${#MODALITY_CONFIGS[@]} -ne ${#PATCH_INIT_MODES[@]} ||
      ${#MODALITY_CONFIGS[@]} -ne ${#INCLUDE_BASE_MODEL[@]} ||
      ${#MODALITY_CONFIGS[@]} -ne ${#EXPERIMENT_NAMES[@]} ]]; then
    echo "Error: experiment arrays must have the same number of entries." >&2
    exit 1
fi

REQUIRED_FEATURES=(observation.images.ego_view)
if [[ -n "${SELECTED_EXPERIMENT_SET[depth]:-}" ||
      -n "${SELECTED_EXPERIMENT_SET[rgbd_gray_separate_views]:-}" ||
      -n "${SELECTED_EXPERIMENT_SET[rgbd_turbo_separate_views]:-}" ||
      -n "${SELECTED_EXPERIMENT_SET[rgbd_turbo_early_fusion]:-}" ||
      -n "${SELECTED_EXPERIMENT_SET[rgbd_late_fusion_pre_adapter]:-}" ||
      -n "${SELECTED_EXPERIMENT_SET[rgbd_turbo_late_fusion_pre_adapter]:-}" ||
      -n "${SELECTED_EXPERIMENT_SET[rgbd_late_fusion_post_adapter]:-}" ]]; then
    REQUIRED_FEATURES+=(observation.images.depth_gray_view)
fi
if [[ -n "${SELECTED_EXPERIMENT_SET[normals]:-}" ||
      -n "${SELECTED_EXPERIMENT_SET[normals_separate_views]:-}" ||
      -n "${SELECTED_EXPERIMENT_SET[normals_late_fusion_pre_adapter]:-}" ||
      -n "${SELECTED_EXPERIMENT_SET[normals_late_fusion_post_adapter]:-}" ]]; then
    REQUIRED_FEATURES+=(observation.images.surface_normals_view)
fi

for dataset in "${REQUIRED_DATASETS[@]}"; do
    if [[ ! -f "$dataset/meta/info.json" ]]; then
        echo "Error: dataset is not ready: $dataset/meta/info.json is missing." >&2
        exit 1
    fi
    for feature in "${REQUIRED_FEATURES[@]}"; do
        if ! grep -Fq "\"$feature\"" "$dataset/meta/info.json"; then
            echo "Error: $dataset is missing required feature $feature." >&2
            exit 1
        fi
    done
    dataset_robot_type="$(
        .venv/bin/python -c \
            'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["robot_type"])' \
            "$dataset/meta/info.json"
    )"
    if [[ "$dataset_robot_type" != "$DATASET_ROBOT_TYPE" ]]; then
        echo "Error: selected dataset robot_type mismatch: $dataset_robot_type" >&2
        exit 1
    fi
done

for i in "${SELECTED_INDICES[@]}"; do
    path="${MODALITY_CONFIGS[$i]}"
    if [[ ! -f "$path" ]]; then
        echo "Error: modality config does not exist: $path" >&2
        exit 1
    fi
done

for i in "${SELECTED_INDICES[@]}"; do
    path="${MODEL_DIRS[$i]}"
    if [[ -n "${RESUME_EXPERIMENT_SET[${EXPERIMENT_NAMES[$i]}]:-}" ]]; then
        .venv/bin/python - "$path" "$MAX_STEPS" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
max_steps = int(sys.argv[2])
checkpoints = [(int(p.name.split("-")[1]), p) for p in root.glob("checkpoint-*")
               if p.is_dir() and p.name.removeprefix("checkpoint-").isdigit()]
if not checkpoints:
    raise SystemExit(f"Error: explicitly requested resume has no checkpoint: {root}")
step, checkpoint = max(checkpoints)
required = ["optimizer.pt", "scheduler.pt", "rng_state.pth", "trainer_state.json",
            "config.json", "processor_config.json", "statistics.json"]
index = checkpoint / "model.safetensors.index.json"
if index.is_file():
    weight_map = json.loads(index.read_text())["weight_map"]
    if not weight_map:
        raise SystemExit(f"Error: empty model shard index: {index}")
    for name in set(weight_map.values()):
        if Path(name).name != name:
            raise SystemExit(f"Error: invalid checkpoint shard name: {name}")
        required.append(name)
else:
    required.append("model.safetensors")
missing = [name for name in required
           if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0]
if missing:
    raise SystemExit(f"Error: latest checkpoint is not resumable: {checkpoint}: {missing}")
state = json.loads((checkpoint / "trainer_state.json").read_text())
if state.get("global_step") != step or not 0 < step < max_steps:
    raise SystemExit(f"Error: checkpoint step must match its directory and be below MAX_STEPS: {checkpoint}")
if state.get("max_steps") != max_steps:
    raise SystemExit(f"Error: resume must retain saved max_steps={state.get('max_steps')}, not {max_steps}")
print(f"Verified resume: {checkpoint} ({step}/{max_steps}); retaining optimizer/scheduler/RNG state")
PY
    elif [[ -e "$path" ]]; then
        echo "Error: output already exists; choose a new RUN_SUFFIX or remove it deliberately: $path" >&2
        exit 1
    fi
done

.venv/bin/python - "$BASE_MODEL_PATH" <<'PY'
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

base = Path(sys.argv[1])
required_base = (
    "config.json",
    "processor_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
)
missing = [name for name in required_base if not (base / name).is_file()]
if missing:
    raise SystemExit(f"Error: local GR00T base is incomplete: {missing}")

snapshot = Path(snapshot_download("nvidia/Cosmos-Reason2-2B", local_files_only=True))
required_backbone = ("config.json", "model.safetensors", "tokenizer.json", "preprocessor_config.json")
missing = [name for name in required_backbone if not (snapshot / name).is_file()]
if missing:
    raise SystemExit(f"Error: local Cosmos cache is incomplete: {missing}")
print(f"Local-only Cosmos cache: {snapshot}")
PY

echo "Dataset root:       $DATASET_ROOT"
echo "Dataset robot type: $DATASET_ROBOT_TYPE"
echo "Model root:         $MODEL_ROOT"
echo "Model prefix:       ${MODEL_PREFIX:-<none>}"
echo "Experiments:        ${SELECTED_EXPERIMENTS[*]}"
echo "Modality dropout:   ${MODALITY_DROPOUT_LABEL:-disabled} (geometry runs only; RGB unchanged)"
echo "Training steps:     $MAX_STEPS (save every $SAVE_STEPS)"
echo "Evaluation split:   $EVALUATION_SPLIT ($EVALUATION_DATASET)"
if [[ "$PRECHECK_ONLY" == "1" ]]; then
    printf 'PRECHECK_ONLY complete; selected training/evaluation is ready.\n'
    for i in "${SELECTED_INDICES[@]}"; do
        printf '  %s\n' "${MODEL_DIRS[$i]}"
    done
    exit 0
fi

if [[ -z "$RESUME_EXPERIMENTS" || ! -s "$EVALUATION_STATUS_FILE" ]]; then
    printf 'stage\tmodel\tstatus\texit_code\n' > "$EVALUATION_STATUS_FILE"
fi

if [[ -n "$RESUME_EXPERIMENTS" ]]; then
    echo "Resume queue: retaining existing train stats.json and relative_stats.json without recomputation."
else
    uv run --no-sync python -m gr00t.data.stats \
        --dataset-path "$TRAIN_DATASET" \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path "${MODALITY_CONFIGS[0]}"
fi

experiment_number=0
for i in "${SELECTED_INDICES[@]}"; do
    experiment_number=$((experiment_number + 1))
    MODALITY_CONFIG_PATH="${MODALITY_CONFIGS[$i]}"
    MODEL_DIR="${MODEL_DIRS[$i]}"
    RESUME_ARGS=()
    if [[ -n "${RESUME_EXPERIMENT_SET[${EXPERIMENT_NAMES[$i]}]:-}" ]]; then
        RESUME_ARGS=(--resume-from-checkpoint)
    fi
    PATCH_EMBED_FLAG="${PATCH_EMBED_FLAGS[$i]}"
    LOAD_BF16="${LOAD_BF16_FLAGS[$i]}"
    BATCH_SIZE="${BATCH_SIZES[$i]}"
    ACCUMULATION_STEP="${ACCUMULATION_STEPS[$i]}"
    PATCH_INIT_MODE="${PATCH_INIT_MODES[$i]}"
    PATCH_INIT_ARGS=()
    if [[ -n "$PATCH_INIT_MODE" ]]; then
        PATCH_INIT_ARGS=(--vision-patch-embed-init "$PATCH_INIT_MODE")
    fi
    MODALITY_DROPOUT_ARGS=()
    EXPERIMENT_DROPOUT_LABEL="disabled"
    if [[ "$MODALITY_DROPOUT" == "1" && "${EXPERIMENT_NAMES[$i]}" != "rgb" ]]; then
        MODALITY_DROPOUT_ARGS=(
            --vision-modality-dropout-rgb-prob 0.05
            --vision-modality-dropout-geometry-prob 0.05
            --vision-modality-dropout-state-policy "$MODALITY_DROPOUT_STATE_POLICY"
        )
        EXPERIMENT_DROPOUT_LABEL="5% RGB + 5% geometry, global, ${MODALITY_DROPOUT_STATE_POLICY}"
    fi
    BACKBONE_STORAGE_ARGS=()
    BACKBONE_STORAGE_LABEL="FP32"
    if [[ "$LOAD_BF16" == "1" ]]; then
        BACKBONE_STORAGE_ARGS=(--load-bf16)
        BACKBONE_STORAGE_LABEL="BF16"
    elif [[ "$LOAD_BF16" != "0" ]]; then
        echo "Error: LOAD_BF16_FLAGS entries must be 0 or 1, got: $LOAD_BF16" >&2
        exit 1
    fi

    echo "============================================================"
    echo "Training experiment $experiment_number/${#SELECTED_INDICES[@]} (${EXPERIMENT_NAMES[$i]})"
    echo "Dataset:         $TRAIN_DATASET"
    echo "Base model:      $BASE_MODEL_PATH"
    echo "Modality config: $MODALITY_CONFIG_PATH"
    echo "Backbone storage: $BACKBONE_STORAGE_LABEL"
    echo "Patch embedding: $PATCH_EMBED_FLAG"
    echo "Patch init:      ${PATCH_INIT_MODE:-default}"
    echo "Modality dropout: $EXPERIMENT_DROPOUT_LABEL"
    echo "Batch:           $BATCH_SIZE x accumulation $ACCUMULATION_STEP"
    echo "Output:          $MODEL_DIR"
    echo "============================================================"

    if CUDA_VISIBLE_DEVICES=0 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
       NO_ALBUMENTATIONS_UPDATE=1 \
       uv run --no-sync python -m gr00t.experiment.launch_finetune \
           --base-model-path "$BASE_MODEL_PATH" \
           --dataset-path "$TRAIN_DATASET" \
           --embodiment-tag NEW_EMBODIMENT \
           --modality-config-path "$MODALITY_CONFIG_PATH" \
           --no-tune-llm \
           --no-tune-visual \
           "$PATCH_EMBED_FLAG" \
           "${PATCH_INIT_ARGS[@]}" \
           "${MODALITY_DROPOUT_ARGS[@]}" \
           --tune-projector \
           "${BACKBONE_STORAGE_ARGS[@]}" \
           --num-gpus 1 \
           --output-dir "$MODEL_DIR" \
           "${RESUME_ARGS[@]}" \
           --global-batch-size "$BATCH_SIZE" \
           --gradient-accumulation-steps "$ACCUMULATION_STEP" \
           --dataloader-num-workers 4 \
           --episode-sampling-rate 0.1 \
           --optim adafactor \
           --learning-rate 1e-4 \
           --warmup-ratio 0.05 \
           --max-steps "$MAX_STEPS" \
           --save-steps "$SAVE_STEPS" \
           --save-total-limit 8 \
           --skip-final-model-save \
           --color-jitter-params \
               brightness 0.20 \
               contrast 0.15 \
               saturation 0.10 \
               hue 0.0; then
        printf 'training\t%s\tPASS\t0\n' "$MODEL_DIR" >> "$EVALUATION_STATUS_FILE"
    else
        training_exit_code=$?
        printf 'training\t%s\tFAIL\t%d\n' \
            "$MODEL_DIR" "$training_exit_code" >> "$EVALUATION_STATUS_FILE"
        printf 'evaluation\t%s\tSKIP\tNA\n' "$MODEL_DIR" >> "$EVALUATION_STATUS_FILE"
        TRAINING_FAILURES+=("$MODEL_DIR (exit $training_exit_code)")
        echo "WARNING: training failed for $MODEL_DIR; skipping its evaluation and continuing." >&2
        continue
    fi

    if ! uv run --no-sync python scripts/analysis_tools/plot_training_history.py \
        --run-dir "$MODEL_DIR" \
        --smooth-window 20; then
        echo "WARNING: training-history plotting failed for $MODEL_DIR; continuing." >&2
    fi

    echo "Finished training: $MODEL_DIR"

    if [[ "$EVALUATION_SPLIT" == "none" ]]; then
        printf 'evaluation\t%s\tSKIP\tNA\n' "$MODEL_DIR" >> "$EVALUATION_STATUS_FILE"
        echo "Evaluation deferred: validation is not held out in a final train+validation fit."
        continue
    fi

    # The legacy evaluator calls its held-out series "validation" internally.
    # Record the real split and fixed-checkpoint protocol without changing CSV schemas.
    .venv/bin/python - "$MODEL_DIR/evaluation_protocol.json" \
        "$EVALUATION_SPLIT" "$EVALUATION_DATASET" "$TRAIN_DATASET" \
        "$MAX_STEPS" "$EXECUTION_HORIZON" "$EVALUATION_DIR_NAME" <<'PY'
import json
from pathlib import Path
import sys

output, split, dataset, train_dataset, final_step, horizon, evaluation_dir = sys.argv[1:]
protocol = {
    "evaluation_split": split,
    "evaluation_dataset_path": str(Path(dataset).resolve()),
    "training_dataset_path": str(Path(train_dataset).resolve()),
    "internal_evaluator_split_label": "validation",
    "checkpoint_policy": "fixed_final_checkpoint" if split == "test" else "all_checkpoints",
    "checkpoint_steps": [int(final_step)] if split == "test" else None,
    "checkpoint_selection_permitted": split != "test",
    "execution_horizon": int(horizon),
    "evaluation_directory": evaluation_dir,
    "normalization": "checkpoint-saved training statistics",
}
Path(output).write_text(json.dumps(protocol, indent=2) + "\n")
PY

    BASE_MODEL_ARGS=()
    if [[ "${INCLUDE_BASE_MODEL[$i]}" == "1" && "$EVALUATION_SPLIT" == "validation" ]]; then
        BASE_MODEL_ARGS=(--base-model-path "$BASE_MODEL_PATH")
    fi

    echo "============================================================"
    echo "Evaluating experiment $experiment_number/${#SELECTED_INDICES[@]} (${EXPERIMENT_NAMES[$i]})"
    echo "Model: $MODEL_DIR"
    echo "============================================================"

    if CUDA_VISIBLE_DEVICES=0 \
       NO_ALBUMENTATIONS_UPDATE=1 \
       uv run --no-sync python -m scripts.analysis_tools.evaluate_checkpoints \
           --run-dir "$MODEL_DIR" \
           "${BASE_MODEL_ARGS[@]}" \
           --dataset-path "$EVALUATION_DATASET" \
           --train-dataset-path "$TRAIN_DATASET" \
           "${EVAL_SELECTION_ARGS[@]}" \
           "${EVAL_CHECKPOINT_ARGS[@]}" \
           --output-dir "$MODEL_DIR/$EVALUATION_DIR_NAME" \
           --steps "$EVAL_STEPS" \
           --execution-horizon "$EXECUTION_HORIZON" \
           --inference-batch-size "$INFERENCE_BATCH_SIZE" \
           --denoising-steps 4 \
           --inference-seed 42 \
           --modality-keys left_arm right_arm left_hand right_hand \
           --train-probe-seed 42 &&
       # Consume the complete saved frame-level predictions without rewriting them.
       uv run --no-sync python -m scripts.analysis_tools.normalized_action_metrics \
           --run-dir "$MODEL_DIR" \
           --evaluation-dir "$MODEL_DIR/$EVALUATION_DIR_NAME" \
           --output-dir "$MODEL_DIR/$NORMALIZED_DIR_NAME" \
           --statistics-path "$MODEL_DIR/experiment_cfg/dataset_statistics.json" \
           --embodiment-tag NEW_EMBODIMENT \
           --execution-horizon "$EXECUTION_HORIZON"; then
        printf 'evaluation\t%s\tPASS\t0\n' "$MODEL_DIR" >> "$EVALUATION_STATUS_FILE"
        echo "Finished evaluation and normalized action metrics: $MODEL_DIR"
    else
        evaluation_exit_code=$?
        printf 'evaluation\t%s\tFAIL\t%d\n' \
            "$MODEL_DIR" "$evaluation_exit_code" >> "$EVALUATION_STATUS_FILE"
        EVALUATION_FAILURES+=("$MODEL_DIR (exit $evaluation_exit_code)")
        echo "WARNING: evaluation or normalized metrics failed for $MODEL_DIR; continuing to the next training run." >&2
    fi
done

echo "Evaluation status: $EVALUATION_STATUS_FILE"
if [[ ${#TRAINING_FAILURES[@]} -gt 0 ]]; then
    echo "${#TRAINING_FAILURES[@]} training run(s) failed; all remaining experiments were still attempted:" >&2
    printf '  %s\n' "${TRAINING_FAILURES[@]}" >&2
fi
if [[ ${#EVALUATION_FAILURES[@]} -gt 0 ]]; then
    echo "${#EVALUATION_FAILURES[@]} evaluation(s) failed; all remaining experiments were still attempted:" >&2
    printf '  %s\n' "${EVALUATION_FAILURES[@]}" >&2
fi
if [[ ${#TRAINING_FAILURES[@]} -gt 0 || ${#EVALUATION_FAILURES[@]} -gt 0 ]]; then
    exit 1
fi

echo "All training and evaluation experiments finished successfully."
