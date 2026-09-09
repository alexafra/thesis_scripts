#!/usr/bin/env bash
set -Eeuo pipefail

# Prepare one geometry-only GR00T ablation. This script never queues itself.
export PATH="/home/alex/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

GROOT_DIR="/home/alex/Development/Isaac-GR00T"
GROOT_PYTHON="$GROOT_DIR/.venv/bin/python"
DATASET_ROOT="${DATASET_ROOT:-/home/alex/Development/Datasets/lerobot2/atomic_combined_09_08_And_10_08_plus_pick_three_cups_right_only_1408_plus_stack_cups_09_08}"
TRAIN_DATASET="${TRAIN_DATASET:-$DATASET_ROOT/train}"
VALIDATION_DATASET="${VALIDATION_DATASET:-$DATASET_ROOT/validation}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/home/alex/Development/Models/GR00T-N1.7-3B}"
MODEL_ROOT="${MODEL_ROOT:-/home/alex/Development/Models}"
LOG_ROOT="${LOG_ROOT:-/home/alex/Development/logs/groot/training}"
VISUAL_MODE="${VISUAL_MODE:-depth}"
DRY_RUN="${DRY_RUN:-0}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
MIN_FREE_GIB="${MIN_FREE_GIB:-110}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%dT%H%M%S%z)}"
EXECUTION_HORIZON=8
INFERENCE_BATCH_SIZE="${INFERENCE_BATCH_SIZE:-8}"

case "$VISUAL_MODE" in
    depth)
        MODALITY_CONFIG="examples/UnitreeG1/g1_dex3_head_depth_gray_only_config.py"
        VIEW_KEY="depth_gray_view"
        FEATURE_KEY="observation.images.depth_gray_view"
        ENCODING_KEY="depth_encoding"
        MODEL_STEM="c_depth_gray_only_3ch_patch_tuned"
        ;;
    normals)
        MODALITY_CONFIG="examples/UnitreeG1/g1_dex3_head_surface_normals_only_config.py"
        VIEW_KEY="surface_normals_view"
        FEATURE_KEY="observation.images.surface_normals_view"
        ENCODING_KEY="surface_normals_encoding"
        MODEL_STEM="c_surface_normals_only_3ch_patch_tuned"
        ;;
    *)
        echo "ERROR: VISUAL_MODE must be depth or normals, got: $VISUAL_MODE" >&2
        exit 1
        ;;
esac

case "$DRY_RUN" in
    0)
        MAX_STEPS=30000
        SAVE_STEPS=5000
        RUN_LABEL="30k"
        EVAL_STEPS=0
        EVAL_SELECTION_ARGS=(--train-probe-episodes 3)
        ;;
    1)
        MAX_STEPS=1
        SAVE_STEPS=1
        RUN_LABEL="dry1"
        EVAL_STEPS=8
        EVAL_SELECTION_ARGS=(--traj-ids 0 --train-traj-ids 0 --trajectory-plot-episodes 0)
        ;;
    *)
        echo "ERROR: DRY_RUN must be 0 or 1, got: $DRY_RUN" >&2
        exit 1
        ;;
esac

case "$PRECHECK_ONLY" in
    0 | 1) ;;
    *)
        echo "ERROR: PRECHECK_ONLY must be 0 or 1, got: $PRECHECK_ONLY" >&2
        exit 1
        ;;
esac

RUN_SUFFIX="${RUN_SUFFIX:-rightonly_1408_stack_0908_${RUN_ID}}"
MODEL_DIR="${MODEL_DIR:-$MODEL_ROOT/${MODEL_STEM}_bf16_batch_32_acc_1_${RUN_LABEL}_${RUN_SUFFIX}}"
LOG_FILE="${LOG_FILE:-$LOG_ROOT/geometry_only_${VISUAL_MODE}_${RUN_LABEL}_${RUN_ID}.log}"
STATUS_FILE="${STATUS_FILE:-$LOG_ROOT/geometry_only_${VISUAL_MODE}_${RUN_LABEL}_${RUN_ID}_status.tsv}"
LOCK_FILE="$LOG_ROOT/.geometry_only_${VISUAL_MODE}.lock"

mkdir -p "$LOG_ROOT"
exec > >(tee -a "$LOG_FILE") 2>&1
exec 9>"$LOCK_FILE"
flock -n 9 || {
    echo "ERROR: another $VISUAL_MODE-only run owns $LOCK_FILE" >&2
    exit 1
}
trap 'flock -u 9 || true' EXIT

export HF_HOME="/home/alex/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export GROOT_HF_LOCAL_FIRST=1
export GROOT_PATCH_MISTRAL=1
export NO_ALBUMENTATIONS_UPDATE=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

echo "RUN_ID=$RUN_ID"
echo "VISUAL_MODE=$VISUAL_MODE"
echo "VISUAL_INPUT=$VIEW_KEY (one three-channel geometry frame; no ego RGB view)"
echo "TRAIN_DATASET=$TRAIN_DATASET"
echo "VALIDATION_DATASET=$VALIDATION_DATASET"
echo "BASE_MODEL_PATH=$BASE_MODEL_PATH"
echo "MODEL_DIR=$MODEL_DIR"
echo "LOG_FILE=$LOG_FILE"

[[ -x "$GROOT_PYTHON" ]] || { echo "ERROR: missing GR00T Python: $GROOT_PYTHON" >&2; exit 1; }
[[ -f "$GROOT_DIR/$MODALITY_CONFIG" ]] || {
    echo "ERROR: missing modality config: $GROOT_DIR/$MODALITY_CONFIG" >&2
    exit 1
}
[[ -f "$BASE_MODEL_PATH/model.safetensors.index.json" ]] || {
    echo "ERROR: local base model is incomplete: $BASE_MODEL_PATH" >&2
    exit 1
}
[[ ! -e "$MODEL_DIR" ]] || {
    echo "ERROR: output already exists; choose a new RUN_ID: $MODEL_DIR" >&2
    exit 1
}

available_kib=$(df -Pk "$MODEL_ROOT" | awk 'NR == 2 {print $4}')
required_kib=$((MIN_FREE_GIB * 1024 * 1024))
if (( available_kib < required_kib )); then
    echo "ERROR: run requires at least ${MIN_FREE_GIB} GiB free; only $((available_kib / 1024 / 1024)) GiB is available" >&2
    exit 1
fi
echo "Disk preflight: $((available_kib / 1024 / 1024)) GiB available."

"$GROOT_PYTHON" - \
    "$TRAIN_DATASET" "$VALIDATION_DATASET" "$GROOT_DIR/$MODALITY_CONFIG" \
    "$VIEW_KEY" "$FEATURE_KEY" "$ENCODING_KEY" "$BASE_MODEL_PATH" <<'PY'
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

from huggingface_hub import snapshot_download

train = Path(sys.argv[1])
validation = Path(sys.argv[2])
config_path = Path(sys.argv[3])
view_key = sys.argv[4]
feature_key = sys.argv[5]
encoding_key = sys.argv[6]
base = Path(sys.argv[7])

for split in (train, validation):
    info_path = split / "meta" / "info.json"
    if not info_path.is_file():
        raise SystemExit(f"Dataset is incomplete: {info_path} is missing")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    feature = info.get("features", {}).get(feature_key)
    if feature is None:
        raise SystemExit(f"{split} is missing {feature_key}")
    if feature.get("dtype") != "video" or feature.get("shape") != [480, 640, 3]:
        raise SystemExit(f"Unexpected {feature_key} feature contract: {feature}")
    encoding = info.get(encoding_key)
    if not isinstance(encoding, dict) or encoding.get("feature_key") != feature_key:
        raise SystemExit(f"Unexpected {encoding_key} contract: {encoding}")

spec = spec_from_file_location("geometry_only_config", config_path)
if spec is None or spec.loader is None:
    raise SystemExit(f"Could not load modality config: {config_path}")
module = module_from_spec(spec)
spec.loader.exec_module(module)
configs = [value for name, value in vars(module).items() if name.endswith("_only_config")]
if len(configs) != 1:
    raise SystemExit(f"Expected one geometry-only config in {config_path}, found {len(configs)}")
video = configs[0]["video"]
if video.modality_keys != [view_key] or video.delta_indices != [0] or video.channel_fusion:
    raise SystemExit(f"Modality config is not strictly {view_key}-only: {video}")

required_base = (
    "config.json",
    "processor_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00002.safetensors",
    "model-00002-of-00002.safetensors",
)
missing = [name for name in required_base if not (base / name).is_file()]
if missing:
    raise SystemExit(f"Local GR00T base is incomplete: {missing}")
snapshot = Path(snapshot_download("nvidia/Cosmos-Reason2-2B", local_files_only=True))
required_backbone = ("config.json", "model.safetensors", "tokenizer.json", "preprocessor_config.json")
missing = [name for name in required_backbone if not (snapshot / name).is_file()]
if missing:
    raise SystemExit(f"Local Cosmos cache is incomplete: {missing}")
print(f"Geometry-only modality validated: {view_key}")
print(f"Local-only GR00T base: {base}")
print(f"Local-only Cosmos cache: {snapshot}")
PY

if [[ "$PRECHECK_ONLY" == "1" ]]; then
    echo "PRECHECK_ONLY complete; no training or evaluation was started."
    exit 0
fi

if pgrep -af '[g]r00t.experiment.launch_finetune|[e]valuate_checkpoints|[r]un_gr00t_server' >/dev/null; then
    echo "ERROR: a GR00T training, evaluation, or policy server is active; refusing GPU contention." >&2
    pgrep -af '[g]r00t.experiment.launch_finetune|[e]valuate_checkpoints|[r]un_gr00t_server' >&2 || true
    exit 1
fi
if ! gpu_compute_apps=$(nvidia-smi \
    --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>&1); then
    echo "ERROR: nvidia-smi failed; refusing to assume the GPU is available." >&2
    echo "$gpu_compute_apps" >&2
    exit 1
fi
if grep -q '[0-9]' <<< "$gpu_compute_apps"; then
    echo "ERROR: a GPU compute process is active; refusing GPU contention." >&2
    echo "$gpu_compute_apps" >&2
    exit 1
fi

printf 'stage\tmodel\tstatus\texit_code\n' > "$STATUS_FILE"

cd "$GROOT_DIR"
uv run --no-sync python -m gr00t.data.stats \
    --dataset-path "$TRAIN_DATASET" \
    --embodiment-tag NEW_EMBODIMENT \
    --modality-config-path "$MODALITY_CONFIG"

echo "Training $VISUAL_MODE-only model for $MAX_STEPS step(s)."
if CUDA_VISIBLE_DEVICES=0 \
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
   uv run --no-sync python -m gr00t.experiment.launch_finetune \
       --base-model-path "$BASE_MODEL_PATH" \
       --dataset-path "$TRAIN_DATASET" \
       --embodiment-tag NEW_EMBODIMENT \
       --modality-config-path "$MODALITY_CONFIG" \
       --no-tune-llm \
       --no-tune-visual \
       --tune-vision-patch-embed \
       --tune-projector \
       --load-bf16 \
       --num-gpus 1 \
       --output-dir "$MODEL_DIR" \
       --global-batch-size 32 \
       --gradient-accumulation-steps 1 \
       --dataloader-num-workers 4 \
       --episode-sampling-rate 0.1 \
       --optim adafactor \
       --learning-rate 1e-4 \
       --warmup-ratio 0.05 \
       --max-steps "$MAX_STEPS" \
       --save-steps "$SAVE_STEPS" \
       --save-total-limit 8; then
    printf 'training\t%s\tPASS\t0\n' "$MODEL_DIR" >> "$STATUS_FILE"
else
    exit_code=$?
    printf 'training\t%s\tFAIL\t%d\n' "$MODEL_DIR" "$exit_code" >> "$STATUS_FILE"
    exit "$exit_code"
fi

if ! uv run --no-sync python scripts/analysis_tools/plot_training_history.py \
    --run-dir "$MODEL_DIR" --smooth-window 20; then
    echo "WARNING: training-history plotting failed; evaluation will continue." >&2
fi

"$GROOT_PYTHON" - "$MODEL_DIR/processor/processor_config.json" "$VIEW_KEY" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
view_key = sys.argv[2]
payload = json.loads(path.read_text(encoding="utf-8"))
configs = payload["processor_kwargs"]["modality_configs"]
config = configs.get("new_embodiment") or configs.get("NEW_EMBODIMENT")
if config is None and len(configs) == 1:
    config = next(iter(configs.values()))
if config is None or config["video"]["modality_keys"] != [view_key]:
    raise SystemExit(f"Saved processor is not strictly {view_key}-only: {configs}")
print(f"Saved processor verified: visual input is only {view_key}")
PY

echo "Evaluating local base step 0 and all retained checkpoints at horizon $EXECUTION_HORIZON."
if CUDA_VISIBLE_DEVICES=0 \
   uv run --no-sync python -m scripts.analysis_tools.evaluate_checkpoints \
       --run-dir "$MODEL_DIR" \
       --base-model-path "$BASE_MODEL_PATH" \
       --dataset-path "$VALIDATION_DATASET" \
       --train-dataset-path "$TRAIN_DATASET" \
       "${EVAL_SELECTION_ARGS[@]}" \
       --output-dir "$MODEL_DIR/evaluation_exec_hor_${EXECUTION_HORIZON}" \
       --steps "$EVAL_STEPS" \
       --execution-horizon "$EXECUTION_HORIZON" \
       --inference-batch-size "$INFERENCE_BATCH_SIZE" \
       --denoising-steps 4 \
       --inference-seed 42 \
       --modality-keys left_arm right_arm left_hand right_hand \
       --train-probe-seed 42; then
    printf 'evaluation\t%s\tPASS\t0\n' "$MODEL_DIR" >> "$STATUS_FILE"
else
    exit_code=$?
    printf 'evaluation\t%s\tFAIL\t%d\n' "$MODEL_DIR" "$exit_code" >> "$STATUS_FILE"
    exit "$exit_code"
fi

echo "GEOMETRY_ONLY_PIPELINE_COMPLETE"
echo "Mode: $VISUAL_MODE"
echo "Model: $MODEL_DIR"
echo "Evaluation: $MODEL_DIR/evaluation_exec_hor_${EXECUTION_HORIZON}"
echo "Status: $STATUS_FILE"
