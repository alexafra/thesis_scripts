#!/usr/bin/env bash
set -Eeuo pipefail

export PATH="/home/alex/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

SCRIPTS_DIR="/home/alex/Development/scripts"
GROOT_DIR="/home/alex/Development/Isaac-GR00T"
UNITREE_DIR="/home/alex/Development/unitree_lerobot"
UNITREE_PYTHON="/home/alex/miniconda3/envs/unitree_lerobot/bin/python"
GROOT_PYTHON="$GROOT_DIR/.venv/bin/python"
DATASETS_DIR="/home/alex/Development/Datasets/lerobot2"

BASE_DATASET="$DATASETS_DIR/atomic_combined_09_08_And_10_08"
ORIGINAL_CUPS_DATASET="$DATASETS_DIR/atomic_combined_09_08_And_10_08_plus_pick_three_cups_1408"
RIGHT_SOURCE="/home/alex/Development/Datasets/processed_raw/pick_three_cups_right_only_1408"
STACK_SOURCE="/home/alex/Development/Datasets/processed_raw/stack_cups_09_08"
TARGET_DATASET="$DATASETS_DIR/atomic_combined_09_08_And_10_08_plus_pick_three_cups_right_only_1408_plus_stack_cups_09_08"
BASE_MODEL="/home/alex/Development/Models/GR00T-N1.7-3B"

RUN_ID="${RUN_ID:-$(date -u +%Y%m%d)}"
WORK_ROOT="$DATASETS_DIR/.three_cups_merge_work_${RUN_ID}"
BUILD_DATASET="$DATASETS_DIR/.atomic_combined_rightonly_stack.build-${RUN_ID}"
RIGHT_STAGE="$WORK_ROOT/pick_three_cups_right_only_1408"
STACK_STAGE="$WORK_ROOT/stack_cups_09_08"
PROVENANCE_STAGE="$WORK_ROOT/provenance"
LOG_ROOT="/home/alex/Development/logs"
PIPELINE_LOG="$LOG_ROOT/data_pipeline/merge_rightonly_stack_${RUN_ID}.log"
TRAIN_LOG="$LOG_ROOT/groot/training/three_models_30k_rightonly_stack_${RUN_ID}.log"
STATUS_FILE="$LOG_ROOT/groot/training/three_models_30k_rightonly_stack_${RUN_ID}_evaluation_status.tsv"
RUN_SUFFIX="rightonly_1408_stack_0908_${RUN_ID}"
LOCK_FILE="$LOG_ROOT/data_pipeline/.merge_rightonly_stack_and_train.lock"
SUPPORT="$SCRIPTS_DIR/three_cups_merge_support.py"

mkdir -p "$LOG_ROOT/data_pipeline" "$LOG_ROOT/groot/training"
exec > >(tee -a "$PIPELINE_LOG") 2>&1
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "ERROR: another right-only/stack merge owns $LOCK_FILE" >&2; exit 1; }

safe_remove_generated() {
    local path="$1"
    [[ -e "$path" ]] || return 0
    case "$path" in
        /home/alex/Development/Datasets/lerobot2/.three_cups_merge_work_*)
            rm -rf -- "$path"
            ;;
        /home/alex/Development/Datasets/lerobot2/.atomic_combined_rightonly_stack.build-*)
            rm -rf -- "$path"
            ;;
        *)
            echo "ERROR: refusing to remove unexpected generated path: $path" >&2
            return 1
            ;;
    esac
}

cleanup() {
    local status=$?
    trap - EXIT
    safe_remove_generated "$WORK_ROOT" || true
    safe_remove_generated "$BUILD_DATASET" || true
    flock -u 9 || true
    exit "$status"
}
trap cleanup EXIT

metadata_digest() {
    local root="$1"
    (
        cd "$root"
        find train test validation -type f \
            \( -path '*/meta/info.json' -o -path '*/meta/tasks.jsonl' \
               -o -path '*/meta/episodes.jsonl' -o -path '*/meta/modality.json' \) \
            -print0 | sort -z | xargs -0 sha256sum
    )
}

data_json_digest() {
    local root="$1"
    (
        cd "$root"
        find . -mindepth 2 -maxdepth 2 -type f -path './episode_*/data.json' \
            -print0 | sort -z | xargs -0 sha256sum
    )
}

source_inventory() {
    local root="$1"
    (
        cd "$root"
        find . -type f -printf '%P\t%s\n' | sort
    )
}

echo "RUN_ID=$RUN_ID"
echo "BASE_DATASET=$BASE_DATASET"
echo "RIGHT_SOURCE=$RIGHT_SOURCE"
echo "STACK_SOURCE=$STACK_SOURCE"
echo "TARGET_DATASET=$TARGET_DATASET"
echo "PIPELINE_LOG=$PIPELINE_LOG"
echo "TRAIN_LOG=$TRAIN_LOG"

for path in "$BASE_DATASET" "$ORIGINAL_CUPS_DATASET" "$RIGHT_SOURCE" "$STACK_SOURCE"; do
    [[ -d "$path" ]] || { echo "ERROR: required source is missing: $path" >&2; exit 1; }
done
[[ -x "$UNITREE_PYTHON" ]] || { echo "ERROR: missing Unitree Python: $UNITREE_PYTHON" >&2; exit 1; }
[[ -x "$GROOT_PYTHON" ]] || { echo "ERROR: missing GR00T Python: $GROOT_PYTHON" >&2; exit 1; }
[[ -f "$SUPPORT" ]] || { echo "ERROR: missing merge support script: $SUPPORT" >&2; exit 1; }
[[ -f "$BASE_MODEL/model.safetensors.index.json" ]] || {
    echo "ERROR: local base model is incomplete: $BASE_MODEL" >&2
    exit 1
}
[[ ! -e "$TARGET_DATASET" ]] || {
    echo "ERROR: target already exists; refusing a duplicate append: $TARGET_DATASET" >&2
    exit 1
}
[[ ! -e "$WORK_ROOT" && ! -e "$BUILD_DATASET" ]] || {
    echo "ERROR: generated work path already exists for RUN_ID=$RUN_ID" >&2
    exit 1
}

available_kib=$(df -Pk "$DATASETS_DIR" | awk 'NR == 2 {print $4}')
required_kib=$((520 * 1024 * 1024))
if (( available_kib < required_kib )); then
    echo "ERROR: merge requires at least 520 GiB free; only $((available_kib / 1024 / 1024)) GiB is available" >&2
    exit 1
fi
echo "Disk preflight: $((available_kib / 1024 / 1024)) GiB available (minimum 520 GiB)."

if pgrep -af '[l]aunch_finetune|[e]valuate_checkpoints|[r]un_gr00t_server' >/dev/null; then
    echo "ERROR: a GR00T training, evaluation, or policy-server process is already active" >&2
    pgrep -af '[l]aunch_finetune|[e]valuate_checkpoints|[r]un_gr00t_server' >&2 || true
    exit 1
fi

export HF_HOME="/home/alex/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export GROOT_HF_LOCAL_FIRST=1
export GROOT_PATCH_MISTRAL=1
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

echo "[preflight] Proving that training can load exclusively from the local base and HF cache..."
(
    cd "$GROOT_DIR"
    "$GROOT_PYTHON" - "$BASE_MODEL" <<'PY'
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
    raise SystemExit(f"Local GR00T base is incomplete: {missing}")
snapshot = Path(snapshot_download("nvidia/Cosmos-Reason2-2B", local_files_only=True))
required_backbone = ("config.json", "model.safetensors", "tokenizer.json", "preprocessor_config.json")
missing = [name for name in required_backbone if not (snapshot / name).is_file()]
if missing:
    raise SystemExit(f"Local Cosmos cache is incomplete: {missing}")
print(f"Local-only GR00T base: {base}")
print(f"Local-only Cosmos cache: {snapshot}")
PY
)

mkdir -p "$WORK_ROOT" "$PROVENANCE_STAGE"
metadata_digest "$BASE_DATASET" > "$PROVENANCE_STAGE/base_metadata_before.sha256"
metadata_digest "$ORIGINAL_CUPS_DATASET" > "$PROVENANCE_STAGE/original_cups_metadata_before.sha256"
data_json_digest "$RIGHT_SOURCE" > "$PROVENANCE_STAGE/right_source_data_json.sha256"
data_json_digest "$STACK_SOURCE" > "$PROVENANCE_STAGE/stack_source_data_json.sha256"
source_inventory "$RIGHT_SOURCE" > "$PROVENANCE_STAGE/right_source_inventory.tsv"
source_inventory "$STACK_SOURCE" > "$PROVENANCE_STAGE/stack_source_inventory.tsv"

echo "[audit] Validating the exact curated processed sources..."
"$UNITREE_PYTHON" "$SUPPORT" audit-source --root "$RIGHT_SOURCE" --kind right
"$UNITREE_PYTHON" "$SUPPORT" audit-source --root "$STACK_SOURCE" --kind stack

echo "[copy] Creating isolated conversion copies; source datasets remain untouched..."
cp -a --reflink=auto "$RIGHT_SOURCE" "$RIGHT_STAGE"
cp -a --reflink=auto "$STACK_SOURCE" "$STACK_STAGE"
data_json_digest "$RIGHT_STAGE" > "$PROVENANCE_STAGE/right_copy_data_json.sha256"
data_json_digest "$STACK_STAGE" > "$PROVENANCE_STAGE/stack_copy_data_json.sha256"
source_inventory "$RIGHT_STAGE" > "$PROVENANCE_STAGE/right_copy_inventory.tsv"
source_inventory "$STACK_STAGE" > "$PROVENANCE_STAGE/stack_copy_inventory.tsv"
diff -u "$PROVENANCE_STAGE/right_source_data_json.sha256" "$PROVENANCE_STAGE/right_copy_data_json.sha256"
diff -u "$PROVENANCE_STAGE/stack_source_data_json.sha256" "$PROVENANCE_STAGE/stack_copy_data_json.sha256"
diff -u "$PROVENANCE_STAGE/right_source_inventory.tsv" "$PROVENANCE_STAGE/right_copy_inventory.tsv"
diff -u "$PROVENANCE_STAGE/stack_source_inventory.tsv" "$PROVENANCE_STAGE/stack_copy_inventory.tsv"

echo "[canonicalize] Recording and removing float32 depth-scale spelling noise in the disposable stack copy..."
"$UNITREE_PYTHON" "$SUPPORT" canonicalize-stack-depth-scale \
    --root "$STACK_STAGE" \
    --source-root "$STACK_SOURCE" \
    --provenance-output "$PROVENANCE_STAGE/stack_depth_scale_canonicalization.json"

echo "[normalize] Canonicalizing task text only in the disposable copies..."
(
    cd "$UNITREE_DIR"
    "$UNITREE_PYTHON" -m unitree_lerobot.utils.normalize_lerobot_tasks \
        "$RIGHT_STAGE" --apply --discard-backup
    "$UNITREE_PYTHON" -m unitree_lerobot.utils.normalize_lerobot_tasks \
        "$STACK_STAGE" --apply --discard-backup
)

echo "[split] Preserving the original 1408 membership for all retained right-only episodes..."
"$UNITREE_PYTHON" "$SUPPORT" split-right --root "$RIGHT_STAGE"
cp "$RIGHT_STAGE/split_manifest.json" "$PROVENANCE_STAGE/right_split_manifest.json"

echo "[split] Applying the established goal-stratified seed-42 split to stack_cups_09_08..."
"$UNITREE_PYTHON" "$SCRIPTS_DIR/split_dataset.py" "$STACK_STAGE" \
    --strategy goal-stratified --seed 42
"$UNITREE_PYTHON" "$SUPPORT" enrich-manifest --root "$STACK_STAGE"
cp "$STACK_STAGE/split_manifest.json" "$PROVENANCE_STAGE/stack_split_manifest.json"

echo "[convert] Converting right-only splits with RGB, gray depth, and lossless surface normals..."
bash "$SCRIPTS_DIR/convert_to_lerobot2.sh" \
    --include-surface-normals "$RIGHT_STAGE" pick_three_cups_right_only_1408 6

echo "[convert] Converting stack_09_08 splits with RGB, gray depth, and lossless surface normals..."
bash "$SCRIPTS_DIR/convert_to_lerobot2.sh" \
    --include-surface-normals "$STACK_STAGE" stack_cups_09_08 6

echo "[append] Building a new independent dataset in base -> right-only -> stack_09_08 order..."
"$UNITREE_PYTHON" "$SCRIPTS_DIR/append_lerobot2.py" \
    "$BUILD_DATASET" "$BASE_DATASET" "$RIGHT_STAGE" "$STACK_STAGE"

echo "[normalize] Installing one canonical nine-task table across all final splits..."
(
    cd "$UNITREE_DIR"
    "$UNITREE_PYTHON" -m unitree_lerobot.utils.normalize_lerobot_tasks \
        "$BUILD_DATASET" --apply --discard-backup
)

echo "[validate] Exhaustively checking indices, media, modalities, split membership, and provenance..."
"$UNITREE_PYTHON" "$SUPPORT" validate-final \
    --base "$BASE_DATASET" \
    --right "$RIGHT_STAGE" \
    --stack "$STACK_STAGE" \
    --right-source "$RIGHT_SOURCE" \
    --stack-source "$STACK_SOURCE" \
    --output "$BUILD_DATASET" \
    --right-manifest "$PROVENANCE_STAGE/right_split_manifest.json" \
    --stack-manifest "$PROVENANCE_STAGE/stack_split_manifest.json" \
    --depth-scale-provenance "$PROVENANCE_STAGE/stack_depth_scale_canonicalization.json" \
    --scripts-dir "$SCRIPTS_DIR"
cp "$PROVENANCE_STAGE/base_metadata_before.sha256" "$BUILD_DATASET/provenance/"
cp "$PROVENANCE_STAGE/original_cups_metadata_before.sha256" "$BUILD_DATASET/provenance/"
cp "$PROVENANCE_STAGE/right_source_data_json.sha256" "$BUILD_DATASET/provenance/"
cp "$PROVENANCE_STAGE/stack_source_data_json.sha256" "$BUILD_DATASET/provenance/"
cp "$PROVENANCE_STAGE/right_source_inventory.tsv" "$BUILD_DATASET/provenance/"
cp "$PROVENANCE_STAGE/stack_source_inventory.tsv" "$BUILD_DATASET/provenance/"

echo "[stats] Generating GR00T statistics against the final 194,692-frame training split..."
(
    cd "$GROOT_DIR"
    uv run --no-sync python -m gr00t.data.stats \
        --dataset-path "$BUILD_DATASET/train" \
        --embodiment-tag NEW_EMBODIMENT \
        --modality-config-path examples/UnitreeG1/g1_dex3_headonly_config.py
)
"$UNITREE_PYTHON" "$SUPPORT" validate-stats --train-root "$BUILD_DATASET/train"

echo "[immutability] Confirming the no-cups base and prior left/right dataset were not changed..."
metadata_digest "$BASE_DATASET" > "$PROVENANCE_STAGE/base_metadata_after.sha256"
metadata_digest "$ORIGINAL_CUPS_DATASET" > "$PROVENANCE_STAGE/original_cups_metadata_after.sha256"
data_json_digest "$RIGHT_SOURCE" > "$PROVENANCE_STAGE/right_source_data_json_after.sha256"
data_json_digest "$STACK_SOURCE" > "$PROVENANCE_STAGE/stack_source_data_json_after.sha256"
source_inventory "$RIGHT_SOURCE" > "$PROVENANCE_STAGE/right_source_inventory_after.tsv"
source_inventory "$STACK_SOURCE" > "$PROVENANCE_STAGE/stack_source_inventory_after.tsv"
diff -u "$PROVENANCE_STAGE/base_metadata_before.sha256" "$PROVENANCE_STAGE/base_metadata_after.sha256"
diff -u "$PROVENANCE_STAGE/original_cups_metadata_before.sha256" "$PROVENANCE_STAGE/original_cups_metadata_after.sha256"
diff -u "$PROVENANCE_STAGE/right_source_data_json.sha256" "$PROVENANCE_STAGE/right_source_data_json_after.sha256"
diff -u "$PROVENANCE_STAGE/stack_source_data_json.sha256" "$PROVENANCE_STAGE/stack_source_data_json_after.sha256"
diff -u "$PROVENANCE_STAGE/right_source_inventory.tsv" "$PROVENANCE_STAGE/right_source_inventory_after.tsv"
diff -u "$PROVENANCE_STAGE/stack_source_inventory.tsv" "$PROVENANCE_STAGE/stack_source_inventory_after.tsv"

echo "[publish] Atomically publishing the fully validated new dataset..."
mv -- "$BUILD_DATASET" "$TARGET_DATASET"
echo "DATASET_PUBLISHED=$TARGET_DATASET"

echo "[cleanup] Removing only the disposable conversion work tree..."
safe_remove_generated "$WORK_ROOT"

if nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>/dev/null \
    | grep -q '[0-9]'; then
    echo "ERROR: GPU compute process detected before queued training; dataset is published but training was not started" >&2
    nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits >&2 || true
    exit 1
fi

echo "[train] Starting three 30k BF16 32x1 models: RGB, 4ch gray-depth, 6ch surface normals."
echo "[train] Local-only model loading is enforced; no Hugging Face login or download is permitted."
echo "[train] Detailed log: $TRAIN_LOG"
if DATASET_ROOT="$TARGET_DATASET" \
   RUN_SUFFIX="$RUN_SUFFIX" \
   EVALUATION_STATUS_FILE="$STATUS_FILE" \
   DRY_RUN=0 \
   bash "$SCRIPTS_DIR/multi_finetune_evaluation.sh" > "$TRAIN_LOG" 2>&1; then
    :
else
    training_status=$?
    echo "ERROR: training/evaluation pipeline failed (exit $training_status); last 120 log lines follow" >&2
    tail -n 120 "$TRAIN_LOG" >&2 || true
    exit "$training_status"
fi

echo "PIPELINE_COMPLETE"
echo "Dataset: $TARGET_DATASET"
echo "Training suffix: $RUN_SUFFIX"
echo "Training/evaluation status: $STATUS_FILE"
echo "Training log: $TRAIN_LOG"
