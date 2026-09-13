#!/usr/bin/env bash
set -Eeuo pipefail

SOURCE_PROCESSED="${SOURCE_PROCESSED:-/home/alex/Development/Datasets/processed_raw/pick_three_cups_1408}"
RAW_SOURCE="${RAW_SOURCE:-/home/alex/Development/Datasets/raw/pick_three_cups_1408}"
RAW_ARCHIVE="${RAW_ARCHIVE:-/home/alex/Development/Datasets/raw/pick_three_cups_1408.zip}"
BASE_DATASET="${BASE_DATASET:-/home/alex/Development/Datasets/lerobot2/atomic_combined_09_08_And_10_08}"
OUTPUT_DATASET="${OUTPUT_DATASET:-/home/alex/Development/Datasets/lerobot2/atomic_combined_09_08_And_10_08_plus_pick_three_cups_1408}"
SCRIPTS_DIR="/home/alex/Development/scripts"
UNITREE_DIR="/home/alex/Development/unitree_lerobot"
UNITREE_PYTHON="/home/alex/miniconda3/envs/unitree_lerobot/bin/python"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%d)}"
WORK_ROOT="${WORK_ROOT:-/home/alex/Development/Datasets/lerobot2/.pick_three_cups_1408_pipeline_${RUN_ID}}"
INCOMING="$WORK_ROOT/pick_three_cups_1408"
LOG_ROOT="${LOG_ROOT:-/home/alex/Development/logs/data_pipeline}"
mkdir -p "$LOG_ROOT"
LOG_FILE="${LOG_FILE:-$LOG_ROOT/add_three_cups_and_train_${RUN_ID}.log}"
LOCK_FILE="/home/alex/Development/Datasets/lerobot2/.add_three_cups_and_train.lock"
RUN_SUFFIX="${RUN_SUFFIX:-three_cups_1408_${RUN_ID}}"

exec > >(tee -a "$LOG_FILE") 2>&1
exec 9>"$LOCK_FILE"
flock -n 9 || { echo "ERROR: another three-cups pipeline owns $LOCK_FILE" >&2; exit 1; }

echo "RUN_ID=$RUN_ID"
echo "SOURCE_PROCESSED=$SOURCE_PROCESSED"
echo "BASE_DATASET=$BASE_DATASET"
echo "OUTPUT_DATASET=$OUTPUT_DATASET"
echo "WORK_ROOT=$WORK_ROOT"
echo "LOG_FILE=$LOG_FILE"

[[ -d "$SOURCE_PROCESSED" ]] || { echo "ERROR: missing processed source: $SOURCE_PROCESSED" >&2; exit 1; }
[[ -d "$RAW_SOURCE" || -f "$RAW_ARCHIVE" ]] || {
    echo "ERROR: neither raw source nor archive exists" >&2
    exit 1
}
[[ -f "$BASE_DATASET/train/meta/info.json" ]] || { echo "ERROR: base dataset is missing" >&2; exit 1; }
[[ ! -e "$OUTPUT_DATASET" ]] || {
    echo "ERROR: output already exists; refusing a possible duplicate append: $OUTPUT_DATASET" >&2
    exit 1
}
[[ ! -e "$WORK_ROOT" ]] || { echo "ERROR: work root already exists: $WORK_ROOT" >&2; exit 1; }

SOURCE_EPISODES=$(find "$SOURCE_PROCESSED" -mindepth 1 -maxdepth 1 -type d -name 'episode_*' | wc -l)
[[ "$SOURCE_EPISODES" -ge 3 ]] || { echo "ERROR: only $SOURCE_EPISODES processed episodes" >&2; exit 1; }
echo "Accepted processed episodes: $SOURCE_EPISODES"

if [[ ! -f "$RAW_ARCHIVE" ]]; then
    echo "[archive] Creating stored ZIP of immutable raw source..."
    archive_tmp="${RAW_ARCHIVE%.zip}.partial-${RUN_ID}.zip"
    rm -f "$archive_tmp"
    (
        cd "$(dirname "$RAW_SOURCE")"
        zip -0 -q -r "$archive_tmp" "$(basename "$RAW_SOURCE")"
    )
    unzip -tq "$archive_tmp"
    mv "$archive_tmp" "$RAW_ARCHIVE"
    sha256sum "$RAW_ARCHIVE" > "${RAW_ARCHIVE}.sha256"
else
    echo "[archive] Existing raw ZIP found; testing it..."
    unzip -tq "$RAW_ARCHIVE"
    if [[ -f "${RAW_ARCHIVE}.sha256" ]]; then
        sha256sum -c "${RAW_ARCHIVE}.sha256"
    else
        sha256sum "$RAW_ARCHIVE" > "${RAW_ARCHIVE}.sha256"
    fi
fi

echo "[copy] Making disposable conversion copy..."
mkdir -p "$WORK_ROOT"
cp -a --reflink=auto "$SOURCE_PROCESSED" "$INCOMING"

COPIED_EPISODES=$(find "$INCOMING" -mindepth 1 -maxdepth 1 -type d -name 'episode_*' | wc -l)
[[ "$COPIED_EPISODES" == "$SOURCE_EPISODES" ]] || {
    echo "ERROR: source changed during copy ($SOURCE_EPISODES -> $COPIED_EPISODES episodes)" >&2
    exit 1
}

echo "[normalize] Normalizing copied processed-raw task text..."
(
    cd "$UNITREE_DIR"
    "$UNITREE_PYTHON" -m unitree_lerobot.utils.normalize_lerobot_tasks "$INCOMING" --apply --discard-backup
)

echo "[split] Goal-stratified 80/10/10 split, seed 42..."
"$UNITREE_PYTHON" "$SCRIPTS_DIR/split_dataset.py" "$INCOMING" --strategy goal-stratified --seed 42

echo "[convert] Converting all three splits with lossless gray depth and surface normals..."
bash "$SCRIPTS_DIR/convert_to_lerobot2.sh" --include-surface-normals "$INCOMING" pick_three_cups_1408 6

echo "[append] Building a new combined dataset; the old combined dataset remains untouched..."
"$UNITREE_PYTHON" "$SCRIPTS_DIR/append_lerobot2.py" "$OUTPUT_DATASET" "$BASE_DATASET" "$INCOMING"

echo "[normalize] Sharing one canonical task table across combined splits..."
(
    cd "$UNITREE_DIR"
    "$UNITREE_PYTHON" -m unitree_lerobot.utils.normalize_lerobot_tasks "$OUTPUT_DATASET" --apply --discard-backup
)

echo "[validate] Checking expected split counts, modalities, and task count..."
"$UNITREE_PYTHON" - "$BASE_DATASET" "$INCOMING" "$OUTPUT_DATASET" <<'PY'
import json
import sys
from pathlib import Path

base, incoming, output = map(Path, sys.argv[1:])
required = {
    "observation.images.ego_view",
    "observation.images.depth_gray_view",
    "observation.images.surface_normals_view",
}
summary = {}
for split in ("train", "test", "validation"):
    infos = []
    for root in (base, incoming, output):
        path = root / split / "meta" / "info.json"
        if not path.is_file():
            raise SystemExit(f"ERROR: missing {path}")
        infos.append(json.loads(path.read_text()))
    old, new, combined = infos
    for key in ("total_episodes", "total_frames"):
        expected = int(old[key]) + int(new[key])
        actual = int(combined[key])
        if actual != expected:
            raise SystemExit(f"ERROR: {split} {key}: expected {expected}, got {actual}")
    if int(combined["total_tasks"]) != 9:
        raise SystemExit(f"ERROR: {split} expected 9 tasks, got {combined['total_tasks']}")
    missing = required.difference(combined.get("features", {}))
    if missing:
        raise SystemExit(f"ERROR: {split} missing modalities: {sorted(missing)}")
    summary[split] = {
        "episodes": int(combined["total_episodes"]),
        "frames": int(combined["total_frames"]),
        "tasks": int(combined["total_tasks"]),
    }
(output / "three_cups_addition_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(summary, indent=2))
PY

echo "[cleanup] Removing the disposable converted source copy..."
case "$WORK_ROOT" in
    /home/alex/Development/Datasets/lerobot2/.pick_three_cups_1408_pipeline_*)
        rm -rf -- "$WORK_ROOT"
        ;;
    *)
        echo "ERROR: refusing to remove unexpected work path: $WORK_ROOT" >&2
        exit 1
        ;;
esac

echo "[train] Regenerating stats, then training/evaluating the three 25k models..."
DATASET_ROOT="$OUTPUT_DATASET" RUN_SUFFIX="$RUN_SUFFIX" \
    bash "$SCRIPTS_DIR/multi_finetune_evaluation.sh"

echo "PIPELINE_COMPLETE"
echo "Dataset: $OUTPUT_DATASET"
echo "Training suffix: $RUN_SUFFIX"
echo "Disposable work tree removed: $WORK_ROOT"
