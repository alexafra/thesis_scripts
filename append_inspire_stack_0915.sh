#!/usr/bin/env bash
set -Eeuo pipefail

# Incrementally append both 2026-09-15 Inspire collections (stacked red cups
# and wooden-block pick/place) to the already converted 452-episode corpus.
# The base and original sources are never converted or mutated in place.
# Exactly one sealed checkpoint advances from combined-component-ready to
# final-build-ready. Unsealed work is retained on every failure.

export PATH="/home/alex/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

usage() {
    cat <<'EOF'
Usage: append_inspire_stack_0915.sh MODE

Modes:
  check           Read-only source/base/converter preflight.
  build           Goal-stratify and convert both incoming collections, append
                  them to the 452-episode base, validate, detach, and publish.

There is deliberately no implicit mode and no training mode. In particular,
check never starts conversion and build always stops after dataset publication.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ $# -eq 1 ]] || { usage >&2; exit 2; }
MODE="$1"
case "$MODE" in
    check|build) ;;
    -h|--help|help) usage; exit 0 ;;
    *) usage >&2; die "Unknown mode: $MODE" ;;
esac

SCRIPTS_DIR="${SCRIPTS_DIR:-/home/alex/Development/scripts}"
GROOT_DIR="${GROOT_DIR:-/home/alex/Development/Isaac-GR00T}"
GROOT_PYTHON="${GROOT_PYTHON:-$GROOT_DIR/.venv/bin/python}"
UNITREE_PYTHON="${UNITREE_PYTHON:-/home/alex/miniconda3/envs/unitree_lerobot/bin/python}"
DATASET_PARENT="${DATASET_PARENT:-/home/alex/Development/Datasets/lerobot2/inspire}"
BASE_DATASET="${BASE_DATASET:-$DATASET_PARENT/all_tasks_452eps_20260915_normals_range_mask_v2}"
STACK_SOURCE="${STACK_SOURCE:-/home/alex/Development/Datasets/processed_raw/inspire/stack_red_cups_09_15}"
WORDS_SOURCE="${WORDS_SOURCE:-/home/alex/Development/Datasets/processed_raw/inspire/woorden_block_09_15}"
TARGET_DATASET="${TARGET_DATASET:-$DATASET_PARENT/all_tasks_655eps_20260916_normals_range_mask_v2}"
TARGET_NAME="$(basename "$TARGET_DATASET")"
RUN_ID="${RUN_ID:-20260916}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "RUN_ID is not a safe name: $RUN_ID"

WORK_ROOT="$DATASET_PARENT/.${TARGET_NAME}.append-work-${RUN_ID}"
COMPONENT="$WORK_ROOT/stack_red_cups_09_15"
PROVENANCE_STAGE="$WORK_ROOT/provenance"
SOURCE_BUNDLE="$PROVENANCE_STAGE/source_bundle"
BUILD_DATASET="$DATASET_PARENT/.${TARGET_NAME}.build-${RUN_ID}"
CHECKPOINT_ROOT="$DATASET_PARENT/.${TARGET_NAME}.resume-checkpoint"
CHECKPOINT_COMPONENT="$CHECKPOINT_ROOT/stack_red_cups_09_15"
CHECKPOINT_BUILD="$CHECKPOINT_ROOT/final_build"
CHECKPOINT_PROVENANCE="$CHECKPOINT_ROOT/provenance"
CHECKPOINT_SOURCE_BUNDLE="$CHECKPOINT_PROVENANCE/source_bundle"
LOG_ROOT="${LOG_ROOT:-/home/alex/Development/logs}"
PIPELINE_LOG="${PIPELINE_LOG:-$LOG_ROOT/data_pipeline/inspire_stack_0915_append_${RUN_ID}.log}"
LOCK_FILE="${LOCK_FILE:-$LOG_ROOT/data_pipeline/.inspire_incremental_append.lock}"
MIN_FREE_GIB="${MIN_FREE_GIB:-360}"
JOBS="${JOBS:-6}"

CONVERT_SCRIPT="${CONVERT_SCRIPT:-$SCRIPTS_DIR/convert_to_lerobot2.sh}"
SPLIT_SCRIPT="${SPLIT_SCRIPT:-$SCRIPTS_DIR/split_dataset.py}"
APPEND_SCRIPT="${APPEND_SCRIPT:-$SCRIPTS_DIR/append_lerobot2.py}"
SUPPORT_SCRIPT="${SUPPORT_SCRIPT:-$SCRIPTS_DIR/inspire_incremental_append_support.py}"
PROCESS_CHECKER="${PROCESS_CHECKER:-pgrep}"
UV="${UV:-uv}"
MODALITY_CONFIG="${MODALITY_CONFIG:-$GROOT_DIR/examples/UnitreeG1/g1_inspire_headonly_config.py}"

BASE_EPISODES=(364 44 44)
BASE_FRAMES=(85496 9423 10854)
COMPONENT_EPISODES=(161 21 21)
COMPONENT_FRAMES=(110957 13296 13712)
FINAL_EPISODES=(525 65 65)
FINAL_FRAMES=(196453 22719 24566)

safe_remove_checkpoint() {
    local path="$1"
    [[ -e "$path" || -L "$path" ]] || return 0
    [[ "$(dirname "$path")" == "$DATASET_PARENT" ]] || \
        die "Refusing to remove checkpoint outside dataset parent: $path"
    [[ "$(basename "$path")" == ".${TARGET_NAME}.resume-checkpoint" ]] || \
        die "Refusing to remove unexpected checkpoint: $path"
    rm -rf -- "$path"
}

refuse_unsealed_scratch() {
    local path
    local -a conflicts=()
    shopt -s nullglob
    for path in \
        "$DATASET_PARENT/.${TARGET_NAME}.append-work-"* \
        "$DATASET_PARENT/.${TARGET_NAME}.build-"*
    do
        [[ "$(dirname "$path")" == "$DATASET_PARENT" ]] || \
            die "Refusing scratch outside dataset parent: $path"
        conflicts+=("$path")
    done
    shopt -u nullglob
    if (( ${#conflicts[@]} > 0 )); then
        printf 'Existing unsealed append scratch was preserved:\n' >&2
        printf '  %s\n' "${conflicts[@]}" >&2
        die "Refusing a second attempt for $TARGET_DATASET; inspect or remove the listed scratch explicitly"
    fi
}

LOCK_HELD=0
release_lock_and_exit() {
    local status=$?
    trap - EXIT INT TERM
    if [[ "$LOCK_HELD" == 1 ]]; then
        flock -u 9 || true
    fi
    exit "$status"
}
trap release_lock_and_exit EXIT INT TERM

require_tools() {
    [[ -x "$GROOT_PYTHON" ]] || die "Missing GR00T Python: $GROOT_PYTHON"
    [[ -x "$UNITREE_PYTHON" ]] || die "Missing Unitree Python: $UNITREE_PYTHON"
    for path in "$CONVERT_SCRIPT" "$SPLIT_SCRIPT" "$APPEND_SCRIPT" "$SUPPORT_SCRIPT"; do
        [[ -f "$path" ]] || die "Required script is missing: $path"
    done
    [[ -f "$MODALITY_CONFIG" ]] || die "Inspire modality config is missing: $MODALITY_CONFIG"
    command -v "$UV" >/dev/null 2>&1 || die "uv is required for exact GR00T statistics"
    [[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "JOBS must be a positive integer"
    [[ "$MIN_FREE_GIB" =~ ^[1-9][0-9]*$ ]] || die "MIN_FREE_GIB must be positive"
}

require_quiescent_writers() {
    command -v "$PROCESS_CHECKER" >/dev/null 2>&1 || \
        die "Process checker is missing: $PROCESS_CHECKER"
    local active
    active="$("$PROCESS_CHECKER" -af '[t]eleop_hand_and_arm\.py|[d]ata_editor_EN_rgbd\.py|[c]onvert_unitree_json_to_lerobot|[l]aunch_finetune|[e]valuate_checkpoints' || true)"
    if [[ -n "$active" ]]; then
        printf 'Conflicting writer process(es):\n%s\n' "$active" >&2
        die "Stop recording, data editing, conversion, training, and evaluation before this append"
    fi
}

validate_paths() {
    "$GROOT_PYTHON" - "$BASE_DATASET" "$STACK_SOURCE" "$WORDS_SOURCE" "$TARGET_DATASET" "$WORK_ROOT" "$BUILD_DATASET" "$CHECKPOINT_ROOT" <<'PY'
from pathlib import Path
import sys

paths = {name: Path(value).expanduser().resolve() for name, value in zip(
    ("base", "stack_source", "words_source", "target", "work", "build", "checkpoint"),
    sys.argv[1:],
    strict=True,
)}
for left_name, left in paths.items():
    for right_name, right in paths.items():
        if left_name >= right_name:
            continue
        if left == right or left in right.parents or right in left.parents:
            raise SystemExit(
                f"ERROR: {left_name} and {right_name} may not overlap: {left}; {right}"
            )
PY
}

base_and_source_preflight() {
    [[ -d "$BASE_DATASET" ]] || die "Converted 452-episode base is missing: $BASE_DATASET"
    [[ -d "$STACK_SOURCE" ]] || die "Stack source is missing: $STACK_SOURCE"
    [[ -d "$WORDS_SOURCE" ]] || die "Wooden-block source is missing: $WORDS_SOURCE"
    validate_paths
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-source \
        --root "$STACK_SOURCE" --episodes 145 --frames 124575 \
        --goal 'stack the three red cups.'
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-source \
        --root "$WORDS_SOURCE" --episodes 58 --frames 13390 \
        --goal 'pick up the wooden block.' \
        --goal 'put down the wooden block.'
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-base \
        --root "$BASE_DATASET" \
        --episodes "${BASE_EPISODES[@]}" --frames "${BASE_FRAMES[@]}"
    bash "$CONVERT_SCRIPT" \
        --preflight-only \
        --include-surface-normals \
        --surface-normals-encoding-version 2 \
        --end-effector inspire-ftp \
        --camera-calibration-profile d435i-254322071415 \
        "$STACK_SOURCE" stack_red_cups_09_15 "$JOBS"
    bash "$CONVERT_SCRIPT" \
        --preflight-only \
        --include-surface-normals \
        --surface-normals-encoding-version 2 \
        --end-effector inspire-ftp \
        --camera-calibration-profile d435i-254322071415 \
        "$WORDS_SOURCE" woorden_block_09_15 "$JOBS"
}

require_free_gib() {
    local required_gib="$1"
    local available_kib required_kib
    available_kib="$(df -Pk "$DATASET_PARENT" | awk 'NR == 2 {print $4}')"
    required_kib=$((required_gib * 1024 * 1024))
    if (( available_kib < required_kib )); then
        die "At least ${required_gib} GiB free is required; only $((available_kib / 1024 / 1024)) GiB is available"
    fi
    printf 'Disk preflight: %d GiB free (minimum %d GiB).\n' \
        "$((available_kib / 1024 / 1024))" "$required_gib"
}

adopt_orphan_final_build() {
    printf '[resume] Validating final build moved before checkpoint state advancement.\n'
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-orphan-final-build-checkpoint \
        --root "$CHECKPOINT_ROOT" --base "$BASE_DATASET" \
        --source "$CHECKPOINT_SOURCE_BUNDLE"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-component \
        --root "$CHECKPOINT_COMPONENT" --base "$BASE_DATASET" \
        --episodes "${COMPONENT_EPISODES[@]}" --frames "${COMPONENT_FRAMES[@]}"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-collection-order \
        --root "$CHECKPOINT_COMPONENT" \
        --component stack_red_cups_09_15 --component woorden_block_09_15
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-final \
        --base "$BASE_DATASET" --component "$CHECKPOINT_COMPONENT" \
        --output "$CHECKPOINT_BUILD" \
        --episodes "${COMPONENT_EPISODES[@]}" --frames "${COMPONENT_FRAMES[@]}"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-training-stats \
        --dataset-root "$CHECKPOINT_BUILD" --expected-frames "${FINAL_FRAMES[0]}" \
        --report "$CHECKPOINT_BUILD/provenance/incremental_append/stats_finalization_report.json"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
        --root "$STACK_SOURCE" \
        --snapshot "$CHECKPOINT_PROVENANCE/stack_source_tree_snapshot.json"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
        --root "$WORDS_SOURCE" \
        --snapshot "$CHECKPOINT_PROVENANCE/words_source_tree_snapshot.json"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" snapshot-tree \
        --root "$CHECKPOINT_BUILD" \
        --output "$CHECKPOINT_PROVENANCE/build_tree_snapshot.json"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" write-build-checkpoint \
        --root "$CHECKPOINT_ROOT" --base "$BASE_DATASET" \
        --source "$CHECKPOINT_SOURCE_BUNDLE"
    printf '[resume] Adopted the fully revalidated orphan final build.\n'
}

build_dataset() {
    [[ ! -e "$TARGET_DATASET" && ! -L "$TARGET_DATASET" ]] || \
        die "Final target already exists; refusing duplicate append: $TARGET_DATASET"
    require_quiescent_writers
    refuse_unsealed_scratch

    local checkpoint_phase_value="none"
    if [[ -e "$CHECKPOINT_ROOT" || -L "$CHECKPOINT_ROOT" ]]; then
        checkpoint_phase_value="$("$UNITREE_PYTHON" "$SUPPORT_SCRIPT" checkpoint-phase \
            --root "$CHECKPOINT_ROOT")"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" normalize-checkpoint \
            --root "$CHECKPOINT_ROOT"
        if [[ "$checkpoint_phase_value" == component_ready && -e "$CHECKPOINT_BUILD" ]]; then
            adopt_orphan_final_build
            checkpoint_phase_value=final_build_ready
        fi
    fi

    if [[ "$checkpoint_phase_value" == final_build_ready ]]; then
        # The sealed build already owns hard links to every retained component
        # payload, so the component can now be retired before base detachment.
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" retire-component-after-build \
            --root "$CHECKPOINT_ROOT" --base "$BASE_DATASET" \
            --source "$CHECKPOINT_SOURCE_BUNDLE"
        require_free_gib 130
    else
        require_free_gib "$MIN_FREE_GIB"
    fi

    if [[ "$checkpoint_phase_value" == none ]]; then
        mkdir -p -- "$PROVENANCE_STAGE"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" snapshot-tree \
            --root "$BASE_DATASET" --output "$PROVENANCE_STAGE/base_tree_snapshot.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" snapshot-tree \
            --root "$STACK_SOURCE" --output "$PROVENANCE_STAGE/stack_source_tree_snapshot.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" snapshot-tree \
            --root "$WORDS_SOURCE" --output "$PROVENANCE_STAGE/words_source_tree_snapshot.json"

        printf '[bundle] Hard-linking an immutable two-source view without duplicating raw payload.\n'
        mkdir -p -- "$SOURCE_BUNDLE"
        cp -al -- "$STACK_SOURCE" "$SOURCE_BUNDLE/stack_red_cups_09_15"
        cp -al -- "$WORDS_SOURCE" "$SOURCE_BUNDLE/woorden_block_09_15"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
            --root "$STACK_SOURCE" --snapshot "$PROVENANCE_STAGE/stack_source_tree_snapshot.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
            --root "$WORDS_SOURCE" --snapshot "$PROVENANCE_STAGE/words_source_tree_snapshot.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" snapshot-tree \
            --root "$SOURCE_BUNDLE" --output "$PROVENANCE_STAGE/source_tree_snapshot.json"

        printf '[split] Independently goal-stratifying stack then woorden with seed 42.\n'
        "$UNITREE_PYTHON" "$SPLIT_SCRIPT" "$SOURCE_BUNDLE" \
            --strategy goal-stratified --seed 42 --collection-output "$COMPONENT"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-raw-split \
            --root "$COMPONENT" \
            --episodes "${COMPONENT_EPISODES[@]}" --frames "${COMPONENT_FRAMES[@]}"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-collection-order \
            --root "$COMPONENT" \
            --component stack_red_cups_09_15 --component woorden_block_09_15

        printf '[convert] Converting only the combined 203-episode incoming component.\n'
        bash "$CONVERT_SCRIPT" \
            --include-surface-normals \
            --surface-normals-encoding-version 2 \
            --end-effector inspire-ftp \
            --camera-calibration-profile d435i-254322071415 \
            "$COMPONENT" stack_red_cups_and_woorden_block_09_15 "$JOBS"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-component \
            --root "$COMPONENT" --base "$BASE_DATASET" \
            --episodes "${COMPONENT_EPISODES[@]}" --frames "${COMPONENT_FRAMES[@]}"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-collection-order \
            --root "$COMPONENT" \
            --component stack_red_cups_09_15 --component woorden_block_09_15
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" snapshot-tree \
            --root "$COMPONENT" --output "$PROVENANCE_STAGE/component_tree_snapshot.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" write-component-checkpoint \
            --root "$WORK_ROOT" --base "$BASE_DATASET" --source "$CHECKPOINT_SOURCE_BUNDLE"
        [[ ! -e "$CHECKPOINT_ROOT" && ! -L "$CHECKPOINT_ROOT" ]] || \
            die "Resume checkpoint appeared during conversion: $CHECKPOINT_ROOT"
        mv -- "$WORK_ROOT" "$CHECKPOINT_ROOT"
        checkpoint_phase_value=component_ready
        printf '[checkpoint] Sealed the converted 203-episode component for safe reuse.\n'
    fi

    if [[ "$checkpoint_phase_value" == component_ready ]]; then
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-component-checkpoint \
            --root "$CHECKPOINT_ROOT" --base "$BASE_DATASET" \
            --source "$CHECKPOINT_SOURCE_BUNDLE"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-component \
            --root "$CHECKPOINT_COMPONENT" --base "$BASE_DATASET" \
            --episodes "${COMPONENT_EPISODES[@]}" --frames "${COMPONENT_FRAMES[@]}"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-collection-order \
            --root "$CHECKPOINT_COMPONENT" \
            --component stack_red_cups_09_15 --component woorden_block_09_15
        require_free_gib 130

        printf '[reuse] Hard-linking the immutable base into a hidden build only.\n'
        [[ "$(stat -c %d "$BASE_DATASET")" == "$(stat -c %d "$DATASET_PARENT")" ]] || \
            die "Base and build parent must be on the same filesystem"
        cp -al -- "$BASE_DATASET" "$BUILD_DATASET"

        printf '[append] Adopting component media by hard link and appending split-to-matching-split.\n'
        "$UNITREE_PYTHON" "$APPEND_SCRIPT" --link-media \
            "$BUILD_DATASET" "$CHECKPOINT_COMPONENT"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" record-provenance \
            --base "$BASE_DATASET" --component "$CHECKPOINT_COMPONENT" --output "$BUILD_DATASET" \
            --source-snapshot "$CHECKPOINT_PROVENANCE/source_tree_snapshot.json" \
            --base-snapshot "$CHECKPOINT_PROVENANCE/base_tree_snapshot.json" \
            --stack-source-snapshot "$CHECKPOINT_PROVENANCE/stack_source_tree_snapshot.json" \
            --words-source-snapshot "$CHECKPOINT_PROVENANCE/words_source_tree_snapshot.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-final \
            --base "$BASE_DATASET" --component "$CHECKPOINT_COMPONENT" --output "$BUILD_DATASET" \
            --episodes "${COMPONENT_EPISODES[@]}" --frames "${COMPONENT_FRAMES[@]}"

        printf '[stats] Regenerating exact low-dimensional and relative-action statistics for final train only.\n'
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" clear-training-stats \
            --train-root "$BUILD_DATASET/train"
        (
            cd "$GROOT_DIR"
            export HF_HOME="${HF_HOME:-/home/alex/.cache/huggingface}"
            export HF_HUB_OFFLINE=1
            export TRANSFORMERS_OFFLINE=1
            export HF_DATASETS_OFFLINE=1
            export HF_HUB_DISABLE_TELEMETRY=1
            "$UV" run --no-sync python -m gr00t.data.stats \
                --dataset-path "$BUILD_DATASET/train" \
                --embodiment-tag NEW_EMBODIMENT \
                --modality-config-path "$MODALITY_CONFIG"
        )
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-training-stats \
            --dataset-root "$BUILD_DATASET" --expected-frames "${FINAL_FRAMES[0]}" \
            --report "$BUILD_DATASET/provenance/incremental_append/stats_finalization_report.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
            --root "$STACK_SOURCE" \
            --snapshot "$CHECKPOINT_PROVENANCE/stack_source_tree_snapshot.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
            --root "$WORDS_SOURCE" \
            --snapshot "$CHECKPOINT_PROVENANCE/words_source_tree_snapshot.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
            --root "$CHECKPOINT_SOURCE_BUNDLE" \
            --snapshot "$CHECKPOINT_PROVENANCE/source_tree_snapshot.json"

        printf '[checkpoint] Advancing the one checkpoint to the validated final build.\n'
        [[ ! -e "$CHECKPOINT_BUILD" && ! -L "$CHECKPOINT_BUILD" ]] || \
            die "Unexpected build payload already exists in checkpoint"
        mv -- "$BUILD_DATASET" "$CHECKPOINT_BUILD"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" snapshot-tree \
            --root "$CHECKPOINT_BUILD" \
            --output "$CHECKPOINT_PROVENANCE/build_tree_snapshot.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" write-build-checkpoint \
            --root "$CHECKPOINT_ROOT" --base "$BASE_DATASET" \
            --source "$CHECKPOINT_SOURCE_BUNDLE"
        checkpoint_phase_value=final_build_ready
    fi

    [[ "$checkpoint_phase_value" == final_build_ready ]] || \
        die "Unexpected checkpoint phase: $checkpoint_phase_value"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" retire-component-after-build \
        --root "$CHECKPOINT_ROOT" --base "$BASE_DATASET" \
        --source "$CHECKPOINT_SOURCE_BUNDLE"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-retired-build-checkpoint \
        --root "$CHECKPOINT_ROOT" --base "$BASE_DATASET" \
        --source "$CHECKPOINT_SOURCE_BUNDLE"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-final \
        --base "$BASE_DATASET" --output "$CHECKPOINT_BUILD" \
        --episodes "${COMPONENT_EPISODES[@]}" --frames "${COMPONENT_FRAMES[@]}"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-training-stats \
        --dataset-root "$CHECKPOINT_BUILD" --expected-frames "${FINAL_FRAMES[0]}" \
        --report "$CHECKPOINT_BUILD/provenance/incremental_append/stats_finalization_report.json"
    require_free_gib 130

    printf '[detach] Making every retained base file physically independent before publication.\n'
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" detach-and-verify \
        --base "$BASE_DATASET" --output "$CHECKPOINT_BUILD" \
        --snapshot "$CHECKPOINT_BUILD/provenance/incremental_append/base_tree_snapshot.json" \
        --report "$CHECKPOINT_BUILD/provenance/incremental_append/detach_report.json"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
        --root "$BASE_DATASET" \
        --snapshot "$CHECKPOINT_BUILD/provenance/incremental_append/base_tree_snapshot.json"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-final \
        --base "$BASE_DATASET" --output "$CHECKPOINT_BUILD" --require-independent \
        --episodes "${COMPONENT_EPISODES[@]}" --frames "${COMPONENT_FRAMES[@]}"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-training-stats \
        --dataset-root "$CHECKPOINT_BUILD" --expected-frames "${FINAL_FRAMES[0]}" \
        --report "$CHECKPOINT_BUILD/provenance/incremental_append/stats_finalization_report.json"

    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" publish-no-replace \
        --build "$CHECKPOINT_BUILD" --target "$TARGET_DATASET"
    safe_remove_checkpoint "$CHECKPOINT_ROOT"
    printf 'DATASET_PUBLISHED=%s\n' "$TARGET_DATASET"
    printf 'Final population: train=%d/%d, validation=%d/%d, test=%d/%d\n' \
        "${FINAL_EPISODES[0]}" "${FINAL_FRAMES[0]}" \
        "${FINAL_EPISODES[1]}" "${FINAL_FRAMES[1]}" \
        "${FINAL_EPISODES[2]}" "${FINAL_FRAMES[2]}"
}

require_tools
mkdir -p -- "$LOG_ROOT/data_pipeline"
exec 9>"$LOCK_FILE"
flock -n 9 || die "Another Inspire append pipeline owns $LOCK_FILE"
LOCK_HELD=1

if [[ "$MODE" == build ]]; then
    exec > >(tee -a "$PIPELINE_LOG") 2>&1
fi

case "$MODE" in
    check)
        require_quiescent_writers
        base_and_source_preflight
        printf 'CHECK_COMPLETE: incremental append is ready; nothing was converted or trained.\n'
        ;;
    build)
        require_quiescent_writers
        base_and_source_preflight
        build_dataset
        ;;
esac
