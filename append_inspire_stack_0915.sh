#!/usr/bin/env bash
set -Eeuo pipefail

# Incrementally append both 2026-09-16 Inspire collections (toothpaste
# pick/place and left-to-right cup pyramids) to the converted 655-episode corpus.
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
                  them to the 655-episode base, validate, detach, and publish.

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
BASE_DATASET="${BASE_DATASET:-$DATASET_PARENT/all_tasks_655eps_20260916_normals_range_mask_v2}"
PYRAMID_SOURCE="${PYRAMID_SOURCE:-/home/alex/Development/Datasets/processed_raw/inspire/cup_pyramid_09_16}"
TOOTHPASTE_SOURCE="${TOOTHPASTE_SOURCE:-/home/alex/Development/Datasets/processed_raw/inspire/toothpaste_09_16}"
TARGET_DATASET="${TARGET_DATASET:-$DATASET_PARENT/all_tasks_713eps_20260917_normals_range_mask_v2}"
TARGET_NAME="$(basename "$TARGET_DATASET")"
RUN_ID="${RUN_ID:-20260917}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "RUN_ID is not a safe name: $RUN_ID"

COMPONENT_NAME="cup_pyramid_and_toothpaste_09_16"
APPEND_GENERATION_ID="20260917_cup_pyramid_and_toothpaste_09_16"
APPEND_PROVENANCE_RELATIVE="provenance/incremental_append_generations/$APPEND_GENERATION_ID"
export INSPIRE_APPEND_CHECKPOINT_COMPONENT="$COMPONENT_NAME"
export INSPIRE_APPEND_PROVENANCE_RELATIVE="$APPEND_PROVENANCE_RELATIVE"
WORK_ROOT="$DATASET_PARENT/.${TARGET_NAME}.append-work-${RUN_ID}"
COMPONENT="$WORK_ROOT/$COMPONENT_NAME"
PROVENANCE_STAGE="$WORK_ROOT/provenance"
SOURCE_BUNDLE="$PROVENANCE_STAGE/source_bundle"
BUILD_DATASET="$DATASET_PARENT/.${TARGET_NAME}.build-${RUN_ID}"
CHECKPOINT_ROOT="$DATASET_PARENT/.${TARGET_NAME}.resume-checkpoint"
CHECKPOINT_COMPONENT="$CHECKPOINT_ROOT/$COMPONENT_NAME"
CHECKPOINT_BUILD="$CHECKPOINT_ROOT/final_build"
CHECKPOINT_PROVENANCE="$CHECKPOINT_ROOT/provenance"
CHECKPOINT_SOURCE_BUNDLE="$CHECKPOINT_PROVENANCE/source_bundle"
CHECKPOINT_APPEND_PROVENANCE="$CHECKPOINT_BUILD/$APPEND_PROVENANCE_RELATIVE"
LOG_ROOT="${LOG_ROOT:-/home/alex/Development/logs}"
PIPELINE_LOG="${PIPELINE_LOG:-$LOG_ROOT/data_pipeline/inspire_toothpaste_pyramid_0916_append_${RUN_ID}.log}"
LOCK_FILE="${LOCK_FILE:-$LOG_ROOT/data_pipeline/.inspire_incremental_append.lock}"
MIN_FREE_GIB="${MIN_FREE_GIB:-360}"
DETACH_HEADROOM_GIB="${DETACH_HEADROOM_GIB:-32}"
JOBS="${JOBS:-6}"

CONVERT_SCRIPT="${CONVERT_SCRIPT:-$SCRIPTS_DIR/convert_to_lerobot2.sh}"
SPLIT_SCRIPT="${SPLIT_SCRIPT:-$SCRIPTS_DIR/split_dataset.py}"
APPEND_SCRIPT="${APPEND_SCRIPT:-$SCRIPTS_DIR/append_lerobot2.py}"
SUPPORT_SCRIPT="${SUPPORT_SCRIPT:-$SCRIPTS_DIR/inspire_incremental_append_support.py}"
PROCESS_CHECKER="${PROCESS_CHECKER:-pgrep}"
UV="${UV:-uv}"
MODALITY_CONFIG="${MODALITY_CONFIG:-$GROOT_DIR/examples/UnitreeG1/g1_inspire_headonly_config.py}"

BASE_EPISODES=(525 65 65)
BASE_FRAMES=(196453 22719 24566)
COMPONENT_EPISODES=(46 6 6)
COMPONENT_FRAMES=(35383 5540 4196)
FINAL_EPISODES=(571 71 71)
FINAL_FRAMES=(231836 28259 28762)

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
    [[ "$DETACH_HEADROOM_GIB" =~ ^[1-9][0-9]*$ ]] || \
        die "DETACH_HEADROOM_GIB must be positive"
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
    "$GROOT_PYTHON" - "$BASE_DATASET" "$PYRAMID_SOURCE" "$TOOTHPASTE_SOURCE" "$TARGET_DATASET" "$WORK_ROOT" "$BUILD_DATASET" "$CHECKPOINT_ROOT" <<'PY'
from pathlib import Path
import sys

paths = {name: Path(value).expanduser().resolve() for name, value in zip(
    ("base", "pyramid_source", "toothpaste_source", "target", "work", "build", "checkpoint"),
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
    [[ -d "$BASE_DATASET" ]] || die "Converted 655-episode base is missing: $BASE_DATASET"
    [[ -d "$PYRAMID_SOURCE" ]] || die "Cup-pyramid source is missing: $PYRAMID_SOURCE"
    [[ -d "$TOOTHPASTE_SOURCE" ]] || die "Toothpaste source is missing: $TOOTHPASTE_SOURCE"
    validate_paths
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-source \
        --root "$PYRAMID_SOURCE" --episodes 28 --frames 36628 \
        --goal 'build a cup pyramid left-to-right.'
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-source \
        --root "$TOOTHPASTE_SOURCE" --episodes 30 --frames 8491 \
        --goal 'pick up the cylinder toothpaste.' \
        --goal 'put down the cylinder toothpaste.'
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-base \
        --root "$BASE_DATASET" \
        --episodes "${BASE_EPISODES[@]}" --frames "${BASE_FRAMES[@]}"
    bash "$CONVERT_SCRIPT" \
        --preflight-only \
        --include-surface-normals \
        --surface-normals-encoding-version 2 \
        --end-effector inspire-ftp \
        --camera-calibration-profile d435i-254322071415 \
        "$PYRAMID_SOURCE" cup_pyramid_09_16 "$JOBS"
    bash "$CONVERT_SCRIPT" \
        --preflight-only \
        --include-surface-normals \
        --surface-normals-encoding-version 2 \
        --end-effector inspire-ftp \
        --camera-calibration-profile d435i-254322071415 \
        "$TOOTHPASTE_SOURCE" toothpaste_09_16 "$JOBS"
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

require_detach_space() {
    local available_kib base_kib required_kib required_gib
    available_kib="$(df -Pk "$DATASET_PARENT" | awk 'NR == 2 {print $4}')"
    base_kib="$(du -sk -- "$BASE_DATASET" | awk '{print $1}')"
    required_kib=$((base_kib + DETACH_HEADROOM_GIB * 1024 * 1024))
    required_gib=$(((required_kib + 1024 * 1024 - 1) / (1024 * 1024)))
    if (( available_kib < required_kib )); then
        die "Detaching the base requires ${required_gib} GiB free (allocated base plus ${DETACH_HEADROOM_GIB} GiB headroom); only $((available_kib / 1024 / 1024)) GiB is available"
    fi
    printf 'Detach disk preflight: %d GiB free (minimum %d GiB: allocated base plus %d GiB headroom).\n' \
        "$((available_kib / 1024 / 1024))" "$required_gib" "$DETACH_HEADROOM_GIB"
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
        --component cup_pyramid_09_16 --component toothpaste_09_16
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-final \
        --base "$BASE_DATASET" --component "$CHECKPOINT_COMPONENT" \
        --output "$CHECKPOINT_BUILD" \
        --episodes "${COMPONENT_EPISODES[@]}" --frames "${COMPONENT_FRAMES[@]}"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-training-stats \
        --dataset-root "$CHECKPOINT_BUILD" --expected-frames "${FINAL_FRAMES[0]}" \
        --report "$CHECKPOINT_APPEND_PROVENANCE/stats_finalization_report.json"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
        --root "$PYRAMID_SOURCE" \
        --snapshot "$CHECKPOINT_PROVENANCE/pyramid_source_tree_snapshot.json"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
        --root "$TOOTHPASTE_SOURCE" \
        --snapshot "$CHECKPOINT_PROVENANCE/toothpaste_source_tree_snapshot.json"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" snapshot-tree \
        --root "$CHECKPOINT_BUILD" \
        --output "$CHECKPOINT_PROVENANCE/build_tree_snapshot.json"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" write-build-checkpoint \
        --root "$CHECKPOINT_ROOT" --base "$BASE_DATASET" \
        --source "$CHECKPOINT_SOURCE_BUNDLE"
    printf '[resume] Adopted the fully revalidated orphan final build.\n'
}

recover_published_target() {
    [[ -d "$TARGET_DATASET" && ! -L "$TARGET_DATASET" ]] || \
        die "Existing target is not a real directory: $TARGET_DATASET"
    [[ -d "$CHECKPOINT_ROOT" && ! -L "$CHECKPOINT_ROOT" ]] || \
        die "Final target exists without its sealed cleanup checkpoint; preserving it for inspection"
    [[ ! -e "$CHECKPOINT_BUILD" && ! -L "$CHECKPOINT_BUILD" ]] || \
        die "Both final target and checkpoint build exist; refusing ambiguous recovery"
    local phase
    phase="$($UNITREE_PYTHON "$SUPPORT_SCRIPT" checkpoint-phase --root "$CHECKPOINT_ROOT")"
    [[ "$phase" == publish_ready ]] || \
        die "Final target exists but checkpoint phase is $phase, not publish_ready"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-published-target-checkpoint \
        --root "$CHECKPOINT_ROOT" --base "$BASE_DATASET" \
        --source "$CHECKPOINT_SOURCE_BUNDLE" --target "$TARGET_DATASET"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-final \
        --base "$BASE_DATASET" --output "$TARGET_DATASET" --require-independent \
        --episodes "${COMPONENT_EPISODES[@]}" --frames "${COMPONENT_FRAMES[@]}"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-training-stats \
        --dataset-root "$TARGET_DATASET" --expected-frames "${FINAL_FRAMES[0]}" \
        --report "$TARGET_DATASET/$APPEND_PROVENANCE_RELATIVE/stats_finalization_report.json" \
        --read-only
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
        --root "$PYRAMID_SOURCE" \
        --snapshot "$CHECKPOINT_PROVENANCE/pyramid_source_tree_snapshot.json"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
        --root "$TOOTHPASTE_SOURCE" \
        --snapshot "$CHECKPOINT_PROVENANCE/toothpaste_source_tree_snapshot.json"
    safe_remove_checkpoint "$CHECKPOINT_ROOT"
    printf '[resume] Verified already-published target and removed only its sealed cleanup checkpoint.\n'
    printf 'DATASET_PUBLISHED=%s\n' "$TARGET_DATASET"
}

build_dataset() {
    require_quiescent_writers
    refuse_unsealed_scratch
    if [[ -e "$TARGET_DATASET" || -L "$TARGET_DATASET" ]]; then
        recover_published_target
        return
    fi

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

    if [[ "$checkpoint_phase_value" == publish_ready ]]; then
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-publish-ready-checkpoint \
            --root "$CHECKPOINT_ROOT" --base "$BASE_DATASET" \
            --source "$CHECKPOINT_SOURCE_BUNDLE"
    elif [[ "$checkpoint_phase_value" == final_build_ready ]]; then
        # The sealed build already owns hard links to every retained component
        # payload, so the component can now be retired before base detachment.
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" retire-component-after-build \
            --root "$CHECKPOINT_ROOT" --base "$BASE_DATASET" \
            --source "$CHECKPOINT_SOURCE_BUNDLE"
        require_detach_space
    else
        require_free_gib "$MIN_FREE_GIB"
    fi

    if [[ "$checkpoint_phase_value" == none ]]; then
        mkdir -p -- "$PROVENANCE_STAGE"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" snapshot-tree \
            --root "$BASE_DATASET" --output "$PROVENANCE_STAGE/base_tree_snapshot.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" snapshot-tree \
            --root "$PYRAMID_SOURCE" --output "$PROVENANCE_STAGE/pyramid_source_tree_snapshot.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" snapshot-tree \
            --root "$TOOTHPASTE_SOURCE" --output "$PROVENANCE_STAGE/toothpaste_source_tree_snapshot.json"

        printf '[bundle] Hard-linking an immutable two-source view without duplicating raw payload.\n'
        mkdir -p -- "$SOURCE_BUNDLE"
        cp -al -- "$PYRAMID_SOURCE" "$SOURCE_BUNDLE/cup_pyramid_09_16"
        cp -al -- "$TOOTHPASTE_SOURCE" "$SOURCE_BUNDLE/toothpaste_09_16"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
            --root "$PYRAMID_SOURCE" --snapshot "$PROVENANCE_STAGE/pyramid_source_tree_snapshot.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
            --root "$TOOTHPASTE_SOURCE" --snapshot "$PROVENANCE_STAGE/toothpaste_source_tree_snapshot.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" snapshot-tree \
            --root "$SOURCE_BUNDLE" --output "$PROVENANCE_STAGE/source_tree_snapshot.json"

        printf '[split] Independently goal-stratifying cup pyramid then toothpaste with seed 42.\n'
        "$UNITREE_PYTHON" "$SPLIT_SCRIPT" "$SOURCE_BUNDLE" \
            --strategy goal-stratified --seed 42 --collection-output "$COMPONENT"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-raw-split \
            --root "$COMPONENT" \
            --episodes "${COMPONENT_EPISODES[@]}" --frames "${COMPONENT_FRAMES[@]}"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-collection-order \
            --root "$COMPONENT" \
            --component cup_pyramid_09_16 --component toothpaste_09_16

        printf '[convert] Converting only the combined 58-episode incoming component.\n'
        bash "$CONVERT_SCRIPT" \
            --include-surface-normals \
            --surface-normals-encoding-version 2 \
            --end-effector inspire-ftp \
            --camera-calibration-profile d435i-254322071415 \
            "$COMPONENT" cup_pyramid_and_toothpaste_09_16 "$JOBS"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-component \
            --root "$COMPONENT" --base "$BASE_DATASET" \
            --episodes "${COMPONENT_EPISODES[@]}" --frames "${COMPONENT_FRAMES[@]}"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-collection-order \
            --root "$COMPONENT" \
            --component cup_pyramid_09_16 --component toothpaste_09_16
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" snapshot-tree \
            --root "$COMPONENT" --output "$PROVENANCE_STAGE/component_tree_snapshot.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" write-component-checkpoint \
            --root "$WORK_ROOT" --base "$BASE_DATASET" --source "$CHECKPOINT_SOURCE_BUNDLE"
        [[ ! -e "$CHECKPOINT_ROOT" && ! -L "$CHECKPOINT_ROOT" ]] || \
            die "Resume checkpoint appeared during conversion: $CHECKPOINT_ROOT"
        mv -- "$WORK_ROOT" "$CHECKPOINT_ROOT"
        checkpoint_phase_value=component_ready
        printf '[checkpoint] Sealed the converted 58-episode component for safe reuse.\n'
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
            --component cup_pyramid_09_16 --component toothpaste_09_16
        require_detach_space

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
            --source-component "cup_pyramid_09_16=$CHECKPOINT_PROVENANCE/pyramid_source_tree_snapshot.json" \
            --source-component "toothpaste_09_16=$CHECKPOINT_PROVENANCE/toothpaste_source_tree_snapshot.json"
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
            --report "$BUILD_DATASET/$APPEND_PROVENANCE_RELATIVE/stats_finalization_report.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
            --root "$PYRAMID_SOURCE" \
            --snapshot "$CHECKPOINT_PROVENANCE/pyramid_source_tree_snapshot.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
            --root "$TOOTHPASTE_SOURCE" \
            --snapshot "$CHECKPOINT_PROVENANCE/toothpaste_source_tree_snapshot.json"
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

    if [[ "$checkpoint_phase_value" == final_build_ready ]]; then
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
            --report "$CHECKPOINT_APPEND_PROVENANCE/stats_finalization_report.json"
        require_detach_space

        printf '[detach] Making every retained base file physically independent before publication.\n'
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" detach-and-verify \
            --base "$BASE_DATASET" --output "$CHECKPOINT_BUILD" \
            --snapshot "$CHECKPOINT_APPEND_PROVENANCE/base_tree_snapshot.json" \
            --report "$CHECKPOINT_APPEND_PROVENANCE/detach_report.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
            --root "$BASE_DATASET" \
            --snapshot "$CHECKPOINT_APPEND_PROVENANCE/base_tree_snapshot.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-final \
            --base "$BASE_DATASET" --output "$CHECKPOINT_BUILD" --require-independent \
            --episodes "${COMPONENT_EPISODES[@]}" --frames "${COMPONENT_FRAMES[@]}"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-training-stats \
            --dataset-root "$CHECKPOINT_BUILD" --expected-frames "${FINAL_FRAMES[0]}" \
            --report "$CHECKPOINT_APPEND_PROVENANCE/stats_finalization_report.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" snapshot-tree \
            --root "$CHECKPOINT_BUILD" \
            --output "$CHECKPOINT_PROVENANCE/published_tree_snapshot.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" write-publish-ready-checkpoint \
            --root "$CHECKPOINT_ROOT" --base "$BASE_DATASET" \
            --source "$CHECKPOINT_SOURCE_BUNDLE"
        checkpoint_phase_value=publish_ready
    fi

    [[ "$checkpoint_phase_value" == publish_ready ]] || \
        die "Unexpected checkpoint phase: $checkpoint_phase_value"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-publish-ready-checkpoint \
        --root "$CHECKPOINT_ROOT" --base "$BASE_DATASET" \
        --source "$CHECKPOINT_SOURCE_BUNDLE"

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
