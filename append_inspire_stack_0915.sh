#!/usr/bin/env bash
set -Eeuo pipefail

# Incrementally append one known pair of Inspire collections to its converted
# base corpus. Presets below pin every source, count, goal, and output identity.
# The base and original sources are never converted or mutated in place.
# Exactly one sealed checkpoint advances from combined-component-ready to
# final-build-ready. Unsealed work is retained on every failure.

export PATH="/home/alex/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

usage() {
    cat <<'EOF'
Usage: append_inspire_stack_0915.sh MODE [PRESET]

Modes:
  check           Read-only source/base/converter preflight.
  prepare         Preflight, split, convert, and seal only the incoming
                  component. This may run while model training reads the base.
  build           Resume a prepared component (or prepare it if absent), append
                  it to the preset base, regenerate train stats, and publish.

Presets:
  toothpaste-pyramid-0916   655 -> 713 (default; existing behavior)
  cereal-pyramid-0917       713 -> 856 (cereal box + cup_pyramid_09_16_02)

There is deliberately no implicit mode and no training mode. In particular,
check never starts conversion, prepare never appends, and build always stops
after dataset publication.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

(( $# >= 1 && $# <= 2 )) || { usage >&2; exit 2; }
MODE="$1"
case "$MODE" in
    check|prepare|build) ;;
    -h|--help|help) usage; exit 0 ;;
    *) usage >&2; die "Unknown mode: $MODE" ;;
esac
PRESET="${2:-${INSPIRE_APPEND_PRESET:-toothpaste-pyramid-0916}}"

SCRIPTS_DIR="${SCRIPTS_DIR:-/home/alex/Development/scripts}"
GROOT_DIR="${GROOT_DIR:-/home/alex/Development/Isaac-GR00T}"
GROOT_PYTHON="${GROOT_PYTHON:-$GROOT_DIR/.venv/bin/python}"
UNITREE_PYTHON="${UNITREE_PYTHON:-/home/alex/miniconda3/envs/unitree_lerobot/bin/python}"
DATASET_PARENT="${DATASET_PARENT:-/home/alex/Development/Datasets/lerobot2/inspire}"

case "$PRESET" in
    toothpaste-pyramid-0916)
        BASE_DATASET="${BASE_DATASET:-$DATASET_PARENT/all_tasks_655eps_20260916_normals_range_mask_v2}"
        SOURCE_A="${SOURCE_A:-${PYRAMID_SOURCE:-/home/alex/Development/Datasets/processed_raw/inspire/cup_pyramid_09_16}}"
        SOURCE_B="${SOURCE_B:-${TOOTHPASTE_SOURCE:-/home/alex/Development/Datasets/processed_raw/inspire/toothpaste_09_16}}"
        TARGET_DATASET="${TARGET_DATASET:-$DATASET_PARENT/all_tasks_713eps_20260917_normals_range_mask_v2}"
        SOURCE_A_NAME="cup_pyramid_09_16"
        SOURCE_B_NAME="toothpaste_09_16"
        SOURCE_A_EPISODES=28
        SOURCE_A_FRAMES=36628
        SOURCE_A_GOALS=('build a cup pyramid left-to-right.')
        SOURCE_B_EPISODES=30
        SOURCE_B_FRAMES=8491
        SOURCE_B_GOALS=('pick up the cylinder toothpaste.' 'put down the cylinder toothpaste.')
        SOURCE_A_SNAPSHOT_NAME="pyramid_source_tree_snapshot.json"
        SOURCE_B_SNAPSHOT_NAME="toothpaste_source_tree_snapshot.json"
        COMPONENT_NAME="cup_pyramid_and_toothpaste_09_16"
        APPEND_GENERATION_ID="20260917_cup_pyramid_and_toothpaste_09_16"
        DEFAULT_RUN_ID="20260917"
        DEFAULT_PIPELINE_LOG_STEM="inspire_toothpaste_pyramid_0916_append"
        BASE_EPISODES=(525 65 65)
        BASE_FRAMES=(196453 22719 24566)
        COMPONENT_EPISODES=(46 6 6)
        COMPONENT_FRAMES=(35383 5540 4196)
        FINAL_EPISODES=(571 71 71)
        FINAL_FRAMES=(231836 28259 28762)
        ;;
    cereal-pyramid-0917)
        BASE_DATASET="${BASE_DATASET:-$DATASET_PARENT/all_tasks_713eps_20260917_normals_range_mask_v2}"
        SOURCE_A="${SOURCE_A:-${CEREAL_SOURCE:-/home/alex/Development/Datasets/processed_raw/inspire/cereal_box_09_16}}"
        SOURCE_B="${SOURCE_B:-${PYRAMID_02_SOURCE:-/home/alex/Development/Datasets/processed_raw/inspire/cup_pyramid_09_16_02}}"
        TARGET_DATASET="${TARGET_DATASET:-$DATASET_PARENT/all_tasks_856eps_20260918_normals_range_mask_v2}"
        SOURCE_A_NAME="cereal_box_09_16"
        SOURCE_B_NAME="cup_pyramid_09_16_02"
        SOURCE_A_EPISODES=88
        SOURCE_A_FRAMES=28360
        SOURCE_A_GOALS=('pick up the cereal box.' 'put down the cereal box.')
        SOURCE_B_EPISODES=55
        SOURCE_B_FRAMES=50466
        SOURCE_B_GOALS=('build a cup pyramid left-to-right.')
        SOURCE_A_SNAPSHOT_NAME="cereal_box_09_16_source_tree_snapshot.json"
        SOURCE_B_SNAPSHOT_NAME="cup_pyramid_09_16_02_source_tree_snapshot.json"
        COMPONENT_NAME="cereal_box_and_cup_pyramid_09_16_02"
        APPEND_GENERATION_ID="20260918_cereal_box_and_cup_pyramid_09_16_02"
        DEFAULT_RUN_ID="20260918"
        DEFAULT_PIPELINE_LOG_STEM="inspire_cereal_pyramid_0917_append"
        BASE_EPISODES=(571 71 71)
        BASE_FRAMES=(231836 28259 28762)
        COMPONENT_EPISODES=(113 15 15)
        COMPONENT_FRAMES=(61514 9888 7424)
        FINAL_EPISODES=(684 86 86)
        FINAL_FRAMES=(293350 38147 36186)
        ;;
    *)
        usage >&2
        die "Unknown preset: $PRESET"
        ;;
esac

TARGET_NAME="$(basename "$TARGET_DATASET")"
RUN_ID="${RUN_ID:-$DEFAULT_RUN_ID}"
[[ "$RUN_ID" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || die "RUN_ID is not a safe name: $RUN_ID"

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
CHECKPOINT_PUBLISH_BUILD="$CHECKPOINT_ROOT/publish_build"
CHECKPOINT_PROVENANCE="$CHECKPOINT_ROOT/provenance"
CHECKPOINT_SOURCE_BUNDLE="$CHECKPOINT_PROVENANCE/source_bundle"
CHECKPOINT_APPEND_PROVENANCE="$CHECKPOINT_BUILD/$APPEND_PROVENANCE_RELATIVE"
LOG_ROOT="${LOG_ROOT:-/home/alex/Development/logs}"
PIPELINE_LOG="${PIPELINE_LOG:-$LOG_ROOT/data_pipeline/${DEFAULT_PIPELINE_LOG_STEM}_${RUN_ID}.log}"
LOCK_FILE="${LOCK_FILE:-$LOG_ROOT/data_pipeline/.inspire_incremental_append.lock}"
MIN_FREE_GIB="${MIN_FREE_GIB:-360}"
BULK_COPY_HEADROOM_GIB="${BULK_COPY_HEADROOM_GIB:-${DETACH_HEADROOM_GIB:-32}}"
JOBS="${JOBS:-6}"
COPY_TOOL="${COPY_TOOL:-cp}"

CONVERT_SCRIPT="${CONVERT_SCRIPT:-$SCRIPTS_DIR/convert_to_lerobot2.sh}"
SPLIT_SCRIPT="${SPLIT_SCRIPT:-$SCRIPTS_DIR/split_dataset.py}"
APPEND_SCRIPT="${APPEND_SCRIPT:-$SCRIPTS_DIR/append_lerobot2.py}"
SUPPORT_SCRIPT="${SUPPORT_SCRIPT:-$SCRIPTS_DIR/inspire_incremental_append_support.py}"
PROCESS_CHECKER="${PROCESS_CHECKER:-pgrep}"
UV="${UV:-uv}"
MODALITY_CONFIG="${MODALITY_CONFIG:-$GROOT_DIR/examples/UnitreeG1/g1_inspire_headonly_config.py}"

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
    [[ "$BULK_COPY_HEADROOM_GIB" =~ ^[1-9][0-9]*$ ]] || \
        die "BULK_COPY_HEADROOM_GIB must be positive"
    command -v "$COPY_TOOL" >/dev/null 2>&1 || die "cp-compatible COPY_TOOL is required"
}

require_quiescent_processes() {
    local scope="$1"
    command -v "$PROCESS_CHECKER" >/dev/null 2>&1 || \
        die "Process checker is missing: $PROCESS_CHECKER"
    local pattern='[t]eleop_hand_and_arm\.py|[d]ata_editor_EN_rgbd\.py|[c]onvert_to_lerobot2\.sh|[c]onvert_unitree_json_to_lerobot'
    if [[ "$scope" == build ]]; then
        pattern+='|[l]aunch_finetune|[e]valuate_checkpoints'
    fi
    local active
    active="$("$PROCESS_CHECKER" -af "$pattern" || true)"
    if [[ -n "$active" ]]; then
        printf 'Conflicting process(es):\n%s\n' "$active" >&2
        if [[ "$scope" == prepare ]]; then
            die "Stop recording, data editing, and other conversion before component preparation"
        fi
        die "Stop recording, data editing, conversion, training, and evaluation before this append"
    fi
}

validate_paths() {
    "$GROOT_PYTHON" - "$BASE_DATASET" "$SOURCE_A" "$SOURCE_B" "$TARGET_DATASET" "$WORK_ROOT" "$BUILD_DATASET" "$CHECKPOINT_ROOT" <<'PY'
from pathlib import Path
import sys

paths = {name: Path(value).expanduser().resolve() for name, value in zip(
    ("base", "source_a", "source_b", "target", "work", "build", "checkpoint"),
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

validate_one_source() {
    local root="$1"
    local episodes="$2"
    local frames="$3"
    shift 3
    local -a goal_arguments=()
    local goal
    for goal in "$@"; do
        goal_arguments+=(--goal "$goal")
    done
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-source \
        --root "$root" --episodes "$episodes" --frames "$frames" \
        "${goal_arguments[@]}"
}

base_and_source_preflight() {
    [[ -d "$BASE_DATASET" ]] || die "Converted preset base is missing: $BASE_DATASET"
    [[ -d "$SOURCE_A" ]] || die "Preset source is missing: $SOURCE_A"
    [[ -d "$SOURCE_B" ]] || die "Preset source is missing: $SOURCE_B"
    validate_paths
    validate_one_source \
        "$SOURCE_A" "$SOURCE_A_EPISODES" "$SOURCE_A_FRAMES" \
        "${SOURCE_A_GOALS[@]}"
    validate_one_source \
        "$SOURCE_B" "$SOURCE_B_EPISODES" "$SOURCE_B_FRAMES" \
        "${SOURCE_B_GOALS[@]}"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-base \
        --root "$BASE_DATASET" \
        --episodes "${BASE_EPISODES[@]}" --frames "${BASE_FRAMES[@]}"
    bash "$CONVERT_SCRIPT" \
        --preflight-only \
        --include-surface-normals \
        --surface-normals-encoding-version 2 \
        --end-effector inspire-ftp \
        --camera-calibration-profile d435i-254322071415 \
        "$SOURCE_A" "$SOURCE_A_NAME" "$JOBS"
    bash "$CONVERT_SCRIPT" \
        --preflight-only \
        --include-surface-normals \
        --surface-normals-encoding-version 2 \
        --end-effector inspire-ftp \
        --camera-calibration-profile d435i-254322071415 \
        "$SOURCE_B" "$SOURCE_B_NAME" "$JOBS"
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

require_bulk_copy_space() {
    local available_kib source_kib required_kib required_gib
    [[ -d "$CHECKPOINT_BUILD" && ! -L "$CHECKPOINT_BUILD" ]] || \
        die "Sealed final build is missing before bulk-copy disk preflight"
    available_kib="$(df -Pk "$DATASET_PARENT" | awk 'NR == 2 {print $4}')"
    source_kib="$(du -sk --apparent-size -- "$CHECKPOINT_BUILD" | awk '{print $1}')"
    required_kib=$((source_kib + BULK_COPY_HEADROOM_GIB * 1024 * 1024))
    required_gib=$(((required_kib + 1024 * 1024 - 1) / (1024 * 1024)))
    if (( available_kib < required_kib )); then
        die "Independent bulk copy requires ${required_gib} GiB free (full candidate plus ${BULK_COPY_HEADROOM_GIB} GiB headroom); only $((available_kib / 1024 / 1024)) GiB is available"
    fi
    printf 'Bulk-copy disk preflight: %d GiB free (minimum %d GiB including %d GiB headroom).\n' \
        "$((available_kib / 1024 / 1024))" "$required_gib" "$BULK_COPY_HEADROOM_GIB"
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
        --component "$SOURCE_A_NAME" --component "$SOURCE_B_NAME"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-final \
        --base "$BASE_DATASET" --component "$CHECKPOINT_COMPONENT" \
        --output "$CHECKPOINT_BUILD" \
        --episodes "${COMPONENT_EPISODES[@]}" --frames "${COMPONENT_FRAMES[@]}"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-training-stats \
        --dataset-root "$CHECKPOINT_BUILD" --expected-frames "${FINAL_FRAMES[0]}" \
        --report "$CHECKPOINT_APPEND_PROVENANCE/stats_finalization_report.json"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
        --root "$SOURCE_A" \
        --snapshot "$CHECKPOINT_PROVENANCE/$SOURCE_A_SNAPSHOT_NAME"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
        --root "$SOURCE_B" \
        --snapshot "$CHECKPOINT_PROVENANCE/$SOURCE_B_SNAPSHOT_NAME"
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
    [[ -d "$CHECKPOINT_BUILD" && ! -L "$CHECKPOINT_BUILD" ]] || \
        die "Published target recovery requires its retained sealed source build"
    [[ ! -e "$CHECKPOINT_PUBLISH_BUILD" && ! -L "$CHECKPOINT_PUBLISH_BUILD" ]] || \
        die "Both final target and publication candidate exist; refusing ambiguous recovery"
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
        --root "$SOURCE_A" \
        --snapshot "$CHECKPOINT_PROVENANCE/$SOURCE_A_SNAPSHOT_NAME"
    "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
        --root "$SOURCE_B" \
        --snapshot "$CHECKPOINT_PROVENANCE/$SOURCE_B_SNAPSHOT_NAME"
    safe_remove_checkpoint "$CHECKPOINT_ROOT"
    printf '[resume] Verified already-published target and removed only its sealed cleanup checkpoint.\n'
    printf 'DATASET_PUBLISHED=%s\n' "$TARGET_DATASET"
}

build_dataset() {
    local prepare_only="$1"
    refuse_unsealed_scratch
    if [[ -e "$TARGET_DATASET" || -L "$TARGET_DATASET" ]]; then
        if [[ "$prepare_only" == 1 ]]; then
            die "Target is already present; component preparation is not applicable: $TARGET_DATASET"
        fi
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
            if [[ "$prepare_only" == 1 ]]; then
                die "A later build phase is pending recovery; run build instead of prepare"
            fi
            adopt_orphan_final_build
            checkpoint_phase_value=final_build_ready
        fi
    fi

    if [[ "$prepare_only" == 1 && "$checkpoint_phase_value" != none && "$checkpoint_phase_value" != component_ready ]]; then
        printf 'PREPARE_COMPLETE: checkpoint is already beyond conversion (%s).\n' \
            "$checkpoint_phase_value"
        return
    fi

    if [[ "$checkpoint_phase_value" == publish_ready ]]; then
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-publish-ready-checkpoint \
            --root "$CHECKPOINT_ROOT" --base "$BASE_DATASET" \
            --source "$CHECKPOINT_SOURCE_BUNDLE"
    elif [[ "$checkpoint_phase_value" == final_build_ready ]]; then
        printf '[resume] Reusing the sealed final build; conversion, append, and stats remain skipped.\n'
    else
        require_free_gib "$MIN_FREE_GIB"
    fi

    if [[ "$checkpoint_phase_value" == none ]]; then
        mkdir -p -- "$PROVENANCE_STAGE"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" snapshot-tree \
            --root "$BASE_DATASET" --output "$PROVENANCE_STAGE/base_tree_snapshot.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" snapshot-tree \
            --root "$SOURCE_A" --output "$PROVENANCE_STAGE/$SOURCE_A_SNAPSHOT_NAME"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" snapshot-tree \
            --root "$SOURCE_B" --output "$PROVENANCE_STAGE/$SOURCE_B_SNAPSHOT_NAME"

        printf '[bundle] Hard-linking an immutable two-source view without duplicating raw payload.\n'
        mkdir -p -- "$SOURCE_BUNDLE"
        cp -al -- "$SOURCE_A" "$SOURCE_BUNDLE/$SOURCE_A_NAME"
        cp -al -- "$SOURCE_B" "$SOURCE_BUNDLE/$SOURCE_B_NAME"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
            --root "$SOURCE_A" --snapshot "$PROVENANCE_STAGE/$SOURCE_A_SNAPSHOT_NAME"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
            --root "$SOURCE_B" --snapshot "$PROVENANCE_STAGE/$SOURCE_B_SNAPSHOT_NAME"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" snapshot-tree \
            --root "$SOURCE_BUNDLE" --output "$PROVENANCE_STAGE/source_tree_snapshot.json"

        printf '[split] Independently goal-stratifying %s then %s with seed 42.\n' \
            "$SOURCE_A_NAME" "$SOURCE_B_NAME"
        "$UNITREE_PYTHON" "$SPLIT_SCRIPT" "$SOURCE_BUNDLE" \
            --strategy goal-stratified --seed 42 --collection-output "$COMPONENT"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-raw-split \
            --root "$COMPONENT" \
            --episodes "${COMPONENT_EPISODES[@]}" --frames "${COMPONENT_FRAMES[@]}"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-collection-order \
            --root "$COMPONENT" \
            --component "$SOURCE_A_NAME" --component "$SOURCE_B_NAME"

        printf '[convert] Converting only the combined %d-episode incoming component.\n' \
            "$((SOURCE_A_EPISODES + SOURCE_B_EPISODES))"
        bash "$CONVERT_SCRIPT" \
            --include-surface-normals \
            --surface-normals-encoding-version 2 \
            --end-effector inspire-ftp \
            --camera-calibration-profile d435i-254322071415 \
            "$COMPONENT" "$COMPONENT_NAME" "$JOBS"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-component \
            --root "$COMPONENT" --base "$BASE_DATASET" \
            --episodes "${COMPONENT_EPISODES[@]}" --frames "${COMPONENT_FRAMES[@]}"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-collection-order \
            --root "$COMPONENT" \
            --component "$SOURCE_A_NAME" --component "$SOURCE_B_NAME"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" snapshot-tree \
            --root "$COMPONENT" --output "$PROVENANCE_STAGE/component_tree_snapshot.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" write-component-checkpoint \
            --root "$WORK_ROOT" --base "$BASE_DATASET" --source "$CHECKPOINT_SOURCE_BUNDLE"
        [[ ! -e "$CHECKPOINT_ROOT" && ! -L "$CHECKPOINT_ROOT" ]] || \
            die "Resume checkpoint appeared during conversion: $CHECKPOINT_ROOT"
        mv -- "$WORK_ROOT" "$CHECKPOINT_ROOT"
        checkpoint_phase_value=component_ready
        printf '[checkpoint] Sealed the converted %d-episode component for safe reuse.\n' \
            "$((SOURCE_A_EPISODES + SOURCE_B_EPISODES))"
    fi

    if [[ "$prepare_only" == 1 ]]; then
        [[ "$checkpoint_phase_value" == component_ready ]] || \
            die "Unexpected prepare checkpoint phase: $checkpoint_phase_value"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-component-checkpoint \
            --root "$CHECKPOINT_ROOT" --base "$BASE_DATASET" \
            --source "$CHECKPOINT_SOURCE_BUNDLE"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-component \
            --root "$CHECKPOINT_COMPONENT" --base "$BASE_DATASET" \
            --episodes "${COMPONENT_EPISODES[@]}" --frames "${COMPONENT_FRAMES[@]}"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-collection-order \
            --root "$CHECKPOINT_COMPONENT" \
            --component "$SOURCE_A_NAME" --component "$SOURCE_B_NAME"
        printf 'PREPARE_COMPLETE: converted component is sealed; base was not appended or published.\n'
        printf 'COMPONENT_PREPARED=%s\n' "$CHECKPOINT_COMPONENT"
        return
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
            --component "$SOURCE_A_NAME" --component "$SOURCE_B_NAME"
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
            --source-component "$SOURCE_A_NAME=$CHECKPOINT_PROVENANCE/$SOURCE_A_SNAPSHOT_NAME" \
            --source-component "$SOURCE_B_NAME=$CHECKPOINT_PROVENANCE/$SOURCE_B_SNAPSHOT_NAME"
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
            --root "$SOURCE_A" \
            --snapshot "$CHECKPOINT_PROVENANCE/$SOURCE_A_SNAPSHOT_NAME"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" verify-tree-snapshot \
            --root "$SOURCE_B" \
            --snapshot "$CHECKPOINT_PROVENANCE/$SOURCE_B_SNAPSHOT_NAME"
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
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-final \
            --base "$BASE_DATASET" --output "$CHECKPOINT_BUILD" \
            --episodes "${COMPONENT_EPISODES[@]}" --frames "${COMPONENT_FRAMES[@]}"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-training-stats \
            --dataset-root "$CHECKPOINT_BUILD" --expected-frames "${FINAL_FRAMES[0]}" \
            --report "$CHECKPOINT_APPEND_PROVENANCE/stats_finalization_report.json"
        require_bulk_copy_space
        [[ ! -e "$CHECKPOINT_PUBLISH_BUILD" && ! -L "$CHECKPOINT_PUBLISH_BUILD" ]] || \
            die "Partial publication copy was preserved; inspect it before retry: $CHECKPOINT_PUBLISH_BUILD"

        printf '[copy] Deep-copying the sealed build with hard links and reflinks disabled.\n'
        mkdir -- "$CHECKPOINT_PUBLISH_BUILD"
        "$COPY_TOOL" --archive --reflink=never --no-preserve=links -- \
            "$CHECKPOINT_BUILD/." "$CHECKPOINT_PUBLISH_BUILD/"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-independent-copy \
            --source "$CHECKPOINT_BUILD" --output "$CHECKPOINT_PUBLISH_BUILD" \
            --report "$CHECKPOINT_PROVENANCE/publish_build_inventory.json"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-final \
            --base "$BASE_DATASET" --output "$CHECKPOINT_PUBLISH_BUILD" --require-independent \
            --episodes "${COMPONENT_EPISODES[@]}" --frames "${COMPONENT_FRAMES[@]}"
        "$UNITREE_PYTHON" "$SUPPORT_SCRIPT" validate-training-stats \
            --dataset-root "$CHECKPOINT_PUBLISH_BUILD" --expected-frames "${FINAL_FRAMES[0]}" \
            --report "$CHECKPOINT_PUBLISH_BUILD/$APPEND_PROVENANCE_RELATIVE/stats_finalization_report.json" \
            --read-only
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
        --build "$CHECKPOINT_PUBLISH_BUILD" --target "$TARGET_DATASET"
    recover_published_target
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

if [[ "$MODE" == prepare || "$MODE" == build ]]; then
    exec > >(tee -a "$PIPELINE_LOG") 2>&1
fi

case "$MODE" in
    check)
        require_quiescent_processes build
        base_and_source_preflight
        printf 'CHECK_COMPLETE: incremental append is ready; nothing was converted or trained.\n'
        ;;
    prepare)
        require_quiescent_processes prepare
        base_and_source_preflight
        build_dataset 1
        ;;
    build)
        require_quiescent_processes build
        if [[ ! -e "$CHECKPOINT_ROOT" && ! -L "$CHECKPOINT_ROOT" && ! -e "$TARGET_DATASET" && ! -L "$TARGET_DATASET" ]]; then
            base_and_source_preflight
        fi
        build_dataset 0
        ;;
esac
