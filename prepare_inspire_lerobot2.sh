#!/usr/bin/env bash
set -Eeuo pipefail

# General processed_raw -> train/validation/test -> LeRobot v2.1 coordinator.
# Dataset-specific values belong in command-line options or environment variables,
# not in copies of this script.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ISAAC_GROOT_REPO="${ISAAC_GROOT_REPO:-/home/alex/Development/Isaac-GR00T}"
GROOT_PYTHON="${GROOT_PYTHON:-$ISAAC_GROOT_REPO/.venv/bin/python}"
SOURCE_ROOT="${SOURCE_ROOT:-}"
DATASET_ROOT="${DATASET_ROOT:-}"
REPO_ID="${REPO_ID:-}"
END_EFFECTOR="${END_EFFECTOR:-inspire-ftp}"
SPLIT_STRATEGY="${SPLIT_STRATEGY:-goal-stratified}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SPLIT_SEARCH_TRIALS="${SPLIT_SEARCH_TRIALS:-20000}"
JOBS="${JOBS:-6}"
RUN_SUFFIX="${RUN_SUFFIX:-}"
DEPTH_NEAR_M="${DEPTH_NEAR_M:-0.25}"
DEPTH_FAR_M="${DEPTH_FAR_M:-1.0}"
CAMERA_CALIBRATION_PROFILE="${CAMERA_CALIBRATION_PROFILE:-d435i-254322071415}"
export DEPTH_NEAR_M DEPTH_FAR_M CAMERA_CALIBRATION_PROFILE

CONVERT_SCRIPT="${CONVERT_SCRIPT:-$SCRIPT_DIR/convert_to_lerobot2.sh}"
SPLIT_SCRIPT="${SPLIT_SCRIPT:-$SCRIPT_DIR/split_dataset.py}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-$SCRIPT_DIR/multi_finetune_evaluation.sh}"

usage() {
    cat <<'EOF'
Usage: prepare_inspire_lerobot2.sh MODE [OPTIONS]

Build one complete Inspire LeRobot dataset from every episode in a curated
processed_raw directory. MODE is required; no mode starts training implicitly.

Modes:
  check           Preflight every raw episode; do not copy or convert anything.
  convert         Copy every episode, split train/validation/test, convert, validate,
                  and publish the completed dataset.
  training-check  Validate the published split population and run the non-training
                  readiness check for the RGB, depth, and surface-normal models.
  train           Validate the published splits, then train/evaluate all three models.
  all             Explicitly run convert followed by train.

Options (the matching uppercase environment variable may be used instead):
  --source PATH             Curated processed_raw root (SOURCE_ROOT).
  --output PATH             Final LeRobot split root (DATASET_ROOT).
  --repo-id ID              Base local LeRobot repo ID; defaults to output basename.
  --end-effector TYPE       inspire-ftp (default) or inspire-dfx.
  --split-strategy NAME     goal-stratified (default) or session-grouped.
  --split-seed INTEGER      Reproducible split seed (default: 42).
  --split-search-trials N   Session-grouped search trials (default: 20000).
  --jobs N                  Video transcoding jobs (default: 6).
  --run-suffix TEXT         Training output suffix; defaults to output basename/date.
  --camera-calibration-profile PROFILE
                            Fallback for legacy episodes without recorded calibration:
                            d435i-254322071415 (Inspire default) or legacy-untagged.
  -h, --help                Show this help.

Depth range environment overrides:
  DEPTH_NEAR_M              Default: 0.25.
  DEPTH_FAR_M               Default: 1.0.

The source is treated as already curated. Every direct episode_* directory is
included; this script has no quality filter or episode-exclusion mechanism. The
source is never modified. Full conversion always includes RGB, gray depth,
lossless depth, and surface normals.

Example:
  prepare_inspire_lerobot2.sh convert \
    --source /data/processed_raw/pick_red_cup \
    --output /data/lerobot2/pick_red_cup
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

path_exists() {
    [[ -e "$1" || -L "$1" ]]
}

[[ $# -gt 0 ]] || { usage >&2; exit 2; }
MODE="$1"
shift

case "$MODE" in
    check|convert|training-check|train|all)
        ;;
    -h|--help|help)
        usage
        exit 0
        ;;
    *)
        usage >&2
        die "Unknown or missing mode: $MODE"
        ;;
esac

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source)
            [[ $# -ge 2 ]] || die "--source requires a path."
            SOURCE_ROOT="$2"
            shift 2
            ;;
        --source=*)
            SOURCE_ROOT="${1#*=}"
            shift
            ;;
        --output)
            [[ $# -ge 2 ]] || die "--output requires a path."
            DATASET_ROOT="$2"
            shift 2
            ;;
        --output=*)
            DATASET_ROOT="${1#*=}"
            shift
            ;;
        --repo-id)
            [[ $# -ge 2 ]] || die "--repo-id requires a value."
            REPO_ID="$2"
            shift 2
            ;;
        --repo-id=*)
            REPO_ID="${1#*=}"
            shift
            ;;
        --end-effector)
            [[ $# -ge 2 ]] || die "--end-effector requires a value."
            END_EFFECTOR="$2"
            shift 2
            ;;
        --end-effector=*)
            END_EFFECTOR="${1#*=}"
            shift
            ;;
        --split-strategy)
            [[ $# -ge 2 ]] || die "--split-strategy requires a value."
            SPLIT_STRATEGY="$2"
            shift 2
            ;;
        --split-strategy=*)
            SPLIT_STRATEGY="${1#*=}"
            shift
            ;;
        --split-seed)
            [[ $# -ge 2 ]] || die "--split-seed requires an integer."
            SPLIT_SEED="$2"
            shift 2
            ;;
        --split-seed=*)
            SPLIT_SEED="${1#*=}"
            shift
            ;;
        --split-search-trials)
            [[ $# -ge 2 ]] || die "--split-search-trials requires an integer."
            SPLIT_SEARCH_TRIALS="$2"
            shift 2
            ;;
        --split-search-trials=*)
            SPLIT_SEARCH_TRIALS="${1#*=}"
            shift
            ;;
        --jobs)
            [[ $# -ge 2 ]] || die "--jobs requires an integer."
            JOBS="$2"
            shift 2
            ;;
        --jobs=*)
            JOBS="${1#*=}"
            shift
            ;;
        --run-suffix)
            [[ $# -ge 2 ]] || die "--run-suffix requires a value."
            RUN_SUFFIX="$2"
            shift 2
            ;;
        --run-suffix=*)
            RUN_SUFFIX="${1#*=}"
            shift
            ;;
        --camera-calibration-profile)
            [[ $# -ge 2 ]] || die "--camera-calibration-profile requires a value."
            CAMERA_CALIBRATION_PROFILE="$2"
            shift 2
            ;;
        --camera-calibration-profile=*)
            CAMERA_CALIBRATION_PROFILE="${1#*=}"
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            [[ $# -eq 0 ]] || die "Unexpected positional arguments: $*"
            ;;
        *)
            die "Unknown option: $1"
            ;;
    esac
done

case "$END_EFFECTOR" in
    inspire-ftp|inspire-dfx) ;;
    *) die "--end-effector must be inspire-ftp or inspire-dfx." ;;
esac
case "$SPLIT_STRATEGY" in
    goal-stratified|session-grouped) ;;
    *) die "--split-strategy must be goal-stratified or session-grouped." ;;
esac
[[ "$SPLIT_SEED" =~ ^-?[0-9]+$ ]] || die "--split-seed must be an integer."
[[ "$SPLIT_SEARCH_TRIALS" =~ ^[1-9][0-9]*$ ]] || \
    die "--split-search-trials must be a positive integer."
[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer."
[[ -n "$CAMERA_CALIBRATION_PROFILE" ]] || \
    die "--camera-calibration-profile must not be empty."
case "$CAMERA_CALIBRATION_PROFILE" in
    legacy-untagged|d435i-254322071415) ;;
    *)
        die "--camera-calibration-profile must be legacy-untagged or d435i-254322071415."
        ;;
esac
[[ -x "$GROOT_PYTHON" ]] || die "Python is missing or not executable: $GROOT_PYTHON"
case "$MODE" in
    check|convert|all)
        [[ -f "$CONVERT_SCRIPT" ]] || die "Conversion script is missing: $CONVERT_SCRIPT"
        ;;
esac
case "$MODE" in
    convert|all)
        [[ -f "$SPLIT_SCRIPT" ]] || die "Split script is missing: $SPLIT_SCRIPT"
        ;;
esac
case "$MODE" in
    training-check|train|all)
        [[ -f "$TRAIN_SCRIPT" ]] || die "Training script is missing: $TRAIN_SCRIPT"
        ;;
esac

if [[ -n "$SOURCE_ROOT" ]]; then
    [[ -d "$SOURCE_ROOT" ]] || die "Source directory does not exist: $SOURCE_ROOT"
    SOURCE_ROOT="$(realpath "$SOURCE_ROOT")"
fi
if [[ -n "$DATASET_ROOT" ]]; then
    DATASET_ROOT="$(realpath -m "$DATASET_ROOT")"
fi

case "$MODE" in
    check)
        [[ -n "$SOURCE_ROOT" ]] || die "Set SOURCE_ROOT or pass --source."
        ;;
    convert|all)
        [[ -n "$SOURCE_ROOT" ]] || die "Set SOURCE_ROOT or pass --source."
        [[ -n "$DATASET_ROOT" ]] || die "Set DATASET_ROOT or pass --output."
        ;;
    training-check|train)
        [[ -n "$DATASET_ROOT" ]] || die "Set DATASET_ROOT or pass --output."
        ;;
esac

if [[ -z "$REPO_ID" ]]; then
    if [[ -n "$DATASET_ROOT" ]]; then
        REPO_ID="$(basename "$DATASET_ROOT")"
    else
        REPO_ID="$(basename "$SOURCE_ROOT")"
    fi
fi
[[ "$REPO_ID" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "--repo-id may contain only letters, numbers, dots, underscores, and hyphens."

if [[ -z "$RUN_SUFFIX" && -n "$DATASET_ROOT" ]]; then
    RUN_SUFFIX="$(basename "$DATASET_ROOT")_$(date -u +%Y%m%d)"
fi

BUILD_ROOT="${BUILD_ROOT:-${DATASET_ROOT}.building}"
if [[ -n "$DATASET_ROOT" ]]; then
    BUILD_ROOT="$(realpath -m "$BUILD_ROOT")"
fi

declare -a RAW_EPISODES=()

discover_raw_episodes() {
    mapfile -d '' RAW_EPISODES < <(
        find "$SOURCE_ROOT" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -name 'episode_*' \
            -print0 |
        sort -z
    )
    [[ ${#RAW_EPISODES[@]} -gt 0 ]] || \
        die "No direct episode_* directories found in $SOURCE_ROOT"
}

raw_preflight() {
    discover_raw_episodes
    bash "$CONVERT_SCRIPT" \
        --preflight-only \
        --include-surface-normals \
        --end-effector "$END_EFFECTOR" \
        --camera-calibration-profile "$CAMERA_CALIBRATION_PROFILE" \
        "$SOURCE_ROOT" \
        "$REPO_ID" \
        "$JOBS"
    printf 'Raw population: %d episodes; all will be included.\n' "${#RAW_EPISODES[@]}"
}

validate_disjoint_paths() {
    "$GROOT_PYTHON" - "$SOURCE_ROOT" "$DATASET_ROOT" "$BUILD_ROOT" <<'PY'
from pathlib import Path
import sys

named_paths = {
    "source": Path(sys.argv[1]).resolve(),
    "output": Path(sys.argv[2]).resolve(),
    "build": Path(sys.argv[3]).resolve(),
}
items = list(named_paths.items())
for index, (left_name, left) in enumerate(items):
    for right_name, right in items[index + 1 :]:
        if left == right or left in right.parents or right in left.parents:
            raise SystemExit(
                f"ERROR: {left_name} and {right_name} paths must not overlap: "
                f"{left} ; {right}"
            )
PY
}

validate_split_population() {
    local root="$1"
    local expected_count="${2:-}"
    local source="${3:-}"
    "$GROOT_PYTHON" - \
        "$root" \
        "$expected_count" \
        "$source" \
        "$END_EFFECTOR" \
        "$SPLIT_STRATEGY" \
        "$SPLIT_SEED" <<'PY'
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys

root = Path(sys.argv[1])
expected_count = int(sys.argv[2]) if sys.argv[2] else None
source = Path(sys.argv[3]) if sys.argv[3] else None
end_effector = sys.argv[4]
expected_strategy = sys.argv[5]
expected_seed = int(sys.argv[6])
split_names = ("train", "validation", "test")
video_features = {
    "observation.images.ego_view",
    "observation.images.depth_gray_view",
    "observation.images.surface_normals_view",
}
required_geometry = {
    "depth_encoding",
    "raw_depth_encoding",
    "aligned_depth_encoding",
    "surface_normals_encoding",
    "surface_normals_lz4",
}
expected_layout = {
    "left_arm": {"start": 0, "end": 7},
    "right_arm": {"start": 7, "end": 14},
    "left_hand": {"start": 14, "end": 20},
    "right_hand": {"start": 20, "end": 26},
}
episode_name = re.compile(r"episode_(\d+)$")
calibration_fingerprint = re.compile(r"sha256:[0-9a-f]{64}$")
split_calibrations = {}


def read_json(path):
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"ERROR: cannot read {path}: {error}") from error
    if not isinstance(value, dict):
        raise SystemExit(f"ERROR: {path} must contain a JSON object")
    return value


def read_jsonl(path):
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as error:
        raise SystemExit(f"ERROR: cannot read {path}: {error}") from error


def calibration_fields_signature(value, *, split, location):
    if value["schema"] != "realsense_rgbd_calibration.v1":
        raise SystemExit(
            f"ERROR: {split} {location} has unsupported schema {value['schema']!r}"
        )
    camera = value["camera"]
    camera_fields = {"model", "serial", "product_id", "firmware"}
    if not isinstance(camera, dict) or set(camera) != camera_fields:
        raise SystemExit(
            f"ERROR: {split} {location}.camera must contain exactly "
            "model, serial, product_id, and firmware"
        )
    if any(not isinstance(camera[key], str) or not camera[key].strip() for key in camera_fields):
        raise SystemExit(
            f"ERROR: {split} {location}.camera fields must be non-empty strings"
        )
    fingerprint = value["fingerprint"]
    if not isinstance(fingerprint, str) or calibration_fingerprint.fullmatch(
        fingerprint
    ) is None:
        raise SystemExit(
            f"ERROR: {split} {location} has invalid fingerprint {fingerprint!r}"
        )
    return value["schema"], camera, fingerprint


def full_calibration_signature(value, *, split):
    expected_fields = {
        "schema",
        "camera",
        "color",
        "depth",
        "depth_to_color",
        "fingerprint",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise SystemExit(
            f"ERROR: {split} camera_calibration must contain exactly "
            "schema, camera, color, depth, depth_to_color, and fingerprint"
        )
    signature = calibration_fields_signature(
        value,
        split=split,
        location="camera_calibration",
    )
    fingerprint_payload = {key: item for key, item in value.items() if key != "fingerprint"}
    try:
        canonical = json.dumps(
            fingerprint_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise SystemExit(
            f"ERROR: {split} camera_calibration is not finite JSON"
        ) from error
    expected_fingerprint = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    if value["fingerprint"] != expected_fingerprint:
        raise SystemExit(
            f"ERROR: {split} camera_calibration fingerprint does not match its payload"
        )
    return signature


def compact_calibration_signature(value, *, split):
    expected_fields = {"source", "schema", "camera", "fingerprint"}
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise SystemExit(
            f"ERROR: {split} surface_normals_encoding.camera_calibration must contain "
            "exactly source, schema, camera, and fingerprint"
        )
    if not isinstance(value["source"], str) or not value["source"].strip():
        raise SystemExit(
            f"ERROR: {split} surface_normals_encoding.camera_calibration.source "
            "must be non-empty"
        )
    return calibration_fields_signature(
        value,
        split=split,
        location="surface_normals_encoding.camera_calibration",
    )


manifest_path = root / "split_manifest.json"
manifest = read_json(manifest_path)
if manifest.get("version") != 1:
    raise SystemExit(f"ERROR: unsupported split-manifest version in {manifest_path}")
if manifest.get("strategy") != expected_strategy or manifest.get("seed") != expected_seed:
    raise SystemExit(
        f"ERROR: split configuration differs from requested "
        f"{expected_strategy!r}/seed {expected_seed}: "
        f"{manifest.get('strategy')!r}/seed {manifest.get('seed')!r}"
    )

records = manifest.get("episodes")
if not isinstance(records, list):
    raise SystemExit(f"ERROR: invalid episodes list in {manifest_path}")
declared_count = manifest.get("episode_count")
if declared_count != len(records):
    raise SystemExit(
        f"ERROR: split manifest count mismatch: {declared_count!r} != {len(records)}"
    )
if expected_count is not None and len(records) != expected_count:
    raise SystemExit(
        f"ERROR: not every raw episode reached the split manifest: "
        f"{len(records)} != {expected_count}"
    )

by_split = {name: [] for name in split_names}
manifest_source = {}
for record in records:
    if not isinstance(record, dict) or record.get("split") not in by_split:
        raise SystemExit(f"ERROR: invalid split-manifest record: {record!r}")
    split = record["split"]
    flattened_episode = record.get("flattened_episode")
    split_episode = record.get("split_episode")
    digest = record.get("data_json_sha256")
    if not isinstance(flattened_episode, str) or episode_name.fullmatch(flattened_episode) is None:
        raise SystemExit(f"ERROR: invalid flattened episode: {flattened_episode!r}")
    if not isinstance(split_episode, str) or episode_name.fullmatch(split_episode) is None:
        raise SystemExit(f"ERROR: invalid split episode: {split_episode!r}")
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise SystemExit(f"ERROR: invalid data.json hash for {flattened_episode}")
    if flattened_episode in manifest_source:
        raise SystemExit(f"ERROR: duplicate flattened episode: {flattened_episode}")
    if not isinstance(record.get("goal"), str):
        raise SystemExit(f"ERROR: invalid goal for {flattened_episode}")
    frame_count = record.get("frame_count")
    if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count <= 0:
        raise SystemExit(f"ERROR: invalid frame count for {flattened_episode}")
    manifest_source[flattened_episode] = digest
    by_split[split].append(record)

counts = Counter({split: len(split_records) for split, split_records in by_split.items()})
if set(counts) != set(split_names) or any(counts[name] <= 0 for name in split_names):
    raise SystemExit(f"ERROR: incomplete train/validation/test population: {dict(counts)}")

if expected_strategy == "goal-stratified":
    total = len(records)
    test_count = max(1, int(total * 0.1 + 0.5))
    validation_count = max(1, int(total * 0.1 + 0.5))
    train_count = total - test_count - validation_count
    if train_count < 1:
        train_count = 1
        remaining = total - train_count
        test_count = max(1, remaining // 2)
        validation_count = remaining - test_count
    wanted = {
        "train": train_count,
        "validation": validation_count,
        "test": test_count,
    }
    if dict(counts) != wanted:
        raise SystemExit(
            f"ERROR: goal-stratified counts differ from the 80/10/10 targets: "
            f"{dict(counts)} != {wanted}"
        )

if source is not None:
    raw_source = {}
    for episode in sorted(source.iterdir()):
        if not episode.is_dir() or episode_name.fullmatch(episode.name) is None:
            continue
        json_path = episode / "data.json"
        try:
            digest = hashlib.sha256(json_path.read_bytes()).hexdigest()
        except OSError as error:
            raise SystemExit(f"ERROR: cannot hash {json_path}: {error}") from error
        raw_source[episode.name] = digest
    if raw_source != manifest_source:
        missing = sorted(set(raw_source) - set(manifest_source))
        unexpected = sorted(set(manifest_source) - set(raw_source))
        changed = sorted(
            name
            for name in set(raw_source).intersection(manifest_source)
            if raw_source[name] != manifest_source[name]
        )
        raise SystemExit(
            "ERROR: converted split provenance does not exactly match processed_raw: "
            f"missing={missing}, unexpected={unexpected}, changed={changed}"
        )

for split in split_names:
    split_root = root / split
    info_path = split_root / "meta" / "info.json"
    info = read_json(info_path)
    if info.get("total_episodes") != counts[split]:
        raise SystemExit(
            f"ERROR: {split} metadata/manifest episode mismatch: "
            f"{info.get('total_episodes')!r} != {counts[split]}"
        )
    if str(info.get("codebase_version")) not in {"v2.1", "2.1"}:
        raise SystemExit(f"ERROR: {split} is not LeRobot v2.1")
    if info.get("robot_type") != "Unitree_G1_Inspire_HeadOnly":
        raise SystemExit(f"ERROR: {split} has the wrong robot type")
    if float(info.get("fps", -1)) != 30.0:
        raise SystemExit(f"ERROR: {split} is not 30 FPS")
    features = info.get("features")
    if not isinstance(features, dict):
        raise SystemExit(f"ERROR: {split} has no feature contract")
    for feature_key in ("observation.state", "action"):
        feature = features.get(feature_key)
        if not isinstance(feature, dict) or feature.get("shape") != [26]:
            raise SystemExit(f"ERROR: {split} {feature_key} is not 26D")
    videos = {
        key
        for key, feature in features.items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    }
    if videos != video_features:
        raise SystemExit(f"ERROR: {split} has unexpected video features: {sorted(videos)}")
    end_effector_metadata = info.get("end_effector")
    expected_protocol = end_effector.removeprefix("inspire-")
    if not isinstance(end_effector_metadata, dict) or (
        end_effector_metadata.get("type"),
        end_effector_metadata.get("protocol"),
        end_effector_metadata.get("hand_dof"),
    ) != ("inspire", expected_protocol, 6):
        raise SystemExit(f"ERROR: {split} has the wrong end-effector contract")
    malformed_geometry = sorted(
        key for key in required_geometry if not isinstance(info.get(key), dict)
    )
    if malformed_geometry:
        raise SystemExit(f"ERROR: {split} is missing geometry metadata: {malformed_geometry}")

    calibration = info.get("camera_calibration")
    compact_calibration = info["surface_normals_encoding"].get("camera_calibration")
    if calibration is None and compact_calibration is None:
        split_calibrations[split] = None
    elif calibration is None or compact_calibration is None:
        raise SystemExit(
            f"ERROR: {split} must carry both full and compact camera calibration"
        )
    else:
        full_signature = full_calibration_signature(calibration, split=split)
        if compact_calibration_signature(compact_calibration, split=split) != full_signature:
            raise SystemExit(
                f"ERROR: {split} full and compact camera calibrations do not agree"
            )
        split_calibrations[split] = calibration

    modality = read_json(split_root / "meta" / "modality.json")
    if modality.get("state") != expected_layout or modality.get("action") != expected_layout:
        raise SystemExit(f"ERROR: {split} has the wrong 26D modality layout")
    modality_videos = modality.get("video")
    if not isinstance(modality_videos, dict) or set(modality_videos) != {
        "ego_view",
        "depth_gray_view",
        "surface_normals_view",
    }:
        raise SystemExit(f"ERROR: {split} has the wrong video modality layout")

    split_records = sorted(
        by_split[split],
        key=lambda record: int(episode_name.fullmatch(record["split_episode"])[1]),
    )
    expected_split_episodes = [
        f"episode_{index:04d}" for index in range(1, len(split_records) + 1)
    ]
    if [record["split_episode"] for record in split_records] != expected_split_episodes:
        raise SystemExit(f"ERROR: {split} split episode names are not canonical")
    episodes = read_jsonl(split_root / "meta" / "episodes.jsonl")
    if len(episodes) != len(split_records):
        raise SystemExit(f"ERROR: {split} episodes.jsonl count differs from the manifest")
    frames = 0
    for episode_index, (record, episode) in enumerate(zip(split_records, episodes, strict=True)):
        if episode.get("episode_index") != episode_index:
            raise SystemExit(f"ERROR: {split} episode indices are not canonical")
        length = episode.get("length")
        if length != record["frame_count"]:
            raise SystemExit(f"ERROR: {split} frame count differs at episode {episode_index}")
        if episode.get("tasks") != [record["goal"]]:
            raise SystemExit(f"ERROR: {split} task differs at episode {episode_index}")
        frames += length
    if info.get("total_frames") != frames:
        raise SystemExit(
            f"ERROR: {split} metadata/manifest frame mismatch: "
            f"{info.get('total_frames')!r} != {frames}"
        )

reference_split = split_names[0]
reference_calibration = split_calibrations[reference_split]
for split in split_names[1:]:
    if split_calibrations[split] != reference_calibration:
        raise SystemExit(
            "ERROR: converted splits have different or known-versus-unknown "
            f"camera calibration: {reference_split} != {split}"
        )

print(
    "Split population: "
    + ", ".join(f"{name}={counts[name]}" for name in split_names)
    + f"; total={len(records)}"
)
PY
}

convert_dataset() {
    validate_disjoint_paths
    ! path_exists "$DATASET_ROOT" || \
        die "Final dataset already exists: $DATASET_ROOT"
    ! path_exists "$BUILD_ROOT" || \
        die "Stale build directory exists; inspect it first: $BUILD_ROOT"
    raw_preflight
    local raw_count="${#RAW_EPISODES[@]}"

    mkdir -p -- "$(dirname "$DATASET_ROOT")"
    mkdir -- "$BUILD_ROOT"
    cp -a -- "${RAW_EPISODES[@]}" "$BUILD_ROOT/"
    if [[ -f "$SOURCE_ROOT/flatten_manifest.json" ]]; then
        cp -a -- "$SOURCE_ROOT/flatten_manifest.json" "$BUILD_ROOT/"
    fi

    local copied_count
    copied_count="$(
        find "$BUILD_ROOT" \
            -mindepth 1 \
            -maxdepth 1 \
            -type d \
            -name 'episode_*' \
            -printf . |
        wc -c
    )"
    [[ "$copied_count" -eq "$raw_count" ]] || \
        die "Raw staging count mismatch: copied $copied_count of $raw_count episodes."

    "$GROOT_PYTHON" "$SPLIT_SCRIPT" \
        "$BUILD_ROOT" \
        --strategy "$SPLIT_STRATEGY" \
        --seed "$SPLIT_SEED" \
        --search-trials "$SPLIT_SEARCH_TRIALS"

    bash "$CONVERT_SCRIPT" \
        --include-surface-normals \
        --end-effector "$END_EFFECTOR" \
        --camera-calibration-profile "$CAMERA_CALIBRATION_PROFILE" \
        "$BUILD_ROOT" \
        "$REPO_ID" \
        "$JOBS"

    validate_split_population "$BUILD_ROOT" "$raw_count" "$SOURCE_ROOT"
    ! path_exists "$DATASET_ROOT" || \
        die "Final dataset appeared during the build; leaving staging untouched: $DATASET_ROOT"
    mv -- "$BUILD_ROOT" "$DATASET_ROOT"
    printf 'Published validated dataset: %s\n' "$DATASET_ROOT"
}

run_training_check() {
    [[ -d "$DATASET_ROOT" ]] || die "Converted dataset does not exist: $DATASET_ROOT"
    validate_split_population "$DATASET_ROOT" "" "$SOURCE_ROOT"
    DATASET_ROOT="$DATASET_ROOT" \
    RUN_SUFFIX="$RUN_SUFFIX" \
    PRECHECK_ONLY=1 \
        bash "$TRAIN_SCRIPT"
}

run_training() {
    [[ -d "$DATASET_ROOT" ]] || die "Converted dataset does not exist: $DATASET_ROOT"
    validate_split_population "$DATASET_ROOT" "" "$SOURCE_ROOT"
    DATASET_ROOT="$DATASET_ROOT" \
    RUN_SUFFIX="$RUN_SUFFIX" \
        bash "$TRAIN_SCRIPT"
}

case "$MODE" in
    check)
        raw_preflight
        ;;
    convert)
        convert_dataset
        ;;
    training-check)
        run_training_check
        ;;
    train)
        run_training
        ;;
    all)
        convert_dataset
        run_training
        ;;
esac
