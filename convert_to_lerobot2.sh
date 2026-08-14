#!/usr/bin/env bash

set -Eeuo pipefail

# Example (run from /home/alex/Development/scripts/):
#   cd /home/alex/Development/scripts/
#   ./convert_to_lerobot2.sh /path/to/copied_split_root
#
# The input may contain episode_* directly, or any subset of:
#   train/  validation/  test/
# Each present split is converted independently and remains a separate folder.
# Raw uint16 depth PNGs are preserved losslessly alongside the standard video
# features under raw_depths/chunk-*/episode_*/frame_*.png.

usage() {
    cat <<'EOF'
Usage:
  convert_to_lerobot2.sh [--include-surface-normals] PATH/TO/INPUT_COPY [REPO_ID] [JOBS]

The input must be either:
  1. A disposable copy containing episode_* folders directly, or
  2. A parent containing any subset of train/, validation/, and test/.

A direct episode root is replaced with LeRobot v2.1. For a split parent, each
present split is converted independently in place and remains separate.

The aligned depth view is converted to model-ready H.264 as before. Set
INCLUDE_SURFACE_NORMALS=1 to additionally derive a camera-frame XYZ surface-
normal view directly from each aligned uint16 depth_0 PNG. Every raw_depth_0
PNG is preserved as before; surface-normal conversions also preserve depth_0
losslessly for reproducibility.

Arguments:
  PATH/TO/INPUT_COPY  Disposable direct-episode or split-parent dataset copy.
  REPO_ID             Base local dataset identifier. Defaults to the input folder
                      name. Split mode adds _train, _validation, or _test.
  JOBS                H.264 transcoding jobs. Defaults to 6.

Environment overrides:
  UNITREE_LEROBOT_REPO  Default: ~/Development/unitree_lerobot
  ISAAC_GROOT_REPO      Default: ~/Development/Isaac-GR00T
  UNITREE_PYTHON        Default: ~/miniconda3/envs/unitree_lerobot/bin/python
  DEPTH_NEAR_M          Default: 0.25
  DEPTH_FAR_M           Default: 1.0
  INCLUDE_SURFACE_NORMALS
                        Default: 0. Set to 1 to add surface_normals_view.
EOF
}

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

INCLUDE_SURFACE_NORMALS="${INCLUDE_SURFACE_NORMALS:-0}"
if [[ ${1:-} == "--include-surface-normals" ]]; then
    INCLUDE_SURFACE_NORMALS=1
    shift
fi

[[ $# -ge 1 && $# -le 3 ]] || { usage >&2; exit 2; }

command -v realpath >/dev/null || die "realpath is required."
command -v uv >/dev/null || die "uv is required."
command -v ffmpeg >/dev/null || die "ffmpeg is required."
command -v ffprobe >/dev/null || die "ffprobe is required."

LEROBOT_DIR="$(realpath "$1")"

[[ -d "$LEROBOT_DIR" ]] || die "Input directory does not exist: $LEROBOT_DIR"
[[ "$LEROBOT_DIR" != "/" ]] || die "Refusing to use the filesystem root as input."
[[ "$LEROBOT_DIR" != "$(realpath "$HOME")" ]] || die "Refusing to use the home directory as input."

TASK_DIR="$(dirname "$LEROBOT_DIR")"

DEFAULT_REPO_ID="$(basename "$LEROBOT_DIR")"
REPO_ID="${2:-$DEFAULT_REPO_ID}"
JOBS="${3:-6}"

[[ "$REPO_ID" =~ ^[A-Za-z0-9._-]+$ ]] || \
    die "REPO_ID may contain only letters, numbers, dots, underscores and hyphens."

[[ "$JOBS" =~ ^[1-9][0-9]*$ ]] || \
    die "JOBS must be a positive integer."

UNITREE_LEROBOT_REPO="${UNITREE_LEROBOT_REPO:-$HOME/Development/unitree_lerobot}"
ISAAC_GROOT_REPO="${ISAAC_GROOT_REPO:-$HOME/Development/Isaac-GR00T}"
UNITREE_PYTHON="${UNITREE_PYTHON:-$HOME/miniconda3/envs/unitree_lerobot/bin/python}"
DEPTH_NEAR_M="${DEPTH_NEAR_M:-0.25}"
DEPTH_FAR_M="${DEPTH_FAR_M:-1.0}"

[[ "$INCLUDE_SURFACE_NORMALS" == 0 || "$INCLUDE_SURFACE_NORMALS" == 1 ]] || \
    die "INCLUDE_SURFACE_NORMALS must be 0 or 1."
export INCLUDE_SURFACE_NORMALS

UNITREE_CONVERTER="$UNITREE_LEROBOT_REPO/unitree_lerobot/utils/convert_unitree_json_to_lerobot.py"
V3_TO_V2_PROJECT="$ISAAC_GROOT_REPO/scripts/lerobot_conversion"
V3_TO_V2_SCRIPT="$V3_TO_V2_PROJECT/convert_v3_to_v2.py"
H264_SCRIPT="$ISAAC_GROOT_REPO/examples/SimplerEnv/convert_av1_to_h264.py"
MODALITY_FILE="$ISAAC_GROOT_REPO/examples/UnitreeG1/modality.json"

[[ -x "$UNITREE_PYTHON" ]] || \
    die "Unitree environment Python is missing: $UNITREE_PYTHON"

[[ -f "$UNITREE_CONVERTER" ]] || \
    die "Unitree converter is missing: $UNITREE_CONVERTER"

[[ -f "$V3_TO_V2_PROJECT/pyproject.toml" ]] || \
    die "v3-to-v2 project is missing: $V3_TO_V2_PROJECT"

[[ -f "$V3_TO_V2_SCRIPT" ]] || \
    die "v3-to-v2 converter is missing: $V3_TO_V2_SCRIPT"

[[ -f "$H264_SCRIPT" ]] || \
    die "AV1-to-H.264 converter is missing: $H264_SCRIPT"

[[ -f "$MODALITY_FILE" ]] || \
    die "Unitree modality file is missing: $MODALITY_FILE"

mapfile -d '' DIRECT_EPISODE_DIRS < <(
    find "$LEROBOT_DIR" \
        -mindepth 1 \
        -maxdepth 1 \
        -type d \
        -name 'episode_*' \
        -print0 |
    sort -z
)

declare -a SPLIT_DIRS=()
declare -a SPLIT_SUFFIXES=()

for split_name in train validation test; do
    split_dir="$LEROBOT_DIR/$split_name"
    if [[ -d "$split_dir" ]]; then
        SPLIT_DIRS+=("$split_dir")
        SPLIT_SUFFIXES+=("$split_name")
    fi
done

# Accept the older validate/ spelling while always using _validation in repo IDs.
if [[ -d "$LEROBOT_DIR/validate" ]]; then
    [[ ! -d "$LEROBOT_DIR/validation" ]] || \
        die "Input contains both validation/ and validate/. Keep only one."
    SPLIT_DIRS+=("$LEROBOT_DIR/validate")
    SPLIT_SUFFIXES+=("validation")
fi

if [[ ${#DIRECT_EPISODE_DIRS[@]} -gt 0 && ${#SPLIT_DIRS[@]} -gt 0 ]]; then
    die "Input mixes direct episode_* folders with split folders. Use one layout."
fi

if [[ ${#SPLIT_DIRS[@]} -gt 0 ]]; then
    SCRIPT_PATH="$(realpath "$0")"

    printf '\nSplit parent: %s\n' "$LEROBOT_DIR"
    printf 'Splits:       %d\n' "${#SPLIT_DIRS[@]}"
    printf 'Base repo ID: %s\n\n' "$REPO_ID"

    for index in "${!SPLIT_DIRS[@]}"; do
        split_dir="${SPLIT_DIRS[$index]}"
        split_suffix="${SPLIT_SUFFIXES[$index]}"
        split_repo_id="${REPO_ID}_${split_suffix}"

        printf '=== Converting %s as %s ===\n' \
            "$(basename "$split_dir")" \
            "$split_repo_id"

        bash "$SCRIPT_PATH" "$split_dir" "$split_repo_id" "$JOBS"
    done

    printf '\nAll available splits converted successfully.\n'
    printf 'Split dataset root: %s\n' "$LEROBOT_DIR"
    exit 0
fi

EPISODE_DIRS=("${DIRECT_EPISODE_DIRS[@]}")

[[ ${#EPISODE_DIRS[@]} -gt 0 ]] || \
    die "No direct episode_* folders or train/validation/test splits found in $LEROBOT_DIR"

for episode_dir in "${EPISODE_DIRS[@]}"; do
    episode_name="$(basename "$episode_dir")"

    [[ "$episode_name" =~ ^episode_[0-9]+$ ]] || \
        die "Invalid episode directory name: $episode_name"

    [[ -f "$episode_dir/data.json" ]] || \
        die "Missing data.json: $episode_dir/data.json"
done

"$UNITREE_PYTHON" - "$INCLUDE_SURFACE_NORMALS" "${EPISODE_DIRS[@]}" <<'PY'
import json
from pathlib import Path
import sys

include_surface_normals = bool(int(sys.argv[1]))
total_frames = 0
for episode_arg in sys.argv[2:]:
    episode = Path(episode_arg).resolve()
    with (episode / "data.json").open(encoding="utf-8") as file:
        payload = json.load(file)

    frames = payload.get("data", [])
    if not frames:
        raise ValueError(f"Episode contains no frames: {episode}")

    for frame_index, frame in enumerate(frames):
        relative = (frame.get("depths") or {}).get("raw_depth_0")
        if not relative:
            raise ValueError(
                f"Missing depths.raw_depth_0 in {episode}/data.json frame {frame_index}"
            )
        raw_path = (episode / relative).resolve()
        try:
            raw_path.relative_to(episode)
        except ValueError as exc:
            raise ValueError(f"Raw-depth path escapes its episode: {relative}") from exc
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing raw-depth PNG: {raw_path}")
        if include_surface_normals:
            aligned_relative = (frame.get("depths") or {}).get("depth_0")
            if not aligned_relative:
                raise ValueError(
                    f"Missing depths.depth_0 in {episode}/data.json frame {frame_index}"
                )
            aligned_path = (episode / aligned_relative).resolve()
            try:
                aligned_path.relative_to(episode)
            except ValueError as exc:
                raise ValueError(
                    f"Aligned-depth path escapes its episode: {aligned_relative}"
                ) from exc
            if not aligned_path.is_file():
                raise FileNotFoundError(f"Missing aligned-depth PNG: {aligned_path}")
    total_frames += len(frames)

print(f"Validated {total_frames} referenced raw_depth_0 PNG files.")
PY

WORK_DIR="$(mktemp -d -p "$TASK_DIR" '.lerobot_conversion.XXXXXX')"
STAGED_RAW="$WORK_DIR/processed_episodes"
STAGED_TASK="$STAGED_RAW/input_task"
HF_HOME_TEMP="$WORK_DIR/huggingface"
DATASET_ROOT="$WORK_DIR/datasets"
V3_DATASET="$HF_HOME_TEMP/lerobot/$REPO_ID"
V2_DATASET="$DATASET_ROOT/$REPO_ID"
V3_BACKUP="$DATASET_ROOT/${REPO_ID}_v3.0"
OLD_LEROBOT="$WORK_DIR/input_before_conversion"
SOURCE_MOVED=false

cleanup() {
    if [[ "$SOURCE_MOVED" == true &&
          ! -e "$LEROBOT_DIR" &&
          -d "$OLD_LEROBOT" ]]; then
        mv -- "$OLD_LEROBOT" "$LEROBOT_DIR"
        printf 'Restored original input directory after an interrupted replacement.\n' >&2
    fi

    if [[ -d "$WORK_DIR" ]]; then
        rm -rf -- "$WORK_DIR"
    fi
}

trap cleanup EXIT

mkdir -p "$STAGED_TASK" "$DATASET_ROOT"
for episode_dir in "${EPISODE_DIRS[@]}"; do
    ln -s -- "$episode_dir" "$STAGED_TASK/$(basename "$episode_dir")"
done

printf '\nInput:       %s\n' "$LEROBOT_DIR"
printf 'Episodes:    %d\n' "${#EPISODE_DIRS[@]}"
printf 'Repository:  %s\n' "$REPO_ID"
if [[ "$INCLUDE_SURFACE_NORMALS" == 1 ]]; then
    printf 'Geometry:    linear depth + camera-frame XYZ surface normals\n'
fi
printf 'Final form:  LeRobot v2.1 with H.264 video and lossless depth sidecars\n\n'

printf '[1/6] Converting processed Unitree episodes to LeRobot v3.0...\n'

(
    cd "$UNITREE_LEROBOT_REPO"

    SURFACE_NORMAL_ARGS=()
    if [[ "$INCLUDE_SURFACE_NORMALS" == 1 ]]; then
        SURFACE_NORMAL_ARGS+=(--include-surface-normals)
    fi

    HF_HOME="$HF_HOME_TEMP" \
    HF_LEROBOT_HOME="$HF_HOME_TEMP/lerobot" \
    "$UNITREE_PYTHON" "$UNITREE_CONVERTER" \
        --raw-dir "$STAGED_RAW" \
        --repo-id "$REPO_ID" \
        --robot-type Unitree_G1_Dex3_HeadOnly \
        --mode video \
        --include-depth \
        --depth-near-m "$DEPTH_NEAR_M" \
        --depth-far-m "$DEPTH_FAR_M" \
        "${SURFACE_NORMAL_ARGS[@]}"
)

[[ -d "$V3_DATASET" ]] || \
    die "Unitree converter did not create the expected dataset: $V3_DATASET"

mv -- "$V3_DATASET" "$V2_DATASET"

printf '[2/6] Converting LeRobot v3.0 to v2.1...\n'

uv run \
    --project "$V3_TO_V2_PROJECT" \
    python "$V3_TO_V2_SCRIPT" \
    --repo-id "$REPO_ID" \
    --root "$DATASET_ROOT"

printf '[3/6] Transcoding dataset video from AV1 to H.264...\n'

(
    cd "$ISAAC_GROOT_REPO"

    uv run --no-sync \
        python "$H264_SCRIPT" \
        "$V2_DATASET" \
        --jobs "$JOBS"
)

printf '[4/6] Preserving uint16 depth PNGs losslessly...\n'

"$UNITREE_PYTHON" - "$STAGED_TASK" "$V2_DATASET" "$INCLUDE_SURFACE_NORMALS" <<'PY'
import cv2
import json
from pathlib import Path
import shutil
import sys

raw_root = Path(sys.argv[1])
dataset = Path(sys.argv[2])
include_surface_normals = bool(int(sys.argv[3]))

episodes = sorted(
    path for path in raw_root.iterdir()
    if path.is_dir() and (path / "data.json").is_file()
)

with (dataset / "meta" / "info.json").open(encoding="utf-8") as file:
    info = json.load(file)

if len(episodes) != int(info["total_episodes"]):
    raise ValueError(
        f"Raw/v2.1 episode mismatch: {len(episodes)} != {info['total_episodes']}"
    )

chunks_size = int(info["chunks_size"])
expected_shape = None
expected_scale = None
copied = 0
aligned_copied = 0

for episode_index, episode in enumerate(episodes):
    with (episode / "data.json").open(encoding="utf-8") as file:
        payload = json.load(file)

    depth_info = payload.get("info", {}).get("depth", {})
    scale = float(depth_info.get("scale_m_per_unit", 0.001))
    shape = (int(depth_info.get("height", 480)), int(depth_info.get("width", 640)))
    if expected_scale is None:
        expected_scale = scale
        expected_shape = shape
    if scale != expected_scale or shape != expected_shape:
        raise ValueError(
            f"Raw-depth metadata differs in {episode}: "
            f"scale={scale}, shape={shape}; expected scale={expected_scale}, "
            f"shape={expected_shape}"
        )

    frames = payload["data"]
    for frame_index, frame in enumerate(frames):
        relative = (frame.get("depths") or {}).get("raw_depth_0")
        if not relative:
            raise ValueError(f"Missing raw_depth_0 in {episode}, frame {frame_index}")
        source = (episode / relative).resolve()
        source.relative_to(episode.resolve())

        if copied == 0:
            image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise RuntimeError(f"Cannot read raw-depth PNG: {source}")
            if image.dtype.name != "uint16" or image.shape != expected_shape:
                raise ValueError(
                    f"Expected uint16 raw depth with shape {expected_shape}; "
                    f"got dtype={image.dtype}, shape={image.shape}: {source}"
                )

        destination = dataset / (
            f"raw_depths/chunk-{episode_index // chunks_size:03d}/"
            f"episode_{episode_index:06d}/frame_{frame_index:06d}.png"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        copied += 1

        if include_surface_normals:
            aligned_relative = (frame.get("depths") or {}).get("depth_0")
            if not aligned_relative:
                raise ValueError(f"Missing depth_0 in {episode}, frame {frame_index}")
            aligned_source = (episode / aligned_relative).resolve()
            aligned_source.relative_to(episode.resolve())
            if aligned_copied == 0:
                aligned_image = cv2.imread(str(aligned_source), cv2.IMREAD_UNCHANGED)
                if aligned_image is None:
                    raise RuntimeError(f"Cannot read aligned-depth PNG: {aligned_source}")
                if aligned_image.dtype.name != "uint16" or aligned_image.shape != expected_shape:
                    raise ValueError(
                        f"Expected uint16 aligned depth with shape {expected_shape}; "
                        f"got dtype={aligned_image.dtype}, shape={aligned_image.shape}: "
                        f"{aligned_source}"
                    )
            aligned_destination = dataset / (
                f"aligned_depths/chunk-{episode_index // chunks_size:03d}/"
                f"episode_{episode_index:06d}/frame_{frame_index:06d}.png"
            )
            aligned_destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(aligned_source, aligned_destination)
            aligned_copied += 1

if copied != int(info["total_frames"]):
    raise ValueError(f"Raw-depth frame mismatch: copied {copied}, expected {info['total_frames']}")

info["raw_depth_encoding"] = {
    "source_key": "raw_depth_0",
    "storage": "lossless_png",
    "path": (
        "raw_depths/chunk-{episode_chunk:03d}/"
        "episode_{episode_index:06d}/frame_{frame_index:06d}.png"
    ),
    "dtype": "uint16",
    "shape": list(expected_shape),
    "scale_m_per_unit": expected_scale,
    "invalid_value": 0,
    "total_files": copied,
}

if include_surface_normals:
    if aligned_copied != int(info["total_frames"]):
        raise ValueError(
            f"Aligned-depth frame mismatch: copied {aligned_copied}, "
            f"expected {info['total_frames']}"
        )
    info["aligned_depth_encoding"] = {
        "source_key": "depth_0",
        "aligned_to": "color_0",
        "storage": "lossless_png",
        "path": (
            "aligned_depths/chunk-{episode_chunk:03d}/"
            "episode_{episode_index:06d}/frame_{frame_index:06d}.png"
        ),
        "dtype": "uint16",
        "shape": list(expected_shape),
        "scale_m_per_unit": expected_scale,
        "invalid_value": 0,
        "total_files": aligned_copied,
    }

with (dataset / "meta" / "info.json").open("w", encoding="utf-8") as file:
    json.dump(info, file, indent=4)
    file.write("\n")

print(f"Preserved {copied} lossless raw-depth PNG files.")
if include_surface_normals:
    print(f"Preserved {aligned_copied} lossless aligned-depth PNG files.")
PY

printf '[5/6] Installing the Unitree G1 modality definition...\n'

install -m 0644 \
    "$MODALITY_FILE" \
    "$V2_DATASET/meta/modality.json"

printf '[6/6] Validating metadata, episode count, raw depth and video codecs...\n'

"$UNITREE_PYTHON" - \
    "$V2_DATASET" \
    "${#EPISODE_DIRS[@]}" \
    "$DEPTH_NEAR_M" \
    "$DEPTH_FAR_M" \
    "$INCLUDE_SURFACE_NORMALS" <<'PY'
import json
from pathlib import Path
import sys

dataset = Path(sys.argv[1])
expected_episodes = int(sys.argv[2])
expected_depth_near_m = float(sys.argv[3])
expected_depth_far_m = float(sys.argv[4])
include_surface_normals = bool(int(sys.argv[5]))

with (dataset / "meta" / "info.json").open(encoding="utf-8") as file:
    info = json.load(file)

assert info["codebase_version"] == "v2.1", info["codebase_version"]
assert info["robot_type"] == "Unitree_G1_Dex3_HeadOnly", info["robot_type"]
assert info["total_episodes"] == expected_episodes, (
    info["total_episodes"],
    expected_episodes,
)
assert (dataset / "meta" / "modality.json").is_file()

features = info["features"]

assert features["observation.state"]["shape"] == [28]
assert features["action"]["shape"] == [28]
assert "observation.images.ego_view" in features
assert "observation.images.depth_gray_view" in features
assert features["observation.images.ego_view"]["dtype"] == "video"
assert features["observation.images.depth_gray_view"]["dtype"] == "video"
if include_surface_normals:
    assert "observation.images.surface_normals_view" in features
    assert features["observation.images.surface_normals_view"]["dtype"] == "video"

depth_encoding = info["depth_encoding"]
assert depth_encoding["feature_key"] == "observation.images.depth_gray_view"
assert depth_encoding["near_m"] == expected_depth_near_m
assert depth_encoding["far_m"] == expected_depth_far_m

raw_depth_encoding = info["raw_depth_encoding"]
assert raw_depth_encoding["source_key"] == "raw_depth_0"
assert raw_depth_encoding["storage"] == "lossless_png"
assert raw_depth_encoding["dtype"] == "uint16"
assert raw_depth_encoding["shape"] == [480, 640]
assert raw_depth_encoding["total_files"] == info["total_frames"]

if include_surface_normals:
    surface_encoding = info["surface_normals_encoding"]
    assert surface_encoding["source_key"] == "depth_0"
    assert surface_encoding["feature_key"] == (
        "observation.images.surface_normals_view"
    )
    assert surface_encoding["aligned_to"] == "color_0"
    assert surface_encoding["encoding"] == "camera_xyz_uint8"
    assert surface_encoding["encoding_version"] == 1
    aligned_depth_encoding = info["aligned_depth_encoding"]
    assert aligned_depth_encoding["source_key"] == "depth_0"
    assert aligned_depth_encoding["aligned_to"] == "color_0"
    assert aligned_depth_encoding["storage"] == "lossless_png"
    assert aligned_depth_encoding["dtype"] == "uint16"
    assert aligned_depth_encoding["shape"] == [480, 640]
    assert aligned_depth_encoding["total_files"] == info["total_frames"]

raw_depth_template = raw_depth_encoding["path"]
chunks_size = int(info["chunks_size"])
for episode_index, episode in enumerate(
    json.loads(line)
    for line in (dataset / "meta" / "episodes.jsonl").read_text().splitlines()
):
    for frame_index in range(int(episode["length"])):
        raw_depth = dataset / raw_depth_template.format(
            episode_chunk=episode_index // chunks_size,
            episode_index=episode_index,
            frame_index=frame_index,
        )
        assert raw_depth.is_file(), raw_depth

with (dataset / "meta" / "modality.json").open(encoding="utf-8") as file:
    modality = json.load(file)

assert modality["video"]["ego_view"]["original_key"] == (
    "observation.images.ego_view"
)
assert modality["video"]["depth_gray_view"]["original_key"] == (
    "observation.images.depth_gray_view"
)
if include_surface_normals:
    assert modality["video"]["surface_normals_view"]["original_key"] == (
        "observation.images.surface_normals_view"
    )

feature_keys = ["observation.images.ego_view", "observation.images.depth_gray_view"]
if include_surface_normals:
    feature_keys.append("observation.images.surface_normals_view")
for feature_key in feature_keys:
    videos = list((dataset / "videos").glob(f"chunk-*/{feature_key}/episode_*.mp4"))
    assert len(videos) == expected_episodes, (feature_key, len(videos), expected_episodes)

print(
    f"Validated v2.1 metadata: {info['total_episodes']} episodes, "
    f"{info['total_frames']} frames at {info['fps']} FPS."
)
PY

mapfile -d '' VIDEOS < <(
    find "$V2_DATASET/videos" \
        -type f \
        -name '*.mp4' \
        -print0 |
    sort -z
)

[[ ${#VIDEOS[@]} -gt 0 ]] || \
    die "No MP4 videos found in $V2_DATASET/videos"

for video in "${VIDEOS[@]}"; do
    codec="$(
        ffprobe \
            -v error \
            -select_streams v:0 \
            -show_entries stream=codec_name \
            -of default=nw=1:nk=1 \
            "$video"
    )"

    [[ "$codec" == "h264" ]] || \
        die "Expected H.264 but found '$codec': $video"

    if [[ "$INCLUDE_SURFACE_NORMALS" == 1 &&
          "$video" == *"/observation.images.surface_normals_view/"* ]]; then
        pixel_format="$(
            ffprobe \
                -v error \
                -select_streams v:0 \
                -show_entries stream=pix_fmt \
                -of default=nw=1:nk=1 \
                "$video"
        )"

        [[ "$pixel_format" == "yuv444p" ]] || \
            die "Expected yuv444p surface-normal video but found '$pixel_format': $video"
    fi
done

printf 'Validated %d H.264 video files.\n' "${#VIDEOS[@]}"

# NVIDIA's converter retains a v3.0 backup. This intermediate copy is no
# longer needed because the caller supplied a disposable input directory.
if [[ -d "$V3_BACKUP" ]]; then
    rm -rf -- "$V3_BACKUP"
fi

# Replace only the explicitly supplied and validated input directory.
SOURCE_MOVED=true
mv -- "$LEROBOT_DIR" "$OLD_LEROBOT"
mv -- "$V2_DATASET" "$LEROBOT_DIR"
SOURCE_MOVED=false

rm -rf -- "$OLD_LEROBOT"

printf '\nConversion complete.\n'
printf 'Final dataset: %s\n' "$LEROBOT_DIR"
