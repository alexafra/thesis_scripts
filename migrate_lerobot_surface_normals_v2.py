#!/home/alex/miniconda3/envs/unitree_lerobot/bin/python
"""Migrate a split LeRobot corpus from surface-normal encoding v1 to v2.

Only the model-visible surface-normal payload is changed.  RGB, gray-depth,
Parquet, raw/aligned depth, split manifests, and low-dimensional statistics are
hard-linked from the immutable source corpus into a sibling dataset.  Existing
canonical v1 normal bytes are decoded from their lossless LZ4 chunks and kept
unchanged wherever the retained aligned uint16 depth passes the v2 five-sample
range predicate.  Invalid pixels alone are zeroed; normals are not recomputed.

The build is resumable per episode.  Completed LZ4 chunks, lossless H.264
backups, and per-episode statistics live in a hidden sibling work tree until
every split validates.  There is no training mode in this tool.
"""

from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
import ctypes
import dataclasses
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any
import uuid

import av
import cv2
from lerobot.datasets.compute_stats import (
    aggregate_feature_stats,
    auto_downsample_height_width,
    get_feature_stats,
    sample_indices,
)
import numpy as np
from unitree_lerobot.utils.surface_normal_encoding import (
    LEGACY_SURFACE_NORMAL_ENCODING_VERSION,
    SURFACE_NORMAL_ENCODING_VERSION,
    PinholeIntrinsics,
    pinhole_intrinsics_from_metadata,
    surface_normals_encoding_metadata,
)


FEATURE_KEY = "observation.images.surface_normals_view"
SPLITS = ("train", "validation", "test")
STATE_FILENAME = ".surface_normals_v2_migration_state.json"
STATE_VERSION = 2
CHUNK_FRAMES = 32
DEFAULT_JOBS = 6
DEFAULT_MIN_FREE_GIB = 100
H264_FILENAME = "h264_backup.mp4"
NORMAL_STATS_KEY = "surface_normal_stats"
MIGRATION_OPERATION = "surface-normal-v1-to-v2-canonical-lz4-range-mask"
MIGRATION_METHOD = "preserve_v1_normal_bytes_and_zero_v2_out_of_range_pixels"
SOURCE_CHUNK_REUSED = "hardlinked_unchanged_source_v1_chunk"
SOURCE_CHUNK_MASKED = "range_masked_and_recompressed"
LZ4_BIN = Path(shutil.which("lz4") or "/home/alex/miniconda3/bin/lz4")
AT_FDCWD = -100
RENAME_NOREPLACE = 1


class MigrationError(RuntimeError):
    """Raised when the source or resumable migration state is unsafe."""


@dataclasses.dataclass(frozen=True)
class EpisodePlan:
    split: str
    episode_index: int
    frame_count: int
    depth_paths: tuple[Path, ...]


@dataclasses.dataclass(frozen=True)
class SplitPlan:
    name: str
    source_root: Path
    episodes: tuple[EpisodePlan, ...]
    info: dict[str, Any]
    intrinsics: PinholeIntrinsics
    depth_scale_m_per_unit: float
    depth_near_m: float
    depth_far_m: float
    max_neighbor_depth_delta_m: float
    old_encoding: dict[str, Any]
    new_encoding: dict[str, Any]
    lz4_relative: Path
    backup_relative: Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("check", "build"))
    parser.add_argument("source", type=Path, help="Immutable v1 split-dataset root")
    parser.add_argument("target", type=Path, help="Absent sibling path to publish as v2")
    parser.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    parser.add_argument("--chunk-frames", type=int, default=CHUNK_FRAMES)
    parser.add_argument("--min-free-gib", type=int, default=DEFAULT_MIN_FREE_GIB)
    parser.add_argument(
        "--max-new-episodes",
        type=int,
        help=(
            "Process at most this many previously incomplete episodes and exit "
            "without publishing. This is useful for bounded/resume checks."
        ),
    )
    return parser.parse_args(argv)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise MigrationError(f"Cannot read JSON {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise MigrationError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise MigrationError(f"Cannot read JSONL {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise MigrationError(f"Invalid JSON on line {line_number} of {path}") from exc
        if not isinstance(value, dict):
            raise MigrationError(f"Line {line_number} of {path} is not an object")
        records.append(value)
    return records


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False, default=_json_default)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_jsonl_atomic(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            for record in records:
                json.dump(record, stream, ensure_ascii=False, default=_json_default)
                stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _jsonable(value: Any) -> Any:
    """Normalize NumPy-backed values to their exact JSON representation."""

    return json.loads(json.dumps(value, default=_json_default))


def rename_directory_noreplace(source: Path, target: Path) -> None:
    """Atomically publish ``source`` only when ``target`` is still absent.

    Plain ``os.rename`` can replace a concurrently created empty target
    directory on Linux.  This migration must instead fail closed and preserve
    its completed work tree for inspection/resume.
    """

    libc = ctypes.CDLL(None, use_errno=True)
    renameat2 = getattr(libc, "renameat2", None)
    if renameat2 is None:
        raise MigrationError(
            "Atomic no-replace publication requires Linux renameat2; "
            "completed work was preserved"
        )
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    result = renameat2(
        AT_FDCWD,
        os.fsencode(source),
        AT_FDCWD,
        os.fsencode(target),
        RENAME_NOREPLACE,
    )
    if result == 0:
        directory_fd = os.open(target.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        except OSError as exc:
            raise MigrationError(
                f"Published {target}, but could not fsync its parent directory: {exc}"
            ) from exc
        finally:
            os.close(directory_fd)
        return
    error_number = ctypes.get_errno()
    if error_number == errno.EEXIST:
        raise FileExistsError(error_number, os.strerror(error_number), target)
    if error_number in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
        raise MigrationError(
            "The filesystem does not support atomic no-replace publication; "
            "completed work was preserved"
        )
    raise OSError(error_number, os.strerror(error_number), target)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _v2_depth_range_mask(
    depth_u16: np.ndarray,
    *,
    scale_m_per_unit: float,
    depth_near_m: float,
    depth_far_m: float,
) -> np.ndarray:
    """Return the exact v2 centre/left/right/up/down inclusive range mask."""

    depth = np.asarray(depth_u16)
    if depth.dtype != np.uint16 or depth.ndim != 2:
        raise ValueError(
            f"Expected an HxW uint16 depth image; got shape={depth.shape}, "
            f"dtype={depth.dtype}"
        )
    depth_m = depth.astype(np.float32) * np.float32(scale_m_per_unit)
    in_range = (depth_m >= depth_near_m) & (depth_m <= depth_far_m)
    valid = np.zeros(depth.shape, dtype=np.bool_)
    valid[1:-1, 1:-1] = (
        in_range[1:-1, 1:-1]
        & in_range[1:-1, :-2]
        & in_range[1:-1, 2:]
        & in_range[:-2, 1:-1]
        & in_range[2:, 1:-1]
    )
    return valid


def _link_file(source: Path, target: Path) -> None:
    """Hard-link a verified immutable source payload into migration scratch."""

    try:
        os.link(source, target)
    except OSError as exc:
        raise MigrationError(f"Cannot hard-link {source} to {target}: {exc}") from exc


def _safe_relative(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise MigrationError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise MigrationError(f"{label} must remain inside its split: {value!r}")
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise MigrationError(f"{label} escapes its split: {value!r}") from exc
    return relative


def _finite_float(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise MigrationError(f"{label} must be finite, got {value!r}") from exc
    if not math.isfinite(parsed):
        raise MigrationError(f"{label} must be finite, got {value!r}")
    return parsed


def source_stat_fingerprint(root: Path) -> str:
    """Fingerprint source paths/sizes/mtimes plus all small metadata bytes.

    Large media are not re-hashed here.  Every aligned depth input is hashed
    while its episode is generated and rechecked before publication.
    """

    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise MigrationError(f"Source corpus may not contain symlinks: {path}")
        if path.is_dir():
            continue
        if not path.is_file():
            raise MigrationError(f"Source corpus contains a non-regular file: {path}")
        relative = path.relative_to(root).as_posix().encode("utf-8")
        stat = path.stat()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(stat.st_size.to_bytes(8, "big"))
        digest.update(stat.st_mtime_ns.to_bytes(8, "big", signed=True))
        if "/meta/" in f"/{relative.decode('utf-8')}" or len(Path(relative.decode()).parts) == 1:
            digest.update(bytes.fromhex(sha256_file(path)))
    return f"sha256:{digest.hexdigest()}"


def _expected_v1_and_v2_encodings(
    info: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], PinholeIntrinsics, float, float, float, float]:
    old = info.get("surface_normals_encoding")
    if not isinstance(old, dict):
        raise MigrationError("info.json has no surface_normals_encoding object")
    if old.get("encoding_version") != LEGACY_SURFACE_NORMAL_ENCODING_VERSION:
        raise MigrationError(
            "Source must use surface-normal encoding v1; "
            f"found {old.get('encoding_version')!r}"
        )
    if "depth_valid_range_m" in old:
        raise MigrationError("A v1 source may not declare depth_valid_range_m")

    depth_encoding = info.get("depth_encoding")
    if not isinstance(depth_encoding, dict):
        raise MigrationError("info.json has no depth_encoding object")
    near = _finite_float(depth_encoding.get("near_m"), "depth_encoding.near_m")
    far = _finite_float(depth_encoding.get("far_m"), "depth_encoding.far_m")
    if near < 0.0 or far <= near:
        raise MigrationError(f"Invalid depth bounds: near={near}, far={far}")

    aligned = info.get("aligned_depth_encoding")
    if not isinstance(aligned, dict):
        raise MigrationError("info.json has no aligned_depth_encoding object")
    scale = _finite_float(
        aligned.get("scale_m_per_unit"),
        "aligned_depth_encoding.scale_m_per_unit",
    )
    if scale <= 0.0:
        raise MigrationError("Aligned-depth scale must be positive")

    try:
        intrinsics = pinhole_intrinsics_from_metadata(old.get("intrinsics"))
    except (TypeError, ValueError) as exc:
        raise MigrationError(f"Invalid source surface-normal intrinsics: {exc}") from exc
    max_delta = _finite_float(
        old.get("max_neighbor_depth_delta_m"),
        "surface_normals_encoding.max_neighbor_depth_delta_m",
    )
    if max_delta <= 0.0:
        raise MigrationError("Surface-normal neighbor threshold must be positive")

    common = {
        "intrinsics": intrinsics,
        "max_neighbor_depth_delta_m": max_delta,
        "camera_calibration": old.get("camera_calibration"),
        "depth_near_m": near,
        "depth_far_m": far,
    }
    expected_v1 = surface_normals_encoding_metadata(
        **common,
        encoding_version=LEGACY_SURFACE_NORMAL_ENCODING_VERSION,
    )
    if old != expected_v1:
        raise MigrationError(
            "Source v1 surface-normal metadata does not match the shared encoder contract"
        )
    expected_v2 = surface_normals_encoding_metadata(
        **common,
        encoding_version=SURFACE_NORMAL_ENCODING_VERSION,
    )
    return expected_v1, expected_v2, intrinsics, scale, near, far, max_delta


def inspect_source(source: Path) -> tuple[SplitPlan, ...]:
    source_input = source.expanduser()
    if source_input.is_symlink():
        raise MigrationError(f"Source may not be a symlink: {source_input}")
    source = source_input.resolve()
    if not source.is_dir():
        raise MigrationError(f"Source must be a real directory: {source}")
    actual_splits = {path.name for path in source.iterdir() if path.is_dir()}
    missing = set(SPLITS) - actual_splits
    if missing:
        raise MigrationError(f"Source is missing split directories: {sorted(missing)}")

    plans: list[SplitPlan] = []
    reference_encoding: dict[str, Any] | None = None
    for split in SPLITS:
        split_root = source / split
        resolved_split_root = split_root.resolve()
        info = read_json(split_root / "meta" / "info.json")
        old, new, intrinsics, scale, near, far, max_delta = _expected_v1_and_v2_encodings(info)
        if reference_encoding is None:
            reference_encoding = old
        elif old != reference_encoding:
            raise MigrationError("All source splits must have exactly one v1 normal contract")

        features = info.get("features")
        feature = features.get(FEATURE_KEY) if isinstance(features, dict) else None
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            raise MigrationError(f"{split} does not declare {FEATURE_KEY} as video")
        expected_shape = [intrinsics.height, intrinsics.width, 3]
        if feature.get("shape") != expected_shape:
            raise MigrationError(
                f"{split} normal feature shape is {feature.get('shape')!r}; "
                f"expected {expected_shape}"
            )

        aligned = info["aligned_depth_encoding"]
        if aligned.get("dtype") != "uint16" or aligned.get("shape") != expected_shape[:2]:
            raise MigrationError(f"{split} aligned-depth metadata is not uint16 {expected_shape[:2]}")
        aligned_template = aligned.get("path")
        if not isinstance(aligned_template, str):
            raise MigrationError(f"{split} aligned-depth path template is missing")

        lz4 = info.get("surface_normals_lz4")
        if not isinstance(lz4, dict):
            raise MigrationError(f"{split} has no surface_normals_lz4 contract")
        expected_lz4 = {
            "feature_key": FEATURE_KEY,
            "storage": "plain_lz4_chunks",
            "dtype": "uint8",
            "layout": "FHWC",
            "lossless_round_trip_verified": True,
        }
        for key, value in expected_lz4.items():
            if lz4.get(key) != value:
                raise MigrationError(f"{split} surface_normals_lz4.{key} is unsupported")
        if int(lz4.get("chunk_frames", -1)) != CHUNK_FRAMES:
            raise MigrationError(f"{split} must use {CHUNK_FRAMES}-frame LZ4 chunks")
        lz4_relative = _safe_relative(split_root, lz4.get("root"), f"{split} LZ4 root")
        backup_relative = _safe_relative(
            split_root,
            lz4.get("h264_backup"),
            f"{split} H.264 backup root",
        )
        if not (split_root / lz4_relative).is_dir():
            raise MigrationError(f"Missing source LZ4 root: {split_root / lz4_relative}")
        if not (split_root / backup_relative).is_dir():
            raise MigrationError(f"Missing source H.264 backup: {split_root / backup_relative}")

        episode_records = read_jsonl(split_root / "meta" / "episodes.jsonl")
        indices = [int(record.get("episode_index", -1)) for record in episode_records]
        if indices != list(range(len(episode_records))):
            raise MigrationError(f"{split} episode indices must be contiguous and ordered")
        if len(episode_records) != int(info.get("total_episodes", -1)):
            raise MigrationError(f"{split} episode count differs from info.json")
        chunks_size = int(info.get("chunks_size", 1000))
        episodes: list[EpisodePlan] = []
        total_frames = 0
        for record in episode_records:
            episode_index = int(record["episode_index"])
            frame_count = int(record.get("length", -1))
            if frame_count <= 0:
                raise MigrationError(f"{split} episode {episode_index} has no frames")
            depth_paths: list[Path] = []
            for frame_index in range(frame_count):
                try:
                    rendered = aligned_template.format(
                        episode_chunk=episode_index // chunks_size,
                        episode_index=episode_index,
                        frame_index=frame_index,
                    )
                except (KeyError, ValueError) as exc:
                    raise MigrationError(
                        f"{split} aligned-depth path template is invalid"
                    ) from exc
                relative = Path(rendered)
                if relative.is_absolute() or ".." in relative.parts:
                    raise MigrationError(
                        f"{split} aligned-depth path escapes its split: {rendered!r}"
                    )
                depth_path = split_root / relative
                try:
                    resolved_depth_path = depth_path.resolve(strict=True)
                    resolved_depth_path.relative_to(resolved_split_root)
                except (OSError, RuntimeError, ValueError) as exc:
                    raise MigrationError(
                        f"Aligned depth frame escapes its split or is missing: {depth_path}"
                    ) from exc
                if resolved_depth_path != depth_path.absolute():
                    raise MigrationError(
                        f"Aligned depth path may not traverse a symlink: {depth_path}"
                    )
                if not resolved_depth_path.is_file() or resolved_depth_path.is_symlink():
                    raise MigrationError(f"Missing aligned depth frame: {depth_path}")
                depth_paths.append(resolved_depth_path)
            episodes.append(
                EpisodePlan(
                    split=split,
                    episode_index=episode_index,
                    frame_count=frame_count,
                    depth_paths=tuple(depth_paths),
                )
            )
            source_normal_episode = (
                split_root / lz4_relative / f"episode_{episode_index:06d}"
            )
            source_normal_index = source_normal_episode / "index.json"
            source_h264 = (
                split_root / backup_relative / f"episode_{episode_index:06d}.mp4"
            )
            for source_payload, label in (
                (source_normal_index, "canonical v1 LZ4 index"),
                (source_h264, "v1 H.264 backup"),
            ):
                if (
                    not source_payload.is_file()
                    or source_payload.is_symlink()
                    or source_payload.resolve() != source_payload.absolute()
                ):
                    raise MigrationError(
                        f"Missing or unsafe {label}: {source_payload}"
                    )
            total_frames += frame_count
        if total_frames != int(info.get("total_frames", -1)):
            raise MigrationError(f"{split} frame count differs from info.json")
        if int(aligned.get("total_files", -1)) != total_frames:
            raise MigrationError(f"{split} aligned-depth total_files is stale")

        split_plan = SplitPlan(
            name=split,
            source_root=split_root,
            episodes=tuple(episodes),
            info=info,
            intrinsics=intrinsics,
            depth_scale_m_per_unit=scale,
            depth_near_m=near,
            depth_far_m=far,
            max_neighbor_depth_delta_m=max_delta,
            old_encoding=old,
            new_encoding=new,
            lz4_relative=lz4_relative,
            backup_relative=backup_relative,
        )
        for episode in split_plan.episodes:
            _read_source_v1_index(split_plan, episode)
        plans.append(split_plan)
    return tuple(plans)


def _allocated_bytes(path: Path) -> int:
    total = 0
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            total += item.stat().st_blocks * 512
    return total


def estimate_migration_bytes(plans: tuple[SplitPlan, ...]) -> int:
    total = 0
    for plan in plans:
        total += _allocated_bytes(plan.source_root / plan.lz4_relative)
        total += _allocated_bytes(plan.source_root / plan.backup_relative)
    return total


def _state_path(work: Path) -> Path:
    return work / STATE_FILENAME


def _new_state(source: Path, target: Path, source_fingerprint: str) -> dict[str, Any]:
    return {
        "version": STATE_VERSION,
        "operation": MIGRATION_OPERATION,
        "source": str(source),
        "target": str(target),
        "source_stat_fingerprint": source_fingerprint,
        "phase": "cloning",
        "encoding_version": SURFACE_NORMAL_ENCODING_VERSION,
        "migration_method": MIGRATION_METHOD,
        "normals_recomputed": False,
        "training_started": False,
    }


def _load_state(work: Path, source: Path, target: Path, fingerprint: str) -> dict[str, Any]:
    state = read_json(_state_path(work))
    expected = {
        "version": STATE_VERSION,
        "operation": MIGRATION_OPERATION,
        "source": str(source),
        "target": str(target),
        "source_stat_fingerprint": fingerprint,
        "encoding_version": SURFACE_NORMAL_ENCODING_VERSION,
        "migration_method": MIGRATION_METHOD,
        "normals_recomputed": False,
        "training_started": False,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise MigrationError(f"Resume state differs at {key}: {state.get(key)!r} != {value!r}")
    return state


def _set_phase(work: Path, state: dict[str, Any], phase: str) -> None:
    state = dict(state)
    state["phase"] = phase
    write_json_atomic(_state_path(work), state)


def hardlink_clone_resume(source: Path, work: Path) -> None:
    """Create or resume a hard-linked clone without touching source inodes."""

    for source_dir, dir_names, file_names in os.walk(source):
        source_directory = Path(source_dir)
        relative_directory = source_directory.relative_to(source)
        target_directory = work / relative_directory
        target_directory.mkdir(parents=True, exist_ok=True)
        dir_names.sort()
        file_names.sort()
        for filename in file_names:
            source_file = source_directory / filename
            target_file = target_directory / filename
            if source_file.is_symlink() or not source_file.is_file():
                raise MigrationError(f"Cannot hard-link non-regular source file: {source_file}")
            if target_file.exists() or target_file.is_symlink():
                if not target_file.is_file() or target_file.is_symlink():
                    raise MigrationError(f"Clone contains an unexpected path: {target_file}")
                left = source_file.stat()
                right = target_file.stat()
                if (left.st_dev, left.st_ino) != (right.st_dev, right.st_ino):
                    raise MigrationError(f"Clone file is not linked to source: {target_file}")
                continue
            os.link(source_file, target_file)


def _build_roots(work_split: Path, plan: SplitPlan) -> tuple[Path, Path]:
    video_parent = (work_split / plan.lz4_relative).parent
    lz4_build = video_parent / f".{FEATURE_KEY}.v2-building"
    backup_build = video_parent / f".{FEATURE_KEY}_h264_backup.v2-building"
    return lz4_build, backup_build


def _source_v1_episode_dir(plan: SplitPlan, episode: EpisodePlan) -> Path:
    return (
        plan.source_root
        / plan.lz4_relative
        / f"episode_{episode.episode_index:06d}"
    )


def _source_v1_h264_path(plan: SplitPlan, episode: EpisodePlan) -> Path:
    return (
        plan.source_root
        / plan.backup_relative
        / f"episode_{episode.episode_index:06d}.mp4"
    )


def _read_source_v1_index(
    plan: SplitPlan,
    episode: EpisodePlan,
) -> tuple[dict[str, Any], Path]:
    """Validate the canonical v1 LZ4 structural and digest contract."""

    episode_dir = _source_v1_episode_dir(plan, episode)
    index_path = episode_dir / "index.json"
    index = read_json(index_path)
    source_version = index.get(
        "surface_normals_encoding_version",
        LEGACY_SURFACE_NORMAL_ENCODING_VERSION,
    )
    if (
        int(index.get("episode_index", -1)) != episode.episode_index
        or index.get("dtype") != "uint8"
        or index.get("layout") != "FHWC"
        or index.get("transform") != "none"
        or int(index.get("chunk_frames", -1)) != CHUNK_FRAMES
        or int(index.get("frame_count", -1)) != episode.frame_count
        or source_version != LEGACY_SURFACE_NORMAL_ENCODING_VERSION
        or index.get("chunks_are_independent") is not True
        or index.get("lossless_round_trip_verified") is not True
        or not _is_sha256(index.get("decoded_reference_sha256"))
    ):
        raise MigrationError(
            f"Unsupported canonical v1 LZ4 index: {index_path}"
        )
    if "depth_valid_range_m" in index:
        raise MigrationError(f"Canonical v1 index declares a v2 depth range: {index_path}")

    next_frame = 0
    chunks = index.get("chunks")
    if not isinstance(chunks, list) or not chunks:
        raise MigrationError(f"Canonical v1 index has no chunks: {index_path}")
    for chunk_index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise MigrationError(f"Invalid canonical v1 chunk record: {index_path}")
        frame_count = int(chunk.get("frame_count", -1))
        filename = chunk.get("filename")
        if (
            int(chunk.get("chunk_index", -1)) != chunk_index
            or int(chunk.get("start_frame", -1)) != next_frame
            or frame_count <= 0
            or frame_count > CHUNK_FRAMES
            or [int(chunk.get(key, -1)) for key in ("height", "width", "channels")]
            != [plan.intrinsics.height, plan.intrinsics.width, 3]
            or int(chunk.get("uncompressed_bytes", -1))
            != frame_count * plan.intrinsics.height * plan.intrinsics.width * 3
            or not isinstance(filename, str)
            or Path(filename).name != filename
            or not _is_sha256(chunk.get("compressed_sha256"))
            or chunk.get("lossless_round_trip_verified") is not True
        ):
            raise MigrationError(
                f"Invalid canonical v1 chunk {chunk_index}: {index_path}"
            )
        chunk_path = episode_dir / filename
        if (
            not chunk_path.is_file()
            or chunk_path.is_symlink()
            or chunk_path.resolve() != chunk_path.absolute()
            or chunk_path.stat().st_size != int(chunk.get("compressed_bytes", -1))
        ):
            raise MigrationError(f"Missing or unsafe canonical v1 chunk: {chunk_path}")
        next_frame += frame_count
    if next_frame != episode.frame_count:
        raise MigrationError(f"Canonical v1 frame count is inconsistent: {index_path}")
    return index, index_path


def _decode_verified_lz4_chunk(
    path: Path,
    *,
    compressed_sha256: str,
    uncompressed_bytes: int,
) -> bytes:
    if sha256_file(path) != compressed_sha256:
        raise MigrationError(f"LZ4 compressed hash mismatch: {path}")
    try:
        raw = subprocess.run(
            [str(LZ4_BIN), "-q", "-d", "-c", str(path)],
            stdout=subprocess.PIPE,
            check=True,
        ).stdout
    except subprocess.SubprocessError as exc:
        raise MigrationError(f"Cannot decode LZ4 chunk {path}: {exc}") from exc
    if len(raw) != uncompressed_bytes:
        raise MigrationError(
            f"LZ4 size mismatch after decode: {path}; "
            f"expected {uncompressed_bytes}, found {len(raw)}"
        )
    return raw


def _compress_and_verify(raw: bytes, target: Path) -> tuple[int, str]:
    if not LZ4_BIN.is_file():
        raise MigrationError(f"LZ4 executable is missing: {LZ4_BIN}")
    with target.open("xb") as stream:
        subprocess.run(
            [str(LZ4_BIN), "-q", "-z", "-c"],
            input=raw,
            stdout=stream,
            check=True,
        )
    reconstructed = subprocess.run(
        [str(LZ4_BIN), "-q", "-d", "-c", str(target)],
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    if reconstructed != raw:
        raise MigrationError(f"LZ4 round trip is not byte exact: {target}")
    return target.stat().st_size, sha256_file(target)


def _normal_stats(sampled_chw: list[np.ndarray]) -> dict[str, Any]:
    images = np.stack(sampled_chw)
    stats = get_feature_stats(images, axis=(0, 2, 3), keepdims=True)
    normalized = {
        key: value if key == "count" else np.squeeze(value / 255.0, axis=0)
        for key, value in stats.items()
    }
    return json.loads(json.dumps(normalized, default=_json_default))


def _decode_video_signature(path: Path) -> tuple[int, str, str]:
    decoded_hash = hashlib.sha256()
    frame_count = 0
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        codec_name = stream.codec_context.name
        pixel_format = stream.codec_context.pix_fmt
        if codec_name != "h264" or pixel_format != "gbrp":
            raise MigrationError(
                f"Expected h264/gbrp backup, found {codec_name}/{pixel_format}: {path}"
            )
        for frame in container.decode(video=0):
            image = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
            decoded_hash.update(image.tobytes())
            frame_count += 1
    return frame_count, decoded_hash.hexdigest(), sha256_file(path)


def _depth_episode_digest(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        raw = path.read_bytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _validate_completed_episode(
    plan: SplitPlan,
    episode: EpisodePlan,
    lz4_build: Path,
    backup_build: Path,
    *,
    verify_source: bool = False,
    verify_payload: bool = False,
) -> dict[str, Any] | None:
    episode_dir = lz4_build / f"episode_{episode.episode_index:06d}"
    index_path = episode_dir / "index.json"
    if not index_path.is_file():
        return None
    try:
        index = read_json(index_path)
        expected_range = plan.new_encoding["depth_valid_range_m"]
        if (
            int(index.get("episode_index", -1)) != episode.episode_index
            or index.get("dtype") != "uint8"
            or index.get("layout") != "FHWC"
            or index.get("transform") != "none"
            or int(index.get("chunk_frames", -1)) != CHUNK_FRAMES
            or int(index.get("frame_count", -1)) != episode.frame_count
            or index.get("surface_normals_encoding_version")
            != SURFACE_NORMAL_ENCODING_VERSION
            or index.get("depth_valid_range_m") != expected_range
            or not index.get("chunks_are_independent")
            or not index.get("lossless_round_trip_verified")
            or not isinstance(index.get(NORMAL_STATS_KEY), dict)
            or index.get("migration_method") != MIGRATION_METHOD
            or index.get("normals_recomputed") is not False
            or index.get("source_v1_semantics")
            != "trusted_declared_dataset_contract"
            or not _is_sha256(index.get("source_v1_lz4_index_sha256"))
            or not _is_sha256(index.get("source_v1_decoded_reference_sha256"))
            or int(index.get("range_mask_invalid_pixels", -1)) < 0
            or int(index.get("range_mask_newly_zeroed_pixels", -1)) < 0
            or int(index.get("source_lz4_chunks_reused", -1)) < 0
            or int(index.get("source_lz4_chunks_rewritten", -1)) < 0
            or not isinstance(index.get("h264_backup_reused_from_source"), bool)
        ):
            return None
        next_frame = 0
        compressed_bytes = 0
        reused_chunks = 0
        rewritten_chunks = 0
        newly_zeroed_pixels = 0
        reconstructed_hash = hashlib.sha256() if verify_payload else None
        chunks = index.get("chunks")
        if not isinstance(chunks, list) or not chunks:
            return None
        for chunk_index, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                return None
            frame_count = int(chunk.get("frame_count", -1))
            filename = chunk.get("filename")
            if (
                int(chunk.get("chunk_index", -1)) != chunk_index
                or int(chunk.get("start_frame", -1)) != next_frame
                or frame_count <= 0
                or frame_count > CHUNK_FRAMES
                or [int(chunk.get(key, -1)) for key in ("height", "width", "channels")]
                != [plan.intrinsics.height, plan.intrinsics.width, 3]
                or int(chunk.get("uncompressed_bytes", -1))
                != frame_count * plan.intrinsics.height * plan.intrinsics.width * 3
                or not isinstance(filename, str)
                or Path(filename).name != filename
                or chunk.get("lossless_round_trip_verified") is not True
                or chunk.get("migration_action")
                not in {SOURCE_CHUNK_REUSED, SOURCE_CHUNK_MASKED}
                or int(chunk.get("range_mask_newly_zeroed_pixels", -1)) < 0
            ):
                return None
            action = chunk["migration_action"]
            chunk_newly_zeroed = int(chunk["range_mask_newly_zeroed_pixels"])
            if (action == SOURCE_CHUNK_REUSED) != (chunk_newly_zeroed == 0):
                return None
            reused_chunks += int(action == SOURCE_CHUNK_REUSED)
            rewritten_chunks += int(action == SOURCE_CHUNK_MASKED)
            newly_zeroed_pixels += chunk_newly_zeroed
            next_frame += frame_count
            chunk_path = episode_dir / filename
            expected_compressed_hash = chunk.get("compressed_sha256")
            if (
                not chunk_path.is_file()
                or chunk_path.stat().st_size != int(chunk.get("compressed_bytes", -1))
                or not isinstance(expected_compressed_hash, str)
                or len(expected_compressed_hash) != 64
                or sha256_file(chunk_path) != expected_compressed_hash
            ):
                return None
            if verify_payload:
                raw = subprocess.run(
                    [str(LZ4_BIN), "-q", "-d", "-c", str(chunk_path)],
                    stdout=subprocess.PIPE,
                    check=True,
                ).stdout
                if len(raw) != int(chunk["uncompressed_bytes"]):
                    return None
                assert reconstructed_hash is not None
                reconstructed_hash.update(raw)
            compressed_bytes += chunk_path.stat().st_size
        if next_frame != episode.frame_count:
            return None
        if (
            int(index["source_lz4_chunks_reused"]) != reused_chunks
            or int(index["source_lz4_chunks_rewritten"]) != rewritten_chunks
            or reused_chunks + rewritten_chunks != len(chunks)
            or int(index["range_mask_newly_zeroed_pixels"])
            != newly_zeroed_pixels
            or bool(index["h264_backup_reused_from_source"])
            != (newly_zeroed_pixels == 0)
        ):
            return None
        total_pixels = (
            episode.frame_count * plan.intrinsics.height * plan.intrinsics.width
        )
        invalid_pixels = int(index["range_mask_invalid_pixels"])
        if (
            invalid_pixels > total_pixels
            or newly_zeroed_pixels > invalid_pixels
        ):
            return None
        expected_decoded_hash = index.get("decoded_reference_sha256")
        if not isinstance(expected_decoded_hash, str) or len(expected_decoded_hash) != 64:
            return None
        if verify_payload and reconstructed_hash.hexdigest() != expected_decoded_hash:
            return None

        internal_video = episode_dir / H264_FILENAME
        external_video = backup_build / f"episode_{episode.episode_index:06d}.mp4"
        videos = [path for path in (internal_video, external_video) if path.is_file()]
        if not videos:
            return None
        expected_h264_hash = index.get("h264_backup_sha256")
        if not isinstance(expected_h264_hash, str) or len(expected_h264_hash) != 64:
            return None
        for video in videos:
            if verify_payload:
                video_frames, video_decoded_hash, video_hash = _decode_video_signature(video)
                if (
                    video_frames != episode.frame_count
                    or video_decoded_hash != expected_decoded_hash
                    or video_hash != expected_h264_hash
                ):
                    return None
            elif sha256_file(video) != expected_h264_hash:
                return None
        expected_depth_hash = index.get("aligned_depth_png_sequence_sha256")
        if not isinstance(expected_depth_hash, str) or len(expected_depth_hash) != 64:
            return None
        if verify_source:
            if _depth_episode_digest(episode.depth_paths) != expected_depth_hash:
                return None
            source_index, source_index_path = _read_source_v1_index(plan, episode)
            if (
                sha256_file(source_index_path)
                != index["source_v1_lz4_index_sha256"]
                or source_index["decoded_reference_sha256"]
                != index["source_v1_decoded_reference_sha256"]
                or len(source_index["chunks"]) != len(chunks)
            ):
                return None
            for target_chunk, source_chunk in zip(
                chunks,
                source_index["chunks"],
                strict=True,
            ):
                if target_chunk["migration_action"] != SOURCE_CHUNK_REUSED:
                    continue
                target_path = episode_dir / target_chunk["filename"]
                source_path = source_index_path.parent / source_chunk["filename"]
                target_stat = target_path.stat()
                source_stat = source_path.stat()
                if (
                    target_chunk["compressed_sha256"]
                    != source_chunk["compressed_sha256"]
                    or (target_stat.st_dev, target_stat.st_ino)
                    != (source_stat.st_dev, source_stat.st_ino)
                ):
                    return None
            if index["h264_backup_reused_from_source"]:
                source_h264 = _source_v1_h264_path(plan, episode)
                for video in videos:
                    video_stat = video.stat()
                    source_video_stat = source_h264.stat()
                    if (video_stat.st_dev, video_stat.st_ino) != (
                        source_video_stat.st_dev,
                        source_video_stat.st_ino,
                    ):
                        return None
        index["compressed_bytes"] = compressed_bytes
        return index
    except (
        KeyError,
        TypeError,
        ValueError,
        OSError,
        MigrationError,
        json.JSONDecodeError,
        subprocess.SubprocessError,
        av.error.FFmpegError,
    ):
        return None


def _encode_h264_from_lz4_chunks(
    plan: SplitPlan,
    episode: EpisodePlan,
    episode_dir: Path,
    chunks: list[dict[str, Any]],
    video_path: Path,
) -> None:
    encoded_frames = 0
    with av.open(str(video_path), "w") as output:
        stream = output.add_stream(
            "libx264rgb",
            int(plan.info["fps"]),
            options={"g": "2", "crf": "0"},
        )
        stream.width = plan.intrinsics.width
        stream.height = plan.intrinsics.height
        stream.pix_fmt = "rgb24"
        for chunk in chunks:
            raw = _decode_verified_lz4_chunk(
                episode_dir / chunk["filename"],
                compressed_sha256=chunk["compressed_sha256"],
                uncompressed_bytes=int(chunk["uncompressed_bytes"]),
            )
            frames = np.frombuffer(raw, dtype=np.uint8).reshape(
                int(chunk["frame_count"]),
                plan.intrinsics.height,
                plan.intrinsics.width,
                3,
            )
            for normal in frames:
                video_frame = av.VideoFrame.from_ndarray(normal, format="rgb24")
                for packet in stream.encode(video_frame):
                    output.mux(packet)
                encoded_frames += 1
        for packet in stream.encode():
            output.mux(packet)
    if encoded_frames != episode.frame_count:
        raise MigrationError(
            f"Encoded {encoded_frames} H.264 frames for "
            f"{plan.name}/{episode.episode_index}; expected {episode.frame_count}"
        )


def _encode_episode(
    plan: SplitPlan,
    episode: EpisodePlan,
    lz4_build: Path,
    backup_build: Path,
) -> tuple[str, int, int, str]:
    completed = _validate_completed_episode(plan, episode, lz4_build, backup_build)
    if completed is not None:
        return plan.name, episode.episode_index, int(completed["compressed_bytes"]), "resumed"

    final_dir = lz4_build / f"episode_{episode.episode_index:06d}"
    if final_dir.exists():
        shutil.rmtree(final_dir)
    temporary = lz4_build / f".episode_{episode.episode_index:06d}.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    video_path = temporary / H264_FILENAME
    source_index, source_index_path = _read_source_v1_index(plan, episode)
    sampled_indices_order = sample_indices(episode.frame_count)
    sampled_index_set = set(sampled_indices_order)
    sampled_by_index: dict[int, np.ndarray] = {}
    source_depth_hash = hashlib.sha256()
    source_v1_decoded_hash = hashlib.sha256()
    target_v2_decoded_hash = hashlib.sha256()
    chunks: list[dict[str, Any]] = []
    frame_count = 0
    range_mask_invalid_pixels = 0
    newly_zeroed_pixels = 0
    reused_chunks = 0
    rewritten_chunks = 0

    try:
        source_episode_dir = source_index_path.parent
        for source_chunk in source_index["chunks"]:
            chunk_index = int(source_chunk["chunk_index"])
            chunk_frame_count = int(source_chunk["frame_count"])
            start_frame = int(source_chunk["start_frame"])
            filename = str(source_chunk["filename"])
            source_chunk_path = source_episode_dir / filename
            raw = _decode_verified_lz4_chunk(
                source_chunk_path,
                compressed_sha256=str(source_chunk["compressed_sha256"]),
                uncompressed_bytes=int(source_chunk["uncompressed_bytes"]),
            )
            source_frames = np.frombuffer(raw, dtype=np.uint8).reshape(
                chunk_frame_count,
                plan.intrinsics.height,
                plan.intrinsics.width,
                3,
            )
            source_v1_decoded_hash.update(raw)
            target_frames = np.array(source_frames, copy=True, order="C")
            chunk_newly_zeroed_pixels = 0

            for local_index in range(chunk_frame_count):
                frame_index = start_frame + local_index
                depth_path = episode.depth_paths[frame_index]
                png_bytes = depth_path.read_bytes()
                source_depth_hash.update(len(png_bytes).to_bytes(8, "big"))
                source_depth_hash.update(png_bytes)
                depth = cv2.imdecode(
                    np.frombuffer(png_bytes, dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED,
                )
                expected_shape = (plan.intrinsics.height, plan.intrinsics.width)
                if depth is None or depth.dtype != np.uint16 or depth.shape != expected_shape:
                    raise MigrationError(
                        f"Expected uint16 {expected_shape} aligned depth: {depth_path}"
                    )
                range_valid = _v2_depth_range_mask(
                    depth,
                    scale_m_per_unit=plan.depth_scale_m_per_unit,
                    depth_near_m=plan.depth_near_m,
                    depth_far_m=plan.depth_far_m,
                )
                source_normal = source_frames[local_index]
                target_normal = target_frames[local_index]
                newly_invalid = np.any(source_normal != 0, axis=-1) & ~range_valid
                frame_newly_zeroed = int(np.count_nonzero(newly_invalid))
                chunk_newly_zeroed_pixels += frame_newly_zeroed
                newly_zeroed_pixels += frame_newly_zeroed
                range_mask_invalid_pixels += int(np.count_nonzero(~range_valid))
                target_normal[~range_valid] = 0

                if frame_index in sampled_index_set:
                    chw = target_normal.transpose(2, 0, 1)
                    sampled_by_index[frame_index] = np.ascontiguousarray(
                        auto_downsample_height_width(chw)
                    )
                frame_count += 1

            target_raw = target_frames.tobytes()
            target_v2_decoded_hash.update(target_raw)
            target_chunk_path = temporary / filename
            if chunk_newly_zeroed_pixels == 0:
                _link_file(source_chunk_path, target_chunk_path)
                compressed_bytes = int(source_chunk["compressed_bytes"])
                compressed_sha = str(source_chunk["compressed_sha256"])
                migration_action = SOURCE_CHUNK_REUSED
                reused_chunks += 1
            else:
                compressed_bytes, compressed_sha = _compress_and_verify(
                    target_raw,
                    target_chunk_path,
                )
                migration_action = SOURCE_CHUNK_MASKED
                rewritten_chunks += 1
            chunks.append(
                {
                    "chunk_index": chunk_index,
                    "start_frame": start_frame,
                    "frame_count": chunk_frame_count,
                    "height": plan.intrinsics.height,
                    "width": plan.intrinsics.width,
                    "channels": 3,
                    "uncompressed_bytes": len(target_raw),
                    "filename": filename,
                    "compressed_bytes": compressed_bytes,
                    "compressed_sha256": compressed_sha,
                    "lossless_round_trip_verified": True,
                    "migration_action": migration_action,
                    "range_mask_newly_zeroed_pixels": chunk_newly_zeroed_pixels,
                }
            )

        if frame_count != episode.frame_count:
            raise MigrationError(
                f"Masked {frame_count} frames for {plan.name}/{episode.episode_index}; "
                f"expected {episode.frame_count}"
            )
        source_decoded_sha = source_v1_decoded_hash.hexdigest()
        if source_decoded_sha != source_index["decoded_reference_sha256"]:
            raise MigrationError(
                f"Canonical v1 decoded hash mismatch: {source_index_path}"
            )
        sampled = [sampled_by_index[index] for index in sampled_indices_order]
        stats = _normal_stats(sampled)
        h264_reused = newly_zeroed_pixels == 0
        if h264_reused:
            source_h264 = _source_v1_h264_path(plan, episode)
            decoded_frames, decoded_sha, h264_sha = _decode_video_signature(source_h264)
            if (
                decoded_frames != episode.frame_count
                or decoded_sha != source_decoded_sha
            ):
                raise MigrationError(
                    f"Source H.264 backup differs from canonical v1 LZ4: {source_h264}"
                )
            _link_file(source_h264, video_path)
        else:
            _encode_h264_from_lz4_chunks(
                plan,
                episode,
                temporary,
                chunks,
                video_path,
            )
            decoded_frames, decoded_sha, h264_sha = _decode_video_signature(video_path)
        if decoded_frames != episode.frame_count:
            raise MigrationError(f"H.264 frame-count mismatch: {video_path}")
        target_decoded_sha = target_v2_decoded_hash.hexdigest()
        if decoded_sha != target_decoded_sha:
            raise MigrationError(f"H.264 round trip is not byte exact: {video_path}")
        index = {
            "episode_index": episode.episode_index,
            "dtype": "uint8",
            "layout": "FHWC",
            "transform": "none",
            "compression": "LZ4 frame, CLI default fast compression",
            "chunks_are_independent": True,
            "chunk_frames": CHUNK_FRAMES,
            "frame_count": frame_count,
            "decoded_reference_sha256": target_decoded_sha,
            "aligned_depth_png_sequence_sha256": source_depth_hash.hexdigest(),
            "source_v1_lz4_index_sha256": sha256_file(source_index_path),
            "source_v1_decoded_reference_sha256": source_decoded_sha,
            "h264_backup_sha256": h264_sha,
            "surface_normals_encoding_version": SURFACE_NORMAL_ENCODING_VERSION,
            "depth_valid_range_m": plan.new_encoding["depth_valid_range_m"],
            "migration_method": MIGRATION_METHOD,
            "normals_recomputed": False,
            "source_v1_semantics": "trusted_declared_dataset_contract",
            "range_mask_invalid_pixels": range_mask_invalid_pixels,
            "range_mask_newly_zeroed_pixels": newly_zeroed_pixels,
            "source_lz4_chunks_reused": reused_chunks,
            "source_lz4_chunks_rewritten": rewritten_chunks,
            "h264_backup_reused_from_source": h264_reused,
            NORMAL_STATS_KEY: stats,
            "lossless_round_trip_verified": True,
            "chunks": chunks,
        }
        write_json_atomic(temporary / "index.json", index)
        os.replace(temporary, final_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return (
        plan.name,
        episode.episode_index,
        sum(int(chunk["compressed_bytes"]) for chunk in chunks),
        "converted",
    )


def _extract_h264_backups(plan: SplitPlan, work: Path) -> None:
    work_split = work / plan.name
    lz4_build, backup_build = _build_roots(work_split, plan)
    backup_build.mkdir(parents=True, exist_ok=True)
    for episode in plan.episodes:
        episode_dir = lz4_build / f"episode_{episode.episode_index:06d}"
        index = read_json(episode_dir / "index.json")
        source_video = episode_dir / H264_FILENAME
        target_video = backup_build / f"episode_{episode.episode_index:06d}.mp4"
        expected_hash = index["h264_backup_sha256"]
        if target_video.exists():
            if sha256_file(target_video) != expected_hash:
                raise MigrationError(f"Corrupt extracted H.264 backup: {target_video}")
            if source_video.exists():
                if sha256_file(source_video) != expected_hash:
                    raise MigrationError(f"Corrupt internal H.264 backup: {source_video}")
                source_video.unlink()
            continue
        if not source_video.is_file():
            raise MigrationError(f"Completed episode has no H.264 backup: {episode_dir}")
        os.replace(source_video, target_video)


def _write_lz4_manifest(plan: SplitPlan, work: Path, target: Path) -> None:
    work_split = work / plan.name
    lz4_build, _ = _build_roots(work_split, plan)
    compressed_bytes = 0
    newly_zeroed_pixels = 0
    reused_chunks = 0
    rewritten_chunks = 0
    reused_h264_episodes = 0
    for episode in plan.episodes:
        index = read_json(lz4_build / f"episode_{episode.episode_index:06d}" / "index.json")
        compressed_bytes += sum(int(chunk["compressed_bytes"]) for chunk in index["chunks"])
        newly_zeroed_pixels += int(index["range_mask_newly_zeroed_pixels"])
        reused_chunks += int(index["source_lz4_chunks_reused"])
        rewritten_chunks += int(index["source_lz4_chunks_rewritten"])
        reused_h264_episodes += int(index["h264_backup_reused_from_source"])
    target_split = target / plan.name
    manifest = {
        "source_dataset": str(plan.source_root),
        "migration": MIGRATION_OPERATION,
        "migration_method": MIGRATION_METHOD,
        "normals_recomputed": False,
        "split": plan.name,
        "feature_key": FEATURE_KEY,
        "format": "independent plain LZ4 Frame chunks",
        "dtype": "uint8",
        "layout": "FHWC",
        "transform": "none",
        "chunk_frames": CHUNK_FRAMES,
        "episode_count": len(plan.episodes),
        "frame_count": sum(episode.frame_count for episode in plan.episodes),
        "compressed_bytes": compressed_bytes,
        "range_mask_newly_zeroed_pixels": newly_zeroed_pixels,
        "source_lz4_chunks_reused": reused_chunks,
        "source_lz4_chunks_rewritten": rewritten_chunks,
        "source_h264_episodes_reused": reused_h264_episodes,
        "all_lossless_round_trips_verified": True,
        "canonical_commit_pending": False,
        "surface_normals_encoding_version": SURFACE_NORMAL_ENCODING_VERSION,
        "depth_valid_range_m": plan.new_encoding["depth_valid_range_m"],
        "canonical_lz4_path": str(target_split / plan.lz4_relative),
        "h264_backup_path": str(target_split / plan.backup_relative),
    }
    write_json_atomic(lz4_build / "manifest.json", manifest)


def _safe_remove_tree(path: Path, work: Path) -> None:
    resolved_work = work.resolve()
    resolved = path.resolve()
    try:
        resolved.relative_to(resolved_work)
    except ValueError as exc:
        raise MigrationError(f"Refusing to remove a path outside migration work: {path}") from exc
    if resolved == resolved_work:
        raise MigrationError(f"Refusing to remove migration root: {path}")
    if path.is_symlink() or not path.is_dir():
        raise MigrationError(f"Expected a real directory before replacement: {path}")
    shutil.rmtree(path)


def _is_v2_lz4_root(root: Path, plan: SplitPlan) -> bool:
    if not root.is_dir():
        return False
    for episode in plan.episodes[:1]:
        index_path = root / f"episode_{episode.episode_index:06d}" / "index.json"
        if not index_path.is_file():
            return False
        if read_json(index_path).get("surface_normals_encoding_version") != (
            SURFACE_NORMAL_ENCODING_VERSION
        ):
            return False
    return True


def _commit_payload(plan: SplitPlan, work: Path) -> None:
    work_split = work / plan.name
    lz4_build, backup_build = _build_roots(work_split, plan)
    canonical = work_split / plan.lz4_relative
    backup = work_split / plan.backup_relative

    if lz4_build.exists():
        if canonical.exists():
            _safe_remove_tree(canonical, work)
        os.replace(lz4_build, canonical)
    elif not _is_v2_lz4_root(canonical, plan):
        raise MigrationError(f"Cannot resume v2 LZ4 commit for {plan.name}")

    if backup_build.exists():
        if backup.exists():
            _safe_remove_tree(backup, work)
        os.replace(backup_build, backup)
    elif not backup.is_dir():
        raise MigrationError(f"Cannot resume H.264 backup commit for {plan.name}")


def _stats_as_arrays(value: dict[str, Any]) -> dict[str, np.ndarray]:
    return {key: np.asarray(item) for key, item in value.items()}


def _update_split_metadata(plan: SplitPlan, work: Path) -> None:
    split_root = work / plan.name
    meta = split_root / "meta"
    episodes_stats_path = meta / "episodes_stats.jsonl"
    records = read_jsonl(episodes_stats_path)
    by_index = {int(record.get("episode_index", -1)): record for record in records}
    if set(by_index) != {episode.episode_index for episode in plan.episodes}:
        raise MigrationError(f"{plan.name} episodes_stats does not cover every episode")

    normal_stats_arrays: list[dict[str, np.ndarray]] = []
    for episode in plan.episodes:
        index = read_json(
            split_root
            / plan.lz4_relative
            / f"episode_{episode.episode_index:06d}"
            / "index.json"
        )
        stats = index[NORMAL_STATS_KEY]
        record = by_index[episode.episode_index]
        if not isinstance(record.get("stats"), dict):
            raise MigrationError(f"Missing stats object for {plan.name}/{episode.episode_index}")
        record["stats"][FEATURE_KEY] = stats
        normal_stats_arrays.append(_stats_as_arrays(stats))
    ordered = [by_index[episode.episode_index] for episode in plan.episodes]
    write_jsonl_atomic(episodes_stats_path, ordered)

    stats_path = meta / "stats.json"
    dataset_stats = read_json(stats_path)
    if FEATURE_KEY in dataset_stats:
        dataset_stats[FEATURE_KEY] = aggregate_feature_stats(normal_stats_arrays)
        write_json_atomic(stats_path, dataset_stats)

    info_path = meta / "info.json"
    info = read_json(info_path)
    info["surface_normals_encoding"] = plan.new_encoding
    write_json_atomic(info_path, info)

    backup_info_path = meta / "info.json.h264_backup"
    backup_info = read_json(backup_info_path) if backup_info_path.is_file() else dict(info)
    backup_info["surface_normals_encoding"] = plan.new_encoding
    backup_info.pop("surface_normals_lz4", None)
    write_json_atomic(backup_info_path, backup_info)


def _changed_source_relative_paths(plans: tuple[SplitPlan, ...]) -> set[Path]:
    changed: set[Path] = set()
    for plan in plans:
        prefix = Path(plan.name)
        changed.update(
            {
                prefix / plan.lz4_relative,
                prefix / plan.backup_relative,
                prefix / "meta/info.json",
                prefix / "meta/info.json.h264_backup",
                prefix / "meta/episodes_stats.jsonl",
                prefix / "meta/stats.json",
            }
        )
    return changed


def _path_is_changed(relative: Path, changed: set[Path]) -> bool:
    return any(relative == prefix or prefix in relative.parents for prefix in changed)


def verify_unchanged_payload_hardlinks(
    source: Path,
    work: Path,
    plans: tuple[SplitPlan, ...],
) -> tuple[int, int]:
    changed = _changed_source_relative_paths(plans)
    checked_files = 0
    checked_bytes = 0
    for source_file in source.rglob("*"):
        if not source_file.is_file() or source_file.is_symlink():
            continue
        relative = source_file.relative_to(source)
        if _path_is_changed(relative, changed):
            continue
        target_file = work / relative
        if not target_file.is_file() or target_file.is_symlink():
            raise MigrationError(f"Sibling lost unchanged source payload: {relative}")
        left = source_file.stat()
        right = target_file.stat()
        if (left.st_dev, left.st_ino) != (right.st_dev, right.st_ino):
            raise MigrationError(f"Unchanged payload is not hard-linked: {relative}")
        checked_files += 1
        checked_bytes += left.st_size
    return checked_files, checked_bytes


def _write_provenance(
    source: Path,
    target: Path,
    work: Path,
    plans: tuple[SplitPlan, ...],
    source_fingerprint: str,
    hardlink_files: int,
    hardlink_bytes: int,
) -> None:
    destination = work / "provenance" / "surface_normals_v2_migration"
    destination.mkdir(parents=True, exist_ok=True)
    split_records: dict[str, Any] = {}
    for plan in plans:
        split_root = work / plan.name
        episode_records = []
        for episode in plan.episodes:
            index = read_json(
                split_root
                / plan.lz4_relative
                / f"episode_{episode.episode_index:06d}"
                / "index.json"
            )
            episode_records.append(
                {
                    "episode_index": episode.episode_index,
                    "frame_count": episode.frame_count,
                    "aligned_depth_png_sequence_sha256": index[
                        "aligned_depth_png_sequence_sha256"
                    ],
                    "source_v1_lz4_index_sha256": index[
                        "source_v1_lz4_index_sha256"
                    ],
                    "source_v1_decoded_normal_frames_sha256": index[
                        "source_v1_decoded_reference_sha256"
                    ],
                    "decoded_normal_frames_sha256": index["decoded_reference_sha256"],
                    "h264_backup_sha256": index["h264_backup_sha256"],
                    "range_mask_invalid_pixels": index[
                        "range_mask_invalid_pixels"
                    ],
                    "range_mask_newly_zeroed_pixels": index[
                        "range_mask_newly_zeroed_pixels"
                    ],
                    "source_lz4_chunks_reused": index[
                        "source_lz4_chunks_reused"
                    ],
                    "source_lz4_chunks_rewritten": index[
                        "source_lz4_chunks_rewritten"
                    ],
                    "h264_backup_reused_from_source": index[
                        "h264_backup_reused_from_source"
                    ],
                    "lz4_compressed_bytes": sum(
                        int(chunk["compressed_bytes"]) for chunk in index["chunks"]
                    ),
                }
            )
        split_records[plan.name] = {
            "episodes": len(plan.episodes),
            "frames": sum(episode.frame_count for episode in plan.episodes),
            "records": episode_records,
        }
    manifest = {
        "version": 2,
        "operation": MIGRATION_OPERATION,
        "migration_method": MIGRATION_METHOD,
        "normals_recomputed": False,
        "source_v1_semantics": "trusted_declared_dataset_contract",
        "source_dataset": str(source),
        "target_dataset": str(target),
        "source_stat_fingerprint": source_fingerprint,
        "source_modified": False,
        "unchanged_payload_storage": "hardlinks_to_immutable_source",
        "unchanged_hardlink_files": hardlink_files,
        "unchanged_hardlink_bytes": hardlink_bytes,
        "old_encoding_version": LEGACY_SURFACE_NORMAL_ENCODING_VERSION,
        "new_encoding": plans[0].new_encoding,
        "splits": split_records,
        "training_started": False,
    }
    write_json_atomic(destination / "manifest.json", manifest)


def _validate_v2_payloads(
    plans: tuple[SplitPlan, ...],
    root: Path,
    *,
    verify_source: bool,
    verify_payload: bool = False,
) -> None:
    for plan in plans:
        split_root = root / plan.name
        info = read_json(split_root / "meta" / "info.json")
        if info.get("surface_normals_encoding") != plan.new_encoding:
            raise MigrationError(f"{plan.name} did not commit the exact v2 encoding")
        lz4_root = split_root / plan.lz4_relative
        backup_root = split_root / plan.backup_relative
        for episode in plan.episodes:
            completed = _validate_completed_episode(
                plan,
                episode,
                lz4_root,
                backup_root,
                verify_source=verify_source,
                verify_payload=verify_payload,
            )
            if completed is None:
                raise MigrationError(
                    f"Invalid v2 payload for {plan.name}/episode_{episode.episode_index:06d}"
                )


def _validate_v2_metadata(
    plans: tuple[SplitPlan, ...],
    root: Path,
    canonical_target: Path,
) -> None:
    """Validate every metadata mutation against source plus generated indexes."""

    for plan in plans:
        split_root = root / plan.name
        meta = split_root / "meta"

        expected_info = json.loads(json.dumps(plan.info))
        expected_info["surface_normals_encoding"] = plan.new_encoding
        if read_json(meta / "info.json") != expected_info:
            raise MigrationError(f"{plan.name} info.json differs beyond the v2 contract")

        source_backup_path = plan.source_root / "meta" / "info.json.h264_backup"
        expected_backup = (
            read_json(source_backup_path)
            if source_backup_path.is_file()
            else dict(expected_info)
        )
        expected_backup["surface_normals_encoding"] = plan.new_encoding
        expected_backup.pop("surface_normals_lz4", None)
        if read_json(meta / "info.json.h264_backup") != expected_backup:
            raise MigrationError(f"{plan.name} H.264 metadata backup is not exact v2")

        source_episode_stats = read_jsonl(plan.source_root / "meta" / "episodes_stats.jsonl")
        target_episode_stats = read_jsonl(meta / "episodes_stats.jsonl")
        if len(source_episode_stats) != len(plan.episodes) or len(target_episode_stats) != len(
            plan.episodes
        ):
            raise MigrationError(f"{plan.name} episode-stat count changed")
        normal_stats_arrays: list[dict[str, np.ndarray]] = []
        for episode, source_record, target_record in zip(
            plan.episodes,
            source_episode_stats,
            target_episode_stats,
            strict=True,
        ):
            if (
                int(source_record.get("episode_index", -1)) != episode.episode_index
                or int(target_record.get("episode_index", -1)) != episode.episode_index
            ):
                raise MigrationError(f"{plan.name} episode-stat order changed")
            index = read_json(
                split_root
                / plan.lz4_relative
                / f"episode_{episode.episode_index:06d}"
                / "index.json"
            )
            normal_stats = index.get(NORMAL_STATS_KEY)
            if not isinstance(normal_stats, dict):
                raise MigrationError(
                    f"{plan.name}/{episode.episode_index} has no generated normal stats"
                )
            expected_record = json.loads(json.dumps(source_record))
            expected_record["stats"][FEATURE_KEY] = normal_stats
            if target_record != expected_record:
                raise MigrationError(
                    f"{plan.name}/{episode.episode_index} metadata differs beyond normal stats"
                )
            normal_stats_arrays.append(_stats_as_arrays(normal_stats))

        source_stats = read_json(plan.source_root / "meta" / "stats.json")
        expected_stats = json.loads(json.dumps(source_stats))
        if FEATURE_KEY in expected_stats:
            expected_stats[FEATURE_KEY] = _jsonable(
                aggregate_feature_stats(normal_stats_arrays)
            )
        if read_json(meta / "stats.json") != expected_stats:
            raise MigrationError(f"{plan.name} aggregate stats are not exact")

        manifest = read_json(split_root / plan.lz4_relative / "manifest.json")
        canonical_split = canonical_target / plan.name
        generated_indexes = [
            read_json(
                split_root
                / plan.lz4_relative
                / f"episode_{episode.episode_index:06d}"
                / "index.json"
            )
            for episode in plan.episodes
        ]
        expected_newly_zeroed = sum(
            int(index["range_mask_newly_zeroed_pixels"])
            for index in generated_indexes
        )
        expected_reused_chunks = sum(
            int(index["source_lz4_chunks_reused"])
            for index in generated_indexes
        )
        expected_rewritten_chunks = sum(
            int(index["source_lz4_chunks_rewritten"])
            for index in generated_indexes
        )
        expected_reused_h264 = sum(
            int(index["h264_backup_reused_from_source"])
            for index in generated_indexes
        )
        if (
            manifest.get("source_dataset") != str(plan.source_root)
            or manifest.get("migration") != MIGRATION_OPERATION
            or manifest.get("migration_method") != MIGRATION_METHOD
            or manifest.get("normals_recomputed") is not False
            or manifest.get("split") != plan.name
            or manifest.get("feature_key") != FEATURE_KEY
            or manifest.get("episode_count") != len(plan.episodes)
            or manifest.get("frame_count")
            != sum(episode.frame_count for episode in plan.episodes)
            or manifest.get("surface_normals_encoding_version")
            != SURFACE_NORMAL_ENCODING_VERSION
            or manifest.get("depth_valid_range_m")
            != plan.new_encoding["depth_valid_range_m"]
            or int(manifest.get("range_mask_newly_zeroed_pixels", -1))
            != expected_newly_zeroed
            or int(manifest.get("source_lz4_chunks_reused", -1))
            != expected_reused_chunks
            or int(manifest.get("source_lz4_chunks_rewritten", -1))
            != expected_rewritten_chunks
            or int(manifest.get("source_h264_episodes_reused", -1))
            != expected_reused_h264
            or manifest.get("canonical_lz4_path")
            != str(canonical_split / plan.lz4_relative)
            or manifest.get("h264_backup_path")
            != str(canonical_split / plan.backup_relative)
            or manifest.get("canonical_commit_pending") is not False
            or manifest.get("all_lossless_round_trips_verified") is not True
        ):
            raise MigrationError(f"{plan.name} LZ4 manifest is not the exact v2 contract")


def validate_published_target(
    source: Path,
    target: Path,
    plans: tuple[SplitPlan, ...],
    source_fingerprint: str,
) -> None:
    """Fail closed unless an existing target is this fully verified migration."""

    manifest_path = target / "provenance/surface_normals_v2_migration/manifest.json"
    if not manifest_path.is_file():
        raise MigrationError(f"Existing target has no migration provenance: {target}")
    manifest = read_json(manifest_path)
    expected_manifest_fields = {
        "version": 2,
        "operation": MIGRATION_OPERATION,
        "migration_method": MIGRATION_METHOD,
        "normals_recomputed": False,
        "source_v1_semantics": "trusted_declared_dataset_contract",
        "source_dataset": str(source),
        "target_dataset": str(target),
        "source_stat_fingerprint": source_fingerprint,
        "source_modified": False,
        "unchanged_payload_storage": "hardlinks_to_immutable_source",
        "old_encoding_version": LEGACY_SURFACE_NORMAL_ENCODING_VERSION,
        "new_encoding": plans[0].new_encoding,
        "training_started": False,
    }
    for key, expected in expected_manifest_fields.items():
        if manifest.get(key) != expected:
            raise MigrationError(
                f"Existing target provenance differs at {key}: "
                f"{manifest.get(key)!r} != {expected!r}"
            )

    state = _load_state(target, source, target, source_fingerprint)
    if state.get("phase") != "ready_to_publish":
        raise MigrationError("Existing target does not have a completed migration state")
    _validate_v2_metadata(plans, target, target)
    _validate_v2_payloads(
        plans,
        target,
        verify_source=True,
        verify_payload=True,
    )
    expected_splits: dict[str, Any] = {}
    for plan in plans:
        records = []
        for episode in plan.episodes:
            index = read_json(
                target
                / plan.name
                / plan.lz4_relative
                / f"episode_{episode.episode_index:06d}"
                / "index.json"
            )
            records.append(
                {
                    "episode_index": episode.episode_index,
                    "frame_count": episode.frame_count,
                    "aligned_depth_png_sequence_sha256": index[
                        "aligned_depth_png_sequence_sha256"
                    ],
                    "source_v1_lz4_index_sha256": index[
                        "source_v1_lz4_index_sha256"
                    ],
                    "source_v1_decoded_normal_frames_sha256": index[
                        "source_v1_decoded_reference_sha256"
                    ],
                    "decoded_normal_frames_sha256": index["decoded_reference_sha256"],
                    "h264_backup_sha256": index["h264_backup_sha256"],
                    "range_mask_invalid_pixels": index[
                        "range_mask_invalid_pixels"
                    ],
                    "range_mask_newly_zeroed_pixels": index[
                        "range_mask_newly_zeroed_pixels"
                    ],
                    "source_lz4_chunks_reused": index[
                        "source_lz4_chunks_reused"
                    ],
                    "source_lz4_chunks_rewritten": index[
                        "source_lz4_chunks_rewritten"
                    ],
                    "h264_backup_reused_from_source": index[
                        "h264_backup_reused_from_source"
                    ],
                    "lz4_compressed_bytes": sum(
                        int(chunk["compressed_bytes"]) for chunk in index["chunks"]
                    ),
                }
            )
        expected_splits[plan.name] = {
            "episodes": len(plan.episodes),
            "frames": sum(episode.frame_count for episode in plan.episodes),
            "records": records,
        }
    if manifest.get("splits") != expected_splits:
        raise MigrationError("Existing target episode provenance does not match its payload")
    hardlink_files, hardlink_bytes = verify_unchanged_payload_hardlinks(
        source,
        target,
        plans,
    )
    if (
        hardlink_files <= 0
        or hardlink_bytes <= 0
        or manifest.get("unchanged_hardlink_files") != hardlink_files
        or manifest.get("unchanged_hardlink_bytes") != hardlink_bytes
    ):
        raise MigrationError("Existing target hard-link verification does not match provenance")


def migrate(
    source: Path,
    target: Path,
    *,
    jobs: int = DEFAULT_JOBS,
    min_free_gib: int = DEFAULT_MIN_FREE_GIB,
    max_new_episodes: int | None = None,
) -> bool:
    if jobs <= 0:
        raise MigrationError("jobs must be positive")
    if min_free_gib < 0:
        raise MigrationError("min_free_gib may not be negative")
    if max_new_episodes is not None and max_new_episodes <= 0:
        raise MigrationError("max_new_episodes must be positive")
    source_input = source.expanduser()
    target_input = target.expanduser()
    if source_input.is_symlink():
        raise MigrationError(f"Source may not be a symlink: {source_input}")
    if target_input.is_symlink():
        raise MigrationError(f"Target may not be a symlink: {target_input}")
    source = source_input.resolve()
    target = target_input.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise MigrationError("Source and target may not overlap")
    if source.parent != target.parent:
        raise MigrationError("Target must be a sibling of the source dataset")

    plans = inspect_source(source)
    source_fingerprint = source_stat_fingerprint(source)
    if target.exists() or target.is_symlink():
        if not target.is_dir() or target.is_symlink():
            raise MigrationError(f"Target already exists and is not a real directory: {target}")
        validate_published_target(source, target, plans, source_fingerprint)
        print(f"Already published and fully verified: {target}")
        return True

    work = target.parent / f".{target.name}.surface-normals-v2-work"
    if not work.exists():
        work.mkdir()
        state = _new_state(source, target, source_fingerprint)
        write_json_atomic(_state_path(work), state)
    elif not work.is_dir() or work.is_symlink() or not _state_path(work).is_file():
        raise MigrationError(f"Unrecognized migration scratch; preserved for inspection: {work}")
    state = _load_state(work, source, target, source_fingerprint)

    if state["phase"] == "cloning":
        hardlink_clone_resume(source, work)
        _set_phase(work, state, "building")
        state = _load_state(work, source, target, source_fingerprint)

    existing_allocated = 0
    for plan in plans:
        lz4_build, backup_build = _build_roots(work / plan.name, plan)
        if lz4_build.exists():
            existing_allocated += _allocated_bytes(lz4_build)
        if backup_build.exists():
            existing_allocated += _allocated_bytes(backup_build)
    required = max(5 * 1024**3, min_free_gib * 1024**3 - existing_allocated)
    free = shutil.disk_usage(target.parent).free
    if free < required:
        raise MigrationError(
            f"Migration requires at least {required / 1024**3:.1f} GiB free at this "
            f"resume point; found {free / 1024**3:.1f} GiB"
        )

    if state["phase"] == "building":
        incomplete: list[tuple[SplitPlan, EpisodePlan, Path, Path]] = []
        for plan in plans:
            lz4_build, backup_build = _build_roots(work / plan.name, plan)
            lz4_build.mkdir(parents=True, exist_ok=True)
            backup_build.mkdir(parents=True, exist_ok=True)
            for episode in plan.episodes:
                if _validate_completed_episode(plan, episode, lz4_build, backup_build) is None:
                    incomplete.append((plan, episode, lz4_build, backup_build))
        if max_new_episodes is not None:
            incomplete = incomplete[:max_new_episodes]
        if incomplete:
            print(f"Generating {len(incomplete)} incomplete episode(s) with {jobs} worker(s).")
            futures: dict[Future, tuple[str, int]] = {}
            with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="normals-v2") as pool:
                for plan, episode, lz4_build, backup_build in incomplete:
                    future = pool.submit(_encode_episode, plan, episode, lz4_build, backup_build)
                    futures[future] = (plan.name, episode.episode_index)
                completed_count = 0
                for future in as_completed(futures):
                    split, episode_index, compressed_bytes, status = future.result()
                    completed_count += 1
                    print(
                        f"[{completed_count}/{len(futures)}] {split}/"
                        f"episode_{episode_index:06d} {status}; "
                        f"LZ4={compressed_bytes / 1024**2:.1f} MiB",
                        flush=True,
                    )

        remaining = []
        for plan in plans:
            lz4_build, backup_build = _build_roots(work / plan.name, plan)
            for episode in plan.episodes:
                if _validate_completed_episode(plan, episode, lz4_build, backup_build) is None:
                    remaining.append((plan.name, episode.episode_index))
        if remaining:
            print(
                f"Migration remains resumable at {work}: {len(remaining)} episode(s) incomplete."
            )
            return False
        if max_new_episodes is not None:
            print(
                "Bounded episode generation finished without publication; "
                f"resume without --max-new-episodes to validate and publish {work}."
            )
            return False

        for plan in plans:
            _extract_h264_backups(plan, work)
            _write_lz4_manifest(plan, work, target)
        _set_phase(work, state, "payload_commit")
        state = _load_state(work, source, target, source_fingerprint)

    if state["phase"] == "payload_commit":
        for plan in plans:
            _commit_payload(plan, work)
        _set_phase(work, state, "metadata_commit")
        state = _load_state(work, source, target, source_fingerprint)

    if state["phase"] == "metadata_commit":
        for plan in plans:
            _update_split_metadata(plan, work)
        hardlink_files, hardlink_bytes = verify_unchanged_payload_hardlinks(
            source, work, plans
        )
        _write_provenance(
            source,
            target,
            work,
            plans,
            source_fingerprint,
            hardlink_files,
            hardlink_bytes,
        )
        _set_phase(work, state, "validating")
        state = _load_state(work, source, target, source_fingerprint)

    if state["phase"] == "validating":
        if source_stat_fingerprint(source) != source_fingerprint:
            raise MigrationError("Source corpus changed during migration")
        _validate_v2_metadata(plans, work, target)
        _validate_v2_payloads(plans, work, verify_source=True)
        hardlink_files, hardlink_bytes = verify_unchanged_payload_hardlinks(
            source, work, plans
        )
        if hardlink_files <= 0 or hardlink_bytes <= 0:
            raise MigrationError("No unchanged payload was verified as hard-linked")
        _set_phase(work, state, "ready_to_publish")
        state = _load_state(work, source, target, source_fingerprint)

    if state["phase"] == "ready_to_publish":
        try:
            rename_directory_noreplace(work, target)
        except FileExistsError as exc:
            raise MigrationError(
                f"Target appeared before atomic publication; completed work is preserved at {work}"
            ) from exc
        print(f"DATASET_PUBLISHED={target}")
        return True

    raise MigrationError(f"Unsupported migration phase: {state['phase']!r}")


def check(source: Path, target: Path, *, min_free_gib: int) -> None:
    source_input = source.expanduser()
    target_input = target.expanduser()
    if source_input.is_symlink():
        raise MigrationError(f"Source may not be a symlink: {source_input}")
    if target_input.is_symlink():
        raise MigrationError(f"Target may not be a symlink: {target_input}")
    source = source_input.resolve()
    target = target_input.resolve()
    if source == target or source in target.parents or target in source.parents:
        raise MigrationError("Source and target may not overlap")
    if source.parent != target.parent:
        raise MigrationError("Target must be a sibling of the source dataset")
    plans = inspect_source(source)
    normal_bytes = estimate_migration_bytes(plans)
    free = shutil.disk_usage(target.parent).free
    minimum = min_free_gib * 1024**3
    print(f"Source: {source}")
    print(f"Target: {target}")
    print(
        "Population: "
        + ", ".join(
            f"{plan.name}={len(plan.episodes)}/"
            f"{sum(episode.frame_count for episode in plan.episodes)}"
            for plan in plans
        )
    )
    print(
        "Conservative normal-payload rewrite ceiling: "
        f"{normal_bytes / 1024**3:.1f} GiB"
    )
    print(f"Free space: {free / 1024**3:.1f} GiB")
    print(f"Required free-space floor: {minimum / 1024**3:.1f} GiB")
    if free < minimum:
        raise MigrationError("Free space is below the requested migration floor")
    print(
        "CHECK_OK: verified canonical-v1 LZ4 range masking; normals are not "
        "recomputed and training is unavailable"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.chunk_frames != CHUNK_FRAMES:
        raise MigrationError(f"Only the canonical {CHUNK_FRAMES}-frame chunks are supported")
    if args.mode == "check":
        check(args.source, args.target, min_free_gib=args.min_free_gib)
        return 0
    published = migrate(
        args.source,
        args.target,
        jobs=args.jobs,
        min_free_gib=args.min_free_gib,
        max_new_episodes=args.max_new_episodes,
    )
    return 0 if published else 3


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MigrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
