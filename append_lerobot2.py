#!/home/alex/miniconda3/envs/unitree_lerobot/bin/python
"""Append LeRobot v2.1 datasets without converting the existing data again.

Examples (run from /home/alex/Development/scripts/):

  ./append_lerobot2.py /data/combined_v21 /data/new_session_v21
  ./append_lerobot2.py /data/combined_v21 /data/session_2_v21 /data/session_3_v21

The same command also accepts roots containing any subset of train/, test/, and
validation/ (or validate/). Matching splits are merged independently.

The destination is replaced transactionally. Sources are always copied and are
never modified. Existing destination Parquet, video, raw-depth, and aligned-depth
files are not rewritten.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


META_FILES = ("info.json", "tasks.jsonl", "episodes.jsonl", "episodes_stats.jsonl")
SPLIT_NAMES = ("train", "test", "validation")
VALIDATION_ALIASES = ("validation", "validate")
SPECIAL_STATS = ("episode_index", "index", "task_index")
SURFACE_NORMALS_LZ4_KEY = "surface_normals_lz4"


class MergeError(RuntimeError):
    pass


def read_json(path: Path) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise MergeError(f"Cannot read JSON file {path}: {exc}") from exc


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise MergeError(f"Invalid JSON on line {line_number} of {path}: {exc}") from exc
                if not isinstance(value, dict):
                    raise MergeError(f"Line {line_number} of {path} is not a JSON object")
                records.append(value)
    except OSError as exc:
        raise MergeError(f"Cannot read {path}: {exc}") from exc
    return records


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        json.dump(value, handle, indent=4, ensure_ascii=False, default=_json_default)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        tmp = Path(handle.name)
        for record in records:
            json.dump(record, handle, ensure_ascii=False, default=_json_default)
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def hardlink_or_copy(source: str, destination: str) -> str:
    try:
        os.link(source, destination)
        return destination
    except OSError:
        return shutil.copy2(source, destination)


def clone_existing(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=hardlink_or_copy, symlinks=True)


def copy_source_tree(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination, copy_function=shutil.copy2, symlinks=True, dirs_exist_ok=True)


def is_dataset(path: Path) -> bool:
    return (path / "meta" / "info.json").is_file()


def split_dirs(root: Path) -> dict[str, Path]:
    found: dict[str, Path] = {}
    for name in ("train", "test"):
        candidate = root / name
        if is_dataset(candidate):
            found[name] = candidate
    validation = [root / name for name in VALIDATION_ALIASES if is_dataset(root / name)]
    if len(validation) > 1:
        raise MergeError(f"{root} contains both validation/ and validate/; keep only one")
    if validation:
        found["validation"] = validation[0]
    return found


def classify(path: Path, *, allow_empty: bool = False) -> str:
    if is_dataset(path):
        return "dataset"
    if path.is_dir() and split_dirs(path):
        return "split-root"
    if allow_empty and (not path.exists() or (path.is_dir() and not any(path.iterdir()))):
        return "empty"
    raise MergeError(
        f"{path} is neither a LeRobot v2.1 dataset nor a root containing train/test/validation datasets"
    )


def ensure_v21(info: dict[str, Any], path: Path) -> None:
    version = str(info.get("codebase_version", ""))
    if version not in {"v2.1", "2.1"}:
        raise MergeError(f"{path} is LeRobot {version or 'unknown'}, not v2.1")


def episode_path(root: Path, info: dict[str, Any], episode_index: int) -> Path:
    chunks_size = int(info["chunks_size"])
    relative = info["data_path"].format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
    )
    return root / relative


def video_path(root: Path, info: dict[str, Any], episode_index: int, video_key: str) -> Path:
    chunks_size = int(info["chunks_size"])
    relative = info["video_path"].format(
        episode_chunk=episode_index // chunks_size,
        episode_index=episode_index,
        video_key=video_key,
    )
    return root / relative


def video_keys(info: dict[str, Any]) -> list[str]:
    return [key for key, feature in info.get("features", {}).items() if feature.get("dtype") == "video"]


def raw_depth_encoding(info: dict[str, Any]) -> dict[str, Any] | None:
    encoding = info.get("raw_depth_encoding")
    if encoding is None:
        return None
    if not isinstance(encoding, dict):
        raise MergeError("info.json raw_depth_encoding must be an object")
    return encoding


def aligned_depth_encoding(info: dict[str, Any]) -> dict[str, Any] | None:
    encoding = info.get("aligned_depth_encoding")
    if encoding is None:
        return None
    if not isinstance(encoding, dict):
        raise MergeError("info.json aligned_depth_encoding must be an object")
    return encoding


def surface_normals_lz4(info: dict[str, Any]) -> dict[str, Any] | None:
    config = info.get(SURFACE_NORMALS_LZ4_KEY)
    if config is None:
        return None
    if not isinstance(config, dict):
        raise MergeError(f"info.json {SURFACE_NORMALS_LZ4_KEY} must be an object")
    required = {
        "feature_key": "observation.images.surface_normals_view",
        "storage": "plain_lz4_chunks",
        "dtype": "uint8",
        "layout": "FHWC",
        "lossless_round_trip_verified": True,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise MergeError(
                f"Unsupported {SURFACE_NORMALS_LZ4_KEY} {key}: {config.get(key)!r}"
            )
    try:
        chunk_frames = int(config["chunk_frames"])
    except (KeyError, TypeError, ValueError) as exc:
        raise MergeError(f"{SURFACE_NORMALS_LZ4_KEY} chunk_frames must be positive") from exc
    if chunk_frames <= 0:
        raise MergeError(f"{SURFACE_NORMALS_LZ4_KEY} chunk_frames must be positive")
    feature = info.get("features", {}).get(config["feature_key"])
    if not isinstance(feature, dict) or feature.get("dtype") != "video":
        raise MergeError(
            f"{SURFACE_NORMALS_LZ4_KEY} feature_key is not a video feature"
        )
    return config


def _internal_dataset_path(root: Path, value: Any, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise MergeError(f"{label} must be a non-empty relative path")
    relative = Path(value)
    if relative.is_absolute():
        raise MergeError(f"{label} must stay inside the dataset: {value}")
    resolved_root = root.resolve()
    resolved = (root / relative).resolve()
    try:
        resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise MergeError(f"{label} escapes dataset root: {value}") from exc
    return resolved


def lz4_storage_root(root: Path, info: dict[str, Any]) -> Path:
    config = surface_normals_lz4(info)
    if config is None:
        raise MergeError(f"{root} has no {SURFACE_NORMALS_LZ4_KEY} metadata")
    return _internal_dataset_path(root, config.get("root"), f"{SURFACE_NORMALS_LZ4_KEY}.root")


def lz4_backup_root(root: Path, info: dict[str, Any]) -> Path:
    config = surface_normals_lz4(info)
    if config is None:
        raise MergeError(f"{root} has no {SURFACE_NORMALS_LZ4_KEY} metadata")
    return _internal_dataset_path(
        root,
        config.get("h264_backup"),
        f"{SURFACE_NORMALS_LZ4_KEY}.h264_backup",
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(8 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def validate_lz4_episode(
    root: Path,
    info: dict[str, Any],
    episode_index: int,
    expected_frames: int,
    *,
    verify_hashes: bool = False,
) -> int:
    config = surface_normals_lz4(info)
    assert config is not None
    episode_dir = lz4_storage_root(root, info) / f"episode_{episode_index:06d}"
    index_path = episode_dir / "index.json"
    index = read_json(index_path)
    if int(index.get("episode_index", -1)) != episode_index:
        raise MergeError(f"LZ4 index episode mismatch in {episode_dir}")
    if (
        index.get("dtype") != "uint8"
        or index.get("layout") != "FHWC"
        or index.get("transform") != "none"
        or int(index.get("chunk_frames", -1)) != int(config["chunk_frames"])
        or not index.get("chunks_are_independent")
        or not index.get("lossless_round_trip_verified")
    ):
        raise MergeError(f"Invalid plain-LZ4 index contract in {episode_dir}")
    shape = info["features"][config["feature_key"]]["shape"]
    if len(shape) != 3:
        raise MergeError(f"Invalid surface-normal feature shape in {root}")
    height, width, channels = (int(value) for value in shape)
    next_frame = 0
    compressed_bytes = 0
    chunks = index.get("chunks")
    if not isinstance(chunks, list):
        raise MergeError(f"Invalid LZ4 chunk list in {episode_dir}")
    for expected_chunk_index, chunk in enumerate(chunks):
        if not isinstance(chunk, dict):
            raise MergeError(f"Invalid LZ4 chunk record in {episode_dir}")
        frame_count = int(chunk.get("frame_count", -1))
        if (
            int(chunk.get("chunk_index", -1)) != expected_chunk_index
            or int(chunk.get("start_frame", -1)) != next_frame
            or frame_count <= 0
            or frame_count > int(config["chunk_frames"])
            or [int(chunk.get(key, -1)) for key in ("height", "width", "channels")]
            != [height, width, channels]
            or int(chunk.get("uncompressed_bytes", -1))
            != frame_count * height * width * channels
        ):
            raise MergeError(f"Invalid LZ4 chunk layout in {episode_dir}")
        filename = chunk.get("filename")
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise MergeError(f"Invalid LZ4 chunk filename in {episode_dir}")
        chunk_path = episode_dir / filename
        if not chunk_path.is_file():
            raise MergeError(f"Missing LZ4 chunk: {chunk_path}")
        size = chunk_path.stat().st_size
        if size != int(chunk.get("compressed_bytes", -1)):
            raise MergeError(f"LZ4 chunk size mismatch: {chunk_path}")
        expected_hash = chunk.get("compressed_sha256")
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise MergeError(f"Invalid LZ4 chunk checksum in {episode_dir}")
        if verify_hashes and sha256_file(chunk_path) != expected_hash:
            raise MergeError(f"LZ4 chunk checksum mismatch: {chunk_path}")
        if not chunk.get("lossless_round_trip_verified"):
            raise MergeError(f"Unverified LZ4 chunk in {episode_dir}")
        next_frame += frame_count
        compressed_bytes += size
    if next_frame != expected_frames or int(index.get("frame_count", -1)) != expected_frames:
        raise MergeError(f"LZ4 frame count mismatch in {episode_dir}")
    backup = lz4_backup_root(root, info) / f"episode_{episode_index:06d}.mp4"
    if not backup.is_file():
        raise MergeError(f"Missing surface-normal H.264 backup: {backup}")
    return compressed_bytes


def raw_depth_path(
    root: Path,
    info: dict[str, Any],
    episode_index: int,
    frame_index: int,
) -> Path:
    encoding = raw_depth_encoding(info)
    if encoding is None:
        raise MergeError(f"{root} has no raw_depth_encoding metadata")

    template = encoding.get("path")
    if not isinstance(template, str) or not template:
        raise MergeError(f"{root}/meta/info.json has no raw-depth path template")

    try:
        relative = template.format(
            episode_chunk=episode_index // int(info["chunks_size"]),
            episode_index=episode_index,
            frame_index=frame_index,
        )
    except (KeyError, ValueError) as exc:
        raise MergeError(f"Invalid raw-depth path template in {root}: {template}") from exc

    resolved_root = root.resolve()
    resolved_path = (root / relative).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise MergeError(f"Raw-depth path escapes dataset root: {relative}") from exc
    return resolved_path


def aligned_depth_path(
    root: Path,
    info: dict[str, Any],
    episode_index: int,
    frame_index: int,
) -> Path:
    encoding = aligned_depth_encoding(info)
    if encoding is None:
        raise MergeError(f"{root} has no aligned_depth_encoding metadata")

    template = encoding.get("path")
    if not isinstance(template, str) or not template:
        raise MergeError(f"{root}/meta/info.json has no aligned-depth path template")

    try:
        relative = template.format(
            episode_chunk=episode_index // int(info["chunks_size"]),
            episode_index=episode_index,
            frame_index=frame_index,
        )
    except (KeyError, ValueError) as exc:
        raise MergeError(f"Invalid aligned-depth path template in {root}: {template}") from exc

    resolved_root = root.resolve()
    resolved_path = (root / relative).resolve()
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as exc:
        raise MergeError(f"Aligned-depth path escapes dataset root: {relative}") from exc
    return resolved_path


def load_metadata(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    meta = root / "meta"
    missing = [name for name in META_FILES if not (meta / name).is_file()]
    if missing:
        raise MergeError(f"{root} is missing metadata: {', '.join(missing)}")
    info = read_json(meta / "info.json")
    ensure_v21(info, root)
    return (
        info,
        read_jsonl(meta / "tasks.jsonl"),
        read_jsonl(meta / "episodes.jsonl"),
        read_jsonl(meta / "episodes_stats.jsonl"),
    )


def validate_tasks(tasks: list[dict[str, Any]], root: Path) -> None:
    indices = [record.get("task_index") for record in tasks]
    if indices != list(range(len(tasks))):
        raise MergeError(f"Task indices in {root}/meta/tasks.jsonl must be contiguous from zero")
    texts = [record.get("task") for record in tasks]
    if any(not isinstance(text, str) for text in texts):
        raise MergeError(f"Every task in {root}/meta/tasks.jsonl must contain text")
    if len(set(texts)) != len(texts):
        raise MergeError(f"{root}/meta/tasks.jsonl contains duplicate task text")


def validate_dataset(root: Path, *, inspect_rows: bool = True) -> dict[str, Any]:
    info, tasks, episodes, episode_stats = load_metadata(root)
    validate_tasks(tasks, root)
    total_episodes = int(info.get("total_episodes", -1))
    total_frames = int(info.get("total_frames", -1))
    expected_episodes = list(range(total_episodes))
    if [record.get("episode_index") for record in episodes] != expected_episodes:
        raise MergeError(f"Episode records in {root} must be contiguous from zero")
    if [record.get("episode_index") for record in episode_stats] != expected_episodes:
        raise MergeError(f"Episode-stat records in {root} must be contiguous from zero")
    if sum(int(record.get("length", -1)) for record in episodes) != total_frames:
        raise MergeError(f"Episode lengths in {root} do not equal info.json total_frames")
    if int(info.get("total_tasks", -1)) != len(tasks):
        raise MergeError(f"Task count in {root} does not match info.json")

    raw_encoding = raw_depth_encoding(info)
    if raw_encoding is not None:
        if raw_encoding.get("storage") != "lossless_png":
            raise MergeError(f"Unsupported raw-depth storage in {root}: {raw_encoding.get('storage')}")
        if int(raw_encoding.get("total_files", -1)) != total_frames:
            raise MergeError(f"Raw-depth file count in {root} does not match info.json total_frames")

    aligned_encoding = aligned_depth_encoding(info)
    if aligned_encoding is not None:
        if aligned_encoding.get("storage") != "lossless_png":
            raise MergeError(
                f"Unsupported aligned-depth storage in {root}: {aligned_encoding.get('storage')}"
            )
        if int(aligned_encoding.get("total_files", -1)) != total_frames:
            raise MergeError(
                f"Aligned-depth file count in {root} does not match info.json total_frames"
            )

    lz4_config = surface_normals_lz4(info)
    lz4_feature_key = lz4_config.get("feature_key") if lz4_config is not None else None
    if lz4_config is not None:
        storage_root = lz4_storage_root(root, info)
        backup_root = lz4_backup_root(root, info)
        if not storage_root.is_dir():
            raise MergeError(f"Missing surface-normal LZ4 root: {storage_root}")
        if not backup_root.is_dir():
            raise MergeError(f"Missing surface-normal H.264 backup root: {backup_root}")

    expected_index = 0
    raw_files = 0
    aligned_files = 0
    required_columns = {"frame_index", "episode_index", "index", "task_index"}
    task_indices = set(range(len(tasks)))
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        length = int(episode["length"])
        parquet_path = episode_path(root, info, episode_index)
        if not parquet_path.is_file():
            raise MergeError(f"Missing episode data: {parquet_path}")
        for key in video_keys(info):
            if key == lz4_feature_key:
                validate_lz4_episode(root, info, episode_index, length)
                continue
            path = video_path(root, info, episode_index, key)
            if not path.is_file():
                raise MergeError(f"Missing episode video: {path}")
        if raw_encoding is not None:
            for frame_index in range(length):
                path = raw_depth_path(root, info, episode_index, frame_index)
                if not path.is_file():
                    raise MergeError(f"Missing raw-depth frame: {path}")
                raw_files += 1
        if aligned_encoding is not None:
            for frame_index in range(length):
                path = aligned_depth_path(root, info, episode_index, frame_index)
                if not path.is_file():
                    raise MergeError(f"Missing aligned-depth frame: {path}")
                aligned_files += 1
        if not inspect_rows:
            continue
        table = pq.read_table(parquet_path, columns=list(required_columns))
        if table.num_rows != length:
            raise MergeError(f"{parquet_path} has {table.num_rows} rows; metadata says {length}")
        if not required_columns.issubset(table.column_names):
            raise MergeError(f"{parquet_path} lacks required index columns")
        frames = table["frame_index"].to_numpy(zero_copy_only=False)
        episode_values = table["episode_index"].to_numpy(zero_copy_only=False)
        indices = table["index"].to_numpy(zero_copy_only=False)
        task_values = table["task_index"].to_numpy(zero_copy_only=False)
        if not np.array_equal(frames, np.arange(length)):
            raise MergeError(f"frame_index is not contiguous in {parquet_path}")
        if not np.all(episode_values == episode_index):
            raise MergeError(f"episode_index is incorrect in {parquet_path}")
        if not np.array_equal(indices, np.arange(expected_index, expected_index + length)):
            raise MergeError(f"global index is not contiguous in {parquet_path}")
        if not set(int(value) for value in np.unique(task_values)).issubset(task_indices):
            raise MergeError(f"task_index is invalid in {parquet_path}")
        expected_index += length
    if raw_encoding is not None and raw_files != total_frames:
        raise MergeError(f"Found {raw_files} raw-depth files in {root}; expected {total_frames}")
    if aligned_encoding is not None and aligned_files != total_frames:
        raise MergeError(
            f"Found {aligned_files} aligned-depth files in {root}; expected {total_frames}"
        )
    return info


def compatibility_value(root: Path, filename: str) -> Any:
    path = root / "meta" / filename
    return read_json(path) if path.exists() else None


def check_compatible(destination: Path, source: Path, destination_info: dict[str, Any], source_info: dict[str, Any]) -> None:
    fields = ("robot_type", "fps", "features", "depth_encoding")
    differences = [field for field in fields if destination_info.get(field) != source_info.get(field)]
    destination_raw = raw_depth_encoding(destination_info)
    source_raw = raw_depth_encoding(source_info)
    destination_raw_signature = (
        {key: value for key, value in destination_raw.items() if key != "total_files"}
        if destination_raw is not None
        else None
    )
    source_raw_signature = (
        {key: value for key, value in source_raw.items() if key != "total_files"}
        if source_raw is not None
        else None
    )
    if destination_raw_signature != source_raw_signature:
        differences.append("raw_depth_encoding")
    destination_aligned = aligned_depth_encoding(destination_info)
    source_aligned = aligned_depth_encoding(source_info)
    destination_aligned_signature = (
        {key: value for key, value in destination_aligned.items() if key != "total_files"}
        if destination_aligned is not None
        else None
    )
    source_aligned_signature = (
        {key: value for key, value in source_aligned.items() if key != "total_files"}
        if source_aligned is not None
        else None
    )
    if destination_aligned_signature != source_aligned_signature:
        differences.append("aligned_depth_encoding")
    destination_lz4 = surface_normals_lz4(destination_info)
    source_lz4 = surface_normals_lz4(source_info)
    if destination_lz4 != source_lz4:
        differences.append(SURFACE_NORMALS_LZ4_KEY)
    if differences:
        raise MergeError(f"{source} is incompatible with {destination}: different {', '.join(differences)}")
    if compatibility_value(destination, "modality.json") != compatibility_value(source, "modality.json"):
        raise MergeError(f"{source} and {destination} have different meta/modality.json files")


def numpy_stats(values: np.ndarray) -> dict[str, np.ndarray]:
    array = np.asarray(values).reshape(-1, 1)
    return {
        "min": np.min(array, axis=0),
        "max": np.max(array, axis=0),
        "mean": np.mean(array, axis=0),
        "std": np.std(array, axis=0),
        "count": np.array([array.shape[0]], dtype=np.int64),
        "q01": np.quantile(array, 0.01, axis=0),
        "q10": np.quantile(array, 0.10, axis=0),
        "q50": np.quantile(array, 0.50, axis=0),
        "q90": np.quantile(array, 0.90, axis=0),
        "q99": np.quantile(array, 0.99, axis=0),
    }


def to_numpy_stats(stats: dict[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    return {
        feature: {key: np.asarray(value) for key, value in feature_stats.items()}
        for feature, feature_stats in stats.items()
    }


def aggregate_feature_stats(items: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    means = np.stack([item["mean"] for item in items])
    variances = np.stack([item["std"] ** 2 for item in items])
    counts = np.stack([item["count"] for item in items])
    total_count = counts.sum(axis=0)
    while counts.ndim < means.ndim:
        counts = np.expand_dims(counts, axis=-1)
    total_mean = (means * counts).sum(axis=0) / total_count
    total_variance = ((variances + (means - total_mean) ** 2) * counts).sum(axis=0) / total_count
    result = {
        "min": np.min(np.stack([item["min"] for item in items]), axis=0),
        "max": np.max(np.stack([item["max"] for item in items]), axis=0),
        "mean": total_mean,
        "std": np.sqrt(total_variance),
        "count": total_count,
    }
    for key in ("q01", "q10", "q50", "q90", "q99"):
        if all(key in item for item in items):
            result[key] = (np.stack([item[key] for item in items]) * counts).sum(axis=0) / total_count
    return result


def aggregate_all_stats(episode_stats: list[dict[str, Any]]) -> dict[str, dict[str, np.ndarray]]:
    converted = [to_numpy_stats(record["stats"]) for record in episode_stats]
    features = {feature for stats in converted for feature in stats}
    return {
        feature: aggregate_feature_stats([stats[feature] for stats in converted if feature in stats])
        for feature in sorted(features)
    }


def replace_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    column_index = table.schema.get_field_index(name)
    if column_index < 0:
        raise MergeError(f"Incoming Parquet lacks required column {name}")
    field = table.schema.field(column_index)
    array = pa.array(values, type=field.type)
    return table.set_column(column_index, field, array)


def copy_reindexed_lz4_episode(
    destination: Path,
    source: Path,
    destination_info: dict[str, Any],
    source_info: dict[str, Any],
    old_episode_index: int,
    new_episode_index: int,
    length: int,
) -> None:
    validate_lz4_episode(
        source,
        source_info,
        old_episode_index,
        length,
        verify_hashes=True,
    )
    source_episode = lz4_storage_root(source, source_info) / f"episode_{old_episode_index:06d}"
    destination_episode = (
        lz4_storage_root(destination, destination_info) / f"episode_{new_episode_index:06d}"
    )
    if destination_episode.exists():
        raise MergeError(f"Destination LZ4 episode already exists: {destination_episode}")
    destination_episode.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_episode, destination_episode, copy_function=shutil.copy2)
    index_path = destination_episode / "index.json"
    index = read_json(index_path)
    index["episode_index"] = new_episode_index
    atomic_write_json(index_path, index)

    source_backup = lz4_backup_root(source, source_info) / f"episode_{old_episode_index:06d}.mp4"
    destination_backup = (
        lz4_backup_root(destination, destination_info) / f"episode_{new_episode_index:06d}.mp4"
    )
    destination_backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_backup, destination_backup)
    validate_lz4_episode(destination, destination_info, new_episode_index, length)


def refresh_lz4_metadata(root: Path, info: dict[str, Any]) -> None:
    config = surface_normals_lz4(info)
    if config is None:
        return
    episodes = read_jsonl(root / "meta" / "episodes.jsonl")
    compressed_bytes = sum(
        validate_lz4_episode(
            root,
            info,
            int(episode["episode_index"]),
            int(episode["length"]),
        )
        for episode in episodes
    )
    manifest_path = lz4_storage_root(root, info) / "manifest.json"
    manifest = read_json(manifest_path)
    manifest.update(
        {
            "feature_key": config["feature_key"],
            "dtype": "uint8",
            "layout": "FHWC",
            "transform": "none",
            "chunk_frames": int(config["chunk_frames"]),
            "episode_count": int(info["total_episodes"]),
            "frame_count": int(info["total_frames"]),
            "compressed_bytes": compressed_bytes,
            "all_lossless_round_trips_verified": True,
            "canonical_commit_pending": False,
        }
    )
    atomic_write_json(manifest_path, manifest)

    h264_info = json.loads(json.dumps(info))
    h264_info.pop(SURFACE_NORMALS_LZ4_KEY, None)
    atomic_write_json(root / "meta" / "info.json.h264_backup", h264_info)


def append_one(destination: Path, source: Path) -> tuple[int, int, int]:
    destination_info, destination_tasks, destination_episodes, destination_stats = load_metadata(destination)
    source_info, source_tasks, source_episodes, source_stats = load_metadata(source)
    validate_tasks(destination_tasks, destination)
    validate_tasks(source_tasks, source)
    check_compatible(destination, source, destination_info, source_info)

    source_stats_by_episode = {int(record["episode_index"]): record for record in source_stats}
    task_text_to_index = {record["task"]: int(record["task_index"]) for record in destination_tasks}
    source_task_map: dict[int, int] = {}
    for task in source_tasks:
        text = task["task"]
        if text not in task_text_to_index:
            new_index = len(destination_tasks)
            destination_tasks.append({"task_index": new_index, "task": text})
            task_text_to_index[text] = new_index
        source_task_map[int(task["task_index"])] = task_text_to_index[text]

    first_episode = len(destination_episodes)
    next_global_index = int(destination_info["total_frames"])
    source_to_destination_episode: dict[int, int] = {}

    for offset, source_episode in enumerate(source_episodes):
        old_episode_index = int(source_episode["episode_index"])
        new_episode_index = first_episode + offset
        source_to_destination_episode[old_episode_index] = new_episode_index
        source_parquet = episode_path(source, source_info, old_episode_index)
        table = pq.read_table(source_parquet)
        length = table.num_rows
        if length != int(source_episode["length"]):
            raise MergeError(f"{source_parquet} length differs from its episode metadata")

        old_task_values = table["task_index"].to_numpy(zero_copy_only=False)
        try:
            new_task_values = np.array([source_task_map[int(value)] for value in old_task_values], dtype=np.int64)
        except KeyError as exc:
            raise MergeError(f"{source_parquet} refers to missing task_index {exc.args[0]}") from exc
        new_global_values = np.arange(next_global_index, next_global_index + length, dtype=np.int64)
        new_episode_values = np.full(length, new_episode_index, dtype=np.int64)
        table = replace_column(table, "episode_index", new_episode_values)
        table = replace_column(table, "index", new_global_values)
        table = replace_column(table, "task_index", new_task_values)

        destination_parquet = episode_path(destination, destination_info, new_episode_index)
        destination_parquet.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, destination_parquet, compression="snappy")
        destination_lz4 = surface_normals_lz4(destination_info)
        lz4_feature_key = (
            destination_lz4.get("feature_key") if destination_lz4 is not None else None
        )
        for key in video_keys(destination_info):
            if key == lz4_feature_key:
                copy_reindexed_lz4_episode(
                    destination,
                    source,
                    destination_info,
                    source_info,
                    old_episode_index,
                    new_episode_index,
                    length,
                )
                continue
            source_video = video_path(source, source_info, old_episode_index, key)
            destination_video = video_path(destination, destination_info, new_episode_index, key)
            destination_video.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_video, destination_video)
        if raw_depth_encoding(destination_info) is not None:
            for frame_index in range(length):
                source_raw_depth = raw_depth_path(
                    source,
                    source_info,
                    old_episode_index,
                    frame_index,
                )
                destination_raw_depth = raw_depth_path(
                    destination,
                    destination_info,
                    new_episode_index,
                    frame_index,
                )
                destination_raw_depth.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_raw_depth, destination_raw_depth)
        if aligned_depth_encoding(destination_info) is not None:
            for frame_index in range(length):
                source_aligned_depth = aligned_depth_path(
                    source,
                    source_info,
                    old_episode_index,
                    frame_index,
                )
                destination_aligned_depth = aligned_depth_path(
                    destination,
                    destination_info,
                    new_episode_index,
                    frame_index,
                )
                destination_aligned_depth.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source_aligned_depth, destination_aligned_depth)

        new_episode = dict(source_episode)
        new_episode["episode_index"] = new_episode_index
        if "tasks" in new_episode:
            new_episode["tasks"] = list(dict.fromkeys(new_episode["tasks"]))
        destination_episodes.append(new_episode)

        if old_episode_index not in source_stats_by_episode:
            raise MergeError(f"No episode statistics for episode {old_episode_index} in {source}")
        new_stats = json.loads(json.dumps(source_stats_by_episode[old_episode_index]))
        new_stats["episode_index"] = new_episode_index
        new_stats["stats"]["episode_index"] = numpy_stats(new_episode_values)
        new_stats["stats"]["index"] = numpy_stats(new_global_values)
        new_stats["stats"]["task_index"] = numpy_stats(new_task_values)
        destination_stats.append(new_stats)
        next_global_index += length

    total_episodes = len(destination_episodes)
    destination_info["total_episodes"] = total_episodes
    destination_info["total_frames"] = next_global_index
    destination_info["total_tasks"] = len(destination_tasks)
    destination_info["total_chunks"] = math.ceil(total_episodes / int(destination_info["chunks_size"]))
    destination_info["total_videos"] = total_episodes * len(video_keys(destination_info))
    destination_info["splits"] = {"train": f"0:{total_episodes}"}
    destination_raw = raw_depth_encoding(destination_info)
    if destination_raw is not None:
        destination_raw["total_files"] = next_global_index
    destination_aligned = aligned_depth_encoding(destination_info)
    if destination_aligned is not None:
        destination_aligned["total_files"] = next_global_index

    meta = destination / "meta"
    atomic_write_jsonl(meta / "tasks.jsonl", destination_tasks)
    atomic_write_jsonl(meta / "episodes.jsonl", destination_episodes)
    atomic_write_jsonl(meta / "episodes_stats.jsonl", destination_stats)
    atomic_write_json(meta / "stats.json", aggregate_all_stats(destination_stats))
    atomic_write_json(meta / "info.json", destination_info)
    refresh_lz4_metadata(destination, destination_info)
    relative_stats = meta / "relative_stats.json"
    if relative_stats.exists():
        relative_stats.unlink()
    return len(source_episodes), int(source_info["total_frames"]), len(destination_tasks)


def merge_direct_in_stage(stage: Path, sources: list[Path]) -> tuple[int, int]:
    added_episodes = 0
    added_frames = 0
    if not is_dataset(stage):
        first, *sources = sources
        copy_source_tree(first, stage)
        validate_dataset(stage)
        info = read_json(stage / "meta" / "info.json")
        added_episodes += int(info["total_episodes"])
        added_frames += int(info["total_frames"])
    else:
        validate_dataset(stage)
    for source in sources:
        validate_dataset(source)
        episodes, frames, _ = append_one(stage, source)
        added_episodes += episodes
        added_frames += frames
    validate_dataset(stage)
    return added_episodes, added_frames


def destination_split_path(root: Path, logical_name: str) -> Path:
    if logical_name != "validation":
        return root / logical_name
    if (root / "validation").exists():
        return root / "validation"
    if (root / "validate").exists():
        return root / "validate"
    return root / "validation"


def merge_split_root_in_stage(stage: Path, sources: list[Path]) -> tuple[int, int]:
    total_episodes = 0
    total_frames = 0
    for logical_name in SPLIT_NAMES:
        incoming = [splits[logical_name] for source in sources if logical_name in (splits := split_dirs(source))]
        if not incoming:
            continue
        destination_split = destination_split_path(stage, logical_name)
        episodes, frames = merge_direct_in_stage(destination_split, incoming)
        total_episodes += episodes
        total_frames += frames
    if not split_dirs(stage):
        raise MergeError("No train, test, or validation datasets were supplied")
    return total_episodes, total_frames


def atomic_merge(destination: Path, sources: list[Path], mode: str) -> tuple[int, int]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.parent / f".{destination.name}.merge-{uuid.uuid4().hex}"
    backup = destination.parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
    destination_existed = destination.exists()
    try:
        if destination_existed and any(destination.iterdir()):
            clone_existing(destination, stage)
        else:
            stage.mkdir()
        if mode == "dataset":
            result = merge_direct_in_stage(stage, sources)
        else:
            result = merge_split_root_in_stage(stage, sources)

        if destination_existed:
            os.replace(destination, backup)
        try:
            os.replace(stage, destination)
        except BaseException:
            if destination_existed and backup.exists():
                os.replace(backup, destination)
            raise
        if backup.exists():
            shutil.rmtree(backup)
        return result
    except BaseException:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Append one or more LeRobot v2.1 datasets into a destination (copying by default)."
    )
    parser.add_argument("destination", type=Path, help="Dataset or split root that absorbs the sources")
    parser.add_argument("sources", type=Path, nargs="+", help="One or more v2.1 datasets or split roots")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    destination = args.destination.expanduser().resolve()
    sources = [path.expanduser().resolve() for path in args.sources]
    if len(set(sources)) != len(sources):
        raise MergeError("The same source was supplied more than once")
    for source in sources:
        if not source.exists():
            raise MergeError(f"Source does not exist: {source}")
        if source == destination or source in destination.parents or destination in source.parents:
            raise MergeError(f"Source and destination may not contain one another: {source}")

    destination_mode = classify(destination, allow_empty=True)
    source_modes = {classify(source) for source in sources}
    if len(source_modes) != 1:
        raise MergeError("Do not mix direct datasets and split roots in one command")
    source_mode = next(iter(source_modes))
    if destination_mode not in {"empty", source_mode}:
        raise MergeError(f"Destination is a {destination_mode}, but sources are {source_mode}s")

    label = "dataset" if source_mode == "dataset" else "split root"
    print(f"Merging {len(sources)} LeRobot v2.1 {label}(s) into {destination}")
    episodes, frames = atomic_merge(destination, sources, source_mode)
    print(f"Done: copied {episodes} episode(s) and {frames} frame(s). Sources were not changed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except MergeError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
