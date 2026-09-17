#!/usr/bin/env python3
"""Replace canonical surface-normal MP4 folders with verified plain LZ4.

The MP4 folders are retained beside the LZ4 folders with an ``_h264_backup``
suffix.  Conversion is resumable and no canonical path or metadata is changed
until every requested split has been converted and verified.
"""

from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any

import av
import numpy as np


DEFAULT_DATASET = Path(
    "/home/alex/Development/Datasets/lerobot2/"
    "atomic_combined_09_08_And_10_08"
)
FEATURE_KEY = "observation.images.surface_normals_view"
BACKUP_KEY = f"{FEATURE_KEY}_h264_backup"
LZ4_BIN = Path("/home/alex/miniconda3/bin/lz4")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    location = parser.add_mutually_exclusive_group()
    location.add_argument("--dataset", type=Path)
    location.add_argument(
        "--split-root",
        type=Path,
        help="Convert one LeRobot split root containing meta/ and videos/ directly.",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        choices=("train", "validation", "test"),
    )
    parser.add_argument("--chunk-frames", type=int, default=32)
    parser.add_argument(
        "--jobs",
        type=int,
        default=6,
        help="Maximum episodes decoded/compressed concurrently (default: 6).",
    )
    return parser.parse_args()


def write_json_atomic(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def compress_and_verify(raw: bytes, target: Path) -> tuple[int, str]:
    temporary = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    with temporary.open("wb") as stream:
        subprocess.run(
            [str(LZ4_BIN), "-q", "-z", "-c"],
            input=raw,
            stdout=stream,
            check=True,
        )
    reconstructed = subprocess.run(
        [str(LZ4_BIN), "-q", "-d", "-c", str(temporary)],
        stdout=subprocess.PIPE,
        check=True,
    ).stdout
    if reconstructed != raw:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Lossless round-trip failed for {target}")
    os.replace(temporary, target)
    return target.stat().st_size, sha256_file(target)


def read_episodes(split_root: Path) -> list[dict[str, Any]]:
    path = split_root / "meta" / "episodes.jsonl"
    episodes = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    indices = [int(item["episode_index"]) for item in episodes]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate episode indices in {path}")
    return sorted(episodes, key=lambda item: int(item["episode_index"]))


def validate_completed_episode(
    episode_dir: Path,
    episode_index: int,
    expected_frames: int,
) -> tuple[int, int] | None:
    index_path = episode_dir / "index.json"
    if not index_path.is_file():
        return None
    try:
        index = json.loads(index_path.read_text(encoding="utf-8"))
        if int(index["episode_index"]) != episode_index:
            return None
        if index["dtype"] != "uint8" or index["layout"] != "FHWC":
            return None
        if index["transform"] != "none":
            return None
        chunks = index["chunks"]
        next_frame = 0
        compressed_bytes = 0
        for chunk in chunks:
            if int(chunk["start_frame"]) != next_frame:
                return None
            next_frame += int(chunk["frame_count"])
            path = episode_dir / chunk["filename"]
            if not path.is_file() or path.stat().st_size != int(
                chunk["compressed_bytes"]
            ):
                return None
            if sha256_file(path) != chunk["compressed_sha256"]:
                return None
            compressed_bytes += path.stat().st_size
        if next_frame != expected_frames:
            return None
        if not index.get("lossless_round_trip_verified"):
            return None
        return next_frame, compressed_bytes
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def convert_episode(
    source_video: Path,
    build_root: Path,
    episode_index: int,
    expected_frames: int,
    expected_shape: tuple[int, int, int],
    chunk_frames: int,
) -> tuple[int, int, str]:
    episode_name = f"episode_{episode_index:06d}"
    final_dir = build_root / episode_name
    completed = validate_completed_episode(
        final_dir, episode_index, expected_frames
    )
    if completed is not None:
        return completed[0], completed[1], "resumed"

    if final_dir.exists():
        shutil.rmtree(final_dir)
    temporary = build_root / f".{episode_name}.tmp-{os.getpid()}"
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.mkdir(parents=True)

    decoded_hash = hashlib.sha256()
    records: list[dict[str, Any]] = []
    pending: list[np.ndarray] = []
    decoded_count = 0

    def write_chunk(frames: list[np.ndarray]) -> None:
        nonlocal decoded_count
        array = np.ascontiguousarray(np.stack(frames))
        chunk_index = len(records)
        start_frame = decoded_count - len(array)
        filename = f"chunk_{chunk_index:06d}.lz4"
        target = temporary / filename
        compressed_bytes, compressed_sha256 = compress_and_verify(
            array.tobytes(), target
        )
        records.append(
            {
                "chunk_index": chunk_index,
                "start_frame": start_frame,
                "frame_count": len(array),
                "height": array.shape[1],
                "width": array.shape[2],
                "channels": array.shape[3],
                "uncompressed_bytes": array.nbytes,
                "filename": filename,
                "compressed_bytes": compressed_bytes,
                "compressed_sha256": compressed_sha256,
                "lossless_round_trip_verified": True,
            }
        )

    try:
        with av.open(str(source_video)) as container:
            for frame in container.decode(video=0):
                image = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
                if image.shape != expected_shape or image.dtype != np.uint8:
                    raise RuntimeError(
                        f"Unexpected frame in {source_video}: "
                        f"shape={image.shape}, dtype={image.dtype}"
                    )
                if decoded_count >= expected_frames:
                    raise RuntimeError(f"Too many frames in {source_video}")
                decoded_hash.update(image)
                pending.append(image)
                decoded_count += 1
                if len(pending) == chunk_frames:
                    write_chunk(pending)
                    pending.clear()
        if pending:
            write_chunk(pending)
            pending.clear()
        if decoded_count != expected_frames:
            raise RuntimeError(
                f"Frame-count mismatch for {source_video}: "
                f"decoded {decoded_count}, expected {expected_frames}"
            )

        index = {
            "episode_index": episode_index,
            "dtype": "uint8",
            "layout": "FHWC",
            "transform": "none",
            "compression": "LZ4 frame, CLI default fast compression",
            "chunks_are_independent": True,
            "chunk_frames": chunk_frames,
            "frame_count": decoded_count,
            "decoded_reference_sha256": decoded_hash.hexdigest(),
            "lossless_round_trip_verified": True,
            "chunks": records,
        }
        write_json_atomic(temporary / "index.json", index)
        os.replace(temporary, final_dir)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return (
        decoded_count,
        sum(int(item["compressed_bytes"]) for item in records),
        "converted",
    )


def prepare_split(
    dataset: Path,
    split: str,
    chunk_frames: int,
    split_root: Path | None = None,
    jobs: int = 1,
) -> dict[str, Any]:
    if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs <= 0:
        raise ValueError(f"jobs must be a positive integer, got {jobs!r}")
    split_root = split_root if split_root is not None else dataset / split
    info_path = split_root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    episodes = read_episodes(split_root)
    shape = tuple(
        int(value)
        for value in info["features"][FEATURE_KEY]["shape"]
    )
    if len(shape) != 3 or shape[-1] != 3:
        raise ValueError(f"Expected HWC normals in {info_path}; found {shape}")

    video_parent = split_root / "videos" / "chunk-000"
    canonical = video_parent / FEATURE_KEY
    backup = video_parent / BACKUP_KEY
    building = video_parent / f".{FEATURE_KEY}.lz4-building"

    configured = info.get("surface_normals_lz4")
    if backup.is_dir() and canonical.is_dir() and configured:
        print(f"[{split}] already installed; validating existing LZ4", flush=True)
        total_frames = 0
        total_bytes = 0
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            expected_frames = int(episode["length"])
            result = validate_completed_episode(
                canonical / f"episode_{episode_index:06d}",
                episode_index,
                expected_frames,
            )
            if result is None:
                raise RuntimeError(
                    f"Invalid installed LZ4 episode {episode_index} in {canonical}"
                )
            total_frames += result[0]
            total_bytes += result[1]
        return {
            "split": split,
            "already_installed": True,
            "frames": total_frames,
            "bytes": total_bytes,
        }

    if backup.exists():
        raise FileExistsError(
            f"Backup exists but split is not fully installed: {backup}"
        )
    if not canonical.is_dir():
        raise FileNotFoundError(f"Canonical normals folder missing: {canonical}")
    source_mp4s = list(canonical.glob("episode_*.mp4"))
    if len(source_mp4s) != len(episodes):
        raise RuntimeError(
            f"{canonical} has {len(source_mp4s)} MP4s; expected {len(episodes)}"
        )
    building.mkdir(parents=False, exist_ok=True)

    split_started = time.monotonic()
    total_frames = 0
    total_bytes = 0
    print(
        f"[{split}] converting {len(episodes)} episodes into {building} "
        f"with {min(jobs, len(episodes))} workers",
        flush=True,
    )

    def submit_episode(
        executor: ThreadPoolExecutor,
        ordinal: int,
    ) -> tuple[int, dict[str, Any], Future]:
        episode = episodes[ordinal - 1]
        episode_index = int(episode["episode_index"])
        expected_frames = int(episode["length"])
        source_video = canonical / f"episode_{episode_index:06d}.mp4"
        return (
            ordinal,
            episode,
            executor.submit(
                convert_episode,
                source_video,
                building,
                episode_index,
                expected_frames,
                shape,
                chunk_frames,
            ),
        )

    worker_count = min(jobs, len(episodes))
    executor = ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="normals-lz4",
    )
    pending: deque[tuple[int, dict[str, Any], Future]] = deque()
    next_ordinal = 1
    try:
        while next_ordinal <= len(episodes) and len(pending) < worker_count:
            pending.append(submit_episode(executor, next_ordinal))
            next_ordinal += 1

        while pending:
            ordinal, episode, future = pending.popleft()
            # Preserve serial episode/error order even when later work finishes first.
            frames, compressed_bytes, status = future.result()
            episode_index = int(episode["episode_index"])
            total_frames += frames
            total_bytes += compressed_bytes
            elapsed = time.monotonic() - split_started
            print(
                f"[{split} {ordinal:03d}/{len(episodes):03d}] "
                f"episode_{episode_index:06d} {status}: {frames} frames, "
                f"{compressed_bytes / (1024 ** 2):.1f} MiB; "
                f"total {total_bytes / (1024 ** 3):.2f} GiB, {elapsed:.1f}s",
                flush=True,
            )
            if next_ordinal <= len(episodes):
                pending.append(submit_episode(executor, next_ordinal))
                next_ordinal += 1
    finally:
        for _, _, future in pending:
            future.cancel()
        executor.shutdown(wait=True, cancel_futures=True)

    expected_total = sum(int(item["length"]) for item in episodes)
    if total_frames != expected_total or total_frames != int(info["total_frames"]):
        raise RuntimeError(
            f"{split} total mismatch: {total_frames}, "
            f"episodes={expected_total}, info={info['total_frames']}"
        )
    manifest = {
        "source_dataset": str(dataset),
        "split": split,
        "feature_key": FEATURE_KEY,
        "format": "independent plain LZ4 Frame chunks",
        "dtype": "uint8",
        "layout": "FHWC",
        "transform": "none",
        "chunk_frames": chunk_frames,
        "episode_count": len(episodes),
        "frame_count": total_frames,
        "compressed_bytes": total_bytes,
        "all_lossless_round_trips_verified": True,
        "canonical_commit_pending": True,
    }
    write_json_atomic(building / "manifest.json", manifest)
    print(
        f"[{split}] READY: {len(episodes)} episodes, {total_frames} frames, "
        f"{total_bytes / (1024 ** 3):.2f} GiB",
        flush=True,
    )
    return {
        "split": split,
        "already_installed": False,
        "split_root": split_root,
        "info_path": info_path,
        "info": info,
        "canonical": canonical,
        "backup": backup,
        "building": building,
        "frames": total_frames,
        "bytes": total_bytes,
    }


def commit_split(prepared: dict[str, Any], chunk_frames: int) -> None:
    if prepared["already_installed"]:
        return
    split = prepared["split"]
    info_path: Path = prepared["info_path"]
    canonical: Path = prepared["canonical"]
    backup: Path = prepared["backup"]
    building: Path = prepared["building"]
    info: dict[str, Any] = prepared["info"]

    info_backup = info_path.with_name("info.json.h264_backup")
    if info_backup.exists():
        raise FileExistsError(f"Refusing to overwrite metadata backup: {info_backup}")
    shutil.copy2(info_path, info_backup)

    info["surface_normals_lz4"] = {
        "feature_key": FEATURE_KEY,
        "storage": "plain_lz4_chunks",
        "root": str(canonical.relative_to(prepared["split_root"])),
        "dtype": "uint8",
        "layout": "FHWC",
        "chunk_frames": chunk_frames,
        "lossless_round_trip_verified": True,
        "h264_backup": str(backup.relative_to(prepared["split_root"])),
    }
    pending_info = info_path.with_name(f".{info_path.name}.lz4-ready-{os.getpid()}")
    pending_info.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")

    canonical.rename(backup)
    try:
        building.rename(canonical)
        os.replace(pending_info, info_path)
    except BaseException:
        if canonical.exists() and not building.exists():
            canonical.rename(building)
        if backup.exists() and not canonical.exists():
            backup.rename(canonical)
        pending_info.unlink(missing_ok=True)
        shutil.copy2(info_backup, info_path)
        raise

    manifest_path = canonical / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["canonical_commit_pending"] = False
    manifest["canonical_lz4_path"] = str(canonical)
    manifest["h264_backup_path"] = str(backup)
    write_json_atomic(manifest_path, manifest)
    print(f"[{split}] COMMITTED LZ4 canonical; H.264 retained at {backup}", flush=True)


def main() -> int:
    args = parse_args()
    if args.chunk_frames <= 0:
        raise ValueError("--chunk-frames must be positive")
    if args.jobs <= 0:
        raise ValueError("--jobs must be positive")
    if not LZ4_BIN.is_file():
        raise RuntimeError(f"The lz4 CLI is required at {LZ4_BIN}")

    if args.split_root is not None:
        split_root = args.split_root.resolve()
        if not split_root.is_dir():
            raise FileNotFoundError(split_root)
        split = split_root.name
        print(f"Split root: {split_root}", flush=True)
        prepared = [
            prepare_split(
                split_root,
                split,
                args.chunk_frames,
                split_root=split_root,
                jobs=args.jobs,
            )
        ]
    else:
        dataset = (args.dataset or DEFAULT_DATASET).resolve()
        if not dataset.is_dir():
            raise FileNotFoundError(dataset)
        print(f"Dataset: {dataset}", flush=True)
        print(f"Splits: {', '.join(args.splits)}", flush=True)
        prepared = [
            prepare_split(dataset, split, args.chunk_frames, jobs=args.jobs)
            for split in args.splits
        ]
    print(f"Chunk frames: {args.chunk_frames}", flush=True)
    print(f"Episode jobs: {args.jobs}", flush=True)
    print("All requested splits verified; beginning canonical commit", flush=True)
    for item in prepared:
        commit_split(item, args.chunk_frames)
    total_frames = sum(int(item["frames"]) for item in prepared)
    total_bytes = sum(int(item["bytes"]) for item in prepared)
    print(
        f"COMPLETE: {len(prepared)} splits, {total_frames} frames, "
        f"{total_bytes / (1024 ** 3):.2f} GiB LZ4",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
