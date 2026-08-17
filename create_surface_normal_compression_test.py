#!/usr/bin/env python3
"""Build a small, losslessly verified surface-normal compression corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import av
import numpy as np


DEFAULT_SOURCE = Path(
    "/home/alex/Development/Datasets/lerobot2/atomic_combined_09_08_And_10_08"
)
DEFAULT_TARGET = Path(
    "/home/alex/Development/Datasets/lerobot2/"
    "atomic_combined_09_08_And_10_08_testing_surface_normal_compression"
)

# One median-length episode from every task, plus the longest and shortest
# selected training episodes. This is more representative than episodes 0-9,
# which contain almost exclusively cereal-box demonstrations.
DEFAULT_EPISODES = [2, 17, 22, 85, 176, 187, 245, 263, 269, 293]

CANONICAL_DIR = "00_canonical_h264_lossless"
RAW_DIR = "01_raw_uint8_npy"
PLAIN_DIR = "02_plain_lz4_chunks"
PLANAR_DIR = "03_planar_channels_lz4_chunks"
DELTA_DIR = "04_temporal_delta_lz4_chunks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--split", default="train")
    parser.add_argument("--chunk-frames", type=int, default=32)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=DEFAULT_EPISODES,
        metavar="INDEX",
    )
    return parser.parse_args()


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(array: np.ndarray, frames_per_block: int = 32) -> str:
    digest = hashlib.sha256()
    for start in range(0, len(array), frames_per_block):
        digest.update(np.ascontiguousarray(array[start : start + frames_per_block]))
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def compress_lz4(data: bytes, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("wb") as stream:
        subprocess.run(
            ["lz4", "-q", "-z", "-c"],
            input=data,
            stdout=stream,
            check=True,
        )


def decompress_lz4(path: Path) -> bytes:
    return subprocess.run(
        ["lz4", "-q", "-d", "-c", str(path)],
        stdout=subprocess.PIPE,
        check=True,
    ).stdout


def temporal_delta_encode(frames: np.ndarray) -> np.ndarray:
    delta = np.empty_like(frames)
    delta[0] = frames[0]
    # uint8 subtraction intentionally wraps modulo 256 and is exactly reversible.
    delta[1:] = frames[1:] - frames[:-1]
    return delta


def temporal_delta_decode(delta: np.ndarray) -> np.ndarray:
    frames = np.empty_like(delta)
    frames[0] = delta[0]
    for frame_index in range(1, len(delta)):
        # uint8 addition intentionally wraps modulo 256.
        frames[frame_index] = frames[frame_index - 1] + delta[frame_index]
    return frames


def variant_index(
    episode_index: int,
    layout: str,
    transform: str,
    chunks: list[dict[str, Any]],
    reference_sha256: str,
) -> dict[str, Any]:
    return {
        "episode_index": episode_index,
        "dtype": "uint8",
        "layout": layout,
        "transform": transform,
        "compression": "LZ4 frame, CLI default fast compression",
        "chunks_are_independent": True,
        "decoded_reference_sha256": reference_sha256,
        "chunks": chunks,
    }


def encode_chunk(
    frames: np.ndarray,
    episode_index: int,
    chunk_index: int,
    start_frame: int,
    staging: Path,
) -> dict[str, dict[str, Any]]:
    name = f"chunk_{chunk_index:06d}.lz4"
    episode_name = f"episode_{episode_index:06d}"

    plain_path = staging / PLAIN_DIR / episode_name / name
    planar_path = staging / PLANAR_DIR / episode_name / name
    delta_path = staging / DELTA_DIR / episode_name / name

    plain = np.ascontiguousarray(frames)
    planar = np.ascontiguousarray(frames.transpose(0, 3, 1, 2))
    delta = temporal_delta_encode(frames)

    compress_lz4(plain.tobytes(), plain_path)
    compress_lz4(planar.tobytes(), planar_path)
    compress_lz4(delta.tobytes(), delta_path)

    plain_round_trip = np.frombuffer(decompress_lz4(plain_path), dtype=np.uint8).reshape(
        plain.shape
    )
    planar_round_trip = np.frombuffer(
        decompress_lz4(planar_path), dtype=np.uint8
    ).reshape(planar.shape).transpose(0, 2, 3, 1)
    delta_round_trip = temporal_delta_decode(
        np.frombuffer(decompress_lz4(delta_path), dtype=np.uint8).reshape(delta.shape)
    )

    for variant, reconstructed in (
        ("plain_lz4", plain_round_trip),
        ("planar_channels_lz4", planar_round_trip),
        ("temporal_delta_lz4", delta_round_trip),
    ):
        if not np.array_equal(reconstructed, frames):
            raise RuntimeError(
                f"Lossless verification failed for {variant}, "
                f"episode {episode_index}, chunk {chunk_index}"
            )

    common = {
        "chunk_index": chunk_index,
        "start_frame": start_frame,
        "frame_count": len(frames),
        "height": frames.shape[1],
        "width": frames.shape[2],
        "channels": frames.shape[3],
        "uncompressed_bytes": frames.nbytes,
        "filename": name,
        "lossless_round_trip_verified": True,
    }
    return {
        "plain": {
            **common,
            "compressed_bytes": plain_path.stat().st_size,
            "compressed_sha256": sha256_file(plain_path),
        },
        "planar": {
            **common,
            "compressed_bytes": planar_path.stat().st_size,
            "compressed_sha256": sha256_file(planar_path),
        },
        "delta": {
            **common,
            "compressed_bytes": delta_path.stat().st_size,
            "compressed_sha256": sha256_file(delta_path),
        },
    }


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    target = args.target.resolve()
    split_root = source / args.split
    info_path = split_root / "meta" / "info.json"
    episodes_path = split_root / "meta" / "episodes.jsonl"

    if args.chunk_frames <= 0:
        raise ValueError("--chunk-frames must be positive")
    if len(set(args.episodes)) != len(args.episodes):
        raise ValueError("Episode indices must be unique")
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing target: {target}")
    if not info_path.is_file() or not episodes_path.is_file():
        raise FileNotFoundError(f"Not a LeRobot split: {split_root}")
    if shutil.which("lz4") is None:
        raise RuntimeError("The lz4 command-line program is required")

    info = json.loads(info_path.read_text(encoding="utf-8"))
    episode_metadata = {
        int(item["episode_index"]): item
        for item in (
            json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines()
        )
    }
    selected = [episode_metadata[index] for index in sorted(args.episodes)]
    shape = tuple(
        info["features"]["observation.images.surface_normals_view"]["shape"]
    )
    if len(shape) != 3 or shape[-1] != 3:
        raise ValueError(f"Expected HWC three-channel normals; found {shape}")
    dataset_chunk_size = int(info["chunks_size"])

    staging = target.parent / f".{target.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(f"Refusing to reuse staging directory: {staging}")
    staging.mkdir(parents=False)

    manifest: dict[str, Any] = {
        "purpose": "Surface-normal storage and loading benchmark corpus",
        "source_dataset": str(source),
        "source_split": args.split,
        "source_dataset_was_modified": False,
        "selected_episode_count": len(selected),
        "selected_episode_indices": [item["episode_index"] for item in selected],
        "selection": (
            "One representative episode from each of eight tasks, plus a long and "
            "a short episode; all selected from the training split."
        ),
        "frame_shape_hwc": list(shape),
        "dtype": "uint8",
        "lz4_chunk_frames": args.chunk_frames,
        "lz4_version": subprocess.run(
            ["lz4", "--version"], capture_output=True, text=True, check=True
        ).stderr.strip(),
        "formats": {
            CANONICAL_DIR: "Copied source lossless H.264/gbrp MP4 files",
            RAW_DIR: "Per-episode C-contiguous HWC uint8 .npy arrays",
            PLAIN_DIR: "Independent LZ4-frame chunks of C-contiguous FHWC bytes",
            PLANAR_DIR: "Independent LZ4-frame chunks after FHWC -> FCHW transform",
            DELTA_DIR: (
                "Independent LZ4-frame chunks after modulo-256 temporal delta; "
                "the first frame of every chunk is absolute"
            ),
        },
        "episodes": [],
    }

    try:
        for directory in (CANONICAL_DIR, RAW_DIR, PLAIN_DIR, PLANAR_DIR, DELTA_DIR):
            (staging / directory).mkdir()

        for ordinal, episode in enumerate(selected, start=1):
            episode_index = int(episode["episode_index"])
            expected_frames = int(episode["length"])
            episode_name = f"episode_{episode_index:06d}"
            source_video = (
                split_root
                / "videos"
                / f"chunk-{episode_index // dataset_chunk_size:03d}"
                / "observation.images.surface_normals_view"
                / f"{episode_name}.mp4"
            )
            if not source_video.is_file():
                raise FileNotFoundError(source_video)

            print(
                f"[{ordinal}/{len(selected)}] {episode_name}: "
                f"{expected_frames} frames, {episode['tasks'][0]}",
                flush=True,
            )
            source_hash_before = sha256_file(source_video)
            canonical_copy = staging / CANONICAL_DIR / source_video.name
            shutil.copy2(source_video, canonical_copy)
            canonical_copy_hash = sha256_file(canonical_copy)
            if canonical_copy_hash != source_hash_before:
                raise RuntimeError(f"Canonical file copy hash mismatch: {source_video}")

            raw_path = staging / RAW_DIR / f"{episode_name}.npy"
            raw = np.lib.format.open_memmap(
                raw_path,
                mode="w+",
                dtype=np.uint8,
                shape=(expected_frames, *shape),
            )
            decoded_hash = hashlib.sha256()
            decoded_count = 0
            pending: list[np.ndarray] = []
            chunk_records = {"plain": [], "planar": [], "delta": []}

            with av.open(str(source_video)) as container:
                for frame in container.decode(video=0):
                    image = np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
                    if image.shape != shape or image.dtype != np.uint8:
                        raise RuntimeError(
                            f"Unexpected decoded frame for {episode_name}: "
                            f"shape={image.shape}, dtype={image.dtype}"
                        )
                    if decoded_count >= expected_frames:
                        raise RuntimeError(f"Too many frames in {source_video}")
                    raw[decoded_count] = image
                    decoded_hash.update(image)
                    decoded_count += 1
                    pending.append(image)

                    if len(pending) == args.chunk_frames:
                        chunk = np.stack(pending)
                        records = encode_chunk(
                            chunk,
                            episode_index,
                            len(chunk_records["plain"]),
                            decoded_count - len(chunk),
                            staging,
                        )
                        for key in chunk_records:
                            chunk_records[key].append(records[key])
                        pending.clear()

            if pending:
                chunk = np.stack(pending)
                records = encode_chunk(
                    chunk,
                    episode_index,
                    len(chunk_records["plain"]),
                    decoded_count - len(chunk),
                    staging,
                )
                for key in chunk_records:
                    chunk_records[key].append(records[key])

            if decoded_count != expected_frames:
                raise RuntimeError(
                    f"Frame-count mismatch for {episode_name}: "
                    f"decoded {decoded_count}, metadata says {expected_frames}"
                )
            raw.flush()
            del raw

            reference_hash = decoded_hash.hexdigest()
            raw_readback = np.load(raw_path, mmap_mode="r")
            if sha256_array(raw_readback) != reference_hash:
                raise RuntimeError(f"Raw NPY verification failed for {episode_name}")
            del raw_readback

            for directory, key, layout, transform in (
                (PLAIN_DIR, "plain", "FHWC", "none"),
                (PLANAR_DIR, "planar", "FCHW", "planar channels"),
                (
                    DELTA_DIR,
                    "delta",
                    "FHWC",
                    "chunk-local modulo-256 temporal delta",
                ),
            ):
                write_json(
                    staging / directory / episode_name / "index.json",
                    variant_index(
                        episode_index,
                        layout,
                        transform,
                        chunk_records[key],
                        reference_hash,
                    ),
                )

            source_hash_after = sha256_file(source_video)
            if source_hash_after != source_hash_before:
                raise RuntimeError(f"Source file changed during conversion: {source_video}")

            manifest["episodes"].append(
                {
                    "episode_index": episode_index,
                    "tasks": episode["tasks"],
                    "frame_count": expected_frames,
                    "source_video": str(source_video.relative_to(source)),
                    "source_and_canonical_mp4_sha256": source_hash_before,
                    "decoded_reference_sha256": reference_hash,
                    "raw_npy_bytes": raw_path.stat().st_size,
                    "all_lossless_round_trips_verified": True,
                }
            )

        manifest["total_selected_frames"] = sum(
            int(item["frame_count"]) for item in manifest["episodes"]
        )
        manifest["storage_bytes"] = {
            directory: directory_bytes(staging / directory)
            for directory in (CANONICAL_DIR, RAW_DIR, PLAIN_DIR, PLANAR_DIR, DELTA_DIR)
        }
        manifest["all_source_copy_hashes_verified"] = True
        manifest["all_lossless_round_trips_verified"] = True
        write_json(staging / "manifest.json", manifest)

        with (staging / "selected_episodes.jsonl").open("w", encoding="utf-8") as stream:
            for item in manifest["episodes"]:
                stream.write(json.dumps(item) + "\n")

        readme = f"""# Surface-normal compression test corpus

Source (read-only): `{source}/{args.split}`

This folder contains the surface-normal stream for ten representative training
episodes. It intentionally excludes unchanged RGB, depth, action, and state data
so the compression comparison does not duplicate irrelevant dataset content.

Formats:

1. `{CANONICAL_DIR}`: copied lossless H.264/gbrp source videos.
2. `{RAW_DIR}`: uncompressed episode-level `uint8` NPY arrays (FHWC).
3. `{PLAIN_DIR}`: plain LZ4, in independent {args.chunk_frames}-frame FHWC chunks.
4. `{PLANAR_DIR}`: reversible FHWC-to-FCHW transform, then LZ4.
5. `{DELTA_DIR}`: reversible chunk-local temporal delta, then LZ4.

Every alternative was decompressed and compared byte-for-byte with the frames
decoded from the canonical video. `manifest.json` records provenance, checksums,
episode selection, format definitions, and storage totals. The conversion script
is `{Path(__file__).resolve()}`.
"""
        (staging / "README.md").write_text(readme, encoding="utf-8")

        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Created: {target}")
    print(f"Episodes: {len(selected)}")
    print(f"Frames: {manifest['total_selected_frames']}")
    for directory, size in manifest["storage_bytes"].items():
        print(f"{directory}: {size / (1024 ** 3):.3f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
