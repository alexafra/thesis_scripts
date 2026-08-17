#!/usr/bin/env python3
"""Benchmark loading and lossless decoding of the normal-storage test corpus."""

from __future__ import annotations

import argparse
import ctypes
import ctypes.util
import hashlib
import io
import json
import os
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import av
import numpy as np


DEFAULT_ROOT = Path(
    "/home/alex/Development/Datasets/lerobot2/"
    "atomic_combined_09_08_And_10_08_testing_surface_normal_compression"
)

FORMATS = {
    "canonical_h264": ("00_canonical_h264_lossless", "h264"),
    "raw_uint8": ("01_raw_uint8_npy", "raw"),
    "plain_lz4": ("02_plain_lz4_chunks", "plain"),
    "planar_lz4": ("03_planar_channels_lz4_chunks", "planar"),
    "temporal_delta_lz4": ("04_temporal_delta_lz4_chunks", "delta"),
}


class Lz4FrameDecoder:
    """Small ctypes binding to the installed liblz4 frame API."""

    VERSION = 100

    def __init__(self) -> None:
        library_name = ctypes.util.find_library("lz4") or "liblz4.so.1"
        self.library_name = library_name
        self.lib = ctypes.CDLL(library_name)
        self.lib.LZ4F_createDecompressionContext.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_uint,
        ]
        self.lib.LZ4F_createDecompressionContext.restype = ctypes.c_size_t
        self.lib.LZ4F_freeDecompressionContext.argtypes = [ctypes.c_void_p]
        self.lib.LZ4F_freeDecompressionContext.restype = ctypes.c_size_t
        self.lib.LZ4F_decompress.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_void_p,
        ]
        self.lib.LZ4F_decompress.restype = ctypes.c_size_t
        self.lib.LZ4F_isError.argtypes = [ctypes.c_size_t]
        self.lib.LZ4F_isError.restype = ctypes.c_uint
        self.lib.LZ4F_getErrorName.argtypes = [ctypes.c_size_t]
        self.lib.LZ4F_getErrorName.restype = ctypes.c_char_p

    def check(self, code: int) -> int:
        if self.lib.LZ4F_isError(code):
            name = self.lib.LZ4F_getErrorName(code).decode("utf-8", "replace")
            raise RuntimeError(f"liblz4 error: {name}")
        return code

    def decompress(self, compressed: bytes, expected_bytes: int) -> bytes:
        context = ctypes.c_void_p()
        self.check(
            self.lib.LZ4F_createDecompressionContext(
                ctypes.byref(context), self.VERSION
            )
        )
        destination = ctypes.create_string_buffer(expected_bytes)
        source_pointer = ctypes.c_char_p(compressed)
        source_base = ctypes.cast(source_pointer, ctypes.c_void_p).value
        if source_base is None:
            raise RuntimeError("Could not obtain compressed-buffer address")

        source_offset = 0
        destination_offset = 0
        try:
            while True:
                source_size = ctypes.c_size_t(len(compressed) - source_offset)
                destination_size = ctypes.c_size_t(
                    expected_bytes - destination_offset
                )
                result = self.check(
                    self.lib.LZ4F_decompress(
                        context,
                        ctypes.c_void_p(
                            ctypes.addressof(destination) + destination_offset
                        ),
                        ctypes.byref(destination_size),
                        ctypes.c_void_p(source_base + source_offset),
                        ctypes.byref(source_size),
                        None,
                    )
                )
                source_offset += source_size.value
                destination_offset += destination_size.value
                if result == 0:
                    break
                if source_size.value == 0 and destination_size.value == 0:
                    raise RuntimeError("liblz4 decompression made no progress")
        finally:
            self.check(self.lib.LZ4F_freeDecompressionContext(context))

        if source_offset != len(compressed):
            raise RuntimeError(
                f"LZ4 frame left {len(compressed) - source_offset} unread bytes"
            )
        if destination_offset != expected_bytes:
            raise RuntimeError(
                f"LZ4 decoded {destination_offset} bytes; expected {expected_bytes}"
            )
        return destination.raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("filesystem", "warm_cache"),
        default=("filesystem", "warm_cache"),
        help=(
            "filesystem includes file reads but uses the uncontrolled OS page cache; "
            "warm_cache preloads encoded files and measures decode/reconstruction"
        ),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=tuple(FORMATS),
        default=tuple(FORMATS),
    )
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        help="Optional subset of the ten episode indices",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output path (default: ROOT/benchmark_results/TIMESTAMP.json)",
    )
    parser.add_argument(
        "--skip-verification",
        action="store_true",
        help="Skip the untimed SHA-256 verification pass",
    )
    return parser.parse_args()


def read_input(path: Path, warm_cache: bool) -> Path | bytes:
    return path.read_bytes() if warm_cache else path


def input_bytes(value: Path | bytes) -> bytes:
    return value if isinstance(value, bytes) else value.read_bytes()


def open_av(value: Path | bytes):
    return av.open(io.BytesIO(value) if isinstance(value, bytes) else str(value))


def prepare_inputs(
    root: Path,
    format_name: str,
    episodes: list[dict[str, Any]],
    warm_cache: bool,
) -> list[dict[str, Any]]:
    directory_name, kind = FORMATS[format_name]
    directory = root / directory_name
    prepared = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        episode_name = f"episode_{episode_index:06d}"
        if kind == "h264":
            payload: Any = read_input(directory / f"{episode_name}.mp4", warm_cache)
        elif kind == "raw":
            payload = read_input(directory / f"{episode_name}.npy", warm_cache)
        else:
            episode_directory = directory / episode_name
            index = json.loads((episode_directory / "index.json").read_text())
            payload = {
                "index": index,
                "chunks": [
                    read_input(episode_directory / chunk["filename"], warm_cache)
                    for chunk in index["chunks"]
                ],
            }
        prepared.append({"episode": episode, "payload": payload})
    return prepared


def decode_episode(
    kind: str,
    payload: Any,
    lz4: Lz4FrameDecoder,
) -> Iterator[np.ndarray]:
    if kind == "h264":
        with open_av(payload) as container:
            for frame in container.decode(video=0):
                yield np.ascontiguousarray(
                    frame.to_ndarray(format="rgb24")
                )[np.newaxis, ...]
        return

    if kind == "raw":
        if isinstance(payload, bytes):
            yield np.load(io.BytesIO(payload), allow_pickle=False)
        else:
            yield np.load(payload, allow_pickle=False)
        return

    index = payload["index"]
    for chunk, encoded in zip(index["chunks"], payload["chunks"], strict=True):
        frame_count = int(chunk["frame_count"])
        height = int(chunk["height"])
        width = int(chunk["width"])
        channels = int(chunk["channels"])
        decoded = lz4.decompress(
            input_bytes(encoded), int(chunk["uncompressed_bytes"])
        )
        if kind == "plain":
            frames = np.frombuffer(decoded, dtype=np.uint8).reshape(
                frame_count, height, width, channels
            )
        elif kind == "planar":
            planar = np.frombuffer(decoded, dtype=np.uint8).reshape(
                frame_count, channels, height, width
            )
            frames = np.ascontiguousarray(planar.transpose(0, 2, 3, 1))
        elif kind == "delta":
            delta = np.frombuffer(decoded, dtype=np.uint8).reshape(
                frame_count, height, width, channels
            )
            frames = np.empty_like(delta)
            frames[0] = delta[0]
            for frame_index in range(1, frame_count):
                frames[frame_index] = (
                    frames[frame_index - 1] + delta[frame_index]
                )
        else:
            raise ValueError(kind)
        yield frames


def consume(
    kind: str,
    prepared: list[dict[str, Any]],
    lz4: Lz4FrameDecoder,
    verify: bool,
) -> dict[str, Any]:
    total_frames = 0
    total_decoded_bytes = 0
    sentinel = 0
    verified_episodes = 0

    for item in prepared:
        expected = item["episode"]
        digest = hashlib.sha256() if verify else None
        episode_frames = 0
        for frames in decode_episode(kind, item["payload"], lz4):
            contiguous = np.ascontiguousarray(frames)
            if contiguous.dtype != np.uint8 or contiguous.shape[-1] != 3:
                raise RuntimeError(
                    f"Unexpected decoded array: {contiguous.shape} {contiguous.dtype}"
                )
            episode_frames += len(contiguous)
            total_frames += len(contiguous)
            total_decoded_bytes += contiguous.nbytes
            flat = contiguous.reshape(-1)
            sentinel = ((sentinel * 131) ^ int(flat[0]) ^ int(flat[-1])) & 0xFFFFFFFF
            if digest is not None:
                digest.update(contiguous)

        if episode_frames != int(expected["frame_count"]):
            raise RuntimeError(
                f"Episode {expected['episode_index']} decoded {episode_frames} frames, "
                f"expected {expected['frame_count']}"
            )
        if digest is not None:
            actual_hash = digest.hexdigest()
            expected_hash = expected["decoded_reference_sha256"]
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Episode {expected['episode_index']} hash mismatch: "
                    f"{actual_hash} != {expected_hash}"
                )
            verified_episodes += 1

    return {
        "frames": total_frames,
        "decoded_bytes": total_decoded_bytes,
        "sentinel": sentinel,
        "verified_episodes": verified_episodes,
    }


def timed_run(
    kind: str,
    prepared: list[dict[str, Any]],
    lz4: Lz4FrameDecoder,
) -> dict[str, Any]:
    cpu_start = time.process_time()
    wall_start = time.perf_counter()
    counts = consume(kind, prepared, lz4, verify=False)
    wall_seconds = time.perf_counter() - wall_start
    cpu_seconds = time.process_time() - cpu_start
    gib = counts["decoded_bytes"] / (1024**3)
    return {
        **counts,
        "wall_seconds": wall_seconds,
        "cpu_seconds": cpu_seconds,
        "frames_per_second": counts["frames"] / wall_seconds,
        "decoded_gib_per_second": gib / wall_seconds,
    }


def main() -> int:
    args = parse_args()
    if args.repeats <= 0 or args.warmups < 0:
        raise ValueError("--repeats must be positive and --warmups non-negative")

    root = args.root.resolve()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    episodes = manifest["episodes"]
    if args.episodes:
        wanted = set(args.episodes)
        episodes = [item for item in episodes if int(item["episode_index"]) in wanted]
        found = {int(item["episode_index"]) for item in episodes}
        if found != wanted:
            raise ValueError(f"Unknown episode indices: {sorted(wanted - found)}")

    lz4 = Lz4FrameDecoder()
    result: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "corpus_root": str(root),
        "episode_indices": [int(item["episode_index"]) for item in episodes],
        "episode_count": len(episodes),
        "frame_count": sum(int(item["frame_count"]) for item in episodes),
        "settings": {
            "repeats": args.repeats,
            "warmups": args.warmups,
            "modes": args.modes,
            "formats": args.formats,
            "verification_enabled": not args.skip_verification,
            "filesystem_mode_note": (
                "Includes file reads but does not clear or control the OS page cache."
            ),
            "warm_cache_mode_note": (
                "Encoded files are preloaded; timings include decode and reversible "
                "layout reconstruction but exclude storage reads."
            ),
        },
        "environment": {
            "hostname": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "numpy": np.__version__,
            "pyav": av.__version__,
            "liblz4": lz4.library_name,
            "pid": os.getpid(),
        },
        "results": [],
    }

    for mode in args.modes:
        warm_cache = mode == "warm_cache"
        for format_name in args.formats:
            directory_name, kind = FORMATS[format_name]
            print(f"Preparing {mode}/{format_name}...", flush=True)
            prepared = prepare_inputs(root, format_name, episodes, warm_cache)

            if not args.skip_verification:
                verified = consume(kind, prepared, lz4, verify=True)
                if verified["verified_episodes"] != len(episodes):
                    raise RuntimeError(f"Incomplete verification for {format_name}")

            for _ in range(args.warmups):
                timed_run(kind, prepared, lz4)

            runs = [timed_run(kind, prepared, lz4) for _ in range(args.repeats)]
            median_wall = statistics.median(run["wall_seconds"] for run in runs)
            median_cpu = statistics.median(run["cpu_seconds"] for run in runs)
            median_fps = statistics.median(
                run["frames_per_second"] for run in runs
            )
            median_gib_s = statistics.median(
                run["decoded_gib_per_second"] for run in runs
            )
            result["results"].append(
                {
                    "mode": mode,
                    "format": format_name,
                    "folder": directory_name,
                    "median_wall_seconds": median_wall,
                    "median_cpu_seconds": median_cpu,
                    "median_frames_per_second": median_fps,
                    "median_decoded_gib_per_second": median_gib_s,
                    "runs": runs,
                }
            )
            print(
                f"  {median_wall:.3f}s, {median_fps:.1f} frames/s, "
                f"{median_gib_s:.3f} decoded GiB/s",
                flush=True,
            )
            del prepared

    output = args.output
    if output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output = root / "benchmark_results" / f"benchmark_{timestamp}.json"
    output = output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)

    print("\nMedian results")
    print("mode\tformat\tseconds\tframes/s\tdecoded GiB/s")
    for item in result["results"]:
        print(
            f"{item['mode']}\t{item['format']}\t"
            f"{item['median_wall_seconds']:.3f}\t"
            f"{item['median_frames_per_second']:.1f}\t"
            f"{item['median_decoded_gib_per_second']:.3f}"
        )
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
