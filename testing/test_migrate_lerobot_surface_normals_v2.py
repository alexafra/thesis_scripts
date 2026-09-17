from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import av
import cv2
import numpy as np
import unitree_lerobot.utils.surface_normal_encoding as surface_normal_encoding
from unitree_lerobot.utils.surface_normal_encoding import (
    LEGACY_SURFACE_NORMAL_ENCODING_VERSION,
    SURFACE_NORMAL_ENCODING_VERSION,
    PinholeIntrinsics,
    encode_surface_normals_rgb,
    surface_normals_encoding_metadata,
)


SCRIPT = Path("/home/alex/Development/scripts/migrate_lerobot_surface_normals_v2.py")
SPEC = importlib.util.spec_from_file_location("migrate_lerobot_surface_normals_v2", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MIGRATION = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MIGRATION
SPEC.loader.exec_module(MIGRATION)

SPLITS = ("train", "validation", "test")
FEATURE_KEY = "observation.images.surface_normals_view"
HEIGHT = 6
WIDTH = 8
INTRINSICS = PinholeIntrinsics(
    width=WIDTH,
    height=HEIGHT,
    fx=8.0,
    fy=8.0,
    cx=3.5,
    cy=2.5,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value) + "\n" for value in values),
        encoding="utf-8",
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def dummy_image_stats(value: float, count: int) -> dict[str, list]:
    channel = [[[value]], [[value]], [[value]]]
    return {
        "min": channel,
        "max": channel,
        "mean": channel,
        "std": [[[0.0]], [[0.0]], [[0.0]]],
        "count": [count],
        "q01": channel,
        "q10": channel,
        "q50": channel,
        "q90": channel,
        "q99": channel,
    }


def fixture_depths() -> dict[int, tuple[np.ndarray, ...]]:
    flat_500 = np.full((HEIGHT, WIDTH), 500, dtype=np.uint16)
    flat_1000 = np.full((HEIGHT, WIDTH), 1000, dtype=np.uint16)
    flat_250 = np.full((HEIGHT, WIDTH), 250, dtype=np.uint16)
    neighbor_invalid = np.full((HEIGHT, WIDTH), 500, dtype=np.uint16)
    neighbor_invalid[2, 3] = 0
    flat_1001 = np.full((HEIGHT, WIDTH), 1001, dtype=np.uint16)
    flat_249 = np.full((HEIGHT, WIDTH), 249, dtype=np.uint16)
    return {
        # Every non-black v1 normal remains valid, so both LZ4 and H.264 can be reused.
        0: (flat_500, flat_1000, flat_250),
        # Exercises the five-pixel neighbor rule and both exclusive sides of the range.
        1: (neighbor_invalid, flat_1001, flat_249),
    }


def encode_v1_fixture_frames(depths: tuple[np.ndarray, ...], episode_index: int) -> np.ndarray:
    frames = np.stack(
        [
            encode_surface_normals_rgb(
                depth,
                scale_m_per_unit=0.001,
                intrinsics=INTRINSICS,
                encoding_version=LEGACY_SURFACE_NORMAL_ENCODING_VERSION,
                depth_near_m=0.25,
                depth_far_m=1.0,
            )
            for depth in depths
        ]
    )
    if episode_index == 0:
        # A model-visible valid byte sentinel proves migration preserves stored v1 bytes;
        # recalculating normals from depth would silently replace it.
        frames[0, 3, 4] = np.array([17, 34, 51], dtype=np.uint8)
    return np.ascontiguousarray(frames)


def write_lossless_h264(path: Path, frames: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with av.open(str(path), "w") as output:
        stream = output.add_stream("libx264rgb", 30, options={"g": "2", "crf": "0"})
        stream.width = WIDTH
        stream.height = HEIGHT
        stream.pix_fmt = "rgb24"
        for frame in frames:
            video_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
            for packet in stream.encode(video_frame):
                output.mux(packet)
        for packet in stream.encode():
            output.mux(packet)


def write_v1_episode(
    split_root: Path,
    lz4_relative: Path,
    backup_relative: Path,
    episode_index: int,
    depths: tuple[np.ndarray, ...],
) -> None:
    aligned = split_root / f"aligned_depths/chunk-000/episode_{episode_index:06d}"
    aligned.mkdir(parents=True)
    for frame_index, depth in enumerate(depths):
        assert cv2.imwrite(str(aligned / f"frame_{frame_index:06d}.png"), depth)

    frames = encode_v1_fixture_frames(depths, episode_index)
    episode_dir = split_root / lz4_relative / f"episode_{episode_index:06d}"
    episode_dir.mkdir(parents=True)
    chunks = []
    for chunk_index, start_frame in enumerate(range(0, len(frames), MIGRATION.CHUNK_FRAMES)):
        chunk_frames = frames[start_frame : start_frame + MIGRATION.CHUNK_FRAMES]
        raw = chunk_frames.tobytes()
        compressed = subprocess.run(
            [str(MIGRATION.LZ4_BIN), "-q", "-z", "-c"],
            input=raw,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout
        filename = f"chunk_{chunk_index:06d}.lz4"
        (episode_dir / filename).write_bytes(compressed)
        chunks.append(
            {
                "chunk_index": chunk_index,
                "start_frame": start_frame,
                "frame_count": len(chunk_frames),
                "height": HEIGHT,
                "width": WIDTH,
                "channels": 3,
                "uncompressed_bytes": len(raw),
                "filename": filename,
                "compressed_bytes": len(compressed),
                "compressed_sha256": sha256_bytes(compressed),
                "lossless_round_trip_verified": True,
            }
        )
    write_json(
        episode_dir / "index.json",
        {
            "episode_index": episode_index,
            "dtype": "uint8",
            "layout": "FHWC",
            "transform": "none",
            "compression": "LZ4 frame, CLI default fast compression",
            "chunks_are_independent": True,
            "chunk_frames": MIGRATION.CHUNK_FRAMES,
            "frame_count": len(frames),
            "decoded_reference_sha256": sha256_bytes(frames.tobytes()),
            "surface_normals_encoding_version": LEGACY_SURFACE_NORMAL_ENCODING_VERSION,
            "lossless_round_trip_verified": True,
            "chunks": chunks,
        },
    )
    write_lossless_h264(
        split_root / backup_relative / f"episode_{episode_index:06d}.mp4",
        frames,
    )


def build_v1_source(root: Path) -> None:
    depths_by_episode = fixture_depths()
    old_encoding = surface_normals_encoding_metadata(
        intrinsics=INTRINSICS,
        encoding_version=LEGACY_SURFACE_NORMAL_ENCODING_VERSION,
        depth_near_m=0.25,
        depth_far_m=1.0,
    )
    total_frames = sum(len(depths) for depths in depths_by_episode.values())
    for split in SPLITS:
        split_root = root / split
        lz4_relative = Path("videos/chunk-000") / FEATURE_KEY
        backup_relative = Path("videos/chunk-000") / f"{FEATURE_KEY}_h264_backup"
        for episode_index, depths in depths_by_episode.items():
            write_v1_episode(
                split_root,
                lz4_relative,
                backup_relative,
                episode_index,
                depths,
            )
            unchanged = split_root / f"data/chunk-000/episode_{episode_index:06d}.parquet"
            unchanged.parent.mkdir(parents=True, exist_ok=True)
            unchanged.write_bytes(f"unchanged-{split}-{episode_index}".encode())

        info = {
            "codebase_version": "v2.1",
            "robot_type": "synthetic",
            "total_episodes": len(depths_by_episode),
            "total_frames": total_frames,
            "total_tasks": 1,
            "chunks_size": 1000,
            "fps": 30,
            "splits": {split: f"0:{len(depths_by_episode)}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "features": {
                FEATURE_KEY: {
                    "dtype": "video",
                    "shape": [HEIGHT, WIDTH, 3],
                    "names": ["height", "width", "channel"],
                }
            },
            "depth_encoding": {"near_m": 0.25, "far_m": 1.0},
            "aligned_depth_encoding": {
                "source_key": "depth_0",
                "aligned_to": "color_0",
                "storage": "lossless_png",
                "path": (
                    "aligned_depths/chunk-{episode_chunk:03d}/"
                    "episode_{episode_index:06d}/frame_{frame_index:06d}.png"
                ),
                "dtype": "uint16",
                "shape": [HEIGHT, WIDTH],
                "scale_m_per_unit": 0.001,
                "invalid_value": 0,
                "total_files": total_frames,
            },
            "surface_normals_encoding": old_encoding,
            "surface_normals_lz4": {
                "feature_key": FEATURE_KEY,
                "storage": "plain_lz4_chunks",
                "root": lz4_relative.as_posix(),
                "dtype": "uint8",
                "layout": "FHWC",
                "chunk_frames": MIGRATION.CHUNK_FRAMES,
                "lossless_round_trip_verified": True,
                "h264_backup": backup_relative.as_posix(),
            },
        }
        write_json(split_root / "meta/info.json", info)
        backup_info = dict(info)
        backup_info.pop("surface_normals_lz4")
        write_json(split_root / "meta/info.json.h264_backup", backup_info)
        write_jsonl(
            split_root / "meta/episodes.jsonl",
            [
                {"episode_index": index, "length": len(depths), "tasks": ["test"]}
                for index, depths in depths_by_episode.items()
            ],
        )
        episode_stats = [
            {
                "episode_index": index,
                "stats": {FEATURE_KEY: dummy_image_stats(0.5, len(depths))},
            }
            for index, depths in depths_by_episode.items()
        ]
        write_jsonl(split_root / "meta/episodes_stats.jsonl", episode_stats)
        write_json(
            split_root / "meta/stats.json",
            {FEATURE_KEY: dummy_image_stats(0.5, total_frames)},
        )


def decode_lz4_episode(split_root: Path, episode_index: int) -> np.ndarray:
    episode = split_root / f"videos/chunk-000/{FEATURE_KEY}/episode_{episode_index:06d}"
    index = json.loads((episode / "index.json").read_text(encoding="utf-8"))
    frames = []
    for chunk in index["chunks"]:
        raw = subprocess.run(
            [str(MIGRATION.LZ4_BIN), "-q", "-d", "-c", str(episode / chunk["filename"])],
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        frames.append(
            np.frombuffer(raw, dtype=np.uint8).reshape(
                chunk["frame_count"], HEIGHT, WIDTH, 3
            )
        )
    return np.concatenate(frames)


def decoded_video_frames(path: Path) -> np.ndarray:
    with av.open(str(path)) as container:
        return np.stack(
            [
                np.ascontiguousarray(frame.to_ndarray(format="rgb24"))
                for frame in container.decode(video=0)
            ]
        )


def expected_depth_mask(depth: np.ndarray) -> np.ndarray:
    in_range = (depth >= 250) & (depth <= 1000)
    mask = np.zeros(depth.shape, dtype=np.bool_)
    mask[1:-1, 1:-1] = (
        in_range[1:-1, 1:-1]
        & in_range[1:-1, :-2]
        & in_range[1:-1, 2:]
        & in_range[:-2, 1:-1]
        & in_range[2:, 1:-1]
    )
    return mask


def masked_v1_frames(source_frames: np.ndarray, depths: tuple[np.ndarray, ...]) -> np.ndarray:
    result = source_frames.copy()
    for frame, depth in zip(result, depths, strict=True):
        frame[~expected_depth_mask(depth)] = 0
    return result


def episode_paths(root: Path, split: str, episode_index: int) -> tuple[Path, Path, dict]:
    episode_dir = root / split / f"videos/chunk-000/{FEATURE_KEY}/episode_{episode_index:06d}"
    index = json.loads((episode_dir / "index.json").read_text(encoding="utf-8"))
    chunk = episode_dir / index["chunks"][0]["filename"]
    video = (
        root
        / split
        / f"videos/chunk-000/{FEATURE_KEY}_h264_backup/episode_{episode_index:06d}.mp4"
    )
    return chunk, video, index


class SurfaceNormalsV2MigrationTest(unittest.TestCase):
    def test_migration_source_has_no_normal_encoder_dependency(self) -> None:
        tree = ast.parse(SCRIPT.read_text(encoding="utf-8"), filename=str(SCRIPT))
        forbidden = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "encode_surface_normals_rgb":
                forbidden.append((node.lineno, node.id))
            elif isinstance(node, ast.Attribute) and node.attr == "encode_surface_normals_rgb":
                forbidden.append((node.lineno, node.attr))
            elif isinstance(node, ast.ImportFrom) and any(
                alias.name == "encode_surface_normals_rgb" for alias in node.names
            ):
                forbidden.append((node.lineno, "import"))
        self.assertEqual(forbidden, [], "migration must never import or call the normal encoder")

    def test_rejects_escaped_or_symlinked_aligned_depth_paths(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source-v1"
            build_v1_source(source)
            info_path = source / "train/meta/info.json"
            info = json.loads(info_path.read_text(encoding="utf-8"))
            info["aligned_depth_encoding"]["path"] = (
                "/tmp/outside/episode_{episode_index:06d}/frame_{frame_index:06d}.png"
            )
            write_json(info_path, info)
            with self.assertRaises(MIGRATION.MigrationError):
                MIGRATION.inspect_source(source)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source-v1"
            build_v1_source(source)
            episode_depths = source / "train/aligned_depths/chunk-000/episode_000000"
            outside_depths = root / "outside-depths"
            episode_depths.rename(outside_depths)
            episode_depths.symlink_to(outside_depths, target_is_directory=True)
            with self.assertRaises(MIGRATION.MigrationError):
                MIGRATION.inspect_source(source)

    def test_atomic_publish_refuses_existing_empty_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "ready-work"
            target = root / "concurrent-target"
            source.mkdir()
            target.mkdir()
            (source / "sentinel").write_bytes(b"complete")

            with self.assertRaises(FileExistsError):
                MIGRATION.rename_directory_noreplace(source, target)

            self.assertEqual((source / "sentinel").read_bytes(), b"complete")
            self.assertTrue(target.is_dir())

    def test_corrupted_source_lz4_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source-v1"
            target = root / "sibling-v2"
            build_v1_source(source)
            chunk, _, _ = episode_paths(source, "train", 0)
            damaged = bytearray(chunk.read_bytes())
            damaged[len(damaged) // 2] ^= 0x80
            chunk.write_bytes(damaged)

            with self.assertRaisesRegex(MIGRATION.MigrationError, "LZ4 compressed hash mismatch"):
                MIGRATION.migrate(source, target, jobs=1, min_free_gib=0)
            self.assertFalse(target.exists())

    def test_resume_masks_only_reuses_unchanged_episode_and_publishes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source-v1"
            target = root / "sibling-v2"
            build_v1_source(source)
            source_fingerprint = MIGRATION.source_stat_fingerprint(source)

            encoder_guard = mock.patch.object(
                surface_normal_encoding,
                "encode_surface_normals_rgb",
                side_effect=AssertionError("migration attempted to calculate normals"),
            )
            with encoder_guard:
                published = MIGRATION.migrate(
                    source,
                    target,
                    jobs=2,
                    min_free_gib=0,
                    max_new_episodes=1,
                )
                self.assertFalse(published)
                self.assertFalse(target.exists())
                work = root / f".{target.name}.surface-normals-v2-work"
                self.assertTrue(work.is_dir())
                completed_indexes = list(
                    work.glob(
                        f"*/videos/chunk-000/.{FEATURE_KEY}.v2-building/episode_*/index.json"
                    )
                )
                self.assertEqual(len(completed_indexes), 1)

                self.assertTrue(MIGRATION.migrate(source, target, jobs=2, min_free_gib=0))
                self.assertTrue(target.is_dir())
                self.assertFalse(work.exists())
                self.assertTrue(MIGRATION.migrate(source, target, jobs=1, min_free_gib=0))

            self.assertEqual(MIGRATION.source_stat_fingerprint(source), source_fingerprint)
            depths_by_episode = fixture_depths()
            for split in SPLITS:
                for episode_index in depths_by_episode:
                    source_unchanged = (
                        source / split / f"data/chunk-000/episode_{episode_index:06d}.parquet"
                    )
                    target_unchanged = (
                        target / split / f"data/chunk-000/episode_{episode_index:06d}.parquet"
                    )
                    self.assertEqual(
                        (source_unchanged.stat().st_dev, source_unchanged.stat().st_ino),
                        (target_unchanged.stat().st_dev, target_unchanged.stat().st_ino),
                    )

                info = json.loads((target / split / "meta/info.json").read_text())
                self.assertEqual(
                    info["surface_normals_encoding"]["encoding_version"],
                    SURFACE_NORMAL_ENCODING_VERSION,
                )
                self.assertEqual(
                    info["surface_normals_encoding"]["depth_valid_range_m"],
                    {
                        "near_m": 0.25,
                        "far_m": 1.0,
                        "inclusive": True,
                        "required_samples": ["center", "left", "right", "up", "down"],
                    },
                )

                for episode_index, depths in depths_by_episode.items():
                    source_frames = decode_lz4_episode(source / split, episode_index)
                    target_frames = decode_lz4_episode(target / split, episode_index)
                    expected = masked_v1_frames(source_frames, depths)
                    np.testing.assert_array_equal(target_frames, expected)
                    for source_frame, target_frame, depth in zip(
                        source_frames, target_frames, depths, strict=True
                    ):
                        mask = expected_depth_mask(depth)
                        np.testing.assert_array_equal(target_frame[mask], source_frame[mask])
                        self.assertFalse(np.any(target_frame[~mask]))

                    source_chunk, source_video, source_index = episode_paths(
                        source, split, episode_index
                    )
                    target_chunk, target_video, target_index = episode_paths(
                        target, split, episode_index
                    )
                    self.assertEqual(target_index["migration_method"], MIGRATION.MIGRATION_METHOD)
                    self.assertFalse(target_index["normals_recomputed"])
                    self.assertEqual(
                        target_index["source_v1_semantics"],
                        "trusted_declared_dataset_contract",
                    )
                    self.assertNotIn("semantic_audit_frame_indices", target_index)
                    self.assertNotIn("semantic_audit_passed", target_index)
                    self.assertEqual(
                        target_index["source_v1_decoded_reference_sha256"],
                        source_index["decoded_reference_sha256"],
                    )
                    np.testing.assert_array_equal(decoded_video_frames(target_video), expected)

                    source_chunk_inode = (source_chunk.stat().st_dev, source_chunk.stat().st_ino)
                    target_chunk_inode = (target_chunk.stat().st_dev, target_chunk.stat().st_ino)
                    source_video_inode = (source_video.stat().st_dev, source_video.stat().st_ino)
                    target_video_inode = (target_video.stat().st_dev, target_video.stat().st_ino)
                    if episode_index == 0:
                        self.assertEqual(source_chunk_inode, target_chunk_inode)
                        self.assertEqual(source_video_inode, target_video_inode)
                        self.assertEqual(target_index["source_lz4_chunks_reused"], 1)
                        self.assertEqual(target_index["source_lz4_chunks_rewritten"], 0)
                        self.assertTrue(target_index["h264_backup_reused_from_source"])
                        self.assertEqual(
                            target_index["chunks"][0]["migration_action"],
                            MIGRATION.SOURCE_CHUNK_REUSED,
                        )
                    else:
                        self.assertNotEqual(source_chunk_inode, target_chunk_inode)
                        self.assertNotEqual(source_video_inode, target_video_inode)
                        self.assertEqual(target_index["source_lz4_chunks_reused"], 0)
                        self.assertEqual(target_index["source_lz4_chunks_rewritten"], 1)
                        self.assertFalse(target_index["h264_backup_reused_from_source"])
                        self.assertGreater(target_index["range_mask_newly_zeroed_pixels"], 0)
                        self.assertEqual(
                            target_index["chunks"][0]["migration_action"],
                            MIGRATION.SOURCE_CHUNK_MASKED,
                        )

            manifest_path = (
                target / "provenance/surface_normals_v2_migration/manifest.json"
            )
            provenance = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(provenance["operation"], MIGRATION.MIGRATION_OPERATION)
            self.assertEqual(provenance["migration_method"], MIGRATION.MIGRATION_METHOD)
            self.assertFalse(provenance["normals_recomputed"])
            self.assertFalse(provenance["training_started"])

            index_path = (
                target
                / "train"
                / f"videos/chunk-000/{FEATURE_KEY}/episode_000000/index.json"
            )
            corrupt_index = json.loads(index_path.read_text(encoding="utf-8"))
            corrupt_index["chunks"][0]["compressed_sha256"] = "0" * 64
            write_json(index_path, corrupt_index)
            with self.assertRaises(MIGRATION.MigrationError):
                MIGRATION.migrate(source, target, jobs=1, min_free_gib=0)


if __name__ == "__main__":
    unittest.main()
