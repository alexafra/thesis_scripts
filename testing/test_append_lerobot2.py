import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq


SCRIPT_DIR = Path(__file__).resolve().parents[1]
APPEND_SCRIPT = SCRIPT_DIR / "append_lerobot2.py"

RAW_DEPTH_TEMPLATE = (
    "raw_depths/chunk-{episode_chunk:03d}/"
    "episode_{episode_index:06d}/frame_{frame_index:06d}.png"
)
ALIGNED_DEPTH_TEMPLATE = (
    "aligned_depths/chunk-{episode_chunk:03d}/"
    "episode_{episode_index:06d}/frame_{frame_index:06d}.png"
)


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=4) + "\n", encoding="utf-8")


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def scalar_stats(values: list[int]) -> dict[str, list[float] | list[int]]:
    count = len(values)
    mean = sum(values) / count
    variance = sum((value - mean) ** 2 for value in values) / count
    ordered = sorted(values)
    return {
        "min": [ordered[0]],
        "max": [ordered[-1]],
        "mean": [mean],
        "std": [variance**0.5],
        "count": [count],
        "q01": [ordered[0]],
        "q10": [ordered[0]],
        "q50": [ordered[count // 2]],
        "q90": [ordered[-1]],
        "q99": [ordered[-1]],
    }


def sidecar_path(root: Path, template: str, episode_index: int, frame_index: int) -> Path:
    return root / template.format(
        episode_chunk=episode_index,
        episode_index=episode_index,
        frame_index=frame_index,
    )


def camera_calibration(serial: str = "254322071415") -> dict:
    profile = {
        "width": 640,
        "height": 480,
        "fx": 600.0,
        "fy": 600.0,
        "cx": 320.0,
        "cy": 240.0,
        "distortion": "distortion.brown_conrady",
        "coeffs": [0.0] * 5,
        "format": "z16",
        "fps": 30,
    }
    calibration = {
        "schema": "realsense_rgbd_calibration.v1",
        "camera": {
            "model": "Intel RealSense D435I",
            "serial": serial,
            "product_id": "0B3A",
            "firmware": "5.15.1.55",
        },
        "color": {**profile, "format": "bgr8"},
        "depth": profile,
        "depth_to_color": {
            "rotation": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            "translation_m": [0.015, 0.0, 0.0],
        },
    }
    encoded = json.dumps(
        calibration,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    calibration["fingerprint"] = f"sha256:{hashlib.sha256(encoded).hexdigest()}"
    return calibration


def make_dataset(
    root: Path,
    marker: bytes,
    *,
    aligned_scale: float = 0.001,
    aligned_total_files: int = 2,
    end_effector: dict | None = None,
    surface_normals_encoding: dict | None = None,
    camera_calibration: dict | None = None,
    camera_calibration_source: str = "episode.info.depth.calibration",
) -> None:
    length = 2
    info = {
        "codebase_version": "v2.1",
        "robot_type": "test_robot",
        "fps": 30,
        "total_episodes": 1,
        "total_frames": length,
        "total_tasks": 1,
        "total_videos": 1,
        "total_chunks": 1,
        "chunks_size": 1,
        "splits": {"train": "0:1"},
        "data_path": (
            "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet"
        ),
        "video_path": (
            "videos/chunk-{episode_chunk:03d}/{video_key}/"
            "episode_{episode_index:06d}.mp4"
        ),
        "features": {
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
            "observation.images.ego_view": {
                "dtype": "video",
                "shape": [3, 2, 2],
            },
        },
        "depth_encoding": {
            "feature_key": "observation.images.depth_gray_view",
            "near_m": 0.25,
            "far_m": 1.0,
        },
        "raw_depth_encoding": {
            "source_key": "raw_depth_0",
            "storage": "lossless_png",
            "path": RAW_DEPTH_TEMPLATE,
            "dtype": "uint16",
            "shape": [480, 640],
            "scale_m_per_unit": 0.001,
            "invalid_value": 0,
            "total_files": length,
        },
        "aligned_depth_encoding": {
            "source_key": "depth_0",
            "aligned_to": "color_0",
            "storage": "lossless_png",
            "path": ALIGNED_DEPTH_TEMPLATE,
            "dtype": "uint16",
            "shape": [480, 640],
            "scale_m_per_unit": aligned_scale,
            "invalid_value": 0,
            "total_files": aligned_total_files,
        },
    }
    if end_effector is not None:
        info["end_effector"] = end_effector
    if surface_normals_encoding is not None:
        info["surface_normals_encoding"] = dict(surface_normals_encoding)
    if camera_calibration is not None:
        info["camera_calibration"] = camera_calibration
        info.setdefault("surface_normals_encoding", {})[
            "camera_calibration"
        ] = {
            "source": camera_calibration_source,
            "schema": camera_calibration["schema"],
            "camera": camera_calibration["camera"],
            "fingerprint": camera_calibration["fingerprint"],
        }
    write_json(root / "meta" / "info.json", info)
    write_json(root / "meta" / "modality.json", {"video": {"ego_view": {}}})
    write_jsonl(root / "meta" / "tasks.jsonl", [{"task_index": 0, "task": "test task"}])
    write_jsonl(
        root / "meta" / "episodes.jsonl",
        [{"episode_index": 0, "tasks": ["test task"], "length": length}],
    )
    write_jsonl(
        root / "meta" / "episodes_stats.jsonl",
        [
            {
                "episode_index": 0,
                "stats": {
                    "frame_index": scalar_stats([0, 1]),
                    "episode_index": scalar_stats([0, 0]),
                    "index": scalar_stats([0, 1]),
                    "task_index": scalar_stats([0, 0]),
                },
            }
        ],
    )

    parquet = root / info["data_path"].format(episode_chunk=0, episode_index=0)
    parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "frame_index": pa.array([0, 1], type=pa.int64()),
                "episode_index": pa.array([0, 0], type=pa.int64()),
                "index": pa.array([0, 1], type=pa.int64()),
                "task_index": pa.array([0, 0], type=pa.int64()),
            }
        ),
        parquet,
    )

    video = root / info["video_path"].format(
        episode_chunk=0,
        episode_index=0,
        video_key="observation.images.ego_view",
    )
    video.parent.mkdir(parents=True, exist_ok=True)
    video.write_bytes(b"video-" + marker)

    for frame_index in range(length):
        raw = sidecar_path(root, RAW_DEPTH_TEMPLATE, 0, frame_index)
        raw.parent.mkdir(parents=True, exist_ok=True)
        raw.write_bytes(b"raw-" + marker + bytes([frame_index]))
        aligned = sidecar_path(root, ALIGNED_DEPTH_TEMPLATE, 0, frame_index)
        aligned.parent.mkdir(parents=True, exist_ok=True)
        aligned.write_bytes(b"aligned-" + marker + bytes([frame_index]))


def make_split_manifest(root: Path, marker: str) -> None:
    records = []
    for split in ("train", "test", "validation"):
        records.append(
            {
                "split": split,
                "split_episode": "episode_0001",
                "flattened_episode": f"episode_{marker}_{split}",
                "source_episode": f"episode_{marker}_{split}",
                "source_session": marker,
                "goal": "test task",
                "frame_count": 2,
                "data_json_sha256": hashlib.sha256(f"{marker}:{split}".encode()).hexdigest(),
            }
        )
    write_json(
        root / "split_manifest.json",
        {
            "version": 1,
            "strategy": "goal-stratified",
            "episode_count": len(records),
            "episodes": records,
        },
    )


def add_canonical_lz4_normals(root: Path, marker: bytes) -> None:
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    feature_key = "observation.images.surface_normals_view"
    storage_relative = "videos/chunk-000/observation.images.surface_normals_view"
    backup_relative = f"{storage_relative}_h264_backup"
    info["features"][feature_key] = {
        "dtype": "video",
        "shape": [2, 2, 3],
    }
    info["total_videos"] = 2
    info["surface_normals_lz4"] = {
        "feature_key": feature_key,
        "storage": "plain_lz4_chunks",
        "root": storage_relative,
        "dtype": "uint8",
        "layout": "FHWC",
        "chunk_frames": 32,
        "lossless_round_trip_verified": True,
        "h264_backup": backup_relative,
    }
    write_json(info_path, info)

    episode_dir = root / storage_relative / "episode_000000"
    episode_dir.mkdir(parents=True)
    chunk_bytes = b"lz4-" + marker
    chunk_name = "chunk_000000.lz4"
    (episode_dir / chunk_name).write_bytes(chunk_bytes)
    write_json(
        episode_dir / "index.json",
        {
            "episode_index": 0,
            "dtype": "uint8",
            "layout": "FHWC",
            "transform": "none",
            "chunks_are_independent": True,
            "chunk_frames": 32,
            "frame_count": 2,
            "decoded_reference_sha256": "0" * 64,
            "lossless_round_trip_verified": True,
            "chunks": [
                {
                    "chunk_index": 0,
                    "start_frame": 0,
                    "frame_count": 2,
                    "height": 2,
                    "width": 2,
                    "channels": 3,
                    "uncompressed_bytes": 24,
                    "filename": chunk_name,
                    "compressed_bytes": len(chunk_bytes),
                    "compressed_sha256": hashlib.sha256(chunk_bytes).hexdigest(),
                    "lossless_round_trip_verified": True,
                }
            ],
        },
    )
    write_json(
        root / storage_relative / "manifest.json",
        {
            "feature_key": feature_key,
            "dtype": "uint8",
            "layout": "FHWC",
            "transform": "none",
            "chunk_frames": 32,
            "episode_count": 1,
            "frame_count": 2,
            "compressed_bytes": len(chunk_bytes),
            "all_lossless_round_trips_verified": True,
            "canonical_commit_pending": False,
        },
    )
    backup = root / backup_relative / "episode_000000.mp4"
    backup.parent.mkdir(parents=True)
    backup.write_bytes(b"normal-video-" + marker)
    h264_info = dict(info)
    h264_info.pop("surface_normals_lz4")
    write_json(root / "meta" / "info.json.h264_backup", h264_info)


def tree_digest(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


class AppendLerobot2AlignedDepthTest(unittest.TestCase):
    def run_append(self, *arguments: Path, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(APPEND_SCRIPT), *map(str, arguments)],
            check=check,
            capture_output=True,
            text=True,
        )

    def test_split_sources_copy_and_renumber_both_depth_sidecars(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
            temporary = Path(temporary_dir)
            first = temporary / "first"
            second = temporary / "second"
            destination = temporary / "combined"
            make_dataset(first / "train", b"first")
            make_dataset(second / "train", b"second")
            first_digest = tree_digest(first)
            second_digest = tree_digest(second)

            result = self.run_append(destination, first, second)

            self.assertIn("copied 2 episode(s) and 4 frame(s)", result.stdout)
            combined = destination / "train"
            info = json.loads((combined / "meta" / "info.json").read_text())
            self.assertEqual(info["total_episodes"], 2)
            self.assertEqual(info["total_frames"], 4)
            self.assertEqual(info["raw_depth_encoding"]["total_files"], 4)
            self.assertEqual(info["aligned_depth_encoding"]["total_files"], 4)

            for episode_index, marker in enumerate((b"first", b"second")):
                for frame_index in range(2):
                    raw = sidecar_path(
                        combined,
                        info["raw_depth_encoding"]["path"],
                        episode_index,
                        frame_index,
                    )
                    aligned = sidecar_path(
                        combined,
                        info["aligned_depth_encoding"]["path"],
                        episode_index,
                        frame_index,
                    )
                    self.assertEqual(raw.read_bytes(), b"raw-" + marker + bytes([frame_index]))
                    self.assertEqual(
                        aligned.read_bytes(),
                        b"aligned-" + marker + bytes([frame_index]),
                    )

            appended = pq.read_table(
                combined / "data" / "chunk-001" / "episode_000001.parquet"
            )
            self.assertEqual(appended["episode_index"].to_pylist(), [1, 1])
            self.assertEqual(appended["index"].to_pylist(), [2, 3])
            self.assertEqual(tree_digest(first), first_digest)
            self.assertEqual(tree_digest(second), second_digest)

    def test_one_call_merges_all_splits_and_updates_provenance(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
            temporary = Path(temporary_dir)
            first = temporary / "first"
            second = temporary / "second"
            destination = temporary / "combined"
            for root, marker in ((first, "first"), (second, "second")):
                for split in ("train", "test", "validation"):
                    make_dataset(root / split, f"{marker}-{split}".encode())
                make_split_manifest(root, marker)

            self.run_append(destination, first)
            result = self.run_append(destination, second)

            self.assertIn("copied 3 episode(s) and 6 frame(s)", result.stdout)
            for split in ("train", "test", "validation"):
                info = json.loads((destination / split / "meta" / "info.json").read_text())
                self.assertEqual(info["total_episodes"], 2)
                self.assertEqual(info["total_frames"], 4)

            combined = json.loads((destination / "split_manifest.json").read_text())
            merge = json.loads(
                (destination / "provenance" / "merge_manifest.json").read_text()
            )
            self.assertEqual(combined["strategy"], "component-append")
            self.assertEqual(combined["episode_count"], 6)
            self.assertEqual(combined["episodes"], merge["episodes"])
            self.assertEqual(len({item["data_json_sha256"] for item in merge["episodes"]}), 6)
            for split in ("train", "test", "validation"):
                records = [item for item in merge["episodes"] if item["split"] == split]
                self.assertEqual([item["final_episode_index"] for item in records], [0, 1])
                self.assertEqual(
                    [item["split_episode"] for item in records],
                    ["episode_000000", "episode_000001"],
                )
                self.assertEqual([item["source"] for item in records], ["first", "second"])

    def test_untracked_payload_collision_does_not_modify_destination(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
            temporary = Path(temporary_dir)
            destination = temporary / "destination"
            source = temporary / "source"
            make_dataset(destination, b"destination")
            make_dataset(source, b"source")
            stale = destination / "data" / "chunk-001" / "episode_000001.parquet"
            stale.parent.mkdir(parents=True)
            stale.write_bytes(b"untracked-payload")
            destination_digest = tree_digest(destination)

            result = self.run_append(destination, source, check=False)

            self.assertEqual(result.returncode, 1)
            self.assertIn("Refusing to overwrite unexpected existing payload", result.stderr)
            self.assertEqual(tree_digest(destination), destination_digest)

    def test_duplicate_source_hashes_are_rejected_transactionally(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
            temporary = Path(temporary_dir)
            source = temporary / "source"
            destination = temporary / "destination"
            for split in ("train", "test", "validation"):
                make_dataset(source / split, split.encode())
            make_split_manifest(source, "source")
            self.run_append(destination, source)
            destination_digest = tree_digest(destination)

            result = self.run_append(destination, source, check=False)

            self.assertEqual(result.returncode, 1)
            self.assertIn("Duplicate source episode hash", result.stderr)
            self.assertEqual(tree_digest(destination), destination_digest)

    def test_split_sources_copy_and_reindex_canonical_lz4_normals(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
            temporary = Path(temporary_dir)
            first = temporary / "first"
            second = temporary / "second"
            destination = temporary / "combined"
            make_dataset(first / "train", b"first")
            make_dataset(second / "train", b"second")
            add_canonical_lz4_normals(first / "train", b"first")
            add_canonical_lz4_normals(second / "train", b"second")
            first_digest = tree_digest(first)
            second_digest = tree_digest(second)

            result = self.run_append(destination, first, second)

            self.assertIn("copied 2 episode(s) and 4 frame(s)", result.stdout)
            combined = destination / "train"
            info = json.loads((combined / "meta" / "info.json").read_text())
            storage = combined / info["surface_normals_lz4"]["root"]
            appended = storage / "episode_000001"
            appended_index = json.loads((appended / "index.json").read_text())
            self.assertEqual(appended_index["episode_index"], 1)
            self.assertEqual(
                (appended / "chunk_000000.lz4").read_bytes(),
                b"lz4-second",
            )
            backup = combined / info["surface_normals_lz4"]["h264_backup"]
            self.assertEqual(
                (backup / "episode_000001.mp4").read_bytes(),
                b"normal-video-second",
            )
            manifest = json.loads((storage / "manifest.json").read_text())
            self.assertEqual(manifest["episode_count"], 2)
            self.assertEqual(manifest["frame_count"], 4)
            h264_info = json.loads(
                (combined / "meta" / "info.json.h264_backup").read_text()
            )
            self.assertEqual(h264_info["total_episodes"], 2)
            self.assertNotIn("surface_normals_lz4", h264_info)
            self.assertEqual(tree_digest(first), first_digest)
            self.assertEqual(tree_digest(second), second_digest)

    def test_aligned_depth_metadata_must_be_compatible(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
            temporary = Path(temporary_dir)
            destination = temporary / "destination"
            source = temporary / "source"
            make_dataset(destination, b"destination")
            make_dataset(source, b"source", aligned_scale=0.002)
            destination_digest = tree_digest(destination)

            result = self.run_append(destination, source, check=False)

            self.assertEqual(result.returncode, 1)
            self.assertIn("different aligned_depth_encoding", result.stderr)
            self.assertEqual(tree_digest(destination), destination_digest)

    def test_inspire_transport_metadata_must_be_compatible(self):
        ftp = {
            "type": "inspire",
            "protocol": "ftp",
            "hand_dof": 6,
            "value_semantics": "normalized_open_fraction",
        }
        dfx = {**ftp, "protocol": "dfx"}
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
            temporary = Path(temporary_dir)
            destination = temporary / "destination"
            source = temporary / "source"
            make_dataset(destination, b"destination", end_effector=ftp)
            make_dataset(source, b"source", end_effector=dfx)
            destination_digest = tree_digest(destination)

            result = self.run_append(destination, source, check=False)

            self.assertEqual(result.returncode, 1)
            self.assertIn("different end_effector", result.stderr)
            self.assertEqual(tree_digest(destination), destination_digest)

    def test_matching_inspire_ftp_metadata_can_be_appended(self):
        ftp = {
            "type": "inspire",
            "protocol": "ftp",
            "hand_dof": 6,
            "value_semantics": "normalized_open_fraction",
        }
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
            temporary = Path(temporary_dir)
            destination = temporary / "destination"
            source = temporary / "source"
            make_dataset(destination, b"destination", end_effector=ftp)
            make_dataset(source, b"source", end_effector=ftp)

            result = self.run_append(destination, source)

            self.assertIn("copied 1 episode(s) and 2 frame(s)", result.stdout)
            info = json.loads((destination / "meta" / "info.json").read_text())
            self.assertEqual(info["total_episodes"], 2)
            self.assertEqual(info["end_effector"], ftp)

    def test_surface_normals_encoding_must_be_compatible(self):
        first_encoding = {
            "method": "camera_space_cross_product",
            "depth_near_m": 0.25,
            "depth_far_m": 1.0,
        }
        second_encoding = {**first_encoding, "depth_far_m": 1.2}
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
            temporary = Path(temporary_dir)
            destination = temporary / "destination"
            source = temporary / "source"
            make_dataset(
                destination,
                b"destination",
                surface_normals_encoding=first_encoding,
            )
            make_dataset(
                source,
                b"source",
                surface_normals_encoding=second_encoding,
            )
            destination_digest = tree_digest(destination)

            result = self.run_append(destination, source, check=False)

            self.assertEqual(result.returncode, 1)
            self.assertIn("different surface_normals_encoding", result.stderr)
            self.assertEqual(tree_digest(destination), destination_digest)

    def test_matching_camera_calibration_is_preserved_and_added_to_provenance(self):
        calibration = camera_calibration()
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
            temporary = Path(temporary_dir)
            first = temporary / "first"
            second = temporary / "second"
            destination = temporary / "combined"
            for root, marker in ((first, "first"), (second, "second")):
                for split in ("train", "test", "validation"):
                    make_dataset(
                        root / split,
                        f"{marker}-{split}".encode(),
                        camera_calibration=calibration,
                    )
                make_split_manifest(root, marker)

            self.run_append(destination, first, second)

            for split in ("train", "test", "validation"):
                info = json.loads((destination / split / "meta" / "info.json").read_text())
                self.assertEqual(
                    info["camera_calibration"],
                    calibration,
                )
            provenance = json.loads(
                (destination / "provenance" / "merge_manifest.json").read_text()
            )
            self.assertTrue(
                all(
                    record["camera_calibration_fingerprint"]
                    == calibration["fingerprint"]
                    for record in provenance["episodes"]
                )
            )
            self.assertEqual(
                provenance["camera_calibration_fingerprints"],
                [calibration["fingerprint"]],
            )

    def test_known_and_unknown_camera_calibration_are_incompatible(self):
        calibration = camera_calibration()
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
            temporary = Path(temporary_dir)
            destination = temporary / "destination"
            source = temporary / "source"
            make_dataset(destination, b"legacy", surface_normals_encoding={})
            make_dataset(source, b"known", camera_calibration=calibration)
            destination_digest = tree_digest(destination)

            result = self.run_append(destination, source, check=False)

            self.assertEqual(result.returncode, 1)
            self.assertIn("different camera_calibration", result.stderr)
            self.assertEqual(tree_digest(destination), destination_digest)

    def test_same_camera_fingerprint_with_different_provenance_source_is_compatible(self):
        calibration = camera_calibration()
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
            temporary = Path(temporary_dir)
            destination = temporary / "destination"
            source = temporary / "source"
            make_dataset(destination, b"recorded", camera_calibration=calibration)
            make_dataset(
                source,
                b"profile",
                camera_calibration=calibration,
                camera_calibration_source="converter.profile.d435i-254322071415",
            )

            result = self.run_append(destination, source)

            self.assertIn("copied 1 episode(s) and 2 frame(s)", result.stdout)
            info = json.loads((destination / "meta" / "info.json").read_text())
            self.assertEqual(
                info["camera_calibration"],
                calibration,
            )

    def test_different_known_camera_calibrations_are_incompatible(self):
        first_calibration = camera_calibration()
        second_calibration = camera_calibration("242322076480")
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
            temporary = Path(temporary_dir)
            destination = temporary / "destination"
            source = temporary / "source"
            make_dataset(
                destination,
                b"first",
                camera_calibration=first_calibration,
            )
            make_dataset(source, b"second", camera_calibration=second_calibration)
            destination_digest = tree_digest(destination)

            result = self.run_append(destination, source, check=False)

            self.assertEqual(result.returncode, 1)
            self.assertIn("different camera_calibration", result.stderr)
            self.assertEqual(tree_digest(destination), destination_digest)

    def test_aligned_depth_count_and_missing_files_are_validated(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
            temporary = Path(temporary_dir)
            bad_count = temporary / "bad_count"
            make_dataset(bad_count, b"count", aligned_total_files=1)

            result = self.run_append(temporary / "count_destination", bad_count, check=False)

            self.assertEqual(result.returncode, 1)
            self.assertIn("Aligned-depth file count", result.stderr)
            self.assertFalse((temporary / "count_destination").exists())

            missing_file = temporary / "missing_file"
            make_dataset(missing_file, b"missing")
            sidecar_path(missing_file, ALIGNED_DEPTH_TEMPLATE, 0, 1).unlink()

            result = self.run_append(
                temporary / "missing_destination",
                missing_file,
                check=False,
            )

            self.assertEqual(result.returncode, 1)
            self.assertIn("Missing aligned-depth frame", result.stderr)
            self.assertFalse((temporary / "missing_destination").exists())


if __name__ == "__main__":
    unittest.main()
