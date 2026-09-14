import hashlib
import json
from collections import Counter
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SPLIT_SCRIPT = Path(__file__).resolve().parents[1] / "split_dataset.py"


class SplitDatasetProvenanceTest(unittest.TestCase):
    @staticmethod
    def make_leaf(root: Path, name: str, count: int) -> dict[str, str]:
        leaf = root / name
        leaf.mkdir(parents=True)
        hashes = {}
        for index in range(count):
            episode = leaf / f"episode_{index:04d}"
            episode.mkdir()
            data_path = episode / "data.json"
            data_path.write_text(
                json.dumps(
                    {
                        "identity": f"{name}/{index}",
                        "text": {"goal": "pick" if index % 2 else "put"},
                        "timing": {
                            "capture_start_utc": f"2026-09-15T00:00:{name}-{index:02d}Z"
                        },
                        "data": [{"idx": frame} for frame in range(index + 1)],
                    }
                ),
                encoding="utf-8",
            )
            hashes[episode.name] = hashlib.sha256(data_path.read_bytes()).hexdigest()
        return hashes

    def test_manifest_records_frame_count_and_source_hash(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
            root = Path(temporary_dir) / "dataset"
            root.mkdir()
            expected_hashes = {}
            for index in range(10):
                episode = root / f"episode_{index:04d}"
                episode.mkdir()
                data_path = episode / "data.json"
                data_path.write_text(
                    json.dumps(
                        {
                            "text": {"goal": "pick" if index % 2 else "put"},
                            "data": [{"idx": frame} for frame in range(index + 1)],
                        }
                    ),
                    encoding="utf-8",
                )
                expected_hashes[episode.name] = hashlib.sha256(
                    data_path.read_bytes()
                ).hexdigest()

            subprocess.run(
                [sys.executable, str(SPLIT_SCRIPT), str(root)],
                check=True,
                capture_output=True,
                text=True,
            )

            manifest = json.loads(
                (root / "split_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(manifest["episode_count"], 10)
            for record in manifest["episodes"]:
                source_name = record["flattened_episode"]
                self.assertEqual(
                    record["data_json_sha256"], expected_hashes[source_name]
                )
                self.assertEqual(record["frame_count"], int(source_name[-4:]) + 1)

    def test_collection_composes_fresh_provenance_and_preserves_membership(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
            temporary = Path(temporary_dir)
            source = temporary / "processed_raw" / "inspire"
            source.mkdir(parents=True)
            old_hashes = self.make_leaf(source, "old_task", 6)
            new_hashes = self.make_leaf(source, "new_task", 10)
            self.make_leaf(source, "data_colour_only_test", 3)
            curation = source / "curation_manifest_20260915.json"
            curation.write_text('{"curated": true}\n', encoding="utf-8")

            preserved_records = []
            expected_membership = {}
            for index, episode_name in enumerate(old_hashes):
                split = (
                    "train" if index < 4 else ("validation" if index == 4 else "test")
                )
                expected_membership[episode_name] = split
                split_index = (
                    sum(record["split"] == split for record in preserved_records) + 1
                )
                preserved_records.append(
                    {
                        "split": split,
                        "split_episode": f"episode_{split_index:04d}",
                        "flattened_episode": episode_name,
                        "source_episode": episode_name,
                        "goal": "stale",
                        "frame_count": 999,
                        "data_json_sha256": "0" * 64,
                    }
                )
            preserved = temporary / "authoritative_split_manifest.json"
            preserved.write_text(
                json.dumps({"version": 1, "episodes": preserved_records}),
                encoding="utf-8",
            )
            output = temporary / "combined"

            result = subprocess.run(
                [
                    sys.executable,
                    str(SPLIT_SCRIPT),
                    str(source),
                    "--collection-output",
                    str(output),
                    "--exclude-dataset",
                    "data_colour_only_test",
                    "--preserve-split",
                    f"old_task={preserved}",
                    "--seed",
                    "42",
                ],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((output / "split_manifest.json").read_text())
            self.assertEqual(manifest["strategy"], "component-preserved")
            self.assertEqual(manifest["episode_count"], 16)
            self.assertEqual(manifest["excluded_datasets"], ["data_colour_only_test"])
            counts = Counter(record["split"] for record in manifest["episodes"])
            self.assertEqual(counts, {"train": 12, "test": 2, "validation": 2})
            self.assertEqual(
                hashlib.sha256(
                    (output / "provenance" / curation.name).read_bytes()
                ).hexdigest(),
                manifest["curation_manifest"]["sha256"],
            )
            preserved_component = next(
                component
                for component in manifest["components"]
                if component["dataset"] == "old_task"
            )
            embedded_assignment = (
                output / preserved_component["assignment_manifest_copy"]
            )
            self.assertEqual(embedded_assignment.read_bytes(), preserved.read_bytes())
            self.assertEqual(
                hashlib.sha256(embedded_assignment.read_bytes()).hexdigest(),
                preserved_component["assignment_manifest_sha256"],
            )

            records_by_identity = {
                record["source_path"]: record for record in manifest["episodes"]
            }
            self.assertEqual(len(records_by_identity), 16)
            for episode_name, expected_split in expected_membership.items():
                record = records_by_identity[f"old_task/{episode_name}"]
                self.assertEqual(record["split"], expected_split)
                self.assertEqual(record["data_json_sha256"], old_hashes[episode_name])
                self.assertNotEqual(record["frame_count"], 999)
            for episode_name, expected_hash in new_hashes.items():
                self.assertEqual(
                    records_by_identity[f"new_task/{episode_name}"]["data_json_sha256"],
                    expected_hash,
                )
            self.assertEqual(len(list(source.glob("*/episode_*/data.json"))), 19)

    def test_collection_rejects_duplicate_episode_content(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
            source = Path(temporary_dir) / "collection"
            for leaf_name in ("first", "second"):
                episode = source / leaf_name / "episode_0000"
                episode.mkdir(parents=True)
                (episode / "data.json").write_text(
                    json.dumps({"text": {"goal": "pick"}, "data": [{"idx": 0}]}),
                    encoding="utf-8",
                )

            result = subprocess.run(
                [sys.executable, str(SPLIT_SCRIPT), str(source), "--collection-check"],
                check=False,
                capture_output=True,
                text=True,
            )

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Duplicate processed episode", result.stderr)


if __name__ == "__main__":
    unittest.main()
