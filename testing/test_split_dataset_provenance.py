import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SPLIT_SCRIPT = Path(__file__).resolve().parents[1] / "split_dataset.py"


class SplitDatasetProvenanceTest(unittest.TestCase):
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
                expected_hashes[episode.name] = hashlib.sha256(data_path.read_bytes()).hexdigest()

            subprocess.run(
                [sys.executable, str(SPLIT_SCRIPT), str(root)],
                check=True,
                capture_output=True,
                text=True,
            )

            manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["episode_count"], 10)
            for record in manifest["episodes"]:
                source_name = record["flattened_episode"]
                self.assertEqual(record["data_json_sha256"], expected_hashes[source_name])
                self.assertEqual(record["frame_count"], int(source_name[-4:]) + 1)


if __name__ == "__main__":
    unittest.main()
