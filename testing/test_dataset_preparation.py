import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


SCRIPT_DIR = Path(__file__).resolve().parent
FLATTEN_SCRIPT = SCRIPT_DIR / "flatten_dataset_sessions.py"
SPLIT_SCRIPT = SCRIPT_DIR / "split_dataset.py"
APPEND_SCRIPT = SCRIPT_DIR / "append_session.py"


class DatasetPreparationTest(unittest.TestCase):
    def make_sessions(self, root: Path, session_count=10):
        for session_index in range(session_count):
            dataset_dir = root / f"session_{session_index:02d}"
            for episode_index, goal in zip(
                (1, 4), ("pick up cup", "open drawer"), strict=True
            ):
                episode_dir = dataset_dir / f"episode_{episode_index:04d}"
                episode_dir.mkdir(parents=True)
                with (episode_dir / "data.json").open("w", encoding="utf-8") as file:
                    json.dump(
                        {
                            "text": {"goal": goal},
                            "data": [{"idx": 0, "timestamp_s": 0.0}],
                        },
                        file,
                    )

    def run_script(self, *arguments):
        subprocess.run(
            [sys.executable, *map(str, arguments)],
            check=True,
            capture_output=True,
            text=True,
        )

    def test_session_grouped_split(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
            root = Path(temporary_dir) / "dataset_copy"
            root.mkdir()
            self.make_sessions(root)

            self.run_script(FLATTEN_SCRIPT, root)
            flat_episodes = sorted(root.glob("episode_*"))
            self.assertEqual(len(flat_episodes), 20)
            self.assertEqual(flat_episodes[0].name, "episode_0001")
            self.assertEqual(flat_episodes[-1].name, "episode_0020")

            self.run_script(SPLIT_SCRIPT, root, "--strategy", "session-grouped")
            self.assertEqual(len(list((root / "train").glob("episode_*"))), 16)
            self.assertEqual(len(list((root / "test").glob("episode_*"))), 2)
            self.assertEqual(len(list((root / "validation").glob("episode_*"))), 2)

            with (root / "split_manifest.json").open(encoding="utf-8") as file:
                manifest = json.load(file)

            sessions_by_split = {
                split: {
                    record["source_session"]
                    for record in manifest["episodes"]
                    if record["split"] == split
                }
                for split in ("train", "test", "validation")
            }
            self.assertTrue(sessions_by_split["train"].isdisjoint(sessions_by_split["test"]))
            self.assertTrue(
                sessions_by_split["train"].isdisjoint(sessions_by_split["validation"])
            )
            self.assertTrue(
                sessions_by_split["test"].isdisjoint(sessions_by_split["validation"])
            )

    def test_goal_stratified_split(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
            root = Path(temporary_dir) / "dataset_copy"
            root.mkdir()
            self.make_sessions(root)

            self.run_script(FLATTEN_SCRIPT, root)
            self.run_script(SPLIT_SCRIPT, root, "--strategy", "goal-stratified")

            with (root / "split_manifest.json").open(encoding="utf-8") as file:
                records = json.load(file)["episodes"]

            goals_by_split = {
                split: [record["goal"] for record in records if record["split"] == split]
                for split in ("train", "test", "validation")
            }
            self.assertEqual(len(goals_by_split["train"]), 16)
            self.assertEqual(sorted(goals_by_split["test"]), ["open drawer", "pick up cup"])
            self.assertEqual(
                sorted(goals_by_split["validation"]),
                ["open drawer", "pick up cup"],
            )

    def test_append_session_preserves_existing_assignments(self):
        with tempfile.TemporaryDirectory(dir="/tmp") as temporary_dir:
            temporary_path = Path(temporary_dir)
            root = temporary_path / "dataset_copy"
            root.mkdir()
            self.make_sessions(root)

            self.run_script(FLATTEN_SCRIPT, root)
            self.run_script(SPLIT_SCRIPT, root)

            existing_episodes = {
                path.relative_to(root): (path / "data.json").read_bytes()
                for split in ("train", "test", "validation")
                for path in (root / split).glob("episode_*")
            }

            new_session = temporary_path / "session_new"
            for index in range(1, 21):
                episode_dir = new_session / f"episode_{index * 2:04d}"
                episode_dir.mkdir(parents=True)
                goal = "pick up cup" if index % 2 else "open drawer"
                with (episode_dir / "data.json").open("w", encoding="utf-8") as file:
                    json.dump({"text": {"goal": goal}, "data": []}, file)

            self.run_script(APPEND_SCRIPT, new_session, root)

            for relative_path, original_json in existing_episodes.items():
                self.assertTrue((root / relative_path).is_dir())
                self.assertEqual(
                    (root / relative_path / "data.json").read_bytes(),
                    original_json,
                )

            with (root / "split_manifest.json").open(encoding="utf-8") as file:
                manifest = json.load(file)
            self.assertEqual(
                manifest["append_history"][-1]["counts"],
                {"train": 16, "test": 2, "validation": 2},
            )


if __name__ == "__main__":
    unittest.main()
