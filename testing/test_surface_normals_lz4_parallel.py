from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import threading
import time
import types
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "convert_canonical_surface_normals_to_lz4.py"


def load_module():
    spec = importlib.util.spec_from_file_location("surface_normals_lz4_under_test", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    # These scheduler tests replace convert_episode and never decode media.
    # Stub PyAV so the scripts test suite remains runnable in its lightweight
    # environment, which intentionally does not install the video dependency.
    with mock.patch.dict(sys.modules, {"av": types.ModuleType("av")}):
        spec.loader.exec_module(module)
    return module


def make_split(root: Path, lengths: list[int], feature_key: str) -> None:
    meta = root / "meta"
    canonical = root / "videos" / "chunk-000" / feature_key
    meta.mkdir(parents=True)
    canonical.mkdir(parents=True)
    info = {
        "total_frames": sum(lengths),
        "features": {feature_key: {"shape": [4, 5, 3]}},
    }
    (meta / "info.json").write_text(json.dumps(info), encoding="utf-8")
    episodes = [{"episode_index": index, "length": length} for index, length in enumerate(lengths)]
    (meta / "episodes.jsonl").write_text(
        "".join(json.dumps(episode) + "\n" for episode in episodes),
        encoding="utf-8",
    )
    for index in range(len(lengths)):
        (canonical / f"episode_{index:06d}.mp4").touch()


class ParallelLz4EpisodeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_module()

    def test_prepare_split_bounds_parallelism_and_keeps_episode_totals(self):
        lengths = [11, 12, 13, 14, 15, 16]
        lock = threading.Lock()
        active = 0
        peak_active = 0

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "train"
            make_split(root, lengths, self.module.FEATURE_KEY)

            def convert(_source, _build, episode_index, expected_frames, _shape, _chunk):
                nonlocal active, peak_active
                with lock:
                    active += 1
                    peak_active = max(peak_active, active)
                try:
                    time.sleep(0.01 * (3 - episode_index % 3))
                    return expected_frames, expected_frames * 7, "converted"
                finally:
                    with lock:
                        active -= 1

            with mock.patch.object(self.module, "convert_episode", side_effect=convert):
                prepared = self.module.prepare_split(
                    root,
                    "train",
                    32,
                    split_root=root,
                    jobs=3,
                )

            self.assertEqual(prepared["frames"], sum(lengths))
            self.assertEqual(prepared["bytes"], sum(lengths) * 7)
            self.assertGreater(peak_active, 1)
            self.assertLessEqual(peak_active, 3)
            self.assertEqual(active, 0)

    def test_prepare_split_reports_earliest_episode_failure(self):
        lengths = [11, 12, 13]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "train"
            make_split(root, lengths, self.module.FEATURE_KEY)

            def convert(_source, _build, episode_index, expected_frames, _shape, _chunk):
                if episode_index == 0:
                    time.sleep(0.04)
                    raise RuntimeError("episode zero")
                if episode_index == 1:
                    raise RuntimeError("episode one")
                return expected_frames, 1, "converted"

            with (
                mock.patch.object(self.module, "convert_episode", side_effect=convert),
                self.assertRaisesRegex(RuntimeError, "episode zero"),
            ):
                self.module.prepare_split(
                    root,
                    "train",
                    32,
                    split_root=root,
                    jobs=3,
                )


if __name__ == "__main__":
    unittest.main()
