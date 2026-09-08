from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import shutil
import sys

import pytest


SCRIPT_PATH = Path(__file__).parents[1] / "three_cups_merge_support.py"
SPEC = spec_from_file_location("three_cups_merge_support", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
support = module_from_spec(SPEC)
sys.modules[SPEC.name] = support
SPEC.loader.exec_module(support)


def _write_episode(root: Path, episode_id: int, scale: float) -> None:
    episode = root / f"episode_{episode_id:04d}"
    episode.mkdir(parents=True)
    payload = {
        "info": {"depth": {"scale_m_per_unit": scale}},
        "text": {"goal": support.TASK},
        "data": [],
    }
    (episode / "data.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_depth_scale_canonicalization_changes_only_staged_copy_and_records_hashes(
    tmp_path: Path,
):
    source = tmp_path / "source"
    _write_episode(source, 1, support.STACK_RECORDED_DEPTH_SCALE)
    _write_episode(source, 2, support.STACK_RECORDED_DEPTH_SCALE)
    stage = tmp_path / "stage"
    shutil.copytree(source, stage)
    provenance_path = tmp_path / "provenance" / "depth_scale.json"

    provenance = support.canonicalize_stack_depth_scale(
        stage,
        provenance_path,
        source_root=source,
        expected_episode_count=2,
    )

    assert provenance["changed_episode_count"] == 2
    assert provenance["original_value"] == support.STACK_RECORDED_DEPTH_SCALE
    assert provenance["canonical_value"] == support.CANONICAL_DEPTH_SCALE
    assert len(provenance["episodes"]) == 2
    assert json.loads(provenance_path.read_text()) == provenance
    for episode_id in (1, 2):
        source_payload = support.read_json(source / f"episode_{episode_id:04d}" / "data.json")
        staged_payload = support.read_json(stage / f"episode_{episode_id:04d}" / "data.json")
        assert (
            source_payload["info"]["depth"]["scale_m_per_unit"]
            == support.STACK_RECORDED_DEPTH_SCALE
        )
        assert staged_payload["info"]["depth"]["scale_m_per_unit"] == support.CANONICAL_DEPTH_SCALE


def test_depth_scale_canonicalization_rejects_unexpected_value_before_any_write(
    tmp_path: Path,
):
    source = tmp_path / "source"
    _write_episode(source, 1, support.STACK_RECORDED_DEPTH_SCALE)
    _write_episode(source, 2, 0.002)
    stage = tmp_path / "stage"
    shutil.copytree(source, stage)
    before = {path: path.read_bytes() for path in sorted(stage.glob("episode_*/data.json"))}
    provenance_path = tmp_path / "depth_scale.json"

    with pytest.raises(ValueError, match="unexpected depth scale"):
        support.canonicalize_stack_depth_scale(
            stage,
            provenance_path,
            source_root=source,
            expected_episode_count=2,
        )

    assert not provenance_path.exists()
    assert {path: path.read_bytes() for path in before} == before
