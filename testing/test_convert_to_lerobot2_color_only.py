from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = SCRIPT_ROOT / "convert_to_lerobot2.sh"
UNITREE_REPO = Path("/home/alex/Development/unitree_lerobot")
ISAAC_REPO = Path("/home/alex/Development/Isaac-GR00T")
UNITREE_PYTHON = Path("/home/alex/miniconda3/envs/unitree_lerobot/bin/python")


def _provenance(protocol: str = "dfx") -> dict:
    return {
        "schema_version": 1,
        "type": "inspire",
        "protocol": protocol,
        "hand_dof": 6,
        "value_unit": "normalized_open_fraction",
        "value_range": [0.0, 1.0],
        "zero_semantics": "fully_closed",
        "one_semantics": "fully_open",
        "left_joint_names": [
            "kLeftHandPinky",
            "kLeftHandRing",
            "kLeftHandMiddle",
            "kLeftHandIndex",
            "kLeftHandThumbBend",
            "kLeftHandThumbRotation",
        ],
        "right_joint_names": [
            "kRightHandPinky",
            "kRightHandRing",
            "kRightHandMiddle",
            "kRightHandIndex",
            "kRightHandThumbBend",
            "kRightHandThumbRotation",
        ],
        "canonical_order": "left_then_right",
    }


def _write_episode(root: Path, *, provenance: bool = True, color: bool = True) -> Path:
    episode = root / "episode_000000"
    episode.mkdir(parents=True)
    colors = episode / "colors"
    colors.mkdir()
    color_path = colors / "000000_color_0.jpg"
    if color:
        color_path.write_bytes(b"preflight-only fixture")

    def components() -> dict:
        return {
            "left_arm": {"qpos": [0.0] * 7},
            "right_arm": {"qpos": [0.0] * 7},
            "left_ee": {"qpos": [0.5] * 6},
            "right_ee": {"qpos": [0.5] * 6},
        }

    info = {}
    if provenance:
        info["end_effector"] = _provenance()
    payload = {
        "info": info,
        "data": [
            {
                "idx": 0,
                "colors": {"color_0": "colors/000000_color_0.jpg"},
                "depths": {},
                "states": components(),
                "actions": components(),
            }
        ],
    }
    (episode / "data.json").write_text(json.dumps(payload), encoding="utf-8")
    return episode


def _run(root: Path, *options: str) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "UNITREE_LEROBOT_REPO": str(UNITREE_REPO),
            "ISAAC_GROOT_REPO": str(ISAAC_REPO),
            "UNITREE_PYTHON": str(UNITREE_PYTHON),
            "INCLUDE_SURFACE_NORMALS": "0",
        }
    )
    return subprocess.run(
        [
            "bash",
            str(WRAPPER),
            *options,
            "--end-effector",
            "inspire-dfx",
            str(root),
            "color_only_test",
            "1",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        env=environment,
        check=False,
    )


def test_color_only_preflight_accepts_episode_without_depth(tmp_path: Path):
    _write_episode(tmp_path)
    result = _run(tmp_path, "--preflight-only", "--color-only")
    assert result.returncode == 0, result.stdout
    assert "Validated 1 color-only frames: color_0=1." in result.stdout
    assert "no files were converted or replaced" in result.stdout


def test_default_preflight_still_requires_depth(tmp_path: Path):
    _write_episode(tmp_path)
    result = _run(tmp_path, "--preflight-only")
    assert result.returncode != 0
    assert "Missing depths.depth_0" in result.stdout


def test_color_only_rejects_surface_normals(tmp_path: Path):
    _write_episode(tmp_path)
    result = _run(
        tmp_path,
        "--preflight-only",
        "--color-only",
        "--include-surface-normals",
    )
    assert result.returncode != 0
    assert "--color-only cannot be combined" in result.stdout


def test_color_only_does_not_weaken_inspire_provenance(tmp_path: Path):
    _write_episode(tmp_path, provenance=False)
    result = _run(tmp_path, "--preflight-only", "--color-only")
    assert result.returncode != 0
    assert "info.end_effector is required" in result.stdout


def test_color_only_requires_referenced_rgb_file(tmp_path: Path):
    _write_episode(tmp_path, color=False)
    result = _run(tmp_path, "--preflight-only", "--color-only")
    assert result.returncode != 0
    assert "Missing color_0 media file" in result.stdout


def test_color_only_is_forwarded_through_split_preflight(tmp_path: Path):
    _write_episode(tmp_path / "train")
    result = _run(tmp_path, "--preflight-only", "--color-only")
    assert result.returncode == 0, result.stdout
    assert "All available splits passed preflight" in result.stdout


def test_nested_episodes_fail_instead_of_being_silently_omitted(tmp_path: Path):
    _write_episode(tmp_path)
    _write_episode(tmp_path / "pick_success")
    result = _run(tmp_path, "--preflight-only", "--color-only")
    assert result.returncode != 0
    assert "Nested episode folders are not consumed automatically" in result.stdout
