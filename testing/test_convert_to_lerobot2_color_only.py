from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
WRAPPER = SCRIPT_ROOT / "convert_to_lerobot2.sh"
UNITREE_REPO = Path("/home/alex/Development/unitree_lerobot")
ISAAC_REPO = Path("/home/alex/Development/Isaac-GR00T")
UNITREE_PYTHON = Path("/home/alex/miniconda3/envs/unitree_lerobot/bin/python")


def _d435i_calibration() -> dict:
    result = subprocess.run(
        [
            str(UNITREE_PYTHON),
            "-c",
            (
                "import json; "
                "from unitree_lerobot.utils.camera_calibration import "
                "NAMED_CAMERA_CALIBRATIONS; "
                "print(json.dumps(NAMED_CAMERA_CALIBRATIONS['d435i-254322071415']))"
            ),
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        cwd=UNITREE_REPO,
    )
    return json.loads(result.stdout)


def _with_updated_fingerprint(calibration: dict) -> dict:
    payload = {key: value for key, value in calibration.items() if key != "fingerprint"}
    canonical = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    calibration["fingerprint"] = f"sha256:{hashlib.sha256(canonical).hexdigest()}"
    return calibration


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


def _write_episode(
    root: Path,
    *,
    provenance: bool = True,
    color: bool = True,
    depth: bool = False,
    depth_scale=...,
    name: str = "episode_000000",
    calibration=...,
) -> Path:
    episode = root / name
    episode.mkdir(parents=True)
    colors = episode / "colors"
    colors.mkdir()
    color_path = colors / "000000_color_0.jpg"
    if color:
        color_path.write_bytes(b"preflight-only fixture")
    depths = episode / "depths"
    if depth:
        depths.mkdir()
        (depths / "000000_depth_0.png").write_bytes(b"aligned depth fixture")
        (depths / "000000_raw_depth_0.png").write_bytes(b"raw depth fixture")

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
    if depth_scale is not ...:
        info["depth"] = {"scale_m_per_unit": depth_scale}
    if calibration is not ...:
        info.setdefault("depth", {})["calibration"] = calibration
    payload = {
        "info": info,
        "data": [
            {
                "idx": 0,
                "colors": {"color_0": "colors/000000_color_0.jpg"},
                "depths": (
                    {
                        "depth_0": "depths/000000_depth_0.png",
                        "raw_depth_0": "depths/000000_raw_depth_0.png",
                    }
                    if depth
                    else {}
                ),
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


def test_preflight_accepts_canonical_embodiment_subdirectories(tmp_path: Path):
    datasets = tmp_path / "Datasets"
    for embodiment in ("dex3", "inspire"):
        source = datasets / "processed_raw" / embodiment / "example_task"
        _write_episode(source)

        result = _run(source, "--preflight-only", "--color-only")

        assert result.returncode == 0, result.stdout
        assert (source / "episode_000000" / "data.json").is_file()
        assert "no files were converted or replaced" in result.stdout


def test_default_preflight_still_requires_depth(tmp_path: Path):
    _write_episode(tmp_path)
    result = _run(tmp_path, "--preflight-only")
    assert result.returncode != 0
    assert "Missing depths.depth_0" in result.stdout


def test_rgbd_preflight_accepts_float32_realsense_scale_spelling(tmp_path: Path):
    _write_episode(
        tmp_path,
        depth=True,
        depth_scale=0.0010000000474974513,
    )

    result = _run(tmp_path, "--preflight-only")

    assert result.returncode == 0, result.stdout
    assert "Validated 1 RGB-D frames" in result.stdout


def test_rgbd_preflight_accepts_missing_legacy_scale(tmp_path: Path):
    _write_episode(tmp_path, depth=True)

    result = _run(tmp_path, "--preflight-only")

    assert result.returncode == 0, result.stdout


def test_rgbd_preflight_rejects_noncanonical_depth_scale(tmp_path: Path):
    _write_episode(tmp_path, depth=True, depth_scale=0.0011)

    result = _run(tmp_path, "--preflight-only")

    assert result.returncode != 0
    assert "not float32-equal to 0.001 m/unit" in result.stdout


def test_rgbd_preflight_rejects_noncanonical_reported_depth_scale(tmp_path: Path):
    episode = _write_episode(tmp_path, depth=True, depth_scale=0.001)
    data_path = episode / "data.json"
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    payload["info"]["depth"]["scale_reported_m_per_unit"] = 0.002
    data_path.write_text(json.dumps(payload), encoding="utf-8")

    result = _run(tmp_path, "--preflight-only")

    assert result.returncode != 0
    assert "info.depth.scale_reported_m_per_unit" in result.stdout


def test_rgbd_preflight_rejects_mixed_calibration_tagging(tmp_path: Path):
    _write_episode(
        tmp_path,
        depth=True,
        name="episode_000001",
        calibration={"fixture": "present"},
    )
    _write_episode(tmp_path, depth=True, name="episode_000002")

    result = _run(tmp_path, "--preflight-only")

    assert result.returncode != 0
    assert "mixes calibration-tagged and legacy episodes" in result.stdout


def test_rgbd_preflight_accepts_profile_backed_mixed_calibration_tagging(
    tmp_path: Path,
):
    _write_episode(
        tmp_path,
        depth=True,
        name="episode_000001",
        calibration=_d435i_calibration(),
    )
    _write_episode(tmp_path, depth=True, name="episode_000002")

    result = _run(
        tmp_path,
        "--preflight-only",
        "--camera-calibration-profile",
        "d435i-254322071415",
    )

    assert result.returncode == 0, result.stdout
    assert "Validated 2 RGB-D frames" in result.stdout


def test_rgbd_preflight_rejects_conflicting_profile_backed_mixed_calibration(
    tmp_path: Path,
):
    different = copy.deepcopy(_d435i_calibration())
    different["camera"]["serial"] = "different-camera"
    _with_updated_fingerprint(different)
    _write_episode(
        tmp_path,
        depth=True,
        name="episode_000001",
        calibration=different,
    )
    _write_episode(tmp_path, depth=True, name="episode_000002")

    result = _run(
        tmp_path,
        "--preflight-only",
        "--camera-calibration-profile",
        "d435i-254322071415",
    )

    assert result.returncode != 0
    assert "conflicts with profile 'd435i-254322071415'" in result.stdout


def test_final_metadata_validation_requires_complete_named_calibration():
    wrapper = WRAPPER.read_text(encoding="utf-8")

    assert "expected_profile_calibration = NAMED_CAMERA_CALIBRATIONS.get(" in wrapper
    assert "camera_calibration == expected_profile_calibration" in wrapper
    assert 'camera_calibration["camera"]["serial"] == "254322071415"' not in wrapper


def test_surface_normal_wrapper_exposes_and_verifies_versioned_contract():
    wrapper = WRAPPER.read_text(encoding="utf-8")

    assert "--surface-normals-encoding-version" in wrapper
    assert 'SURFACE_NORMALS_ENCODING_VERSION="${SURFACE_NORMALS_ENCODING_VERSION:-2}"' in wrapper
    assert '--surface-normals-encoding-version "$SURFACE_NORMALS_ENCODING_VERSION"' in wrapper
    assert 'surface_encoding["encoding_version"] == expected_surface_normals_encoding_version' in wrapper
    assert '"depth_valid_range_m" not in surface_encoding' in wrapper
    assert 'surface_encoding["depth_valid_range_m"] == {' in wrapper
    assert '"required_samples": ["center", "left", "right", "up", "down"]' in wrapper


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
