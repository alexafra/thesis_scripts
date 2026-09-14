import json
import os
from pathlib import Path
import stat
import subprocess
import sys


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
PIPELINE = SCRIPTS_DIR / "prepare_inspire_lerobot2.sh"
SPLITTER = SCRIPTS_DIR / "split_dataset.py"


def _write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def _make_raw(root: Path, episode_count: int) -> None:
    root.mkdir(parents=True)
    for episode_index in range(1, episode_count + 1):
        episode = root / f"episode_{episode_index:04d}"
        episode.mkdir()
        (episode / "data.json").write_text(
            json.dumps(
                {
                    "text": {
                        "goal": "pick up cup" if episode_index % 2 else "put down cup"
                    },
                    "data": [{"frame_index": 0}],
                }
            ),
            encoding="utf-8",
        )


def _fake_environment(tmp_path: Path) -> dict[str, str]:
    converter = tmp_path / "fake_convert.sh"
    trainer = tmp_path / "fake_train.sh"
    convert_log = tmp_path / "convert.log"
    train_log = tmp_path / "train.log"

    _write_executable(
        converter,
        """#!/usr/bin/env bash
set -euo pipefail
printf '%s | near=%s far=%s profile=%s\n' \
    "$*" "$DEPTH_NEAR_M" "$DEPTH_FAR_M" "$CAMERA_CALIBRATION_PROFILE" >> "$CONVERT_LOG"
if [[ " $* " == *" --preflight-only "* ]]; then
    exit 0
fi
root="${@: -3:1}"
"$GROOT_PYTHON" - "$root" <<'PY'
import json
from pathlib import Path
import sys

root = Path(sys.argv[1])
layout = {
    "left_arm": {"start": 0, "end": 7},
    "right_arm": {"start": 7, "end": 14},
    "left_hand": {"start": 14, "end": 20},
    "right_hand": {"start": 20, "end": 26},
}
video = {
    "ego_view": {"original_key": "observation.images.ego_view"},
    "depth_gray_view": {"original_key": "observation.images.depth_gray_view"},
    "surface_normals_view": {
        "original_key": "observation.images.surface_normals_view"
    },
}
for split in ("train", "validation", "test"):
    split_root = root / split
    raw_episodes = sorted(split_root.glob("episode_*"))
    episode_rows = []
    total_frames = 0
    for episode_index, episode in enumerate(raw_episodes):
        payload = json.loads((episode / "data.json").read_text(encoding="utf-8"))
        length = len(payload["data"])
        goal = payload["text"]["goal"]
        episode_rows.append(
            {"episode_index": episode_index, "length": length, "tasks": [goal]}
        )
        total_frames += length
    metadata = split_root / "meta"
    metadata.mkdir(parents=True)
    features = {
        "observation.state": {"shape": [26], "dtype": "float32"},
        "action": {"shape": [26], "dtype": "float32"},
        "observation.images.ego_view": {"dtype": "video"},
        "observation.images.depth_gray_view": {"dtype": "video"},
        "observation.images.surface_normals_view": {"dtype": "video"},
    }
    info = {
        "codebase_version": "v2.1",
        "robot_type": "Unitree_G1_Inspire_HeadOnly",
        "fps": 30,
        "total_episodes": len(raw_episodes),
        "total_frames": total_frames,
        "features": features,
        "end_effector": {"type": "inspire", "protocol": "ftp", "hand_dof": 6},
        "depth_encoding": {},
        "raw_depth_encoding": {},
        "aligned_depth_encoding": {},
        "surface_normals_encoding": {},
        "surface_normals_lz4": {},
    }
    (metadata / "info.json").write_text(json.dumps(info), encoding="utf-8")
    (metadata / "modality.json").write_text(
        json.dumps({"state": layout, "action": layout, "video": video}),
        encoding="utf-8",
    )
    (metadata / "episodes.jsonl").write_text(
        "".join(json.dumps(row) + "\\n" for row in episode_rows),
        encoding="utf-8",
    )
PY
""",
    )
    _write_executable(
        trainer,
        """#!/usr/bin/env bash
set -euo pipefail
printf 'precheck=%s dataset=%s suffix=%s\n' \
    "${PRECHECK_ONLY:-0}" "$DATASET_ROOT" "$RUN_SUFFIX" >> "$TRAIN_LOG"
""",
    )

    return {
        **os.environ,
        "GROOT_PYTHON": sys.executable,
        "CONVERT_SCRIPT": str(converter),
        "SPLIT_SCRIPT": str(SPLITTER),
        "TRAIN_SCRIPT": str(trainer),
        "CONVERT_LOG": str(convert_log),
        "TRAIN_LOG": str(train_log),
    }


def _run(
    mode: str,
    source: Path,
    output: Path,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(PIPELINE),
            mode,
            "--source",
            str(source),
            "--output",
            str(output),
            "--repo-id",
            "test_inspire",
        ],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def _run_dataset_name(
    mode: str,
    dataset_name: str,
    environment: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PIPELINE), mode, "--dataset-name", dataset_name],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_no_mode_does_not_run_any_stage(tmp_path: Path) -> None:
    environment = _fake_environment(tmp_path)

    result = subprocess.run(
        [str(PIPELINE)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "MODE is required" in result.stderr
    assert not Path(environment["CONVERT_LOG"]).exists()
    assert not Path(environment["TRAIN_LOG"]).exists()


def test_convert_includes_all_137_episodes_and_builds_three_splits(tmp_path: Path) -> None:
    environment = _fake_environment(tmp_path)
    source = tmp_path / "processed_raw"
    output = tmp_path / "lerobot2"
    _make_raw(source, 137)

    result = _run("convert", source, output, environment)

    assert "Raw population: 137 episodes; all will be included." in result.stdout
    assert "Split population: train=109, validation=14, test=14; total=137" in result.stdout
    assert len(list(source.glob("episode_*"))) == 137
    assert not Path(f"{output}.building").exists()
    manifest = json.loads((output / "split_manifest.json").read_text(encoding="utf-8"))
    assert manifest["episode_count"] == 137
    counts = {
        split: sum(record["split"] == split for record in manifest["episodes"])
        for split in ("train", "validation", "test")
    }
    assert counts == {"train": 109, "validation": 14, "test": 14}

    calls = Path(environment["CONVERT_LOG"]).read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert "--preflight-only" in calls[0]
    assert all("--include-surface-normals" in call for call in calls)
    assert all("--end-effector inspire-ftp" in call for call in calls)
    assert all("--camera-calibration-profile d435i-254322071415" in call for call in calls)
    assert all("near=0.25 far=1.0 profile=d435i-254322071415" in call for call in calls)
    assert not Path(environment["TRAIN_LOG"]).exists()


def test_convert_accepts_canonical_inspire_stage_hierarchy(tmp_path: Path) -> None:
    environment = _fake_environment(tmp_path)
    datasets = tmp_path / "Datasets"
    task_name = "pick_place_red_cup_08_13"
    raw = datasets / "raw" / "inspire" / task_name
    source = datasets / "processed_raw" / "inspire" / task_name
    output = datasets / "lerobot2" / "inspire" / task_name

    raw.mkdir(parents=True)
    raw_sentinel = raw / "collection_is_not_pipeline_input.txt"
    raw_sentinel.write_text("raw remains untouched", encoding="utf-8")
    _make_raw(source, 10)

    result = _run("convert", source, output, environment)

    assert f"Published validated dataset: {output}" in result.stdout
    assert output.is_dir()
    assert (output / "train" / "meta" / "info.json").is_file()
    assert (output / "validation" / "meta" / "info.json").is_file()
    assert (output / "test" / "meta" / "info.json").is_file()
    assert len(list(source.glob("episode_*"))) == 10
    assert raw_sentinel.read_text(encoding="utf-8") == "raw remains untouched"
    assert not Path(environment["TRAIN_LOG"]).exists()


def test_dataset_name_derives_canonical_inspire_source_and_output(
    tmp_path: Path,
) -> None:
    environment = _fake_environment(tmp_path)
    datasets = tmp_path / "Datasets"
    environment["DATASETS_ROOT"] = str(datasets)
    task_name = "pick_place_red_cup_08_13"
    source = datasets / "processed_raw" / "inspire" / task_name
    output = datasets / "lerobot2" / "inspire" / task_name
    _make_raw(source, 10)

    result = _run_dataset_name("convert", task_name, environment)

    assert f"Published validated dataset: {output}" in result.stdout
    assert output.is_dir()
    assert len(list(source.glob("episode_*"))) == 10
    assert not Path(environment["TRAIN_LOG"]).exists()


def test_dataset_name_environment_derives_paths(tmp_path: Path) -> None:
    environment = _fake_environment(tmp_path)
    datasets = tmp_path / "Datasets"
    task_name = "pick_place_red_cup_08_13"
    source = datasets / "processed_raw" / "inspire" / task_name
    _make_raw(source, 3)
    environment.update(
        {
            "DATASETS_ROOT": str(datasets),
            "DATASET_NAME": task_name,
        }
    )

    result = subprocess.run(
        [str(PIPELINE), "check"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "Raw population: 3 episodes; all will be included." in result.stdout


def test_explicit_paths_override_dataset_name_derived_paths(tmp_path: Path) -> None:
    environment = _fake_environment(tmp_path)
    environment["DATASETS_ROOT"] = str(tmp_path / "unused_Datasets")
    source = tmp_path / "explicit_source"
    output = tmp_path / "explicit_output"
    _make_raw(source, 10)

    result = subprocess.run(
        [
            str(PIPELINE),
            "convert",
            "--dataset-name",
            "derived_name_must_not_win",
            "--source",
            str(source),
            "--output",
            str(output),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert f"Published validated dataset: {output}" in result.stdout
    assert not (tmp_path / "unused_Datasets" / "lerobot2").exists()


def test_dataset_name_must_be_one_safe_path_component(tmp_path: Path) -> None:
    for index, unsafe_name in enumerate(("../escape", "nested/task", ".", "..")):
        case_root = tmp_path / f"case_{index}"
        case_root.mkdir()
        environment = _fake_environment(case_root)
        environment["DATASETS_ROOT"] = str(tmp_path / "Datasets")
        result = subprocess.run(
            [str(PIPELINE), "check", "--dataset-name", unsafe_name],
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )

        assert result.returncode != 0
        assert "dataset-name" in result.stderr.lower()
        assert not Path(environment["CONVERT_LOG"]).exists()
        assert not Path(environment["TRAIN_LOG"]).exists()


def test_check_rejects_embodiment_container_instead_of_combining_tasks(
    tmp_path: Path,
) -> None:
    environment = _fake_environment(tmp_path)
    inspire_root = tmp_path / "Datasets" / "processed_raw" / "inspire"
    inspire_root.mkdir(parents=True)
    _make_raw(inspire_root / "first_task", 3)
    _make_raw(inspire_root / "second_task", 3)

    result = subprocess.run(
        [str(PIPELINE), "check", "--source", str(inspire_root)],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert f"No direct episode_* directories found in {inspire_root}" in result.stderr
    assert not Path(environment["CONVERT_LOG"]).exists()
    assert not Path(environment["TRAIN_LOG"]).exists()


def test_depth_range_and_calibration_profile_overrides_reach_every_conversion_call(
    tmp_path: Path,
) -> None:
    environment = _fake_environment(tmp_path)
    environment.update(
        {
            "DEPTH_NEAR_M": "0.3",
            "DEPTH_FAR_M": "3.0",
            "CAMERA_CALIBRATION_PROFILE": "legacy-untagged",
        }
    )
    source = tmp_path / "processed_raw"
    output = tmp_path / "lerobot2"
    _make_raw(source, 10)

    _run("convert", source, output, environment)

    calls = Path(environment["CONVERT_LOG"]).read_text(encoding="utf-8").splitlines()
    assert len(calls) == 2
    assert all("--camera-calibration-profile legacy-untagged" in call for call in calls)
    assert all("near=0.3 far=3.0 profile=legacy-untagged" in call for call in calls)


def test_training_check_train_and_all_are_distinct_explicit_modes(tmp_path: Path) -> None:
    environment = _fake_environment(tmp_path)
    source = tmp_path / "processed_raw"
    output = tmp_path / "lerobot2"
    _make_raw(source, 10)
    _run("convert", source, output, environment)

    _run("training-check", source, output, environment)
    _run("train", source, output, environment)
    training_calls = Path(environment["TRAIN_LOG"]).read_text(encoding="utf-8").splitlines()
    assert training_calls[0].startswith("precheck=1 ")
    assert training_calls[1].startswith("precheck=0 ")

    all_source = tmp_path / "all_processed_raw"
    all_output = tmp_path / "all_lerobot2"
    _make_raw(all_source, 3)
    _run("all", all_source, all_output, environment)
    training_calls = Path(environment["TRAIN_LOG"]).read_text(encoding="utf-8").splitlines()
    assert len(training_calls) == 3
    assert training_calls[-1].startswith("precheck=0 ")
    assert all_output.is_dir()


def test_training_check_validates_the_test_split_contract(tmp_path: Path) -> None:
    environment = _fake_environment(tmp_path)
    source = tmp_path / "processed_raw"
    output = tmp_path / "lerobot2"
    _make_raw(source, 10)
    _run("convert", source, output, environment)
    test_info_path = output / "test" / "meta" / "info.json"
    test_info = json.loads(test_info_path.read_text(encoding="utf-8"))
    test_info["features"]["action"]["shape"] = [28]
    test_info_path.write_text(json.dumps(test_info), encoding="utf-8")

    result = subprocess.run(
        [
            str(PIPELINE),
            "training-check",
            "--source",
            str(source),
            "--output",
            str(output),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "test action is not 26D" in result.stderr
    assert not Path(environment["TRAIN_LOG"]).exists()


def test_training_check_rejects_changed_raw_provenance(tmp_path: Path) -> None:
    environment = _fake_environment(tmp_path)
    source = tmp_path / "processed_raw"
    output = tmp_path / "lerobot2"
    _make_raw(source, 10)
    _run("convert", source, output, environment)
    changed = source / "episode_0001" / "data.json"
    changed.write_text(changed.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    result = subprocess.run(
        [
            str(PIPELINE),
            "training-check",
            "--source",
            str(source),
            "--output",
            str(output),
        ],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "does not exactly match processed_raw" in result.stderr
    assert "episode_0001" in result.stderr
    assert not Path(environment["TRAIN_LOG"]).exists()
