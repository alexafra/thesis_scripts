import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SCRIPTS_DIR = Path(__file__).resolve().parents[1]
SUPPORT = SCRIPTS_DIR / "inspire_incremental_append_support.py"
PIPELINE = SCRIPTS_DIR / "append_inspire_stack_0915.sh"
V2_BASE_NAME = "all_tasks_452eps_20260915_normals_range_mask_v2"


def run_support(*arguments: object, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SUPPORT), *map(str, arguments)],
        check=check,
        capture_output=True,
        text=True,
    )


def write_executable(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def schema_fingerprint(name: str, dtype: str, shape: list[int]) -> str:
    payload = {"feature": name, "dtype": dtype, "shape": shape}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def relative_fingerprint(name: str) -> str:
    payload = {
        "embodiment_tag": "new_embodiment",
        "action_key": name,
        "action_delta_indices": list(range(32)),
        "state_delta_indices": [0],
        "rep": "RELATIVE",
        "type": "NON_EEF",
        "format": "DEFAULT",
        "state_key": None,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def stat_entry(shape: tuple[int, ...]) -> dict[str, object]:
    def filled(value: float) -> object:
        import numpy as np

        return np.full(shape, value, dtype=float).tolist()

    return {
        "mean": filled(0.0),
        "std": filled(0.25),
        "min": filled(-1.0),
        "max": filled(1.0),
        "q01": filled(-0.5),
        "q99": filled(0.5),
    }


def fake_pipeline_environment(
    tmp_path: Path,
    *,
    fail_split: bool = False,
    race_target: bool = False,
    fail_detach_once: bool = False,
    fail_append_once: bool = False,
    fail_build_snapshot_once: bool = False,
) -> tuple[dict[str, str], dict[str, Path]]:
    dataset_parent = tmp_path / "lerobot2" / "inspire"
    base = dataset_parent / "base_452"
    source = tmp_path / "processed_raw" / "stack_red_cups_09_15"
    words_source = tmp_path / "processed_raw" / "woorden_block_09_15"
    target = dataset_parent / "target_655"
    (base / "train" / "data").mkdir(parents=True)
    (base / "train" / "data" / "base.bin").write_bytes(b"base-bytes")
    (base / "split_manifest.json").write_text(
        '{"version": 1, "episodes": []}\n', encoding="utf-8"
    )
    source.mkdir(parents=True)
    (source / "raw.bin").write_bytes(b"source-bytes")
    (source / "split_manifest.json").write_text(
        '{"version": 1, "episodes": []}\n', encoding="utf-8"
    )
    words_source.mkdir(parents=True)
    (words_source / "raw.bin").write_bytes(b"words-source-bytes")
    (words_source / "split_manifest.json").write_text(
        '{"version": 1, "episodes": []}\n', encoding="utf-8"
    )

    fake_log = tmp_path / "fake-commands.log"
    support = tmp_path / "support.py"
    write_executable(
        support,
        """#!/usr/bin/env python3
import json
import os
from pathlib import Path
import shutil
import sys

args = sys.argv[1:]
command = args[0]
with open(os.environ["FAKE_LOG"], "a", encoding="utf-8") as handle:
    handle.write("support " + " ".join(args) + "\\n")

def value(flag):
    return Path(args[args.index(flag) + 1])

if command == "snapshot-tree":
    root = value("--root")
    fail_marker = Path(os.environ["FAIL_BUILD_SNAPSHOT_MARKER"])
    if (
        os.environ.get("FAIL_BUILD_SNAPSHOT_ONCE") == "1"
        and root.name == "final_build"
        and not fail_marker.exists()
    ):
        fail_marker.write_text("failed once\\n", encoding="utf-8")
        raise SystemExit(73)
    output = value("--output")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text('{"tree_sha256": "fake"}\\n', encoding="utf-8")
elif command == "write-component-checkpoint":
    root = value("--root")
    (root / "component_checkpoint.json").write_text(
        '{"version": 1, "phase": "component_ready"}\\n', encoding="utf-8"
    )
elif command == "checkpoint-phase":
    state = json.loads((value("--root") / "component_checkpoint.json").read_text())
    print(state["phase"])
elif command == "normalize-checkpoint":
    root = value("--root")
    state = json.loads((root / "component_checkpoint.json").read_text())
    if state["phase"] == "final_build_ready":
        (
            root
            / "final_build"
            / "provenance"
            / "incremental_append"
            / "detach_report.json"
        ).unlink(missing_ok=True)
elif command == "validate-component-checkpoint":
    root = value("--root")
    if (root / "corrupt.marker").exists() or not (root / "stack_red_cups_09_15").is_dir():
        raise SystemExit(31)
elif command == "write-build-checkpoint":
    root = value("--root")
    (root / "component_checkpoint.json").write_text(
        '{"version": 1, "phase": "final_build_ready"}\\n', encoding="utf-8"
    )
elif command == "validate-build-checkpoint":
    root = value("--root")
    if (root / "corrupt.marker").exists() or not (root / "final_build").is_dir():
        raise SystemExit(32)
elif command == "retire-component-after-build":
    root = value("--root")
    if not (root / "final_build").is_dir():
        raise SystemExit(33)
    shutil.rmtree(root / "stack_red_cups_09_15", ignore_errors=True)
    marker = root / "provenance" / "component_retirement.json"
    marker.write_text('{"retired": true}\\n', encoding="utf-8")
elif command == "validate-retired-build-checkpoint":
    root = value("--root")
    if (root / "stack_red_cups_09_15").exists() or not (root / "final_build").is_dir():
        raise SystemExit(34)
elif command == "record-provenance":
    output = value("--output") / "provenance" / "incremental_append"
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(value("--base") / "split_manifest.json", output / "base_split_manifest.json")
    shutil.copy2(value("--component") / "split_manifest.json", output / "component_split_manifest.json")
    shutil.copy2(value("--base-snapshot"), output / "base_tree_snapshot.json")
    (output / "append_manifest.json").write_text("{}\\n", encoding="utf-8")
elif command == "detach-and-verify":
    assert not Path(os.environ["EXPECTED_COMPONENT"]).exists(), "component was not retired before detach"
    fail_marker = Path(os.environ["FAIL_MARKER"])
    if os.environ.get("FAIL_DETACH_ONCE") == "1" and not fail_marker.exists():
        fail_marker.write_text("failed once\\n", encoding="utf-8")
        raise SystemExit(71)
    base = value("--base")
    output = value("--output")
    for source in sorted(path for path in base.rglob("*") if path.is_file()):
        destination = output / source.relative_to(base)
        if not destination.is_file():
            continue
        left = source.stat()
        right = destination.stat()
        if (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino):
            temporary = destination.with_name(destination.name + ".detached")
            shutil.copy2(source, temporary)
            temporary.replace(destination)
    report = value("--report")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text('{"remaining_shared_inodes": 0}\\n', encoding="utf-8")
elif command == "validate-final" and "--require-independent" in args and os.environ.get("RACE_TARGET") == "1":
    target = Path(os.environ["TARGET_DATASET"])
    target.mkdir()
    (target / "racer.txt").write_text("must survive\\n", encoding="utf-8")
elif command == "publish-no-replace":
    build = value("--build")
    target = value("--target")
    if target.exists() or target.is_symlink():
        raise SystemExit(17)
    build.rename(target)
""",
    )
    splitter = tmp_path / "split.py"
    if fail_split:
        write_executable(splitter, "#!/usr/bin/env python3\nraise SystemExit(42)\n")
    else:
        write_executable(
            splitter,
            "#!/usr/bin/env python3\nfrom pathlib import Path\nimport shutil,sys\n"
            "source=Path(sys.argv[1]); output=Path(sys.argv[sys.argv.index('--collection-output')+1])\n"
            "shutil.copytree(source,output)\n"
            "(output/'split_manifest.json').write_text('{\"version\":1,\"components\":[],\"episodes\":[]}\\n',encoding='utf-8')\n",
        )
    converter = tmp_path / "convert.sh"
    write_executable(
        converter,
        "#!/usr/bin/env bash\nprintf '%s\\n' \"convert $*\" >> \"$FAKE_LOG\"\n",
    )
    appender = tmp_path / "append.py"
    write_executable(
        appender,
        "#!/usr/bin/env python3\nfrom pathlib import Path\nimport os,sys\n"
        "with open(os.environ['FAKE_LOG'],'a',encoding='utf-8') as h: h.write('append ' + ' '.join(sys.argv[1:]) + '\\n')\n"
        "marker=Path(os.environ['FAIL_APPEND_MARKER'])\n"
        "if os.environ.get('FAIL_APPEND_ONCE') == '1' and not marker.exists(): marker.write_text('failed once\\n',encoding='utf-8'); raise SystemExit(72)\n"
        "args=[value for value in sys.argv[1:] if value != '--link-media']\n"
        "(Path(args[0]) / 'appended.marker').write_text('203 combined\\n', encoding='utf-8')\n",
    )
    quiet_pgrep = tmp_path / "quiet-pgrep.sh"
    write_executable(quiet_pgrep, "#!/usr/bin/env bash\nexit 1\n")
    fake_uv = tmp_path / "uv"
    write_executable(
        fake_uv,
        "#!/usr/bin/env bash\nprintf '%s\\n' \"uv $*\" >> \"$FAKE_LOG\"\n",
    )
    modality_config = tmp_path / "g1_inspire_headonly_config.py"
    modality_config.write_text("# test config\n", encoding="utf-8")
    log_root = tmp_path / "logs"
    run_id = "test"
    work = dataset_parent / f".{target.name}.append-work-{run_id}"
    build = dataset_parent / f".{target.name}.build-{run_id}"
    checkpoint = dataset_parent / f".{target.name}.resume-checkpoint"
    fail_marker = tmp_path / "detach-failed-once.marker"
    fail_append_marker = tmp_path / "append-failed-once.marker"
    fail_build_snapshot_marker = tmp_path / "build-snapshot-failed-once.marker"
    environment = {
        **os.environ,
        "DATASET_PARENT": str(dataset_parent),
        "BASE_DATASET": str(base),
        "STACK_SOURCE": str(source),
        "WORDS_SOURCE": str(words_source),
        "TARGET_DATASET": str(target),
        "RUN_ID": run_id,
        "LOG_ROOT": str(log_root),
        "LOCK_FILE": str(log_root / "pipeline.lock"),
        "PIPELINE_LOG": str(log_root / "pipeline.log"),
        "GROOT_PYTHON": sys.executable,
        "UNITREE_PYTHON": sys.executable,
        "CONVERT_SCRIPT": str(converter),
        "SPLIT_SCRIPT": str(splitter),
        "APPEND_SCRIPT": str(appender),
        "SUPPORT_SCRIPT": str(support),
        "PROCESS_CHECKER": str(quiet_pgrep),
        "UV": str(fake_uv),
        "MODALITY_CONFIG": str(modality_config),
        "MIN_FREE_GIB": "1",
        "FAKE_LOG": str(fake_log),
        "EXPECTED_COMPONENT": str(checkpoint / "stack_red_cups_09_15"),
        "RACE_TARGET": "1" if race_target else "0",
        "FAIL_DETACH_ONCE": "1" if fail_detach_once else "0",
        "FAIL_MARKER": str(fail_marker),
        "FAIL_APPEND_ONCE": "1" if fail_append_once else "0",
        "FAIL_APPEND_MARKER": str(fail_append_marker),
        "FAIL_BUILD_SNAPSHOT_ONCE": "1" if fail_build_snapshot_once else "0",
        "FAIL_BUILD_SNAPSHOT_MARKER": str(fail_build_snapshot_marker),
    }
    return environment, {
        "base": base,
        "source": source,
        "words_source": words_source,
        "target": target,
        "work": work,
        "build": build,
        "checkpoint": checkpoint,
        "log": fake_log,
    }


def test_hidden_hardlinks_are_detached_and_relative_stats_may_be_regenerated(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    payload = base / "train" / "data" / "chunk-000" / "episode_000000.parquet"
    payload.parent.mkdir(parents=True)
    payload.write_bytes(b"base-payload")
    metadata = base / "train" / "meta" / "info.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"total_episodes": 1}\n', encoding="utf-8")
    relative_stats = base / "train" / "meta" / "relative_stats.json"
    relative_stats.write_text('{"derived": true}\n', encoding="utf-8")
    normal_manifest = (
        base
        / "test"
        / "videos"
        / "chunk-000"
        / "observation.images.surface_normals_view"
        / "manifest.json"
    )
    normal_manifest.parent.mkdir(parents=True)
    normal_manifest.write_text(
        '{"episode_count": 44, "frame_count": 10854}\n', encoding="utf-8"
    )
    provenance = base / "split_manifest.json"
    provenance.write_text('{"episodes": []}\n', encoding="utf-8")

    snapshot = tmp_path / "base-snapshot.json"
    run_support("snapshot-tree", "--root", base, "--output", snapshot)

    output = tmp_path / "hidden-build"
    shutil.copytree(base, output, copy_function=os.link)
    replacement = output / "train" / "meta" / ".info.json.new"
    replacement.write_text('{"total_episodes": 2}\n', encoding="utf-8")
    replacement.replace(output / "train" / "meta" / "info.json")
    (output / "train" / "meta" / "relative_stats.json").unlink()
    output_manifest = output / normal_manifest.relative_to(base)
    replacement_manifest = output_manifest.with_name(".manifest.json.new")
    replacement_manifest.write_text(
        '{"episode_count": 59, "frame_count": 23168}\n', encoding="utf-8"
    )
    replacement_manifest.replace(output_manifest)

    report = output / "detach-report.json"
    run_support(
        "detach-and-verify",
        "--base",
        base,
        "--output",
        output,
        "--snapshot",
        snapshot,
        "--report",
        report,
    )

    assert payload.read_bytes() == (output / payload.relative_to(base)).read_bytes()
    assert payload.stat().st_ino != (output / payload.relative_to(base)).stat().st_ino
    assert metadata.read_text(encoding="utf-8") == '{"total_episodes": 1}\n'
    assert (output / metadata.relative_to(base)).read_text(encoding="utf-8") == (
        '{"total_episodes": 2}\n'
    )
    assert not (output / relative_stats.relative_to(base)).exists()
    assert normal_manifest.read_text(encoding="utf-8") == (
        '{"episode_count": 44, "frame_count": 10854}\n'
    )
    assert output_manifest.read_text(encoding="utf-8") == (
        '{"episode_count": 59, "frame_count": 23168}\n'
    )
    assert normal_manifest.stat().st_ino != output_manifest.stat().st_ino
    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["remaining_shared_inodes"] == 0
    assert result["detached_file_count"] == 2
    run_support("verify-tree-snapshot", "--root", base, "--snapshot", snapshot)


def test_tree_snapshot_detects_a_base_mutation(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    payload = base / "payload.bin"
    payload.write_bytes(b"before")
    snapshot = tmp_path / "snapshot.json"
    run_support("snapshot-tree", "--root", base, "--output", snapshot)
    payload.write_bytes(b"after!")

    result = run_support(
        "verify-tree-snapshot",
        "--root",
        base,
        "--snapshot",
        snapshot,
        check=False,
    )

    assert result.returncode != 0
    assert "Tree snapshot mismatch" in result.stderr


def test_atomic_publish_never_replaces_an_existing_target(tmp_path: Path) -> None:
    build = tmp_path / "build"
    target = tmp_path / "target"
    build.mkdir()
    target.mkdir()
    (build / "candidate.txt").write_text("candidate\n", encoding="utf-8")
    (target / "owner.txt").write_text("existing\n", encoding="utf-8")

    refused = run_support(
        "publish-no-replace",
        "--build",
        build,
        "--target",
        target,
        check=False,
    )

    assert refused.returncode != 0
    assert (build / "candidate.txt").read_text(encoding="utf-8") == "candidate\n"
    assert (target / "owner.txt").read_text(encoding="utf-8") == "existing\n"

    fresh_target = tmp_path / "fresh-target"
    run_support(
        "publish-no-replace",
        "--build",
        build,
        "--target",
        fresh_target,
    )
    assert not build.exists()
    assert (fresh_target / "candidate.txt").read_text(encoding="utf-8") == "candidate\n"


def test_final_build_checkpoint_validation_retains_component(tmp_path: Path) -> None:
    base = tmp_path / "base"
    source = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    component = checkpoint / "stack_red_cups_09_15"
    final_build = checkpoint / "final_build"
    provenance = checkpoint / "provenance"
    for root, payload in (
        (base, b"base"),
        (source, b"source"),
        (component, b"component"),
    ):
        root.mkdir(parents=True)
        (root / "payload.bin").write_bytes(payload)
    provenance.mkdir()
    run_support("snapshot-tree", "--root", base, "--output", provenance / "base_tree_snapshot.json")
    run_support(
        "snapshot-tree",
        "--root",
        source,
        "--output",
        provenance / "source_tree_snapshot.json",
    )
    run_support(
        "snapshot-tree",
        "--root",
        component,
        "--output",
        provenance / "component_tree_snapshot.json",
    )
    run_support(
        "write-component-checkpoint",
        "--root",
        checkpoint,
        "--base",
        base,
        "--source",
        source,
    )
    final_build.mkdir()
    (final_build / "payload.bin").write_bytes(b"build")
    run_support(
        "snapshot-tree",
        "--root",
        final_build,
        "--output",
        provenance / "build_tree_snapshot.json",
    )
    run_support(
        "validate-orphan-final-build-checkpoint",
        "--root",
        checkpoint,
        "--base",
        base,
        "--source",
        source,
    )
    run_support(
        "write-build-checkpoint",
        "--root",
        checkpoint,
        "--base",
        base,
        "--source",
        source,
    )

    run_support(
        "validate-build-checkpoint",
        "--root",
        checkpoint,
        "--base",
        base,
        "--source",
        source,
    )

    assert (component / "payload.bin").read_bytes() == b"component"
    assert (final_build / "payload.bin").read_bytes() == b"build"


def test_component_is_retired_only_after_sealed_build_and_is_resumable(
    tmp_path: Path,
) -> None:
    base = tmp_path / "base"
    source = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    component = checkpoint / "stack_red_cups_09_15"
    final_build = checkpoint / "final_build"
    provenance = checkpoint / "provenance"
    for root, payload in ((base, b"base"), (source, b"source"), (component, b"shared")):
        root.mkdir(parents=True)
        (root / "payload.bin").write_bytes(payload)
    provenance.mkdir()
    run_support("snapshot-tree", "--root", base, "--output", provenance / "base_tree_snapshot.json")
    run_support("snapshot-tree", "--root", source, "--output", provenance / "source_tree_snapshot.json")
    run_support("snapshot-tree", "--root", component, "--output", provenance / "component_tree_snapshot.json")
    run_support(
        "write-component-checkpoint",
        "--root",
        checkpoint,
        "--base",
        base,
        "--source",
        source,
    )
    final_build.mkdir()
    os.link(component / "payload.bin", final_build / "payload.bin")
    run_support("snapshot-tree", "--root", final_build, "--output", provenance / "build_tree_snapshot.json")
    run_support(
        "write-build-checkpoint",
        "--root",
        checkpoint,
        "--base",
        base,
        "--source",
        source,
    )

    run_support(
        "retire-component-after-build",
        "--root",
        checkpoint,
        "--base",
        base,
        "--source",
        source,
    )
    run_support(
        "retire-component-after-build",
        "--root",
        checkpoint,
        "--base",
        base,
        "--source",
        source,
    )

    assert not component.exists()
    assert (final_build / "payload.bin").read_bytes() == b"shared"
    marker = json.loads((provenance / "component_retirement.json").read_text())
    assert marker["reason"] == "media hardlinks adopted by sealed final build"


def test_collection_order_rejects_interleaving(tmp_path: Path) -> None:
    root = tmp_path / "component"
    root.mkdir()
    components = [
        {"dataset": "stack_red_cups_09_15"},
        {"dataset": "woorden_block_09_15"},
    ]
    records = []
    for split in ("train", "validation", "test"):
        records.extend(
            [
                {
                    "split": split,
                    "split_episode": "episode_0001",
                    "source_dataset": "stack_red_cups_09_15",
                },
                {
                    "split": split,
                    "split_episode": "episode_0002",
                    "source_dataset": "woorden_block_09_15",
                },
            ]
        )
    write_json(root / "split_manifest.json", {"components": components, "episodes": records})
    command = (
        "validate-collection-order",
        "--root",
        root,
        "--component",
        "stack_red_cups_09_15",
        "--component",
        "woorden_block_09_15",
    )
    run_support(*command)

    records[0]["source_dataset"] = "woorden_block_09_15"
    write_json(root / "split_manifest.json", {"components": components, "episodes": records})
    rejected = run_support(*command, check=False)
    assert rejected.returncode != 0
    assert "does not preserve component order" in rejected.stderr


def test_exact_train_statistics_validation_is_26d_finite_and_train_only(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset"
    features = {
        "observation.state": {"dtype": "float32", "shape": [26]},
        "action": {"dtype": "float32", "shape": [26]},
        "timestamp": {"dtype": "float32", "shape": [1]},
        "episode_index": {"dtype": "int64", "shape": [1]},
    }
    write_json(
        dataset / "train" / "meta" / "info.json",
        {"total_frames": 40, "features": features},
    )
    (dataset / "train" / "meta" / "episodes.jsonl").write_text(
        json.dumps({"episode_index": 0, "length": 40}) + "\n",
        encoding="utf-8",
    )
    stats = {
        key: stat_entry(tuple(feature["shape"]))
        for key, feature in features.items()
        if "float" in feature["dtype"]
    }
    stats["__fingerprints__"] = {
        key: schema_fingerprint(key, feature["dtype"], feature["shape"])
        for key, feature in features.items()
        if "float" in feature["dtype"]
    }
    write_json(dataset / "train" / "meta" / "stats.json", stats)
    relative = {
        "left_arm": stat_entry((32, 7)),
        "right_arm": stat_entry((32, 7)),
        "__fingerprints__": {
            "left_arm": relative_fingerprint("left_arm"),
            "right_arm": relative_fingerprint("right_arm"),
        },
    }
    write_json(dataset / "train" / "meta" / "relative_stats.json", relative)
    nontrain_hashes = {}
    for split in ("validation", "test"):
        path = dataset / split / "meta" / "stats.json"
        write_json(path, {"split": split})
        nontrain_hashes[split] = hashlib.sha256(path.read_bytes()).hexdigest()
    write_json(
        dataset
        / "provenance"
        / "incremental_append"
        / "append_manifest.json",
        {"nontrain_stats_sha256": nontrain_hashes},
    )
    report = (
        dataset
        / "provenance"
        / "incremental_append"
        / "stats_finalization_report.json"
    )

    run_support(
        "validate-training-stats",
        "--dataset-root",
        dataset,
        "--expected-frames",
        40,
        "--report",
        report,
    )

    result = json.loads(report.read_text(encoding="utf-8"))
    assert result["scope"] == "train-only"
    assert result["dataset_statistics_sample_count"] == 40
    assert result["relative_trajectory_count_per_arm"] == 9
    assert result["state_action_dimension"] == 26

    report_bytes = report.read_bytes()
    report_mtime_ns = report.stat().st_mtime_ns
    run_support(
        "validate-training-stats",
        "--dataset-root",
        dataset,
        "--expected-frames",
        40,
        "--report",
        report,
        "--read-only",
    )
    assert report.read_bytes() == report_bytes
    assert report.stat().st_mtime_ns == report_mtime_ns

    stats["action"]["q01"][0] = 2.0
    write_json(dataset / "train" / "meta" / "stats.json", stats)
    invalid = run_support(
        "validate-training-stats",
        "--dataset-root",
        dataset,
        "--expected-frames",
        40,
        "--report",
        report,
        check=False,
    )
    assert invalid.returncode != 0
    assert "min <= q01 <= q99 <= max" in invalid.stderr


def test_provenance_rejects_cross_split_duplicates_and_component_reordering() -> None:
    sys.path.insert(0, str(SCRIPTS_DIR))
    import inspire_incremental_append_support as support

    def source_record(split: str, number: int, label: str, frames: int) -> dict[str, object]:
        return {
            "split": split,
            "split_episode": f"episode_{number:04d}",
            "data_json_sha256": hashlib.sha256(label.encode()).hexdigest(),
            "frame_count": frames,
        }

    base = [
        source_record("train", 1, "base-train", 3),
        source_record("validation", 1, "base-validation", 4),
        source_record("test", 1, "base-test", 5),
    ]
    component = [
        source_record("train", 1, "component-train-1", 6),
        source_record("train", 2, "component-train-2", 7),
        source_record("validation", 1, "component-validation", 8),
        source_record("test", 1, "component-test", 9),
    ]
    merged = []
    output_frames = {}
    for split in ("train", "validation", "test"):
        ordered = [
            *[record for record in base if record["split"] == split],
            *[record for record in component if record["split"] == split],
        ]
        index_start = 0
        for episode_index, record in enumerate(ordered):
            frame_count = int(record["frame_count"])
            merged.append(
                {
                    **record,
                    "split_episode": f"episode_{episode_index:06d}",
                    "final_episode_index": episode_index,
                    "final_index_start": index_start,
                    "final_index_end_exclusive": index_start + frame_count,
                }
            )
            index_start += frame_count
        output_frames[split] = index_start

    support._validate_provenance_merge(base, component, merged, output_frames)

    reordered = [dict(record) for record in merged]
    train_records = sorted(
        (record for record in reordered if record["split"] == "train"),
        key=lambda record: int(record["final_episode_index"]),
    )
    train_records[1]["data_json_sha256"], train_records[2]["data_json_sha256"] = (
        train_records[2]["data_json_sha256"],
        train_records[1]["data_json_sha256"],
    )
    with pytest.raises(support.AppendSafetyError, match="changes data_json_sha256"):
        support._validate_provenance_merge(base, component, reordered, output_frames)

    duplicate_component = [dict(record) for record in component]
    duplicate_component[-1]["data_json_sha256"] = base[0]["data_json_sha256"]
    with pytest.raises(support.AppendSafetyError, match="repeats source hash"):
        support._validate_provenance_merge(
            base,
            duplicate_component,
            merged,
            output_frames,
        )


def test_dataset_wrapper_has_no_training_mode() -> None:
    no_mode = subprocess.run(
        ["bash", str(PIPELINE)],
        check=False,
        capture_output=True,
        text=True,
    )
    train_mode = subprocess.run(
        ["bash", str(PIPELINE), "train"],
        check=False,
        capture_output=True,
        text=True,
    )
    script = PIPELINE.read_text(encoding="utf-8")

    assert no_mode.returncode == 2
    assert train_mode.returncode != 0
    assert "Unknown mode: train" in train_mode.stderr
    assert "EXPERIMENTS=" not in script
    assert "multi_finetune_evaluation.sh" not in script


def test_check_defaults_to_migrated_v2_base_and_pins_converter_v2(
    tmp_path: Path,
) -> None:
    environment, paths = fake_pipeline_environment(tmp_path)
    default_base = paths["base"].parent / V2_BASE_NAME
    paths["base"].rename(default_base)
    environment.pop("BASE_DATASET")

    result = subprocess.run(
        ["bash", str(PIPELINE), "check"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    calls = paths["log"].read_text(encoding="utf-8").splitlines()
    validate_base_call = next(line for line in calls if line.startswith("support validate-base"))
    assert f"--root {default_base}" in validate_base_call
    convert_calls = [line for line in calls if line.startswith("convert ")]
    assert convert_calls == [
        "convert --preflight-only --include-surface-normals "
        "--surface-normals-encoding-version 2 --end-effector inspire-ftp "
        "--camera-calibration-profile d435i-254322071415 "
        f"{paths['source']} stack_red_cups_09_15 6",
        "convert --preflight-only --include-surface-normals "
        "--surface-normals-encoding-version 2 --end-effector inspire-ftp "
        "--camera-calibration-profile d435i-254322071415 "
        f"{paths['words_source']} woorden_block_09_15 6",
    ]


def test_build_failure_retains_hidden_work_and_preserves_inputs(
    tmp_path: Path,
) -> None:
    environment, paths = fake_pipeline_environment(tmp_path, fail_split=True)
    base_before = (paths["base"] / "train" / "data" / "base.bin").read_bytes()
    source_before = (paths["source"] / "raw.bin").read_bytes()

    result = subprocess.run(
        ["bash", str(PIPELINE), "build"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 42
    assert paths["work"].is_dir()
    assert (
        paths["work"]
        / "provenance"
        / "source_bundle"
        / "stack_red_cups_09_15"
        / "raw.bin"
    ).read_bytes() == source_before
    assert not paths["build"].exists()
    assert not paths["target"].exists()
    assert (paths["base"] / "train" / "data" / "base.bin").read_bytes() == base_before
    assert (paths["source"] / "raw.bin").read_bytes() == source_before


def test_build_retires_component_after_seal_then_cleans_after_publish(
    tmp_path: Path,
) -> None:
    environment, paths = fake_pipeline_environment(tmp_path)
    base_payload = paths["base"] / "train" / "data" / "base.bin"

    result = subprocess.run(
        ["bash", str(PIPELINE), "build"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert paths["target"].is_dir()
    assert not paths["work"].exists()
    assert not paths["build"].exists()
    assert not paths["checkpoint"].exists()
    target_payload = paths["target"] / base_payload.relative_to(paths["base"])
    assert target_payload.read_bytes() == base_payload.read_bytes()
    assert target_payload.stat().st_ino != base_payload.stat().st_ino
    assert (paths["target"] / "appended.marker").read_text(encoding="utf-8") == "203 combined\n"
    assert (paths["target"] / "provenance" / "incremental_append" / "detach_report.json").is_file()

    calls = paths["log"].read_text(encoding="utf-8").splitlines()
    detach_index = next(index for index, line in enumerate(calls) if "support detach-and-verify" in line)
    final_validate_indices = [
        index for index, line in enumerate(calls) if "support validate-final" in line
    ]
    fresh_base_verify = next(
        index
        for index, line in enumerate(calls)
        if index > detach_index and "support verify-tree-snapshot" in line and str(paths["base"]) in line
    )
    assert final_validate_indices[0] < detach_index < fresh_base_verify < final_validate_indices[-1]
    assert "--require-independent" in calls[final_validate_indices[-1]]
    retire_index = next(
        index
        for index, line in enumerate(calls)
        if "support retire-component-after-build" in line
    )
    assert retire_index < detach_index
    assert "--episodes 161 21 21" in calls[final_validate_indices[-1]]
    assert "--frames 110957 13296 13712" in calls[final_validate_indices[-1]]
    assert "Final population: train=525/196453, validation=65/22719, test=65/24566" in result.stdout
    combined_calls = "\n".join(calls)
    convert_calls = [line for line in calls if line.startswith("convert ")]
    assert len(convert_calls) == 3
    assert all("--surface-normals-encoding-version 2" in line for line in convert_calls)
    assert all("--preflight-only" in line for line in convert_calls[:2])
    assert "--preflight-only" not in convert_calls[2]
    append_call = next(line for line in calls if line.startswith("append "))
    assert "--link-media" in append_call
    assert "support normalize-checkpoint" not in combined_calls
    assert "launch_finetune" not in combined_calls
    assert "multi_finetune_evaluation" not in combined_calls
    stats_call = next(line for line in calls if line.startswith("uv "))
    assert f"--dataset-path {paths['build']}/train" in stats_call
    assert f"{paths['build']}/validation" not in stats_call
    assert f"{paths['build']}/test" not in stats_call


def test_detach_failure_retains_sealed_final_build_after_component_retirement(
    tmp_path: Path,
) -> None:
    environment, paths = fake_pipeline_environment(tmp_path, fail_detach_once=True)

    result = subprocess.run(
        ["bash", str(PIPELINE), "build"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 71
    assert not paths["target"].exists()
    assert not (paths["checkpoint"] / "stack_red_cups_09_15").exists()
    assert (paths["checkpoint"] / "final_build").is_dir()
    assert (paths["checkpoint"] / "final_build" / "appended.marker").is_file()

    resumed = subprocess.run(
        ["bash", str(PIPELINE), "build"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert resumed.returncode == 0, resumed.stderr
    assert paths["target"].is_dir()


def test_crash_after_final_build_move_rolls_forward_without_deleting_payload(
    tmp_path: Path,
) -> None:
    environment, paths = fake_pipeline_environment(
        tmp_path,
        fail_build_snapshot_once=True,
    )

    interrupted = subprocess.run(
        ["bash", str(PIPELINE), "build"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert interrupted.returncode == 73
    assert not paths["target"].exists()
    checkpoint = paths["checkpoint"]
    state = json.loads((checkpoint / "component_checkpoint.json").read_text())
    assert state["phase"] == "component_ready"
    assert (checkpoint / "stack_red_cups_09_15").is_dir()
    assert (checkpoint / "final_build" / "appended.marker").is_file()

    resumed = subprocess.run(
        ["bash", str(PIPELINE), "build"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert resumed.returncode == 0, resumed.stderr
    assert paths["target"].is_dir()
    assert (paths["target"] / "appended.marker").read_text(encoding="utf-8") == (
        "203 combined\n"
    )
    calls = paths["log"].read_text(encoding="utf-8")
    assert "support validate-orphan-final-build-checkpoint" in calls
    assert "[resume] Adopted the fully revalidated orphan final build." in resumed.stdout


def test_append_failure_retains_hidden_build_and_component(tmp_path: Path) -> None:
    environment, paths = fake_pipeline_environment(tmp_path, fail_append_once=True)

    result = subprocess.run(
        ["bash", str(PIPELINE), "build"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 72
    assert not paths["target"].exists()
    assert paths["build"].is_dir()
    assert (paths["build"] / "train" / "data" / "base.bin").is_file()
    assert (paths["checkpoint"] / "stack_red_cups_09_15").is_dir()


def test_startup_refuses_and_lists_existing_unsealed_scratch(tmp_path: Path) -> None:
    environment, paths = fake_pipeline_environment(tmp_path)
    stale_work = paths["work"].with_name(f".{paths['target'].name}.append-work-prior")
    stale_build = paths["build"].with_name(f".{paths['target'].name}.build-prior")
    stale_work.mkdir()
    stale_build.mkdir()
    (stale_work / "keep.txt").write_text("work\n", encoding="utf-8")
    (stale_build / "keep.txt").write_text("build\n", encoding="utf-8")

    result = subprocess.run(
        ["bash", str(PIPELINE), "build"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    output = result.stdout + result.stderr
    assert str(stale_work) in output
    assert str(stale_build) in output
    assert "inspect or remove the listed scratch explicitly" in output
    assert (stale_work / "keep.txt").read_text(encoding="utf-8") == "work\n"
    assert (stale_build / "keep.txt").read_text(encoding="utf-8") == "build\n"
    assert "append " not in paths["log"].read_text(encoding="utf-8")


def test_target_created_at_publish_boundary_is_never_replaced(tmp_path: Path) -> None:
    environment, paths = fake_pipeline_environment(tmp_path, race_target=True)

    result = subprocess.run(
        ["bash", str(PIPELINE), "build"],
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 17
    assert (paths["target"] / "racer.txt").read_text(encoding="utf-8") == "must survive\n"
    assert not paths["work"].exists()
    assert not paths["build"].exists()
    assert not (paths["checkpoint"] / "stack_red_cups_09_15").exists()
    assert (paths["checkpoint"] / "final_build").is_dir()
    calls = paths["log"].read_text(encoding="utf-8")
    assert "support publish-no-replace" in calls
