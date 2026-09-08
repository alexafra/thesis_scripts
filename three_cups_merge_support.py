#!/usr/bin/env python3
"""Deterministic split, audit, and provenance checks for the 2026-09 cups merge."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np
import pyarrow.parquet as pq


TASK = "stack the three red cups."
SPLITS = ("train", "test", "validation")
STACK_RECORDED_DEPTH_SCALE = 0.0010000000474974513
CANONICAL_DEPTH_SCALE = 0.001
DEPTH_SCALE_ABS_TOLERANCE = 1e-15
RIGHT_IDS = {
    "train": (
        "0013",
        "0014",
        "0015",
        "0016",
        "0017",
        "0020",
        "0022",
        "0024",
        "0025",
        "0027",
        "0028",
        "0029",
        "0033",
        "0034",
        "0035",
        "0037",
        "0040",
        "0041",
        "0044",
        "0046",
    ),
    "test": ("0026", "0031", "0038"),
    "validation": ("0021", "0047"),
}
STACK_EXCLUDED_IDS = {"0001", "0013", "0018", "0053", "0082", "0092"}
EXPECTED_SOURCE = {
    "right": {"episodes": 25, "frames": 40_517},
    "stack": {"episodes": 86, "frames": 97_272},
}
EXPECTED_FINAL = {
    "train": {"episodes": 420, "frames": 194_692},
    "test": {"episodes": 54, "frames": 26_616},
    "validation": {"episodes": 53, "frames": 25_845},
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def episode_number(name: str) -> int:
    prefix = "episode_"
    if not name.startswith(prefix) or not name[len(prefix) :].isdigit():
        raise ValueError(f"Invalid episode name: {name}")
    return int(name[len(prefix) :])


def combined_data_json_digest(root: Path, episode_dirs: list[Path]) -> str:
    digest = hashlib.sha256()
    for episode in episode_dirs:
        path = episode / "data.json"
        digest.update(episode.name.encode())
        digest.update(b"\0")
        with path.open("rb") as handle:
            while block := handle.read(1024 * 1024):
                digest.update(block)
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def canonicalize_stack_depth_scale(
    root: Path,
    provenance_output: Path,
    *,
    source_root: Path,
    expected_episode_count: int = EXPECTED_SOURCE["stack"]["episodes"],
) -> dict:
    """Canonicalize float32 spelling noise in an isolated stack-source copy."""

    root = root.resolve()
    source_root = source_root.resolve()
    episodes = sorted(root.glob("episode_*"), key=lambda path: episode_number(path.name))
    if len(episodes) != expected_episode_count:
        raise ValueError(
            f"{root}: expected {expected_episode_count} episodes before depth-scale "
            f"canonicalization, found {len(episodes)}"
        )
    if provenance_output.exists():
        raise ValueError(f"Refusing to overwrite provenance: {provenance_output}")

    prepared = []
    for episode in episodes:
        path = episode / "data.json"
        source_path = source_root / episode.name / "data.json"
        if not source_path.is_file():
            raise ValueError(f"Missing immutable source JSON: {source_path}")
        original_sha256 = file_sha256(path)
        source_sha256 = file_sha256(source_path)
        if original_sha256 != source_sha256:
            raise ValueError(f"Staged JSON no longer matches its source: {path}")
        payload = read_json(path)
        try:
            scale = float(payload["info"]["depth"]["scale_m_per_unit"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Missing numeric info.depth.scale_m_per_unit: {path}") from error
        if not math.isfinite(scale) or not math.isclose(
            scale,
            STACK_RECORDED_DEPTH_SCALE,
            rel_tol=0.0,
            abs_tol=DEPTH_SCALE_ABS_TOLERANCE,
        ):
            raise ValueError(
                f"{path}: unexpected depth scale {scale!r}; expected the recorded "
                f"value {STACK_RECORDED_DEPTH_SCALE!r}"
            )
        payload["info"]["depth"]["scale_m_per_unit"] = CANONICAL_DEPTH_SCALE
        prepared.append((episode.name, path, payload, original_sha256))

    records = []
    for episode_name, path, payload, original_sha256 in prepared:
        temporary = path.with_name("data.json.depth-scale.tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
        verified = read_json(path)
        actual = verified["info"]["depth"]["scale_m_per_unit"]
        if actual != CANONICAL_DEPTH_SCALE:
            raise ValueError(f"{path}: canonical depth scale did not persist")
        records.append(
            {
                "episode": episode_name,
                "original_data_json_sha256": original_sha256,
                "canonicalized_data_json_sha256": file_sha256(path),
            }
        )

    provenance = {
        "version": 1,
        "operation": "canonicalize_depth_scale_float_spelling",
        "field": "info.depth.scale_m_per_unit",
        "source_root": str(source_root),
        "original_value": STACK_RECORDED_DEPTH_SCALE,
        "canonical_value": CANONICAL_DEPTH_SCALE,
        "absolute_tolerance": DEPTH_SCALE_ABS_TOLERANCE,
        "reason": (
            "The recorded float32 spelling is physically equivalent to 0.001 m/unit, "
            "but exact LeRobot append metadata compatibility requires one canonical value."
        ),
        "changed_episode_count": len(records),
        "episodes": records,
    }
    provenance_output.parent.mkdir(parents=True, exist_ok=True)
    provenance_output.write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return provenance


def audit_source(root: Path, kind: str) -> None:
    root = root.resolve()
    episodes = sorted(root.glob("episode_*"), key=lambda path: episode_number(path.name))
    expected = EXPECTED_SOURCE[kind]
    if len(episodes) != expected["episodes"]:
        raise ValueError(f"{root}: expected {expected['episodes']} episodes, found {len(episodes)}")

    ids = {path.name.removeprefix("episode_") for path in episodes}
    if kind == "right":
        expected_ids = {value for values in RIGHT_IDS.values() for value in values}
    else:
        expected_ids = {f"{value:04d}" for value in range(1, 93)} - STACK_EXCLUDED_IDS
    if ids != expected_ids:
        raise ValueError(
            f"{root}: episode membership mismatch; missing={sorted(expected_ids - ids)}, "
            f"unexpected={sorted(ids - expected_ids)}"
        )

    total_frames = 0
    for episode in episodes:
        payload = read_json(episode / "data.json")
        goal = (payload.get("text") or {}).get("goal")
        if goal != TASK:
            raise ValueError(f"{episode}: goal is {goal!r}, expected {TASK!r}")
        frames = payload.get("data")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"{episode}: data must be a non-empty list")
        previous_timestamp = -math.inf
        for index, frame in enumerate(frames):
            if frame.get("idx") != index:
                raise ValueError(f"{episode}: frame {index} has idx={frame.get('idx')!r}")
            timestamp = float(frame.get("timestamp_s"))
            if not math.isfinite(timestamp) or timestamp < previous_timestamp:
                raise ValueError(f"{episode}: invalid timestamp at frame {index}")
            previous_timestamp = timestamp
            for section in ("states", "actions"):
                payload_section = frame.get(section) or {}
                for key in ("left_arm", "right_arm", "left_ee", "right_ee"):
                    values = (payload_section.get(key) or {}).get("qpos")
                    if not isinstance(values, list) or len(values) != 7:
                        raise ValueError(
                            f"{episode}: {section}.{key}.qpos is not 7-D at frame {index}"
                        )
                    if not all(math.isfinite(float(value)) for value in values):
                        raise ValueError(
                            f"{episode}: non-finite {section}.{key}.qpos at frame {index}"
                        )
            media = (
                (frame.get("colors") or {}).get("color_0"),
                (frame.get("depths") or {}).get("depth_0"),
                (frame.get("depths") or {}).get("raw_depth_0"),
            )
            for relative in media:
                if not isinstance(relative, str) or not (episode / relative).is_file():
                    raise ValueError(
                        f"{episode}: missing referenced media at frame {index}: {relative!r}"
                    )
        total_frames += len(frames)

    if total_frames != expected["frames"]:
        raise ValueError(f"{root}: expected {expected['frames']} frames, found {total_frames}")
    digest = combined_data_json_digest(root, episodes)
    print(
        json.dumps(
            {
                "source": str(root),
                "kind": kind,
                "episodes": len(episodes),
                "frames": total_frames,
                "data_json_sha256": digest,
            },
            sort_keys=True,
        )
    )


def enrich_manifest(root: Path) -> dict:
    manifest_path = root / "split_manifest.json"
    manifest = read_json(manifest_path)
    records = manifest.get("episodes")
    if not isinstance(records, list):
        raise ValueError(f"Invalid episodes list: {manifest_path}")
    for record in records:
        episode = root / str(record["split"]) / str(record["split_episode"])
        payload_path = episode / "data.json"
        payload = read_json(payload_path)
        frames = payload.get("data")
        if not isinstance(frames, list) or not frames:
            raise ValueError(f"No frame data in {payload_path}")
        record["frame_count"] = len(frames)
        record["data_json_sha256"] = hashlib.sha256(payload_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def split_right(root: Path) -> None:
    scripts_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(scripts_dir))
    import split_dataset  # noqa: PLC0415

    root = root.resolve()
    episodes = split_dataset.load_episodes(root, split_dataset.load_flatten_manifest(root))
    by_id = {item["flattened_episode"].removeprefix("episode_"): item for item in episodes}
    expected_ids = {value for values in RIGHT_IDS.values() for value in values}
    if set(by_id) != expected_ids:
        raise ValueError(
            f"Right-only membership mismatch; missing={sorted(expected_ids - set(by_id))}, "
            f"unexpected={sorted(set(by_id) - expected_ids)}"
        )
    splits = {split: [by_id[value] for value in values] for split, values in RIGHT_IDS.items()}
    split_dataset.apply_split(
        root,
        splits,
        "preserved-original-pick-three-cups-1408-seed-42-membership",
        42,
    )
    manifest = enrich_manifest(root)
    manifest["preservation"] = {
        "original_source_episode_count": 31,
        "removed_train_ids": ["0049", "0050", "0054", "0055", "0056"],
        "removed_validation_ids": ["0051"],
        "trimmed_retained": {
            "episode_0013": {"original_frame_start": 784, "retained_frames": 1022},
            "episode_0025": {"original_frame_start": 250, "retained_frames": 1114},
        },
    }
    (root / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    split_dataset.print_summary(root, splits)


def _load_append_module(scripts_dir: Path):
    path = scripts_dir / "append_lerobot2.py"
    spec = importlib.util.spec_from_file_location("append_lerobot2_for_validation", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _split_info(root: Path, split: str) -> dict:
    return read_json(root / split / "meta" / "info.json")


def _episode_records(root: Path, split: str) -> list[dict]:
    return read_jsonl(root / split / "meta" / "episodes.jsonl")


def validate_final(
    base: Path,
    right: Path,
    stack: Path,
    right_source: Path,
    stack_source: Path,
    output: Path,
    right_manifest_path: Path,
    stack_manifest_path: Path,
    depth_scale_provenance_path: Path,
    scripts_dir: Path,
) -> None:
    append_module = _load_append_module(scripts_dir)
    right_manifest = read_json(right_manifest_path)
    stack_manifest = read_json(stack_manifest_path)
    depth_scale_provenance = read_json(depth_scale_provenance_path)
    if (
        depth_scale_provenance.get("changed_episode_count") != EXPECTED_SOURCE["stack"]["episodes"]
        or depth_scale_provenance.get("original_value") != STACK_RECORDED_DEPTH_SCALE
        or depth_scale_provenance.get("canonical_value") != CANONICAL_DEPTH_SCALE
    ):
        raise ValueError("Stack depth-scale provenance does not match the locked transform")
    provenance_records: list[dict] = []

    for split in SPLITS:
        base_info = _split_info(base, split)
        right_info = _split_info(right, split)
        stack_info = _split_info(stack, split)
        output_info = append_module.validate_dataset(output / split, inspect_rows=True)
        expected = EXPECTED_FINAL[split]
        if int(output_info["total_episodes"]) != expected["episodes"]:
            raise ValueError(f"{split}: wrong final episode count")
        if int(output_info["total_frames"]) != expected["frames"]:
            raise ValueError(f"{split}: wrong final frame count")
        if int(output_info["total_tasks"]) != 9:
            raise ValueError(f"{split}: expected 9 tasks")
        expected_episodes = (
            int(base_info["total_episodes"])
            + int(right_info["total_episodes"])
            + int(stack_info["total_episodes"])
        )
        expected_frames = (
            int(base_info["total_frames"])
            + int(right_info["total_frames"])
            + int(stack_info["total_frames"])
        )
        if expected_episodes != expected["episodes"] or expected_frames != expected["frames"]:
            raise ValueError(f"{split}: component totals do not match the locked expectation")

        tasks = read_jsonl(output / split / "meta" / "tasks.jsonl")
        task_lookup = {record["task"]: int(record["task_index"]) for record in tasks}
        if task_lookup.get(TASK) != 8 or len(tasks) != 9:
            raise ValueError(f"{split}: cups task is not the single task index 8")

        output_episodes = _episode_records(output, split)
        source_specs = (
            ("pick_three_cups_right_only_1408", right, right_manifest),
            ("stack_cups_09_08", stack, stack_manifest),
        )
        final_offset = int(base_info["total_episodes"])
        for source_name, source_root, manifest in source_specs:
            manifest_records = sorted(
                (record for record in manifest["episodes"] if record["split"] == split),
                key=lambda record: episode_number(record["split_episode"]),
            )
            source_records = _episode_records(source_root, split)
            if len(manifest_records) != len(source_records):
                raise ValueError(f"{source_name}/{split}: manifest/dataset episode mismatch")
            source_info = _split_info(source_root, split)
            for local_index, (manifest_record, source_record) in enumerate(
                zip(manifest_records, source_records, strict=True)
            ):
                final_index = final_offset + local_index
                final_record = output_episodes[final_index]
                frame_count = int(manifest_record["frame_count"])
                if int(source_record["length"]) != frame_count:
                    raise ValueError(
                        f"{source_name}/{split}: source length mismatch at {local_index}"
                    )
                if int(final_record["length"]) != frame_count or final_record.get("tasks") != [
                    TASK
                ]:
                    raise ValueError(
                        f"{source_name}/{split}: final episode mismatch at {final_index}"
                    )
                parquet_path = append_module.episode_path(output / split, output_info, final_index)
                task_values = pq.read_table(parquet_path, columns=["task_index"])[
                    "task_index"
                ].to_numpy()
                if not np.all(task_values == 8):
                    raise ValueError(f"{parquet_path}: appended task_index is not uniformly 8")
                provenance_records.append(
                    {
                        "source": source_name,
                        "split": split,
                        "source_episode": manifest_record["flattened_episode"],
                        "converted_episode_index": local_index,
                        "final_episode_index": final_index,
                        "frame_count": frame_count,
                        "data_json_sha256": manifest_record["data_json_sha256"],
                    }
                )
            final_offset += int(source_info["total_episodes"])
        if final_offset != expected["episodes"]:
            raise ValueError(f"{split}: final provenance range ends at {final_offset}")

    provenance = output / "provenance"
    provenance.mkdir(parents=True, exist_ok=False)
    shutil.copy2(
        right_manifest_path, provenance / "pick_three_cups_right_only_1408_split_manifest.json"
    )
    shutil.copy2(stack_manifest_path, provenance / "stack_cups_09_08_split_manifest.json")
    shutil.copy2(
        depth_scale_provenance_path,
        provenance / "stack_cups_09_08_depth_scale_canonicalization.json",
    )
    merge_manifest = {
        "version": 1,
        "task": TASK,
        "task_index": 8,
        "sources": {
            "base": str(base.resolve()),
            "right_only": str(right_source.resolve()),
            "stack_09_08": str(stack_source.resolve()),
        },
        "expected_final": EXPECTED_FINAL,
        "total": {"episodes": 527, "frames": 247_153, "tasks": 9},
        "transformations": {
            "stack_depth_scale": {
                "field": depth_scale_provenance["field"],
                "original_value": depth_scale_provenance["original_value"],
                "canonical_value": depth_scale_provenance["canonical_value"],
                "changed_episode_count": depth_scale_provenance["changed_episode_count"],
                "provenance_file": (
                    "provenance/stack_cups_09_08_depth_scale_canonicalization.json"
                ),
            }
        },
        "episodes": provenance_records,
    }
    (provenance / "merge_manifest.json").write_text(
        json.dumps(merge_manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps({"final": EXPECTED_FINAL, "mapped_additions": len(provenance_records)}, indent=2)
    )


def validate_stats(train_root: Path) -> None:
    paths = (train_root / "meta" / "stats.json", train_root / "meta" / "relative_stats.json")
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))

        def walk(item, label: str) -> None:
            if isinstance(item, dict):
                for key, child in item.items():
                    walk(child, f"{label}.{key}")
            elif isinstance(item, list):
                for index, child in enumerate(item):
                    walk(child, f"{label}[{index}]")
            elif isinstance(item, float) and not math.isfinite(item):
                raise ValueError(f"Non-finite value at {label} in {path}")

        walk(value, path.name)
    print(f"Validated finite GR00T stats: {paths[0]} and {paths[1]}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    audit = subparsers.add_parser("audit-source")
    audit.add_argument("--root", type=Path, required=True)
    audit.add_argument("--kind", choices=("right", "stack"), required=True)

    right = subparsers.add_parser("split-right")
    right.add_argument("--root", type=Path, required=True)

    enrich = subparsers.add_parser("enrich-manifest")
    enrich.add_argument("--root", type=Path, required=True)

    canonicalize = subparsers.add_parser("canonicalize-stack-depth-scale")
    canonicalize.add_argument("--root", type=Path, required=True)
    canonicalize.add_argument("--source-root", type=Path, required=True)
    canonicalize.add_argument("--provenance-output", type=Path, required=True)

    final = subparsers.add_parser("validate-final")
    final.add_argument("--base", type=Path, required=True)
    final.add_argument("--right", type=Path, required=True)
    final.add_argument("--stack", type=Path, required=True)
    final.add_argument("--right-source", type=Path, required=True)
    final.add_argument("--stack-source", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--right-manifest", type=Path, required=True)
    final.add_argument("--stack-manifest", type=Path, required=True)
    final.add_argument("--depth-scale-provenance", type=Path, required=True)
    final.add_argument("--scripts-dir", type=Path, required=True)

    stats = subparsers.add_parser("validate-stats")
    stats.add_argument("--train-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "audit-source":
        audit_source(args.root, args.kind)
    elif args.command == "split-right":
        split_right(args.root)
    elif args.command == "enrich-manifest":
        enrich_manifest(args.root)
        print(f"Enriched {args.root / 'split_manifest.json'}")
    elif args.command == "canonicalize-stack-depth-scale":
        provenance = canonicalize_stack_depth_scale(
            args.root,
            args.provenance_output,
            source_root=args.source_root,
        )
        print(json.dumps(provenance, indent=2, sort_keys=True))
    elif args.command == "validate-final":
        validate_final(
            args.base,
            args.right,
            args.stack,
            args.right_source,
            args.stack_source,
            args.output,
            args.right_manifest,
            args.stack_manifest,
            args.depth_scale_provenance,
            args.scripts_dir,
        )
    elif args.command == "validate-stats":
        validate_stats(args.train_root)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(f"ERROR: {error}") from error
