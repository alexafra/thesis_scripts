#!/usr/bin/env python3
"""Destructively split flattened episodes into train/test/validation folders.

Example (run from /home/alex/Development/scripts/):
    cd /home/alex/Development/scripts/
    ./split_dataset.py /path/to/copied_root

The default strategy is goal-stratified with an 80/10/10 split.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import tempfile


EPISODE_PATTERN = re.compile(r"episode_(\d+)$")
FLATTEN_MANIFEST = "flatten_manifest.json"
SPLIT_MANIFEST = "split_manifest.json"
SPLIT_NAMES = ("train", "test", "validation")
RATIOS = {"train": 0.8, "test": 0.1, "validation": 0.1}
COLLECTION_STRATEGY = "component-preserved"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Destructively move flat episode_* folders into "
            "train/test/validation. The default stratifies episodes by goal."
        )
    )
    parser.add_argument("root", type=Path, help="Flattened disposable dataset root")
    parser.add_argument(
        "--strategy",
        choices=("session-grouped", "goal-stratified"),
        default="goal-stratified",
        help=(
            "goal-stratified balances goals episode-by-episode (default); "
            "session-grouped prevents scene leakage"
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Reproducible split seed")
    parser.add_argument(
        "--search-trials",
        type=int,
        default=20000,
        help="Candidate assignments considered for session-grouped splitting",
    )
    collection_mode = parser.add_mutually_exclusive_group()
    collection_mode.add_argument(
        "--collection-check",
        action="store_true",
        help=(
            "Validate and plan an immediate-child dataset collection without copying "
            "or changing anything"
        ),
    )
    collection_mode.add_argument(
        "--collection-output",
        type=Path,
        help=(
            "Copy an immediate-child dataset collection into one combined raw "
            "train/test/validation tree at this new path"
        ),
    )
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=[],
        metavar="NAME",
        help="Exact immediate-child dataset name to exclude (repeatable)",
    )
    parser.add_argument(
        "--preserve-split",
        action="append",
        default=[],
        metavar="DATASET=MANIFEST",
        help=(
            "Preserve split membership and ordering for one child dataset from an "
            "existing split_manifest.json (repeatable)"
        ),
    )
    return parser.parse_args()


def target_sizes(total: int) -> dict[str, int]:
    if total < 3:
        raise ValueError(
            "At least 3 episodes are required to create three non-empty splits"
        )

    test_count = max(1, int(total * RATIOS["test"] + 0.5))
    validation_count = max(1, int(total * RATIOS["validation"] + 0.5))
    train_count = total - test_count - validation_count

    if train_count < 1:
        train_count = 1
        remaining = total - train_count
        test_count = max(1, remaining // 2)
        validation_count = remaining - test_count

    return {
        "train": train_count,
        "test": test_count,
        "validation": validation_count,
    }


def load_flatten_manifest(root: Path) -> dict[str, dict]:
    manifest_path = root / FLATTEN_MANIFEST
    if not manifest_path.is_file():
        return {}

    try:
        with manifest_path.open(encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {manifest_path}: {error}") from error

    records = manifest.get("episodes")
    if not isinstance(records, list):
        raise ValueError(f"Invalid episodes list in {manifest_path}")

    return {
        record["flattened_episode"]: record
        for record in records
        if isinstance(record, dict) and isinstance(record.get("flattened_episode"), str)
    }


def load_episodes(root: Path, source_records: dict[str, dict]) -> list[dict]:
    episode_paths = sorted(
        (
            path
            for path in root.iterdir()
            if path.is_dir() and EPISODE_PATTERN.fullmatch(path.name)
        ),
        key=lambda path: int(EPISODE_PATTERN.fullmatch(path.name).group(1)),
    )

    if not episode_paths:
        raise ValueError(f"No flat episode_* directories found directly under {root}")

    episodes = []
    for episode_path in episode_paths:
        json_path = episode_path / "data.json"
        if not json_path.is_file():
            raise ValueError(f"Episode is missing data.json: {episode_path}")

        try:
            with json_path.open(encoding="utf-8") as file:
                episode_json = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read {json_path}: {error}") from error

        text_info = episode_json.get("text")
        goal = text_info.get("goal", "") if isinstance(text_info, dict) else ""
        if not isinstance(goal, str):
            goal = str(goal)
        frames = episode_json.get("data")
        if not isinstance(frames, list):
            raise ValueError(f"Episode data is not a list: {json_path}")
        timing = episode_json.get("timing")
        capture_start_utc = (
            timing.get("capture_start_utc") if isinstance(timing, dict) else None
        )
        if capture_start_utc is not None and (
            not isinstance(capture_start_utc, str) or not capture_start_utc.strip()
        ):
            raise ValueError(f"Invalid timing.capture_start_utc in {json_path}")

        source = source_records.get(episode_path.name, {})
        session = source.get("source_session")
        if not isinstance(session, str) or not session:
            session = episode_path.name

        episodes.append(
            {
                "path": episode_path,
                "flattened_episode": episode_path.name,
                "source_episode": source.get("source_episode", episode_path.name),
                "source_session": session,
                "goal": goal.strip(),
                "frame_count": len(frames),
                "data_json_sha256": hashlib.sha256(json_path.read_bytes()).hexdigest(),
                "capture_start_utc": capture_start_utc,
            }
        )

    return episodes


def assignment_score(
    assignments: dict[str, str],
    group_summaries: dict[str, tuple[int, Counter]],
    targets: dict[str, int],
    goal_totals: Counter,
) -> float:
    split_counts = Counter()
    split_goals = {name: Counter() for name in SPLIT_NAMES}

    for group_name, split_name in assignments.items():
        group_size, group_goals = group_summaries[group_name]
        split_counts[split_name] += group_size
        split_goals[split_name].update(group_goals)

    total = sum(targets.values())
    size_error = (
        sum(abs(split_counts[name] - targets[name]) for name in SPLIT_NAMES) / total
    )

    goal_error = 0.0
    for goal, goal_total in goal_totals.items():
        goal_error += sum(
            abs(split_goals[name][goal] - RATIOS[name] * goal_total)
            for name in SPLIT_NAMES
        ) / (2 * goal_total)
    goal_error /= max(1, len(goal_totals))

    return 4.0 * size_error + goal_error


def split_session_grouped(
    episodes: list[dict], targets: dict[str, int], seed: int, trials: int
) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for episode in episodes:
        groups[episode["source_session"]].append(episode)

    if len(groups) < 3:
        raise ValueError(
            "session-grouped splitting needs at least 3 source sessions; "
            "use --strategy goal-stratified to split individual episodes"
        )
    if trials < 1:
        raise ValueError("--search-trials must be positive")

    rng = random.Random(seed)
    group_names = sorted(groups)
    goal_totals = Counter(episode["goal"] for episode in episodes)
    group_summaries = {
        group_name: (
            len(group_episodes),
            Counter(episode["goal"] for episode in group_episodes),
        )
        for group_name, group_episodes in groups.items()
    }
    best_assignment = None
    best_score = float("inf")

    for _ in range(trials):
        shuffled_groups = group_names.copy()
        rng.shuffle(shuffled_groups)
        assignments = {}

        anchor_splits = list(SPLIT_NAMES)
        rng.shuffle(anchor_splits)
        for group_name, split_name in zip(
            shuffled_groups[:3], anchor_splits, strict=True
        ):
            assignments[group_name] = split_name

        for group_name in shuffled_groups[3:]:
            assignments[group_name] = rng.choices(
                SPLIT_NAMES,
                weights=[RATIOS[name] for name in SPLIT_NAMES],
                k=1,
            )[0]

        score = assignment_score(assignments, group_summaries, targets, goal_totals)
        if score < best_score:
            best_score = score
            best_assignment = assignments
            if score == 0:
                break

    result = {name: [] for name in SPLIT_NAMES}
    for group_name, split_name in best_assignment.items():
        result[split_name].extend(groups[group_name])
    return result


def proportional_counts(
    group_sizes: dict[str, int], total_to_select: int, capacities: dict[str, int]
) -> dict[str, int]:
    total = sum(group_sizes.values())
    quotas = {
        goal: total_to_select * size / total for goal, size in group_sizes.items()
    }
    counts = {goal: min(int(quotas[goal]), capacities[goal]) for goal in group_sizes}

    while sum(counts.values()) < total_to_select:
        candidates = [goal for goal in group_sizes if counts[goal] < capacities[goal]]
        if not candidates:
            raise ValueError("Could not allocate requested stratified split size")
        selected_goal = max(
            candidates,
            key=lambda goal: (quotas[goal] - counts[goal], group_sizes[goal], goal),
        )
        counts[selected_goal] += 1

    return counts


def split_goal_stratified(
    episodes: list[dict], targets: dict[str, int], seed: int
) -> dict[str, list[dict]]:
    rng = random.Random(seed)
    by_goal: dict[str, list[dict]] = defaultdict(list)
    for episode in episodes:
        by_goal[episode["goal"]].append(episode)
    for goal_episodes in by_goal.values():
        rng.shuffle(goal_episodes)

    sizes = {goal: len(goal_episodes) for goal, goal_episodes in by_goal.items()}
    test_counts = proportional_counts(sizes, targets["test"], sizes)
    remaining_capacities = {goal: sizes[goal] - test_counts[goal] for goal in sizes}
    validation_counts = proportional_counts(
        sizes, targets["validation"], remaining_capacities
    )

    result = {name: [] for name in SPLIT_NAMES}
    for goal, goal_episodes in by_goal.items():
        test_end = test_counts[goal]
        validation_end = test_end + validation_counts[goal]
        result["test"].extend(goal_episodes[:test_end])
        result["validation"].extend(goal_episodes[test_end:validation_end])
        result["train"].extend(goal_episodes[validation_end:])
    return result


def _safe_dataset_name(value: str, *, label: str) -> str:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
        raise ValueError(
            f"{label} must be one safe immediate-child directory name: {value!r}"
        )
    if value in {".", ".."}:
        raise ValueError(f"{label} must not be {value!r}")
    return value


def parse_preserved_splits(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        dataset_name, separator, manifest_text = value.partition("=")
        if not separator or not manifest_text:
            raise ValueError(
                "--preserve-split must use DATASET=/path/to/split_manifest.json"
            )
        dataset_name = _safe_dataset_name(dataset_name, label="preserved dataset")
        if dataset_name in result:
            raise ValueError(f"Duplicate --preserve-split dataset: {dataset_name}")
        manifest_arg = Path(manifest_text).expanduser()
        if manifest_arg.is_symlink():
            raise ValueError(
                f"Preserved split manifest may not be a symlink: {manifest_arg}"
            )
        manifest_path = manifest_arg.resolve()
        if not manifest_path.is_file():
            raise ValueError(
                f"Preserved split manifest does not exist: {manifest_path}"
            )
        result[dataset_name] = manifest_path
    return result


def discover_collection_datasets(
    root: Path, excluded_names: list[str]
) -> dict[str, Path]:
    if root.is_symlink():
        raise ValueError(f"Collection root may not be a symlink: {root}")
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Collection root does not exist: {root}")

    excluded = [
        _safe_dataset_name(name, label="excluded dataset") for name in excluded_names
    ]
    if len(set(excluded)) != len(excluded):
        raise ValueError("Duplicate --exclude-dataset value")

    child_directories: dict[str, Path] = {}
    for child in root.iterdir():
        if child.is_symlink():
            raise ValueError(f"Collection entries may not be symlinks: {child}")
        if child.is_dir():
            child_directories[child.name] = child

    missing_exclusions = sorted(set(excluded) - set(child_directories))
    if missing_exclusions:
        raise ValueError(
            f"Excluded dataset(s) do not exist directly under {root}: {missing_exclusions}"
        )
    direct_episodes = sorted(
        name for name in child_directories if EPISODE_PATTERN.fullmatch(name)
    )
    if direct_episodes:
        raise ValueError(
            f"Collection root contains direct episode directories: {direct_episodes[:5]}"
        )

    selected = {
        name: path
        for name, path in child_directories.items()
        if name not in set(excluded)
    }
    if not selected:
        raise ValueError(f"No child datasets remain after exclusions under {root}")

    for dataset_name, dataset_path in sorted(selected.items()):
        _safe_dataset_name(dataset_name, label="dataset")
        entries = list(dataset_path.iterdir())
        episode_directories = [
            path
            for path in entries
            if path.is_dir()
            and not path.is_symlink()
            and EPISODE_PATTERN.fullmatch(path.name)
        ]
        unexpected_directories = [
            path.name
            for path in entries
            if path.is_dir() and not EPISODE_PATTERN.fullmatch(path.name)
        ]
        direct_symlinks = [path for path in entries if path.is_symlink()]
        if direct_symlinks:
            raise ValueError(
                f"Dataset entries may not be symlinks: {direct_symlinks[0]}"
            )
        if unexpected_directories:
            raise ValueError(
                f"Dataset {dataset_name} contains nested wrapper directories: "
                f"{sorted(unexpected_directories)}"
            )
        if not episode_directories:
            raise ValueError(
                f"Dataset has no direct episode_* directories: {dataset_path}"
            )

        for current_dir, directory_names, file_names in os.walk(
            dataset_path, followlinks=False
        ):
            current = Path(current_dir)
            for name in directory_names + file_names:
                candidate = current / name
                if candidate.is_symlink():
                    raise ValueError(
                        f"Dataset trees may not contain symlinks: {candidate}"
                    )
            if current != dataset_path and current.parent != dataset_path:
                nested_episodes = [
                    name for name in directory_names if EPISODE_PATTERN.fullmatch(name)
                ]
                if nested_episodes:
                    raise ValueError(
                        f"Dataset contains nested episode directories below an episode: "
                        f"{current / nested_episodes[0]}"
                    )

    return dict(sorted(selected.items()))


def collection_curation_manifest(root: Path) -> tuple[Path | None, str | None]:
    candidates = sorted(root.glob("curation_manifest*.json"))
    if len(candidates) > 1:
        raise ValueError(
            f"Collection has multiple curation manifests; keep one canonical file: {candidates}"
        )
    if not candidates:
        return None, None
    path = candidates[0]
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Curation manifest must be a regular file: {path}")
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def load_collection_episodes(
    datasets: dict[str, Path],
) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    digest_owners: dict[str, str] = {}
    capture_owners: dict[str, str] = {}

    for dataset_name, dataset_path in datasets.items():
        episodes = load_episodes(dataset_path, {})
        for episode in episodes:
            source_episode = episode["path"].name
            source_identity = f"{dataset_name}/{source_episode}"
            episode["source_dataset"] = dataset_name
            episode["source_episode"] = source_episode
            episode["source_identity"] = source_identity

            digest = episode["data_json_sha256"]
            previous_digest_owner = digest_owners.get(digest)
            if previous_digest_owner is not None:
                raise ValueError(
                    "Duplicate processed episode data.json content: "
                    f"{previous_digest_owner} and {source_identity}"
                )
            digest_owners[digest] = source_identity

            capture_start = episode.get("capture_start_utc")
            if capture_start:
                previous_capture_owner = capture_owners.get(capture_start)
                if previous_capture_owner is not None:
                    raise ValueError(
                        "Duplicate or conflicting timing.capture_start_utc: "
                        f"{previous_capture_owner} and {source_identity} ({capture_start})"
                    )
                capture_owners[capture_start] = source_identity
        result[dataset_name] = episodes
    return result


def load_preserved_assignment(
    dataset_name: str,
    episodes: list[dict],
    manifest_path: Path,
) -> tuple[dict[str, list[dict]], dict]:
    manifest_bytes = manifest_path.read_bytes()
    try:
        manifest = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Could not parse preserved split manifest {manifest_path}: {error}"
        ) from error
    if not isinstance(manifest, dict) or not isinstance(manifest.get("episodes"), list):
        raise ValueError(f"Invalid preserved split manifest: {manifest_path}")

    indexed: dict[str, dict] = {}
    for record in manifest["episodes"]:
        if not isinstance(record, dict):
            raise ValueError(
                f"Invalid episode record in preserved manifest: {manifest_path}"
            )
        split = (
            "validation" if record.get("split") == "validate" else record.get("split")
        )
        if split not in SPLIT_NAMES:
            raise ValueError(
                f"Invalid preserved split name in {manifest_path}: {split!r}"
            )
        source_episode_value = record.get(
            "source_episode", record.get("flattened_episode")
        )
        source_episode = Path(str(source_episode_value)).name
        if EPISODE_PATTERN.fullmatch(source_episode) is None:
            raise ValueError(
                f"Invalid source episode in preserved manifest {manifest_path}: "
                f"{source_episode_value!r}"
            )
        if source_episode in indexed:
            raise ValueError(
                f"Duplicate source episode in preserved manifest: {source_episode}"
            )
        split_episode = str(record.get("split_episode", ""))
        split_match = EPISODE_PATTERN.fullmatch(split_episode)
        if split_match is None:
            raise ValueError(
                f"Invalid split_episode in preserved manifest {manifest_path}: {split_episode!r}"
            )
        indexed[source_episode] = {
            "record": record,
            "split": split,
            "order": int(split_match.group(1)),
        }

    current_names = {episode["source_episode"] for episode in episodes}
    manifest_names = set(indexed)
    if current_names != manifest_names:
        raise ValueError(
            f"Preserved manifest/source identity mismatch for {dataset_name}: "
            f"missing={sorted(current_names - manifest_names)}, "
            f"unexpected={sorted(manifest_names - current_names)}"
        )

    result = {name: [] for name in SPLIT_NAMES}
    changed_hashes = []
    changed_frame_counts = []
    for episode in episodes:
        reference = indexed[episode["source_episode"]]
        old_record = reference["record"]
        if old_record.get("data_json_sha256") != episode["data_json_sha256"]:
            changed_hashes.append(episode["source_episode"])
        if old_record.get("frame_count") != episode["frame_count"]:
            changed_frame_counts.append(episode["source_episode"])
        episode["preserved_order"] = reference["order"]
        result[reference["split"]].append(episode)

    for split in SPLIT_NAMES:
        orders = [episode["preserved_order"] for episode in result[split]]
        if len(set(orders)) != len(orders):
            raise ValueError(
                f"Duplicate {split} ordering in preserved manifest {manifest_path}"
            )
        result[split].sort(key=lambda episode: episode["preserved_order"])
        if not result[split]:
            raise ValueError(
                f"Preserved manifest has an empty {split} split: {manifest_path}"
            )

    provenance = {
        "dataset": dataset_name,
        "strategy": "preserved-membership",
        "episode_count": len(episodes),
        "splits": {split: len(result[split]) for split in SPLIT_NAMES},
        "assignment_manifest": str(manifest_path),
        "assignment_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "assignment_manifest_copy": f"provenance/preserved_split_{dataset_name}.json",
        "changed_data_json_hashes": sorted(changed_hashes),
        "changed_frame_counts": sorted(changed_frame_counts),
    }
    return result, provenance


def plan_collection(
    root: Path,
    excluded_names: list[str],
    preserved_manifests: dict[str, Path],
    strategy: str,
    seed: int,
    search_trials: int,
) -> tuple[list[dict], list[dict], dict[str, list[dict]]]:
    datasets = discover_collection_datasets(root, excluded_names)
    unknown_preserved = sorted(set(preserved_manifests) - set(datasets))
    if unknown_preserved:
        raise ValueError(
            f"--preserve-split names are not selected collection datasets: {unknown_preserved}"
        )
    episodes_by_dataset = load_collection_episodes(datasets)

    preserved_order = list(preserved_manifests)
    component_order = preserved_order + [
        name for name in datasets if name not in preserved_manifests
    ]
    component_splits: dict[str, dict[str, list[dict]]] = {}
    components: list[dict] = []
    for dataset_name in component_order:
        episodes = episodes_by_dataset[dataset_name]
        manifest_path = preserved_manifests.get(dataset_name)
        if manifest_path is not None:
            splits, component = load_preserved_assignment(
                dataset_name, episodes, manifest_path
            )
        else:
            targets = target_sizes(len(episodes))
            if strategy == "session-grouped":
                splits = split_session_grouped(episodes, targets, seed, search_trials)
            else:
                splits = split_goal_stratified(episodes, targets, seed)
            for split in SPLIT_NAMES:
                splits[split].sort(
                    key=lambda episode: int(
                        EPISODE_PATTERN.fullmatch(episode["source_episode"]).group(1)
                    )
                )
            component = {
                "dataset": dataset_name,
                "strategy": strategy,
                "seed": seed,
                "episode_count": len(episodes),
                "splits": {split: len(splits[split]) for split in SPLIT_NAMES},
            }
        component_splits[dataset_name] = splits
        components.append(component)

    flattened_names: dict[str, str] = {}
    next_flattened_index = 1
    for dataset_name in component_order:
        for episode in sorted(
            episodes_by_dataset[dataset_name],
            key=lambda item: int(
                EPISODE_PATTERN.fullmatch(item["source_episode"]).group(1)
            ),
        ):
            flattened_names[episode["source_identity"]] = (
                f"episode_{next_flattened_index:04d}"
            )
            next_flattened_index += 1

    planned_records: list[dict] = []
    for split in SPLIT_NAMES:
        next_split_index = 1
        for dataset_name in component_order:
            for episode in component_splits[dataset_name][split]:
                record = {
                    "split": split,
                    "split_episode": f"episode_{next_split_index:04d}",
                    "flattened_episode": flattened_names[episode["source_identity"]],
                    "source_dataset": dataset_name,
                    "source_episode": episode["source_episode"],
                    "source_session": dataset_name,
                    "source_path": episode["source_identity"],
                    "goal": episode["goal"],
                    "frame_count": episode["frame_count"],
                    "data_json_sha256": episode["data_json_sha256"],
                    "_source_path": episode["path"],
                }
                if episode.get("capture_start_utc"):
                    record["capture_start_utc"] = episode["capture_start_utc"]
                planned_records.append(record)
                next_split_index += 1

    return planned_records, components, episodes_by_dataset


def _collection_snapshot(episodes_by_dataset: dict[str, list[dict]]) -> dict[str, str]:
    return {
        episode["source_identity"]: episode["data_json_sha256"]
        for episodes in episodes_by_dataset.values()
        for episode in episodes
    }


def print_collection_summary(
    root: Path,
    records: list[dict],
    components: list[dict],
    excluded_names: list[str],
) -> None:
    counts = Counter(record["split"] for record in records)
    print(f"Collection source: {root}")
    print(
        f"Included datasets: {', '.join(component['dataset'] for component in components)}"
    )
    print(
        f"Excluded datasets: {', '.join(excluded_names) if excluded_names else '<none>'}"
    )
    print(
        "Combined split plan: "
        + ", ".join(f"{split}={counts[split]}" for split in SPLIT_NAMES)
        + f"; total={len(records)}"
    )
    for component in components:
        split_counts = component["splits"]
        print(
            f"  {component['dataset']}: {component['episode_count']} episodes; "
            f"strategy={component['strategy']}; "
            + ", ".join(f"{split}={split_counts[split]}" for split in SPLIT_NAMES)
        )


def compose_collection(
    root: Path,
    output: Path | None,
    excluded_names: list[str],
    preserved_manifests: dict[str, Path],
    strategy: str,
    seed: int,
    search_trials: int,
) -> None:
    root_arg = root.expanduser()
    records, components, initial_episodes = plan_collection(
        root_arg,
        excluded_names,
        preserved_manifests,
        strategy,
        seed,
        search_trials,
    )
    resolved_root = root_arg.resolve()
    curation_path, curation_digest = collection_curation_manifest(resolved_root)
    print_collection_summary(resolved_root, records, components, excluded_names)
    if curation_path is not None:
        print(f"Curation manifest: {curation_path} ({curation_digest})")
    if output is None:
        print("Collection check complete; no files were copied or changed.")
        return

    output_arg = output.expanduser()
    if output_arg.is_symlink():
        raise ValueError(f"Collection output may not be a symlink: {output_arg}")
    output = output_arg.resolve()
    if output.exists():
        raise ValueError(f"Collection output already exists: {output}")
    if (
        output == resolved_root
        or output in resolved_root.parents
        or resolved_root in output.parents
    ):
        raise ValueError(
            f"Collection source and output must not overlap: {resolved_root}, {output}"
        )
    output.parent.mkdir(parents=True, exist_ok=True)

    stage = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.composing.", dir=output.parent)
    )
    published = False
    try:
        for split in SPLIT_NAMES:
            (stage / split).mkdir()
        if curation_path is not None or preserved_manifests:
            provenance_dir = stage / "provenance"
            provenance_dir.mkdir()
        if curation_path is not None:
            shutil.copy2(curation_path, provenance_dir / curation_path.name)
        for component in components:
            if component["strategy"] != "preserved-membership":
                continue
            assignment_source = Path(component["assignment_manifest"])
            assignment_copy = stage / component["assignment_manifest_copy"]
            shutil.copy2(assignment_source, assignment_copy)
            copied_digest = hashlib.sha256(assignment_copy.read_bytes()).hexdigest()
            current_digest = hashlib.sha256(assignment_source.read_bytes()).hexdigest()
            expected_digest = component["assignment_manifest_sha256"]
            if copied_digest != expected_digest or current_digest != expected_digest:
                raise ValueError(
                    f"Preserved split manifest changed while staging: {assignment_source}"
                )
        for record in records:
            source_path = record["_source_path"]
            destination_path = stage / record["split"] / record["split_episode"]
            shutil.copytree(source_path, destination_path, copy_function=shutil.copy2)

        current_datasets = discover_collection_datasets(resolved_root, excluded_names)
        current_episodes = load_collection_episodes(current_datasets)
        current_curation_path, current_curation_digest = collection_curation_manifest(
            resolved_root
        )
        if (
            current_curation_path != curation_path
            or current_curation_digest != curation_digest
        ):
            raise ValueError("Collection curation manifest changed while staging")
        initial_snapshot = _collection_snapshot(initial_episodes)
        current_snapshot = _collection_snapshot(current_episodes)
        if current_snapshot != initial_snapshot:
            missing = sorted(set(initial_snapshot) - set(current_snapshot))
            added = sorted(set(current_snapshot) - set(initial_snapshot))
            changed = sorted(
                identity
                for identity in set(initial_snapshot).intersection(current_snapshot)
                if initial_snapshot[identity] != current_snapshot[identity]
            )
            raise ValueError(
                "Collection source changed while it was being staged: "
                f"missing={missing}, added={added}, changed={changed}"
            )

        for record in records:
            staged_json = (
                stage / record["split"] / record["split_episode"] / "data.json"
            )
            staged_digest = hashlib.sha256(staged_json.read_bytes()).hexdigest()
            if staged_digest != record["data_json_sha256"]:
                raise ValueError(
                    f"Staged data.json hash mismatch for {record['source_path']}"
                )

        manifest_records = [
            {key: value for key, value in record.items() if not key.startswith("_")}
            for record in records
        ]
        manifest = {
            "version": 1,
            "strategy": COLLECTION_STRATEGY,
            "seed": seed,
            "ratios": RATIOS,
            "source_root": str(resolved_root),
            "excluded_datasets": list(excluded_names),
            "episode_count": len(manifest_records),
            "components": components,
            "episodes": manifest_records,
        }
        if curation_path is not None:
            manifest["curation_manifest"] = {
                "source": str(curation_path),
                "path": f"provenance/{curation_path.name}",
                "sha256": curation_digest,
            }
        temporary_manifest = stage / f".{SPLIT_MANIFEST}.tmp"
        with temporary_manifest.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        temporary_manifest.replace(stage / SPLIT_MANIFEST)

        if output.exists() or output.is_symlink():
            raise ValueError(f"Collection output appeared during staging: {output}")
        stage.rename(output)
        published = True
    finally:
        if not published and stage.exists():
            shutil.rmtree(stage)

    print(f"Published combined raw split tree: {output}")


def apply_split(
    root: Path, splits: dict[str, list[dict]], strategy: str, seed: int
) -> None:
    for split_name in SPLIT_NAMES:
        split_path = root / split_name
        if split_path.exists():
            raise ValueError(f"Split output already exists: {split_path}")

    planned_records = []
    for split_name in SPLIT_NAMES:
        split_episodes = sorted(
            splits[split_name],
            key=lambda item: int(
                EPISODE_PATTERN.fullmatch(item["flattened_episode"]).group(1)
            ),
        )
        for split_index, episode in enumerate(split_episodes, start=1):
            planned_records.append(
                {
                    "split": split_name,
                    "split_episode": f"episode_{split_index:04d}",
                    "flattened_episode": episode["flattened_episode"],
                    "source_episode": episode["source_episode"],
                    "source_session": episode["source_session"],
                    "goal": episode["goal"],
                    "frame_count": episode["frame_count"],
                    "data_json_sha256": episode["data_json_sha256"],
                    "source_path": episode["path"],
                }
            )

    moved: list[tuple[Path, Path]] = []
    try:
        for split_name in SPLIT_NAMES:
            (root / split_name).mkdir()

        for record in planned_records:
            source_path = record["source_path"]
            destination_path = root / record["split"] / record["split_episode"]
            source_path.rename(destination_path)
            moved.append((source_path, destination_path))

        manifest_records = []
        for record in planned_records:
            manifest_records.append(
                {key: value for key, value in record.items() if key != "source_path"}
            )

        manifest = {
            "version": 1,
            "strategy": strategy,
            "seed": seed,
            "ratios": RATIOS,
            "episode_count": len(manifest_records),
            "episodes": manifest_records,
        }
        temporary_manifest = root / f".{SPLIT_MANIFEST}.tmp"
        with temporary_manifest.open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
            file.write("\n")
        temporary_manifest.replace(root / SPLIT_MANIFEST)

    except Exception:
        for source_path, destination_path in reversed(moved):
            if destination_path.exists() and not source_path.exists():
                destination_path.rename(source_path)
        for split_name in SPLIT_NAMES:
            split_path = root / split_name
            if split_path.is_dir() and not any(split_path.iterdir()):
                split_path.rmdir()
        raise


def print_summary(root: Path, splits: dict[str, list[dict]]) -> None:
    print(f"Split dataset in place: {root}")
    for split_name in SPLIT_NAMES:
        goals = Counter(episode["goal"] for episode in splits[split_name])
        sessions = {episode["source_session"] for episode in splits[split_name]}
        print(
            f"  {split_name:8s}: {len(splits[split_name]):5d} episodes, {len(sessions):4d} sessions"
        )
        for goal, count in sorted(goals.items()):
            print(f"      {count:5d}  {goal or '<empty>'}")
    print(f"Manifest: {root / SPLIT_MANIFEST}")


def main() -> None:
    args = parse_args()
    collection_mode = args.collection_check or args.collection_output is not None
    if collection_mode:
        if args.search_trials < 1:
            raise ValueError("--search-trials must be positive")
        compose_collection(
            args.root,
            args.collection_output,
            args.exclude_dataset,
            parse_preserved_splits(args.preserve_split),
            args.strategy,
            args.seed,
            args.search_trials,
        )
        return
    if args.exclude_dataset or args.preserve_split:
        raise ValueError(
            "--exclude-dataset and --preserve-split require --collection-check "
            "or --collection-output"
        )
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Root directory does not exist: {root}")
    if root == Path(root.anchor) or root == Path.home().resolve():
        raise ValueError(f"Refusing unsafe root directory: {root}")

    source_records = load_flatten_manifest(root)
    episodes = load_episodes(root, source_records)
    targets = target_sizes(len(episodes))

    if args.strategy == "session-grouped":
        splits = split_session_grouped(episodes, targets, args.seed, args.search_trials)
    else:
        splits = split_goal_stratified(episodes, targets, args.seed)

    apply_split(root, splits, args.strategy, args.seed)
    print_summary(root, splits)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
