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
from pathlib import Path
import random
import re


EPISODE_PATTERN = re.compile(r"episode_(\d+)$")
FLATTEN_MANIFEST = "flatten_manifest.json"
SPLIT_MANIFEST = "split_manifest.json"
SPLIT_NAMES = ("train", "test", "validation")
RATIOS = {"train": 0.8, "test": 0.1, "validation": 0.1}


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
    return parser.parse_args()


def target_sizes(total: int) -> dict[str, int]:
    if total < 3:
        raise ValueError("At least 3 episodes are required to create three non-empty splits")

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
    size_error = sum(
        abs(split_counts[name] - targets[name]) for name in SPLIT_NAMES
    ) / total

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
        for group_name, split_name in zip(shuffled_groups[:3], anchor_splits, strict=True):
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
    counts = {
        goal: min(int(quotas[goal]), capacities[goal]) for goal in group_sizes
    }

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
    remaining_capacities = {
        goal: sizes[goal] - test_counts[goal] for goal in sizes
    }
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


def apply_split(root: Path, splits: dict[str, list[dict]], strategy: str, seed: int) -> None:
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
            f"  {split_name:8s}: {len(splits[split_name]):5d} episodes, "
            f"{len(sessions):4d} sessions"
        )
        for goal, count in sorted(goals.items()):
            print(f"      {count:5d}  {goal or '<empty>'}")
    print(f"Manifest: {root / SPLIT_MANIFEST}")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Root directory does not exist: {root}")
    if root == Path(root.anchor) or root == Path.home().resolve():
        raise ValueError(f"Refusing unsafe root directory: {root}")

    source_records = load_flatten_manifest(root)
    episodes = load_episodes(root, source_records)
    targets = target_sizes(len(episodes))

    if args.strategy == "session-grouped":
        splits = split_session_grouped(
            episodes, targets, args.seed, args.search_trials
        )
    else:
        splits = split_goal_stratified(episodes, targets, args.seed)

    apply_split(root, splits, args.strategy, args.seed)
    print_summary(root, splits)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
