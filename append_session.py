#!/usr/bin/env python3
"""Merge partial train/test/validation roots into the first root.

Example (run from /home/alex/Development/scripts/):
    cd /home/alex/Development/scripts/
    ./append_session.py /path/to/destination_root /path/to/source_root

The destination may be empty, and more than one source may be supplied:
    ./append_session.py /path/to/destination_root /path/to/root_1 /path/to/root_2

Episodes are copied by default. To remove them from the sources, add --move.
Each root may contain any subset of train, test, and validation. The legacy
name "validate" is also accepted.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
import re
import shutil


EPISODE_PATTERN = re.compile(r"^episode_(\d+)$")
LOGICAL_SPLITS = ("train", "test", "validation")


def direct_episodes(directory: Path) -> list[Path]:
    episodes = [
        path
        for path in directory.iterdir()
        if path.is_dir() and EPISODE_PATTERN.fullmatch(path.name)
    ]
    return sorted(
        episodes,
        key=lambda path: int(EPISODE_PATTERN.fullmatch(path.name).group(1)),
    )


def validate_episode(episode_path: Path) -> None:
    if not (episode_path / "data.json").is_file():
        raise ValueError(f"Episode is missing data.json: {episode_path}")


def find_split_directories(root: Path, allow_empty: bool) -> dict[str, Path]:
    split_directories = {}
    for split_name in ("train", "test"):
        split_path = root / split_name
        if split_path.exists():
            if not split_path.is_dir():
                raise ValueError(f"Expected a directory: {split_path}")
            split_directories[split_name] = split_path

    validation_paths = [
        root / name for name in ("validation", "validate") if (root / name).exists()
    ]
    if len(validation_paths) > 1:
        raise ValueError(
            f"{root} contains both validation/ and validate/; keep only one"
        )
    if validation_paths:
        if not validation_paths[0].is_dir():
            raise ValueError(f"Expected a directory: {validation_paths[0]}")
        split_directories["validation"] = validation_paths[0]

    if not allow_empty and not split_directories:
        raise ValueError(
            f"No train/, test/, validation/, or validate/ directory found in {root}"
        )
    return split_directories


def validate_roots(
    destination_root: Path,
    source_roots: list[Path],
) -> tuple[Path, list[Path]]:
    destination_root = destination_root.expanduser().resolve()
    source_roots = [path.expanduser().resolve() for path in source_roots]

    if not destination_root.is_dir():
        raise ValueError(f"Destination root does not exist: {destination_root}")
    if len(set(source_roots)) != len(source_roots):
        raise ValueError("The same source root was supplied more than once")

    for source_root in source_roots:
        if not source_root.is_dir():
            raise ValueError(f"Source root does not exist: {source_root}")
        if (
            destination_root == source_root
            or destination_root in source_root.parents
            or source_root in destination_root.parents
        ):
            raise ValueError(
                "Destination and source roots must not overlap: "
                f"{destination_root}, {source_root}"
            )
    return destination_root, source_roots


def destination_split_paths(destination_root: Path) -> dict[str, Path]:
    existing = find_split_directories(destination_root, allow_empty=True)
    paths = {}
    for split_name in LOGICAL_SPLITS:
        if split_name in existing:
            paths[split_name] = existing[split_name]
        else:
            paths[split_name] = destination_root / split_name
    return paths


def merge_session_roots(
    destination_root: Path,
    source_roots: list[Path],
    copy: bool,
) -> None:
    destination_root, source_roots = validate_roots(destination_root, source_roots)
    destination_paths = destination_split_paths(destination_root)
    incoming = defaultdict(list)

    for source_root in source_roots:
        for split_name, split_path in find_split_directories(
            source_root, allow_empty=False
        ).items():
            episodes = direct_episodes(split_path)
            for episode_path in episodes:
                validate_episode(episode_path)
            incoming[split_name].extend(episodes)

    if not any(incoming.values()):
        raise ValueError("The source roots do not contain any episode_* directories")

    planned = []
    created_split_paths = []
    for split_name in LOGICAL_SPLITS:
        source_episodes = incoming[split_name]
        if not source_episodes:
            continue

        destination_path = destination_paths[split_name]
        if destination_path.exists():
            existing_episodes = direct_episodes(destination_path)
            for episode_path in existing_episodes:
                validate_episode(episode_path)
        else:
            existing_episodes = []

        existing_numbers = [
            int(EPISODE_PATTERN.fullmatch(path.name).group(1))
            for path in existing_episodes
        ]
        first_number = max(existing_numbers, default=0) + 1
        final_number = first_number + len(source_episodes) - 1
        width = max(4, len(str(final_number)))

        for offset, source_path in enumerate(source_episodes):
            planned.append(
                (
                    split_name,
                    source_path,
                    destination_path / f"episode_{first_number + offset:0{width}d}",
                )
            )

    completed = []
    try:
        for split_name in LOGICAL_SPLITS:
            if not incoming[split_name]:
                continue
            destination_path = destination_paths[split_name]
            if not destination_path.exists():
                destination_path.mkdir()
                created_split_paths.append(destination_path)

        for _, source_path, destination_path in planned:
            if destination_path.exists():
                raise ValueError(f"Refusing to overwrite: {destination_path}")
            if copy:
                shutil.copytree(source_path, destination_path)
            else:
                shutil.move(str(source_path), str(destination_path))
            completed.append((source_path, destination_path))

    except Exception:
        for source_path, destination_path in reversed(completed):
            if not destination_path.exists():
                continue
            if copy:
                shutil.rmtree(destination_path)
            elif not source_path.exists():
                source_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(destination_path), str(source_path))
        for split_path in reversed(created_split_paths):
            if split_path.is_dir() and not any(split_path.iterdir()):
                split_path.rmdir()
        raise

    operation = "Copied" if copy else "Moved"
    print(
        f"{operation} episodes from {len(source_roots)} source root(s) into "
        f"{destination_root}:"
    )
    for split_name in LOGICAL_SPLITS:
        count = len(incoming[split_name])
        if count:
            print(f"  {split_name:10s} +{count:5d} episodes")

    manifest_path = destination_root / "split_manifest.json"
    if manifest_path.exists():
        print(
            f"Warning: {manifest_path} was not merged and may no longer describe "
            "every episode."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Make the first raw split root absorb one or more partial "
            "train/test/validation roots. Existing destination episodes remain fixed."
        )
    )
    parser.add_argument(
        "destination_root",
        type=Path,
        help="Existing root that receives episodes; it may be empty",
    )
    parser.add_argument(
        "source_roots",
        type=Path,
        nargs="+",
        help="One or more roots containing any subset of the splits",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move episodes instead of copying them",
    )
    args = parser.parse_args()

    try:
        merge_session_roots(
            args.destination_root,
            args.source_roots,
            copy=not args.move,
        )
    except (OSError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()
