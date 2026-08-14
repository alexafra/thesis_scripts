#!/usr/bin/env python3
"""Move/copy episodes from one or more folders into a destination and renumber them.

Example (run from /home/alex/Development/scripts/):
    cd /home/alex/Development/scripts/
    ./simple_append.py /path/to/destination_folder /path/to/source1 /path/to/source2

Episodes are copied by default. To remove them from the sources after appending:
    ./simple_append.py /path/to/destination_folder /path/to/source1 /path/to/source2 --move
"""
from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil

EPISODE_PATTERN = re.compile(r"^episode_(\d+)$")


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


def append_episodes(destination: Path, sources: list[Path], copy: bool) -> None:
    destination = destination.expanduser().resolve()

    if not destination.is_dir():
        raise ValueError(f"Destination folder does not exist: {destination}")

    resolved_sources: list[Path] = []
    for source in sources:
        source = source.expanduser().resolve()
        if not source.is_dir():
            raise ValueError(f"Source folder does not exist: {source}")
        if (
            destination == source
            or destination in source.parents
            or source in destination.parents
        ):
            raise ValueError(f"Destination and source folders must not overlap: {source}")
        resolved_sources.append(source)

    destination_episodes = direct_episodes(destination)

    # Collect episodes from all sources in the order given
    all_source_episodes: list[Path] = []
    for source in resolved_sources:
        source_episodes = direct_episodes(source)
        if not source_episodes:
            raise ValueError(f"No direct episode_* directories found in {source}")
        all_source_episodes.extend(source_episodes)

    # Validate everything
    for episode_path in destination_episodes + all_source_episodes:
        validate_episode(episode_path)

    existing_numbers = [
        int(EPISODE_PATTERN.fullmatch(path.name).group(1))
        for path in destination_episodes
    ]
    first_number = max(existing_numbers, default=0) + 1
    final_number = first_number + len(all_source_episodes) - 1
    width = max(4, len(str(final_number)))

    planned = [
        (
            source_path,
            destination / f"episode_{first_number + offset:0{width}d}",
        )
        for offset, source_path in enumerate(all_source_episodes)
    ]

    completed = []
    try:
        for source_path, destination_path in planned:
            if destination_path.exists():
                raise ValueError(f"Refusing to overwrite: {destination_path}")
            if copy:
                shutil.copytree(source_path, destination_path)
            else:
                shutil.move(str(source_path), str(destination_path))
            completed.append((source_path, destination_path))
    except Exception:
        # Roll back on failure
        for source_path, destination_path in reversed(completed):
            if not destination_path.exists():
                continue
            if copy:
                shutil.rmtree(destination_path)
            elif not source_path.exists():
                shutil.move(str(destination_path), str(source_path))
        raise

    operation = "Copied" if copy else "Moved"
    print(
        f"{operation} {len(planned)} episode(s) from {len(resolved_sources)} source folder(s) "
        f"into {destination} as episode_{first_number:0{width}d} through "
        f"episode_{final_number:0{width}d}."
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Append episode_* folders from one or more source folders into a destination. "
            "Incoming episodes are renumbered after the destination's highest existing number."
        )
    )
    parser.add_argument(
        "destination",
        type=Path,
        help="Existing folder that will receive episodes",
    )
    parser.add_argument(
        "sources",
        type=Path,
        nargs="+",
        help="One or more existing folders containing episodes to append",
    )
    parser.add_argument(
        "--move",
        action="store_true",
        help="Move episodes instead of copying them",
    )
    args = parser.parse_args()

    try:
        append_episodes(args.destination, args.sources, copy=not args.move)
    except (OSError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error


if __name__ == "__main__":
    main()