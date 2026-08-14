#!/usr/bin/env python3
"""Flatten nested recording sessions into one sequential episode directory.

Example (run from /home/alex/Development/scripts/):
    cd /home/alex/Development/scripts/
    ./flatten_dataset_sessions.py /path/to/copied_root

This command destructively replaces ROOT using same-filesystem moves with
rollback. Use it only on a disposable dataset copy.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re
import shutil
import tempfile


EPISODE_PATTERN = re.compile(r"episode_(\d+)$")
MANIFEST_NAME = "flatten_manifest.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Destructively flatten every nested episode_* directory under ROOT "
            "into ROOT/episode_0001, ROOT/episode_0002, ..."
        )
    )
    parser.add_argument("root", type=Path, help="Disposable root containing session datasets")
    return parser.parse_args()


def validate_root(root: Path) -> Path:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Root directory does not exist: {root}")
    if root == Path(root.anchor) or root == Path.home().resolve():
        raise ValueError(f"Refusing unsafe root directory: {root}")
    return root


def find_episodes(root: Path) -> list[Path]:
    episodes: list[Path] = []

    for current_dir, directory_names, _ in os.walk(root, followlinks=False):
        directory_names.sort()
        current_path = Path(current_dir)

        if EPISODE_PATTERN.fullmatch(current_path.name):
            if not (current_path / "data.json").is_file():
                raise ValueError(f"Episode is missing data.json: {current_path}")
            episodes.append(current_path)
            directory_names[:] = []

    episodes.sort(
        key=lambda path: (
            str(path.parent.relative_to(root)).casefold(),
            int(EPISODE_PATTERN.fullmatch(path.name).group(1)),
        )
    )
    return episodes


def read_goal(episode_path: Path) -> str:
    json_path = episode_path / "data.json"
    try:
        with json_path.open(encoding="utf-8") as file:
            episode_json = json.load(file)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Could not read {json_path}: {error}") from error

    text_info = episode_json.get("text")
    goal = text_info.get("goal", "") if isinstance(text_info, dict) else ""
    return goal if isinstance(goal, str) else str(goal)


def main() -> None:
    args = parse_args()
    root = validate_root(args.root)
    episodes = find_episodes(root)

    if not episodes:
        raise ValueError(f"No episode_* directories containing data.json found under {root}")

    records = []
    goals = Counter()

    for new_index, episode_path in enumerate(episodes, start=1):
        goal = read_goal(episode_path)
        source_relative = episode_path.relative_to(root)
        session_relative = episode_path.parent.relative_to(root)
        flattened_name = f"episode_{new_index:04d}"

        records.append(
            {
                "flattened_episode": flattened_name,
                "source_episode": source_relative.as_posix(),
                "source_session": session_relative.as_posix(),
                "goal": goal,
            }
        )
        goals[goal] += 1

    staging_path = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.flattening.", dir=root.parent)
    )
    backup_path = root.parent / f".{root.name}.before_flatten"

    if backup_path.exists():
        shutil.rmtree(staging_path)
        raise ValueError(
            f"Refusing to overwrite recovery directory from an earlier run: {backup_path}"
        )

    moved: list[tuple[Path, Path]] = []
    replaced = False
    try:
        os.chmod(staging_path, root.stat().st_mode)

        for episode_path, record in zip(episodes, records, strict=True):
            destination_path = staging_path / record["flattened_episode"]
            episode_path.rename(destination_path)
            moved.append((episode_path, destination_path))

        manifest = {
            "version": 1,
            "source_root": str(root),
            "episode_count": len(records),
            "episodes": records,
        }
        with (staging_path / MANIFEST_NAME).open("w", encoding="utf-8") as file:
            json.dump(manifest, file, ensure_ascii=False, indent=2)
            file.write("\n")

        root.rename(backup_path)
        try:
            staging_path.rename(root)
            replaced = True
        except Exception:
            backup_path.rename(root)
            raise

        try:
            shutil.rmtree(backup_path)
        except OSError as error:
            print(f"WARNING: could not remove old wrappers at {backup_path}: {error}")

    finally:
        if not replaced and staging_path.exists():
            rollback_errors = []
            for source_path, destination_path in reversed(moved):
                if not destination_path.exists():
                    continue
                try:
                    source_path.parent.mkdir(parents=True, exist_ok=True)
                    destination_path.rename(source_path)
                except OSError as error:
                    rollback_errors.append(str(error))

            if rollback_errors:
                print(
                    f"WARNING: rollback was incomplete; recovery data remains at "
                    f"{staging_path}"
                )
            else:
                shutil.rmtree(staging_path)

    print(f"Flattened {len(records)} episodes in place: {root}")
    print(f"Recorded source-session mappings in: {root / MANIFEST_NAME}")
    print("Goals:")
    for goal, count in sorted(goals.items()):
        display_goal = goal or "<empty>"
        print(f"  {count:5d}  {display_goal}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}") from error
