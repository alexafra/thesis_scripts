#!/usr/bin/env python3

import argparse
import os
import shutil
from pathlib import Path


def make_symlink_dir_real(path: Path):
    target = path.resolve(strict=True)

    if not target.is_dir():
        return

    print(f"Converting: {path}")
    print(f"        -> {target}")

    temp = path.parent / f".{path.name}.real_tmp"

    if temp.exists():
        shutil.rmtree(temp)

    shutil.copytree(
        target,
        temp,
        symlinks=False,   # follow symlinks inside target and make them real too
        copy_function=shutil.copy2,
    )

    path.unlink()
    temp.rename(path)

    print(f"Done: {path}")


def convert_tree(root: Path):
    root = root.expanduser().resolve()

    if not root.is_dir():
        raise ValueError(f"Not a directory: {root}")

    # Bottom-up so nested symlinks are handled safely.
    for current, dirs, files in os.walk(root, topdown=False, followlinks=False):
        current = Path(current)

        for name in dirs:
            path = current / name

            if path.is_symlink():
                make_symlink_dir_real(path)


def main():
    parser = argparse.ArgumentParser(
        description="Recursively convert symlinked child directories into real directories."
    )
    parser.add_argument("folder", help="Root folder to process")
    args = parser.parse_args()

    convert_tree(Path(args.folder))
    print("\nAll symlinked child directories converted.")


if __name__ == "__main__":
    main()