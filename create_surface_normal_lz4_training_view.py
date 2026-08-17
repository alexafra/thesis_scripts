#!/usr/bin/env python3
"""Create a non-destructive LeRobot view over the 10-episode LZ4 corpus."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil


DEFAULT_SOURCE = Path(
    "/home/alex/Development/Datasets/lerobot2/"
    "atomic_combined_09_08_And_10_08/train"
)
DEFAULT_CORPUS = Path(
    "/home/alex/Development/Datasets/lerobot2/"
    "atomic_combined_09_08_And_10_08_testing_surface_normal_compression"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--target", type=Path)
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def link(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(os.path.relpath(source, destination.parent))


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    corpus = args.corpus.resolve()
    target = (
        args.target.resolve()
        if args.target is not None
        else corpus / "train_plain_lz4"
    )
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite existing target: {target}")

    corpus_manifest = json.loads((corpus / "manifest.json").read_text())
    selected_ids = [int(value) for value in corpus_manifest["selected_episode_indices"]]
    selected_set = set(selected_ids)
    if len(selected_ids) != 10 or len(selected_set) != 10:
        raise ValueError(f"Expected ten unique episodes; found {selected_ids}")

    source_meta = source / "meta"
    info = json.loads((source_meta / "info.json").read_text())
    episodes = [
        json.loads(line)
        for line in (source_meta / "episodes.jsonl").read_text().splitlines()
    ]
    episodes = [
        episode
        for episode in episodes
        if int(episode["episode_index"]) in selected_set
    ]
    episodes.sort(key=lambda episode: selected_ids.index(int(episode["episode_index"])))
    if {int(episode["episode_index"]) for episode in episodes} != selected_set:
        raise RuntimeError("Selected episode metadata is incomplete")

    staging = target.parent / f".{target.name}.building-{os.getpid()}"
    if staging.exists():
        raise FileExistsError(staging)
    staging.mkdir()

    try:
        (staging / "meta").mkdir()
        for name in ("tasks.jsonl", "modality.json", "stats.json", "relative_stats.json"):
            shutil.copy2(source_meta / name, staging / "meta" / name)

        with (staging / "meta" / "episodes.jsonl").open("w", encoding="utf-8") as stream:
            for episode in episodes:
                stream.write(json.dumps(episode) + "\n")

        source_episode_stats = source_meta / "episodes_stats.jsonl"
        if source_episode_stats.exists():
            selected_stats = [
                json.loads(line)
                for line in source_episode_stats.read_text().splitlines()
                if int(json.loads(line)["episode_index"]) in selected_set
            ]
            selected_stats.sort(
                key=lambda item: selected_ids.index(int(item["episode_index"]))
            )
            with (staging / "meta" / "episodes_stats.jsonl").open(
                "w", encoding="utf-8"
            ) as stream:
                for item in selected_stats:
                    stream.write(json.dumps(item) + "\n")

        chunks_size = int(info["chunks_size"])
        video_count = 0
        for episode in episodes:
            episode_index = int(episode["episode_index"])
            chunk_name = f"chunk-{episode_index // chunks_size:03d}"
            episode_name = f"episode_{episode_index:06d}"

            link(
                source / "data" / chunk_name / f"{episode_name}.parquet",
                staging / "data" / chunk_name / f"{episode_name}.parquet",
            )

            source_video_chunk = source / "videos" / chunk_name
            for video_directory in sorted(source_video_chunk.iterdir()):
                if video_directory.name == "observation.images.surface_normals_view":
                    # Deliberately omit the canonical normals MP4 so this test cannot
                    # silently fall back to H.264 if the LZ4 loader hook is bypassed.
                    continue
                source_video = video_directory / f"{episode_name}.mp4"
                if source_video.is_file():
                    link(
                        source_video,
                        staging
                        / "videos"
                        / chunk_name
                        / video_directory.name
                        / f"{episode_name}.mp4",
                    )
                    video_count += 1

            for depth_root in ("raw_depths", "aligned_depths"):
                source_episode_dir = source / depth_root / chunk_name / episode_name
                if source_episode_dir.is_dir():
                    link(
                        source_episode_dir,
                        staging / depth_root / chunk_name / episode_name,
                    )

        total_frames = sum(int(episode["length"]) for episode in episodes)
        info["total_episodes"] = len(episodes)
        info["total_frames"] = total_frames
        info["total_videos"] = video_count
        info["total_chunks"] = len(
            {int(episode["episode_index"]) // chunks_size for episode in episodes}
        )
        info["splits"] = {"train": f"0:{len(episodes)}"}
        for key in ("raw_depth_encoding", "aligned_depth_encoding"):
            if key in info:
                info[key]["total_files"] = total_frames
        info["surface_normals_lz4"] = {
            "feature_key": "observation.images.surface_normals_view",
            "storage": "plain_lz4_chunks",
            "root": os.path.relpath(corpus / "02_plain_lz4_chunks", target),
            "dtype": "uint8",
            "layout": "FHWC",
            "chunk_frames": int(corpus_manifest["lz4_chunk_frames"]),
            "lossless_round_trip_verified": bool(
                corpus_manifest["all_lossless_round_trips_verified"]
            ),
        }
        write_json(staging / "meta" / "info.json", info)
        write_json(
            staging / "lz4_training_view.json",
            {
                "source_dataset": str(source),
                "source_dataset_modified": False,
                "normal_storage": str(corpus / "02_plain_lz4_chunks"),
                "episode_indices": selected_ids,
                "episodes": len(episodes),
                "frames": total_frames,
                "linked_videos": video_count,
            },
        )
        (staging / "README.md").write_text(
            "# Plain-LZ4 surface-normal training view\n\n"
            "This is a non-destructive ten-episode LeRobot view. Parquet, RGB/depth "
            "videos, and depth directories are relative symlinks to the canonical "
            "training dataset. Only `surface_normals_view` is overridden by "
            "`meta/info.json` to load the verified plain-LZ4 chunks.\n",
            encoding="utf-8",
        )

        os.replace(staging, target)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    print(f"Created: {target}")
    print(f"Episodes: {len(episodes)}")
    print(f"Frames: {total_frames}")
    print(f"Linked videos: {video_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
