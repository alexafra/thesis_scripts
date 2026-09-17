#!/usr/bin/env python3
"""Measure whether a GR00T policy uses correctly aligned geometry.

It evaluates paired observations whose RGB, robot state, language, target
actions, diffusion noise, and inference frame are identical.  The intervention
changes only ``depth_gray_view`` or ``surface_normals_view`` using a same-task
cross-episode donor, a same-episode temporal offset, or an all-zero geometry
frame.

Dataset files are read-only.  Geometry replacement happens only in memory.
Positive counterfactual-minus-intact error means the intact geometry helped
open-loop action prediction.  Prediction change measures sensitivity, which is
not by itself evidence of benefit or closed-loop task success.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import hashlib
import json
import logging
from pathlib import Path
import random
import re
import subprocess
import time
from typing import Any, Iterable

import gr00t
from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import ModalityConfig
from gr00t.data.utils import parse_observation_gr00t
from gr00t.eval._horizon_contract import PolicyHorizonSpec
from gr00t.eval.open_loop_eval import batch_policy_observations
from gr00t.policy.gr00t_policy import Gr00tPolicy
import numpy as np
import pandas as pd
import torch


LOGGER = logging.getLogger("geometry_correspondence_evaluation")
GEOMETRY_KEYS = ("depth_gray_view", "surface_normals_view")
CROSS_EPISODE_INTERVENTIONS = ("phase_matched", "out_of_phase")
SAME_EPISODE_OFFSETS = {"offset_10pct": 0.10, "offset_50pct": 0.50}
INTERVENTIONS = (
    *CROSS_EPISODE_INTERVENTIONS,
    *SAME_EPISODE_OFFSETS,
    "zero_geometry",
)
CONDITIONS = ("intact", "counterfactual")


@dataclass(frozen=True)
class EvaluationTarget:
    step: int
    model_path: Path


@dataclass(frozen=True)
class EpisodeRecord:
    loader_position: int
    episode_index: int
    length: int
    tasks: tuple[str, ...]
    task: str
    cohort: str
    source: str | None = None
    source_episode: str | None = None
    source_data_json_sha256: str | None = None


@dataclass(frozen=True)
class DonorAssignment:
    shuffle_repeat: int
    task: str
    recipient_loader_position: int
    recipient_episode_index: int
    recipient_length: int
    donor_loader_position: int
    donor_episode_index: int
    donor_length: int


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare paired intact and counterfactual geometry on a held-out "
            "LeRobot evaluation split."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--geometry-key", choices=GEOMETRY_KEYS, required=True)
    parser.add_argument(
        "--intervention", choices=INTERVENTIONS, default="phase_matched"
    )
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--checkpoint-steps", type=int, nargs="*")
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument(
        "--task",
        action="append",
        help="Exact held-out task to include; repeat for multiple tasks. Omit for all tasks.",
    )
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument(
        "--pair-batch-size",
        type=int,
        default=4,
        help=(
            "Number of intact/counterfactual pairs per inference call (model batch is twice this)."
        ),
    )
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--inference-seed", type=int, default=42)
    parser.add_argument("--shuffle-seed", type=int, default=42)
    parser.add_argument("--shuffle-repeats", type=int, default=1)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=0,
        help="Deterministic smoke-test cap on recipient episodes; 0 evaluates all.",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    if args.execution_horizon <= 0:
        parser.error("--execution-horizon must be positive")
    if args.pair_batch_size <= 0:
        parser.error("--pair-batch-size must be positive")
    if args.denoising_steps <= 0:
        parser.error("--denoising-steps must be positive")
    if args.shuffle_repeats <= 0:
        parser.error("--shuffle-repeats must be positive")
    if args.bootstrap_replicates < 0:
        parser.error("--bootstrap-replicates must be non-negative")
    if args.max_episodes < 0:
        parser.error("--max-episodes must be non-negative")
    if args.task:
        args.task = list(dict.fromkeys(value.strip() for value in args.task))
        if any(not value for value in args.task):
            parser.error("--task values must not be empty")
    if args.split == "test" and (
        args.checkpoint_steps is None or len(set(args.checkpoint_steps)) != 1
    ):
        parser.error("test evaluation requires exactly one preselected checkpoint")
    if args.output_dir.exists():
        parser.error(f"--output-dir already exists: {args.output_dir}")
    return args


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match is None:
        raise ValueError(f"Not a checkpoint directory: {path}")
    return int(match.group(1))


def find_targets(
    run_dir: Path, selected_steps: list[int] | None
) -> list[EvaluationTarget]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Training run does not exist: {run_dir}")
    if re.fullmatch(r"checkpoint-\d+", run_dir.name):
        checkpoints = [run_dir]
    else:
        checkpoints = sorted(
            (path for path in run_dir.glob("checkpoint-*") if path.is_dir()),
            key=checkpoint_step,
        )
    selected = set(selected_steps) if selected_steps else None
    targets = [
        EvaluationTarget(checkpoint_step(path), path)
        for path in checkpoints
        if selected is None or checkpoint_step(path) in selected
    ]
    if selected is not None:
        missing = selected - {target.step for target in targets}
        if missing:
            raise FileNotFoundError(f"Missing checkpoint step(s): {sorted(missing)}")
    if not targets:
        raise FileNotFoundError(f"No checkpoints found under {run_dir}")
    return targets


def _stable_seed(base_seed: int, *parts: Any) -> int:
    payload = json.dumps([int(base_seed), *parts], sort_keys=True, default=str).encode(
        "utf-8"
    )
    return int.from_bytes(
        hashlib.blake2b(payload, digest_size=8).digest(), "little"
    ) % (2**63)


def _task_tuple(metadata: dict[str, Any]) -> tuple[str, ...]:
    tasks = tuple(
        dict.fromkeys(str(task) for task in metadata.get("tasks", []) if str(task))
    )
    if not tasks:
        raise ValueError(f"Episode {metadata.get('episode_index')} has no task text")
    return tasks


def _task_label(tasks: tuple[str, ...]) -> str:
    return " / ".join(tasks)


def _provenance_map(dataset_path: Path, split: str) -> dict[int, dict[str, Any]]:
    path = dataset_path.parent / "provenance" / "merge_manifest.json"
    if not path.is_file():
        LOGGER.warning(
            "No merge provenance manifest at %s; all episodes labelled base", path
        )
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    result = {}
    for row in payload.get("episodes", []):
        if str(row.get("split")) != split:
            continue
        index = int(row["final_episode_index"])
        if index in result:
            raise ValueError(f"Duplicate provenance for {split} episode {index}")
        result[index] = row
    return result


def _cohort_for(provenance: dict[str, Any] | None) -> str:
    if provenance is None:
        return "base"
    source = str(provenance.get("source", "unknown"))
    if source == "pick_three_cups_right_only_1408":
        return "right_only_1408"
    if source == "stack_cups_09_08":
        return "stack_09_08"
    return source


def episode_records(
    loader: LeRobotEpisodeLoader,
    *,
    split: str,
    tasks: Iterable[str] | None,
) -> list[EpisodeRecord]:
    provenance = _provenance_map(loader.dataset_path, split)
    task_filter = set(tasks or [])
    records = []
    for position, metadata in enumerate(loader.episodes_metadata):
        episode_tasks = _task_tuple(metadata)
        label = _task_label(episode_tasks)
        if task_filter and label not in task_filter:
            continue
        episode_index = int(metadata["episode_index"])
        source = provenance.get(episode_index)
        records.append(
            EpisodeRecord(
                loader_position=position,
                episode_index=episode_index,
                length=int(metadata["length"]),
                tasks=episode_tasks,
                task=label,
                cohort=_cohort_for(source),
                source=(str(source["source"]) if source else None),
                source_episode=(str(source["source_episode"]) if source else None),
                source_data_json_sha256=(
                    str(source["data_json_sha256"])
                    if source and source.get("data_json_sha256")
                    else None
                ),
            )
        )
    if not records:
        raise ValueError(f"No {split} episodes match task filter {sorted(task_filter)}")
    return records


def build_donor_assignments(
    records: list[EpisodeRecord],
    *,
    shuffle_seed: int,
    shuffle_repeat: int,
) -> dict[int, DonorAssignment]:
    """Build a deterministic within-task cyclic derangement."""

    by_task: dict[tuple[str, ...], list[EpisodeRecord]] = defaultdict(list)
    for record in records:
        by_task[record.tasks].append(record)
    assignments = {}
    for task_key, group in sorted(by_task.items(), key=lambda item: item[0]):
        group = sorted(group, key=lambda record: record.episode_index)
        if len(group) < 2:
            raise ValueError(
                f"Task {_task_label(task_key)!r} has {len(group)} held-out episode; "
                "same-task cross-episode shuffling requires at least two"
            )
        seed = _stable_seed(shuffle_seed, "donor", shuffle_repeat, task_key)
        rng = np.random.default_rng(seed)
        order = [group[int(index)] for index in rng.permutation(len(group))]
        for index, recipient in enumerate(order):
            donor = order[(index + 1) % len(order)]
            if donor.episode_index == recipient.episode_index:
                raise RuntimeError("Donor derangement produced a self-pair")
            assignments[recipient.loader_position] = DonorAssignment(
                shuffle_repeat=shuffle_repeat,
                task=recipient.task,
                recipient_loader_position=recipient.loader_position,
                recipient_episode_index=recipient.episode_index,
                recipient_length=recipient.length,
                donor_loader_position=donor.loader_position,
                donor_episode_index=donor.episode_index,
                donor_length=donor.length,
            )
    if len(assignments) != len(records):
        raise RuntimeError("Donor assignment did not cover every eligible episode")
    return assignments


def donor_progress_for_phase(
    target_frame: int,
    target_length: int,
    *,
    donor_phase: str,
) -> float:
    if target_length <= 0:
        raise ValueError("Episode lengths must be positive")
    if not 0 <= target_frame < target_length:
        raise IndexError(
            f"Target frame {target_frame} is outside 0..{target_length - 1}"
        )
    target_progress = target_frame / (target_length - 1) if target_length > 1 else 0.0
    if donor_phase == "phase_matched":
        return target_progress
    if donor_phase == "out_of_phase":
        return (target_progress + 0.5) % 1.0
    raise ValueError(f"Unknown donor phase mode: {donor_phase}")


def donor_frame_for_phase(
    target_frame: int,
    target_length: int,
    donor_length: int,
    *,
    donor_phase: str,
) -> int:
    if donor_length <= 0:
        raise ValueError("Episode lengths must be positive")
    donor_progress = donor_progress_for_phase(
        target_frame,
        target_length,
        donor_phase=donor_phase,
    )
    if donor_length == 1:
        return 0
    mapped = int(np.rint(donor_progress * (donor_length - 1)))
    return min(max(mapped, 0), donor_length - 1)


def phase_matched_frame(
    target_frame: int, target_length: int, donor_length: int
) -> int:
    return donor_frame_for_phase(
        target_frame,
        target_length,
        donor_length,
        donor_phase="phase_matched",
    )


def same_episode_progress_for_offset(
    target_frame: int,
    target_length: int,
    *,
    intervention: str,
) -> float:
    """Return normalized progress for a discrete circular same-episode shift."""

    mapped_frame = same_episode_frame_for_offset(
        target_frame,
        target_length,
        intervention=intervention,
    )
    return mapped_frame / (target_length - 1) if target_length > 1 else 0.0


def same_episode_shift_frames(target_length: int, *, intervention: str) -> int:
    """Resolve a fractional episode offset to its nearest integer frame shift."""

    if target_length <= 0:
        raise ValueError("Episode lengths must be positive")
    try:
        offset = SAME_EPISODE_OFFSETS[intervention]
    except KeyError as exc:
        raise ValueError(
            f"Unknown same-episode offset intervention: {intervention}"
        ) from exc
    return int(round(offset * target_length))


def same_episode_frame_for_offset(
    target_frame: int,
    target_length: int,
    *,
    intervention: str,
) -> int:
    """Return the same-episode frame after a discrete circular frame shift."""

    if target_length <= 0:
        raise ValueError("Episode lengths must be positive")
    if not 0 <= target_frame < target_length:
        raise IndexError(
            f"Target frame {target_frame} is outside 0..{target_length - 1}"
        )
    shift = same_episode_shift_frames(
        target_length,
        intervention=intervention,
    )
    return (target_frame + shift) % target_length


def _validate_contract(
    modality: dict[str, ModalityConfig],
    *,
    geometry_key: str,
    execution_horizon: int,
) -> int:
    spec = PolicyHorizonSpec.from_modality_config(
        modality, n_action_steps=execution_horizon
    )
    video = modality.get("video")
    if video is None:
        raise ValueError("Policy has no video modality")
    expected_keys = ["ego_view", geometry_key]
    if list(video.modality_keys) != expected_keys:
        raise ValueError(
            f"Expected early-fusion video keys {expected_keys}, got {video.modality_keys}"
        )
    if [int(value) for value in video.delta_indices] != [0]:
        raise ValueError(
            f"Geometry shuffle requires video delta_indices=[0], got {video.delta_indices}"
        )
    if video.channel_fusion is None:
        raise ValueError("Expected an early-fusion channel contract")
    source_keys = [source.key for source in video.channel_fusion]
    if source_keys != expected_keys:
        raise ValueError(f"Unexpected channel-fusion sources: {source_keys}")
    expected_channels = 4 if geometry_key == "depth_gray_view" else 6
    if video.vision_input_channels != expected_channels:
        raise ValueError(
            f"Expected {expected_channels} fused channels for {geometry_key}, "
            f"got {video.vision_input_channels}"
        )
    for name, config in modality.items():
        if name == "action":
            continue
        positive = [int(delta) for delta in config.delta_indices if int(delta) > 0]
        if positive:
            raise ValueError(f"Input modality {name} leaks future indices {positive}")
    return spec.action_horizon


def _donor_modality(
    modality: dict[str, ModalityConfig], geometry_key: str
) -> dict[str, ModalityConfig]:
    return {
        "video": ModalityConfig(delta_indices=[0], modality_keys=[geometry_key]),
        "state": deepcopy(modality["state"]),
    }


def _flat_observation(
    trajectory: pd.DataFrame,
    anchor: int,
    modality: dict[str, ModalityConfig],
    embodiment_tag: EmbodimentTag,
) -> dict[str, Any]:
    inputs = deepcopy(modality)
    inputs.pop("action")
    point = extract_step_data(trajectory, anchor, inputs, embodiment_tag)
    flat: dict[str, Any] = {}
    for key, value in point.states.items():
        flat[f"state.{key}"] = value
    for key, value in point.images.items():
        flat[f"video.{key}"] = np.asarray(value)
    for key in modality["language"].modality_keys:
        flat[key] = point.text
    return flat


def replace_geometry(
    flat: dict[str, Any], geometry_key: str, replacement_frame: Any
) -> dict[str, Any]:
    """Return a shallow copy with exactly one batched video tensor replaced."""

    column = f"video.{geometry_key}"
    if column not in flat:
        raise KeyError(f"Target observation is missing {column}")
    replacement = np.asarray([replacement_frame])
    target = np.asarray(flat[column])
    if replacement.shape != target.shape:
        raise ValueError(
            f"Replacement {geometry_key} shape {replacement.shape} "
            f"does not match target {target.shape}"
        )
    if replacement.dtype != target.dtype:
        replacement = replacement.astype(target.dtype, copy=False)
    replaced = dict(flat)
    replaced[column] = replacement
    if any(replaced[key] is not value for key, value in flat.items() if key != column):
        raise RuntimeError(
            "Geometry intervention copied or changed a non-geometry input"
        )
    return replaced


def _paired_observations(
    *,
    target_trajectory: pd.DataFrame,
    target_frame: int,
    replacement_frame: Any,
    modality: dict[str, ModalityConfig],
    embodiment_tag: EmbodimentTag,
    geometry_key: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    flat = _flat_observation(target_trajectory, target_frame, modality, embodiment_tag)
    counterfactual = replace_geometry(flat, geometry_key, replacement_frame)
    return parse_observation_gr00t(flat, modality), parse_observation_gr00t(
        counterfactual, modality
    )


def zero_geometry_frame(target_frame: Any) -> np.ndarray:
    """Create an all-zero frame with the exact shape and dtype of a geometry frame."""

    target = np.asarray(target_frame)
    return np.zeros_like(target)


def _stack_columns(frame: pd.DataFrame, prefix: str, keys: Iterable[str]) -> np.ndarray:
    arrays = []
    for key in keys:
        column = f"{prefix}.{key}"
        if column not in frame:
            raise ValueError(f"Episode is missing {column}")
        arrays.append(
            np.vstack(
                [
                    np.asarray(value, dtype=np.float64).reshape(-1)
                    for value in frame[column]
                ]
            )
        )
    values = np.concatenate(arrays, axis=1)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"Non-finite values in {prefix} columns")
    return values


def _action_schema(
    trajectory: pd.DataFrame, action_keys: list[str]
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray]]:
    actions = _stack_columns(trajectory, "action", action_keys)
    labels = []
    slices = {}
    cursor = 0
    for key in action_keys:
        width = np.asarray(trajectory[f"action.{key}"].iloc[0]).reshape(-1).size
        slices[key] = np.arange(cursor, cursor + width, dtype=np.int64)
        labels.extend(f"{key}[{index}]" for index in range(width))
        cursor += width
    groups: dict[str, np.ndarray] = {"all": np.arange(cursor, dtype=np.int64), **slices}
    selectors = {
        "arms": lambda key: "arm" in key,
        "hands": lambda key: "hand" in key,
        "left_side": lambda key: key.startswith("left_"),
        "right_side": lambda key: key.startswith("right_"),
    }
    for name, predicate in selectors.items():
        selected = [slices[key] for key in action_keys if predicate(key)]
        if selected:
            groups[name] = np.concatenate(selected)
    return actions, labels, groups


def _chunk_array(
    chunks: dict[str, Any], batch_index: int, action_keys: list[str]
) -> np.ndarray:
    arrays = []
    horizon = None
    for key in action_keys:
        value = np.asarray(chunks[key][batch_index], dtype=np.float64)
        if value.ndim == 1:
            value = value[:, None]
        if value.ndim != 2:
            raise ValueError(f"Policy action {key} has unexpected shape {value.shape}")
        if horizon is not None and len(value) != horizon:
            raise ValueError("Policy action groups have different horizons")
        horizon = len(value)
        arrays.append(value)
    result = np.concatenate(arrays, axis=1)
    if not np.all(np.isfinite(result)):
        raise ValueError("Policy returned non-finite physical actions")
    return result


def _error_metrics(
    prediction: np.ndarray, expert: np.ndarray
) -> dict[str, float | int]:
    prediction = np.asarray(prediction, dtype=np.float64)
    expert = np.asarray(expert, dtype=np.float64)
    if prediction.shape != expert.shape or prediction.size == 0:
        raise ValueError(
            f"Invalid metric shapes: prediction={prediction.shape}, expert={expert.shape}"
        )
    error = prediction - expert
    absolute = np.abs(error)
    squared = np.square(error)
    return {
        "sample_count": int(error.size),
        "sum_absolute_error": float(np.sum(absolute)),
        "sum_squared_error": float(np.sum(squared)),
        "sum_error": float(np.sum(error)),
        "mae": float(np.mean(absolute)),
        "mse": float(np.mean(squared)),
        "rmse": float(np.sqrt(np.mean(squared))),
        "median_absolute_error": float(np.median(absolute)),
        "p95_absolute_error": float(np.quantile(absolute, 0.95)),
        "bias": float(np.mean(error)),
        "max_absolute_error": float(np.max(absolute)),
    }


def _prediction_change(
    intact: np.ndarray, counterfactual: np.ndarray, indices: np.ndarray
) -> dict[str, float]:
    difference = np.abs(counterfactual[:, indices] - intact[:, indices])
    return {
        "sum_absolute_prediction_change": float(np.sum(difference)),
        "mean_absolute_prediction_change": float(np.mean(difference)),
        "max_absolute_prediction_change": float(np.max(difference)),
    }


def _repeat_label(value: str, count: int) -> np.ndarray:
    return np.repeat(np.asarray([value]), count)


def _evaluate_checkpoint(
    *,
    policy: Gr00tPolicy,
    target_loader: LeRobotEpisodeLoader,
    donor_loader: LeRobotEpisodeLoader | None,
    records: list[EpisodeRecord],
    selected_records: list[EpisodeRecord],
    assignments_by_repeat: dict[int, dict[int, DonorAssignment]],
    checkpoint_step_value: int,
    split: str,
    geometry_key: str,
    intervention: str,
    modality: dict[str, ModalityConfig],
    embodiment_tag: EmbodimentTag,
    execution_horizon: int,
    pair_batch_size: int,
    inference_seed: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    action_keys = [str(key) for key in modality["action"].modality_keys]
    record_by_position = {record.loader_position: record for record in records}
    episode_condition_rows = []
    episode_pair_rows = []
    raw: dict[str, list[np.ndarray]] = defaultdict(list)
    canonical_labels = None
    canonical_groups = None

    for recipient_number, recipient in enumerate(selected_records, start=1):
        anchors = list(range(0, recipient.length, execution_horizon))
        target_trajectory = target_loader.load_episode(
            recipient.loader_position, frame_indices=anchors
        )
        if len(target_trajectory) != recipient.length:
            raise ValueError(
                f"Recipient episode {recipient.episode_index} metadata length "
                f"{recipient.length} != loaded length {len(target_trajectory)}"
            )
        actions, labels, groups = _action_schema(target_trajectory, action_keys)
        if canonical_labels is None:
            canonical_labels = labels
            canonical_groups = groups
        elif labels != canonical_labels:
            raise ValueError("Action schema changed between episodes")

        for repeat, assignments in sorted(assignments_by_repeat.items()):
            assignment = assignments.get(recipient.loader_position)
            donor_trajectory = None
            donor_frames: dict[int, int] = {}
            replacement_episode_index = None
            if intervention == "zero_geometry":
                if assignment is not None or donor_loader is not None:
                    raise RuntimeError(
                        "zero_geometry must not construct or decode donors"
                    )
            elif intervention in SAME_EPISODE_OFFSETS:
                if assignment is not None or donor_loader is None:
                    raise RuntimeError(
                        f"{intervention} requires no donor assignment and a geometry loader"
                    )
                donor_frames = {
                    anchor: same_episode_frame_for_offset(
                        anchor,
                        recipient.length,
                        intervention=intervention,
                    )
                    for anchor in anchors
                }
                donor_trajectory = donor_loader.load_episode(
                    recipient.loader_position,
                    frame_indices=sorted(set(donor_frames.values())),
                )
                if len(donor_trajectory) != recipient.length:
                    raise ValueError(
                        f"Same-episode geometry length for episode {recipient.episode_index} "
                        f"was {len(donor_trajectory)}, expected {recipient.length}"
                    )
                replacement_episode_index = recipient.episode_index
            else:
                if assignment is None or donor_loader is None:
                    raise RuntimeError(
                        f"{intervention} requires a donor assignment and loader"
                    )
                donor_record = record_by_position[assignment.donor_loader_position]
                donor_frames = {
                    anchor: donor_frame_for_phase(
                        anchor,
                        recipient.length,
                        donor_record.length,
                        donor_phase=intervention,
                    )
                    for anchor in anchors
                }
                donor_trajectory = donor_loader.load_episode(
                    donor_record.loader_position,
                    frame_indices=sorted(set(donor_frames.values())),
                )
                if len(donor_trajectory) != donor_record.length:
                    raise ValueError(
                        f"Donor episode {donor_record.episode_index} metadata length "
                        f"{donor_record.length} != loaded length {len(donor_trajectory)}"
                    )
                replacement_episode_index = assignment.donor_episode_index

            episode_expert = []
            episode_intact = []
            episode_counterfactual = []
            episode_frames = []
            episode_horizon_positions = []
            for batch_start in range(0, len(anchors), pair_batch_size):
                batch_anchors = anchors[batch_start : batch_start + pair_batch_size]
                observations = []
                seeds = []
                for anchor in batch_anchors:
                    if intervention == "zero_geometry":
                        target_value = target_trajectory[f"video.{geometry_key}"].iloc[
                            anchor
                        ]
                        if target_value is None:
                            raise ValueError(
                                f"Target geometry was not decoded for frame {anchor}"
                            )
                        replacement_frame = zero_geometry_frame(target_value)
                    else:
                        assert donor_trajectory is not None
                        replacement_frame = donor_trajectory[
                            f"video.{geometry_key}"
                        ].iloc[donor_frames[anchor]]
                        if replacement_frame is None:
                            raise ValueError(
                                "Replacement geometry was not decoded for frame "
                                f"{donor_frames[anchor]}"
                            )
                    intact_observation, counterfactual_observation = (
                        _paired_observations(
                            target_trajectory=target_trajectory,
                            target_frame=anchor,
                            replacement_frame=replacement_frame,
                            modality=modality,
                            embodiment_tag=embodiment_tag,
                            geometry_key=geometry_key,
                        )
                    )
                    noise_seed = _stable_seed(
                        inference_seed, split, recipient.episode_index, anchor, repeat
                    )
                    observations.extend(
                        (intact_observation, counterfactual_observation)
                    )
                    seeds.extend((noise_seed, noise_seed))
                chunks, _ = policy.get_action(
                    batch_policy_observations(observations),
                    options={"inference_mode": "synchronous", "noise_seeds": seeds},
                )
                for batch_index, anchor in enumerate(batch_anchors):
                    window = min(execution_horizon, recipient.length - anchor)
                    intact_chunk = _chunk_array(chunks, batch_index * 2, action_keys)[
                        :window
                    ]
                    counterfactual_chunk = _chunk_array(
                        chunks, batch_index * 2 + 1, action_keys
                    )[:window]
                    expert = actions[anchor : anchor + window]
                    if (
                        intact_chunk.shape != expert.shape
                        or counterfactual_chunk.shape != expert.shape
                    ):
                        raise ValueError(
                            f"Prediction/target mismatch at episode {recipient.episode_index}, "
                            f"frame {anchor}: {intact_chunk.shape}, "
                            f"{counterfactual_chunk.shape}, "
                            f"{expert.shape}"
                        )
                    episode_expert.append(expert)
                    episode_intact.append(intact_chunk)
                    episode_counterfactual.append(counterfactual_chunk)
                    episode_frames.append(
                        np.arange(anchor, anchor + window, dtype=np.int64)
                    )
                    episode_horizon_positions.append(np.arange(window, dtype=np.int64))

            expert = np.concatenate(episode_expert, axis=0)
            intact = np.concatenate(episode_intact, axis=0)
            counterfactual = np.concatenate(episode_counterfactual, axis=0)
            frames = np.concatenate(episode_frames)
            horizon_positions = np.concatenate(episode_horizon_positions)
            if len(expert) != recipient.length:
                raise RuntimeError(
                    f"Episode {recipient.episode_index} produced {len(expert)} predictions, "
                    f"expected {recipient.length}"
                )
            base = {
                "checkpoint_step": checkpoint_step_value,
                "split": split,
                "shuffle_repeat": repeat,
                "episode_index": recipient.episode_index,
                "loader_position": recipient.loader_position,
                "task": recipient.task,
                "cohort": recipient.cohort,
                "source": recipient.source,
                "source_episode": recipient.source_episode,
                "intervention": intervention,
                "donor_episode_index": replacement_episode_index,
            }
            for group_name, indices in groups.items():
                intact_metrics = _error_metrics(intact[:, indices], expert[:, indices])
                counterfactual_metrics = _error_metrics(
                    counterfactual[:, indices], expert[:, indices]
                )
                for condition, metrics in (
                    ("intact", intact_metrics),
                    ("counterfactual", counterfactual_metrics),
                ):
                    episode_condition_rows.append(
                        {**base, "condition": condition, "group": group_name, **metrics}
                    )
                episode_pair_rows.append(
                    {
                        **base,
                        "group": group_name,
                        "sample_count": intact_metrics["sample_count"],
                        "intact_sum_absolute_error": intact_metrics[
                            "sum_absolute_error"
                        ],
                        "counterfactual_sum_absolute_error": counterfactual_metrics[
                            "sum_absolute_error"
                        ],
                        "intact_sum_squared_error": intact_metrics["sum_squared_error"],
                        "counterfactual_sum_squared_error": counterfactual_metrics[
                            "sum_squared_error"
                        ],
                        "intact_mae": intact_metrics["mae"],
                        "counterfactual_mae": counterfactual_metrics["mae"],
                        "delta_mae_counterfactual_minus_intact": (
                            counterfactual_metrics["mae"] - intact_metrics["mae"]
                        ),
                        "intact_mse": intact_metrics["mse"],
                        "counterfactual_mse": counterfactual_metrics["mse"],
                        "delta_mse_counterfactual_minus_intact": (
                            counterfactual_metrics["mse"] - intact_metrics["mse"]
                        ),
                        **_prediction_change(intact, counterfactual, indices),
                    }
                )

            row_count = len(expert)
            raw["expert"].append(expert.astype(np.float32))
            raw["intact"].append(intact.astype(np.float32))
            raw["counterfactual"].append(counterfactual.astype(np.float32))
            raw["episode_index"].append(
                np.full(row_count, recipient.episode_index, dtype=np.int32)
            )
            raw["donor_episode_index"].append(
                np.full(
                    row_count,
                    replacement_episode_index
                    if replacement_episode_index is not None
                    else -1,
                    dtype=np.int32,
                )
            )
            raw["target_frame"].append(frames.astype(np.int32))
            raw["horizon_position"].append(horizon_positions.astype(np.int16))
            raw["shuffle_repeat"].append(np.full(row_count, repeat, dtype=np.int16))
            raw["intervention"].append(_repeat_label(intervention, row_count))
            raw["task"].append(_repeat_label(recipient.task, row_count))
            raw["cohort"].append(_repeat_label(recipient.cohort, row_count))

        LOGGER.info(
            "checkpoint %d | %s/%s: recipient %d/%d (episode=%d, task=%s)",
            checkpoint_step_value,
            geometry_key,
            intervention,
            recipient_number,
            len(selected_records),
            recipient.episode_index,
            recipient.task,
        )

    if canonical_labels is None or canonical_groups is None:
        raise ValueError("No episodes were evaluated")
    arrays = {key: np.concatenate(values, axis=0) for key, values in raw.items()}
    arrays["action_labels"] = np.asarray(canonical_labels, dtype=str)
    for group_name, indices in canonical_groups.items():
        arrays[f"group__{group_name}"] = indices.astype(np.int16)
    return arrays, episode_condition_rows, episode_pair_rows


def _scope_masks(arrays: dict[str, np.ndarray]) -> list[tuple[str, str, np.ndarray]]:
    count = len(arrays["episode_index"])
    scopes = [("all_tasks", "all", np.ones(count, dtype=bool))]
    for task in sorted(set(arrays["task"].tolist())):
        scopes.append(("task", str(task), arrays["task"] == task))
    for cohort in sorted(set(arrays["cohort"].tolist())):
        scopes.append(("cohort", str(cohort), arrays["cohort"] == cohort))
    return scopes


def _constant_label(arrays: dict[str, np.ndarray], key: str) -> str:
    values = np.unique(arrays[key])
    if len(values) != 1:
        raise ValueError(f"Expected one {key} value, got {values.tolist()}")
    return str(values[0])


def summarize_conditions(
    arrays: dict[str, np.ndarray], *, checkpoint_step_value: int, split: str
) -> pd.DataFrame:
    rows = []
    groups = {
        key.removeprefix("group__"): value
        for key, value in arrays.items()
        if key.startswith("group__")
    }
    intervention = _constant_label(arrays, "intervention")
    for scope, scope_value, mask in _scope_masks(arrays):
        for condition in CONDITIONS:
            for group, indices in groups.items():
                rows.append(
                    {
                        "checkpoint_step": checkpoint_step_value,
                        "split": split,
                        "intervention": intervention,
                        "scope": scope,
                        "scope_value": scope_value,
                        "condition": condition,
                        "group": group,
                        "episodes": int(np.unique(arrays["episode_index"][mask]).size),
                        "frames": int(np.count_nonzero(mask)),
                        **_error_metrics(
                            arrays[condition][mask][:, indices],
                            arrays["expert"][mask][:, indices],
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize_horizon(
    arrays: dict[str, np.ndarray], *, checkpoint_step_value: int, split: str
) -> pd.DataFrame:
    rows = []
    groups = {
        key.removeprefix("group__"): value
        for key, value in arrays.items()
        if key.startswith("group__")
    }
    intervention = _constant_label(arrays, "intervention")
    for position in sorted(np.unique(arrays["horizon_position"]).tolist()):
        mask = arrays["horizon_position"] == position
        for condition in CONDITIONS:
            for group, indices in groups.items():
                rows.append(
                    {
                        "checkpoint_step": checkpoint_step_value,
                        "split": split,
                        "intervention": intervention,
                        "horizon_position": int(position),
                        "condition": condition,
                        "group": group,
                        "frames": int(np.count_nonzero(mask)),
                        **_error_metrics(
                            arrays[condition][mask][:, indices],
                            arrays["expert"][mask][:, indices],
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize_joints(
    arrays: dict[str, np.ndarray], *, checkpoint_step_value: int, split: str
) -> pd.DataFrame:
    rows = []
    intervention = _constant_label(arrays, "intervention")
    for joint_index, joint in enumerate(arrays["action_labels"].tolist()):
        index = np.asarray([joint_index], dtype=np.int64)
        intact = _error_metrics(arrays["intact"][:, index], arrays["expert"][:, index])
        counterfactual = _error_metrics(
            arrays["counterfactual"][:, index], arrays["expert"][:, index]
        )
        rows.append(
            {
                "checkpoint_step": checkpoint_step_value,
                "split": split,
                "intervention": intervention,
                "joint_index": joint_index,
                "joint": joint,
                "intact_mae": intact["mae"],
                "counterfactual_mae": counterfactual["mae"],
                "delta_mae_counterfactual_minus_intact": (
                    counterfactual["mae"] - intact["mae"]
                ),
                "intact_mse": intact["mse"],
                "counterfactual_mse": counterfactual["mse"],
                "delta_mse_counterfactual_minus_intact": (
                    counterfactual["mse"] - intact["mse"]
                ),
                **_prediction_change(arrays["intact"], arrays["counterfactual"], index),
            }
        )
    return pd.DataFrame(rows)


def _stratified_task_bootstrap_ci(
    episode_effects: pd.DataFrame,
    *,
    value_column: str,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if replicates == 0 or episode_effects.empty:
        return float("nan"), float("nan")
    per_episode = (
        episode_effects.groupby(["task", "episode_index"], sort=True)[value_column]
        .mean()
        .reset_index()
    )
    clusters = [
        group[value_column].to_numpy(dtype=np.float64)
        for _task, group in per_episode.groupby("task", sort=True)
    ]
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        task_means = []
        for values in clusters:
            selected = rng.integers(0, len(values), size=len(values))
            task_means.append(float(np.mean(values[selected])))
        samples[replicate] = float(np.mean(task_means))
    low, high = np.quantile(samples, [0.025, 0.975])
    return float(low), float(high)


def summarize_paired_effects(
    episode_effects: pd.DataFrame,
    *,
    checkpoint_step_value: int,
    split: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows = []
    interventions = episode_effects["intervention"].drop_duplicates().tolist()
    if len(interventions) != 1:
        raise ValueError(f"Expected one intervention, got {interventions}")
    intervention = str(interventions[0])
    scopes: list[tuple[str, str, pd.DataFrame]] = [
        ("all_tasks", "all", episode_effects)
    ]
    scopes.extend(
        ("task", str(value), group)
        for value, group in episode_effects.groupby("task", sort=True)
    )
    scopes.extend(
        ("cohort", str(value), group)
        for value, group in episode_effects.groupby("cohort", sort=True)
    )
    for scope, scope_value, scoped in scopes:
        for group_name, group in scoped.groupby("group", sort=True):
            per_episode = (
                group.groupby(["task", "episode_index"], sort=True)
                .agg(
                    delta_mae=("delta_mae_counterfactual_minus_intact", "mean"),
                    delta_mse=("delta_mse_counterfactual_minus_intact", "mean"),
                    prediction_change=("mean_absolute_prediction_change", "mean"),
                )
                .reset_index()
            )
            task_means = per_episode.groupby("task", sort=True)["delta_mae"].mean()
            low, high = _stratified_task_bootstrap_ci(
                group,
                value_column="delta_mae_counterfactual_minus_intact",
                replicates=bootstrap_replicates,
                seed=_stable_seed(
                    bootstrap_seed,
                    checkpoint_step_value,
                    scope,
                    scope_value,
                    group_name,
                ),
            )
            sample_count = float(group["sample_count"].sum())
            rows.append(
                {
                    "checkpoint_step": checkpoint_step_value,
                    "split": split,
                    "intervention": intervention,
                    "scope": scope,
                    "scope_value": scope_value,
                    "group": group_name,
                    "episodes": int(per_episode["episode_index"].nunique()),
                    "tasks": int(per_episode["task"].nunique()),
                    "shuffle_repeats": int(group["shuffle_repeat"].nunique()),
                    "micro_intact_mae": float(
                        group["intact_sum_absolute_error"].sum() / sample_count
                    ),
                    "micro_counterfactual_mae": float(
                        group["counterfactual_sum_absolute_error"].sum() / sample_count
                    ),
                    "micro_delta_mae_counterfactual_minus_intact": float(
                        (
                            group["counterfactual_sum_absolute_error"].sum()
                            - group["intact_sum_absolute_error"].sum()
                        )
                        / sample_count
                    ),
                    "micro_intact_mse": float(
                        group["intact_sum_squared_error"].sum() / sample_count
                    ),
                    "micro_counterfactual_mse": float(
                        group["counterfactual_sum_squared_error"].sum() / sample_count
                    ),
                    "micro_delta_mse_counterfactual_minus_intact": float(
                        (
                            group["counterfactual_sum_squared_error"].sum()
                            - group["intact_sum_squared_error"].sum()
                        )
                        / sample_count
                    ),
                    "episode_macro_mean_delta_mae": float(
                        per_episode["delta_mae"].mean()
                    ),
                    "episode_macro_median_delta_mae": float(
                        per_episode["delta_mae"].median()
                    ),
                    "task_balanced_mean_delta_mae": float(task_means.mean()),
                    "task_balanced_delta_mae_ci95_low": low,
                    "task_balanced_delta_mae_ci95_high": high,
                    "episode_win_rate_counterfactual_worse": float(
                        np.mean(per_episode["delta_mae"] > 0)
                    ),
                    "episode_macro_prediction_change_mae": float(
                        per_episode["prediction_change"].mean()
                    ),
                    "interpretation": (
                        "positive error deltas mean correctly aligned geometry predicted "
                        "the recorded actions more accurately"
                    ),
                }
            )
    return pd.DataFrame(rows)


def _donor_manifest(
    assignments_by_repeat: dict[int, dict[int, DonorAssignment]],
    *,
    intervention: str,
) -> list[dict[str, Any]]:
    if intervention == "zero_geometry":
        return [
            {
                "intervention": intervention,
                "donor_required": False,
                "replacement": "all-zero frame with target shape and dtype",
            }
        ]
    if intervention in SAME_EPISODE_OFFSETS:
        return [
            {
                "intervention": intervention,
                "donor_required": False,
                "replacement": "same-episode geometry with a circular integer-frame shift",
                "phase_offset_fraction": SAME_EPISODE_OFFSETS[intervention],
            }
        ]
    return [
        {"intervention": intervention, "donor_required": True, **asdict(assignment)}
        for _repeat, assignments in sorted(assignments_by_repeat.items())
        for _position, assignment in sorted(assignments.items())
    ]


def _frame_mapping(
    selected_records: list[EpisodeRecord],
    assignments_by_repeat: dict[int, dict[int, DonorAssignment]],
    execution_horizon: int,
    *,
    intervention: str,
) -> list[dict[str, Any]]:
    rows = []
    for repeat, assignments in sorted(assignments_by_repeat.items()):
        for recipient in selected_records:
            assignment = assignments.get(recipient.loader_position)
            for anchor in range(0, recipient.length, execution_horizon):
                recipient_progress = (
                    anchor / (recipient.length - 1) if recipient.length > 1 else 0.0
                )
                if intervention == "zero_geometry":
                    if assignment is not None:
                        raise RuntimeError(
                            "zero_geometry must not have donor assignments"
                        )
                    donor_progress = None
                    donor_frame = None
                    donor_episode_index = None
                    donor_length = None
                    replacement_source = "all_zeros"
                elif intervention in SAME_EPISODE_OFFSETS:
                    if assignment is not None:
                        raise RuntimeError(
                            f"{intervention} must not have a cross-episode donor assignment"
                        )
                    donor_progress = same_episode_progress_for_offset(
                        anchor,
                        recipient.length,
                        intervention=intervention,
                    )
                    donor_frame = same_episode_frame_for_offset(
                        anchor,
                        recipient.length,
                        intervention=intervention,
                    )
                    donor_episode_index = recipient.episode_index
                    donor_length = recipient.length
                    replacement_source = "same_episode"
                else:
                    if assignment is None:
                        raise RuntimeError(
                            f"{intervention} is missing a donor assignment"
                        )
                    donor_progress = donor_progress_for_phase(
                        anchor,
                        recipient.length,
                        donor_phase=intervention,
                    )
                    donor_frame = donor_frame_for_phase(
                        anchor,
                        recipient.length,
                        assignment.donor_length,
                        donor_phase=intervention,
                    )
                    donor_episode_index = assignment.donor_episode_index
                    donor_length = assignment.donor_length
                    replacement_source = "same_task_donor"
                rows.append(
                    {
                        "intervention": intervention,
                        "shuffle_repeat": repeat,
                        "task": recipient.task,
                        "recipient_episode_index": recipient.episode_index,
                        "recipient_length": recipient.length,
                        "recipient_frame": anchor,
                        "recipient_progress": recipient_progress,
                        "replacement_source": replacement_source,
                        "donor_progress": donor_progress,
                        "donor_episode_index": donor_episode_index,
                        "donor_length": donor_length,
                        "donor_frame": donor_frame,
                    }
                )
    return rows


def _write_csv(path: Path, rows: pd.DataFrame | list[dict[str, Any]]) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(path, index=False)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_state(path: Path) -> dict[str, Any]:
    try:
        root = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        commit = subprocess.run(
            ["git", "-C", root, "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", root, "status", "--short"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        return {"root": root, "commit": commit, "dirty": bool(status), "status": status}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"error": str(exc), "path": str(path)}


def _artifact_fingerprints(model_path: Path, run_dir: Path) -> dict[str, Any]:
    candidates = [
        model_path / "config.json",
        model_path / "model.safetensors.index.json",
        model_path / "processor_config.json",
        model_path / "statistics.json",
        run_dir / "processor" / "processor_config.json",
        run_dir / "processor" / "statistics.json",
    ]
    result = {}
    for path in candidates:
        if path.is_file():
            result[str(path.resolve())] = {
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
    for shard in sorted(model_path.glob("model-*.safetensors")):
        result[str(shard.resolve())] = {"size": shard.stat().st_size}
    return result


def _modality_signature(modality: dict[str, ModalityConfig]) -> dict[str, Any]:
    return {
        name: {
            "keys": [str(key) for key in config.modality_keys],
            "delta_indices": [int(value) for value in config.delta_indices],
            "vision_channel_layout": (
                config.vision_channel_layout if name == "video" else None
            ),
        }
        for name, config in modality.items()
    }


def validate_heldout_dataset_path(dataset_path: Path, split: str) -> Path:
    resolved = dataset_path.resolve()
    if split not in {"validation", "test"}:
        raise ValueError(
            f"Only held-out validation/test splits are allowed, got {split!r}"
        )
    if resolved.name != split:
        raise ValueError(
            f"--dataset-path must resolve to the named held-out split {split!r}; got {resolved}"
        )
    return resolved


def _run_metadata(
    *,
    args: argparse.Namespace,
    target_metadata: list[dict[str, Any]],
    modality_signature: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    dataset_meta = args.dataset_path / "meta"
    merge_manifest = args.dataset_path.parent / "provenance" / "merge_manifest.json"
    script_path = Path(__file__).resolve()
    runner_path = script_path.with_name("multi_geometry_correspondence_evaluation.sh")
    if args.intervention == "phase_matched":
        intervention_detail = {
            "name": args.intervention,
            "replacement": "same-task cross-episode donor geometry",
            "donor_rule": (
                "different episode, exact same task, deterministic cyclic derangement"
            ),
            "frame_rule": "donor_progress = recipient_progress",
            "caveat": (
                "normalized episode progress is approximate, not semantic phase alignment"
            ),
        }
    elif args.intervention == "out_of_phase":
        intervention_detail = {
            "name": args.intervention,
            "replacement": "same-task cross-episode donor geometry",
            "donor_rule": (
                "different episode, exact same task, deterministic cyclic derangement"
            ),
            "frame_rule": "donor_progress = (recipient_progress + 0.5) modulo 1",
            "caveat": (
                "the half-episode shift deliberately destroys phase correspondence"
            ),
        }
    elif args.intervention in SAME_EPISODE_OFFSETS:
        offset = SAME_EPISODE_OFFSETS[args.intervention]
        intervention_detail = {
            "name": args.intervention,
            "replacement": "geometry from the same episode after a circular frame shift",
            "donor_rule": "same episode as the intact observation",
            "frame_rule": (
                "replacement_frame = (recipient_frame + "
                f"round({offset:.2f} * episode_length)) modulo episode_length"
            ),
            "phase_offset_fraction": offset,
            "caveat": (
                "the integer circular shift wraps near the episode end and intentionally breaks "
                "instantaneous RGB/geometry correspondence"
            ),
        }
    elif args.intervention == "zero_geometry":
        intervention_detail = {
            "name": args.intervention,
            "replacement": "all-zero frame with the target geometry shape and dtype",
            "donor_rule": None,
            "frame_rule": None,
            "caveat": (
                "all-zero geometry is a strong out-of-distribution ablation, not a "
                "realistic sensor sample"
            ),
        }
    else:
        raise ValueError(f"Unknown intervention: {args.intervention}")
    return {
        "version": 3,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "offline_only": True,
        "dataset_mutated": False,
        "intervention": {
            "changed_input": args.geometry_key,
            "unchanged_inputs": ["ego_view", "state", "language", "expert_action"],
            **intervention_detail,
        },
        "interpretation": (
            "Prediction change measures geometry sensitivity. Positive "
            "counterfactual-minus-intact error means intact geometry predicted the recorded "
            "actions more accurately; neither quantity proves closed-loop task success."
        ),
        "arguments": arguments,
        "targets": target_metadata,
        "modality_signature": modality_signature,
        "dataset": {
            "path": str(args.dataset_path.resolve()),
            "info_sha256": _sha256(dataset_meta / "info.json"),
            "episodes_sha256": _sha256(dataset_meta / "episodes.jsonl"),
            "tasks_sha256": _sha256(dataset_meta / "tasks.jsonl"),
            "modality_sha256": _sha256(dataset_meta / "modality.json"),
            "merge_manifest_sha256": _sha256(merge_manifest),
        },
        "analysis_files": {
            str(script_path): _sha256(script_path),
            str(runner_path): _sha256(runner_path),
        },
        "git": {
            "isaac_groot": _git_state(Path(gr00t.__file__).resolve().parent.parent),
            "analysis_scripts": _git_state(script_path.parent),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    validate_heldout_dataset_path(args.dataset_path, args.split)
    if not (args.dataset_path / "meta" / "info.json").is_file():
        raise FileNotFoundError(f"Held-out dataset is incomplete: {args.dataset_path}")
    targets = find_targets(args.run_dir, args.checkpoint_steps)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    incomplete = args.output_dir / "INCOMPLETE"
    incomplete.write_text(
        "Evaluation is incomplete until this marker is removed.\n", encoding="utf-8"
    )
    started = time.perf_counter()
    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    canonical_signature = None
    records = None
    selected_records = None
    assignments_by_repeat = None
    all_condition_summaries = []
    all_paired_summaries = []
    target_metadata = []

    try:
        for target_number, target in enumerate(targets, start=1):
            LOGGER.info(
                "Loading target %d/%d: checkpoint %d from %s",
                target_number,
                len(targets),
                target.step,
                target.model_path,
            )
            random.seed(args.inference_seed)
            np.random.seed(args.inference_seed)
            torch.manual_seed(args.inference_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(args.inference_seed)
            policy = Gr00tPolicy(
                embodiment_tag=embodiment_tag,
                model_path=str(target.model_path),
                device=args.device,
            )
            policy.model.action_head.num_inference_timesteps = args.denoising_steps
            modality = policy.get_modality_config()
            _validate_contract(
                modality,
                geometry_key=args.geometry_key,
                execution_horizon=args.execution_horizon,
            )
            signature = _modality_signature(modality)
            target_loader = LeRobotEpisodeLoader(
                args.dataset_path, modality_configs=modality
            )
            donor_loader = (
                None
                if args.intervention == "zero_geometry"
                else LeRobotEpisodeLoader(
                    args.dataset_path,
                    modality_configs=_donor_modality(modality, args.geometry_key),
                )
            )
            if canonical_signature is None:
                canonical_signature = signature
                records = episode_records(
                    target_loader, split=args.split, tasks=args.task
                )
                selected_records = records[: args.max_episodes or None]
                if args.intervention not in CROSS_EPISODE_INTERVENTIONS:
                    assignments_by_repeat = {
                        repeat: {} for repeat in range(args.shuffle_repeats)
                    }
                else:
                    assignments_by_repeat = {
                        repeat: build_donor_assignments(
                            records,
                            shuffle_seed=args.shuffle_seed,
                            shuffle_repeat=repeat,
                        )
                        for repeat in range(args.shuffle_repeats)
                    }
                _write_csv(
                    args.output_dir / "episode_manifest.csv",
                    [asdict(row) for row in records],
                )
                _write_csv(
                    args.output_dir / "donor_manifest.csv",
                    _donor_manifest(
                        assignments_by_repeat, intervention=args.intervention
                    ),
                )
                _write_csv(
                    args.output_dir / "frame_mapping.csv",
                    _frame_mapping(
                        selected_records,
                        assignments_by_repeat,
                        args.execution_horizon,
                        intervention=args.intervention,
                    ),
                )
            elif signature != canonical_signature:
                raise ValueError(
                    "Checkpoint modality contract changed within this evaluation"
                )

            checkpoint_dir = args.output_dir / f"checkpoint-{target.step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=False)
            arrays, condition_rows, pair_rows = _evaluate_checkpoint(
                policy=policy,
                target_loader=target_loader,
                donor_loader=donor_loader,
                records=records,
                selected_records=selected_records,
                assignments_by_repeat=assignments_by_repeat,
                checkpoint_step_value=target.step,
                split=args.split,
                geometry_key=args.geometry_key,
                intervention=args.intervention,
                modality=modality,
                embodiment_tag=embodiment_tag,
                execution_horizon=args.execution_horizon,
                pair_batch_size=args.pair_batch_size,
                inference_seed=args.inference_seed,
            )
            np.savez_compressed(checkpoint_dir / "frame_predictions.npz", **arrays)
            episode_condition = pd.DataFrame(condition_rows)
            episode_pair = pd.DataFrame(pair_rows)
            condition_summary = summarize_conditions(
                arrays, checkpoint_step_value=target.step, split=args.split
            )
            paired_summary = summarize_paired_effects(
                episode_pair,
                checkpoint_step_value=target.step,
                split=args.split,
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.shuffle_seed,
            )
            _write_csv(
                checkpoint_dir / "episode_condition_metrics.csv", episode_condition
            )
            _write_csv(checkpoint_dir / "paired_episode_effects.csv", episode_pair)
            _write_csv(checkpoint_dir / "condition_summary.csv", condition_summary)
            _write_csv(checkpoint_dir / "paired_summary.csv", paired_summary)
            _write_csv(
                checkpoint_dir / "horizon_metrics.csv",
                summarize_horizon(
                    arrays, checkpoint_step_value=target.step, split=args.split
                ),
            )
            _write_csv(
                checkpoint_dir / "joint_metrics.csv",
                summarize_joints(
                    arrays, checkpoint_step_value=target.step, split=args.split
                ),
            )
            all_condition_summaries.append(condition_summary)
            all_paired_summaries.append(paired_summary)
            target_metadata.append(
                {
                    "step": target.step,
                    "model_path": str(target.model_path.resolve()),
                    "artifact_fingerprints": _artifact_fingerprints(
                        target.model_path, args.run_dir
                    ),
                }
            )
            del arrays, policy, target_loader, donor_loader
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        combined_conditions = pd.concat(all_condition_summaries, ignore_index=True)
        combined_paired = pd.concat(all_paired_summaries, ignore_index=True)
        _write_csv(
            args.output_dir / "condition_summary_all_checkpoints.csv",
            combined_conditions,
        )
        _write_csv(
            args.output_dir / "paired_summary_all_checkpoints.csv", combined_paired
        )
        primary = combined_paired[
            (combined_paired["scope"] == "all_tasks")
            & (combined_paired["group"] == "all")
        ]
        _write_csv(args.output_dir / "primary_task_balanced_result.csv", primary)
        metadata = _run_metadata(
            args=args,
            target_metadata=target_metadata,
            modality_signature=canonical_signature,
            elapsed_seconds=time.perf_counter() - started,
        )
        (args.output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        incomplete.unlink()
        LOGGER.info(
            "Finished %d checkpoint(s) in %.1f minutes. Results: %s",
            len(targets),
            (time.perf_counter() - started) / 60,
            args.output_dir,
        )
        return 0
    except BaseException:
        LOGGER.exception(
            "Geometry correspondence evaluation failed; INCOMPLETE retained"
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
