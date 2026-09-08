#!/usr/bin/env python3
"""Offline comparison of GR00T behavior around recorded hand-state plateaus.

This script reads LeRobot datasets and local checkpoints only. It never imports
the robot controller, opens DDS, or publishes robot commands.

For a maximal constant hand-state run q[f] == ... == q[e], f is explicitly a
fresh/cache-fill proxy because the historical recorder did not save DDS receipt
timestamps. At decision offset d, every strategy is compared against the same
recorded command window A[f+d:f+d+H]:

* stale_state_inference: infer from the exact recorded observation at f+d;
* hold_pre_dropout: repeat the full recorded command A[f];
* continue_previous_chunk: infer once at f and use chunk slice d:d+H.

Policy outputs are already decoded into physical, unnormalised action units.
The scores measure open-loop command agreement, not closed-loop robot safety.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import gc
import gzip
import hashlib
import json
import logging
from pathlib import Path
import random
import re
import subprocess
import time
from typing import Any, Iterable

from gr00t.data.dataset.lerobot_episode_loader import LeRobotEpisodeLoader
from gr00t.data.dataset.sharded_single_step_dataset import extract_step_data
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.utils import parse_observation_gr00t
from gr00t.eval._horizon_contract import PolicyHorizonSpec
from gr00t.eval.open_loop_eval import batch_policy_observations
from gr00t.policy.gr00t_policy import Gr00tPolicy
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import torch


plt.switch_backend("Agg")


LOGGER = logging.getLogger("stale_state_counterfactual_evaluation")
STRATEGIES = (
    "stale_state_inference",
    "hold_pre_dropout",
    "continue_previous_chunk",
)
REQUIRED_HAND_KEYS = ("left_hand", "right_hand")
DEFAULT_DECISION_OFFSETS = (1, 8, 16)


@dataclass(frozen=True)
class DatasetSpec:
    split: str
    path: Path


@dataclass(frozen=True)
class EvaluationTarget:
    step: int
    model_path: Path
    processor_path: Path | None = None


@dataclass(frozen=True)
class PlateauEvent:
    split: str
    dataset_path: str
    loader_position: int
    episode_index: int
    event_index: int
    fresh_proxy_frame: int
    plateau_end_frame: int
    plateau_frames: int
    stale_candidate_frames: int
    fps: float
    duration_s: float
    left_action_range: float
    right_action_range: float
    strong: bool
    moving_hand: str
    task: str
    source: str | None = None
    source_episode: str | None = None
    source_data_json_sha256: str | None = None

    @property
    def event_id(self) -> str:
        return (
            f"{self.split}:episode-{self.episode_index:06d}:"
            f"frames-{self.fresh_proxy_frame}-{self.plateau_end_frame}"
        )


@dataclass(frozen=True)
class Decision:
    event: PlateauEvent
    offset: int
    anchor_frame: int
    target_start_frame: int
    target_end_frame: int
    window_all_stale: bool


def _dataset_argument(value: str) -> DatasetSpec:
    if "=" not in value:
        raise argparse.ArgumentTypeError("datasets must use SPLIT=/absolute/path")
    split, raw_path = value.split("=", 1)
    split = split.strip()
    path = Path(raw_path).expanduser()
    if not split or not re.fullmatch(r"[A-Za-z0-9_.-]+", split):
        raise argparse.ArgumentTypeError(f"invalid dataset split label: {split!r}")
    if not path.is_absolute():
        raise argparse.ArgumentTypeError(f"dataset path must be absolute: {path}")
    return DatasetSpec(split=split, path=path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare stale-input inference, pre-dropout HOLD, and continuation of a "
            "previously predicted chunk using recorded LeRobot episodes."
        )
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset",
        type=_dataset_argument,
        action="append",
        required=True,
        metavar="SPLIT=/ABSOLUTE/PATH",
        help="Held-out dataset. Repeat for validation and test.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path)
    parser.add_argument("--checkpoint-steps", type=int, nargs="*")
    parser.add_argument("--embodiment-tag", default="NEW_EMBODIMENT")
    parser.add_argument(
        "--task",
        action="append",
        help="Exact task text to include. Repeat for multiple tasks; omit for all tasks.",
    )
    parser.add_argument("--execution-horizon", type=int, default=8)
    parser.add_argument(
        "--decision-offsets",
        type=int,
        nargs="+",
        default=list(DEFAULT_DECISION_OFFSETS),
        help=(
            "Frames after the fresh proxy at which to compare the strategies. "
            "Offsets are summarized separately."
        ),
    )
    parser.add_argument("--minimum-plateau-frames", type=int, default=25)
    parser.add_argument("--strong-action-range", type=float, default=0.1)
    parser.add_argument(
        "--event-filter",
        choices=("strong", "all"),
        default="all",
        help=(
            "strong keeps plateaus where either hand command spans the configured range; "
            "all includes ambiguous stationary plateaus too."
        ),
    )
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--denoising-steps", type=int, default=4)
    parser.add_argument("--inference-seed", type=int, default=42)
    parser.add_argument("--noise-repeats", type=int, default=1)
    parser.add_argument("--bootstrap-replicates", type=int, default=1000)
    parser.add_argument(
        "--max-events-per-split",
        type=int,
        default=0,
        help="Deterministic smoke-test cap; 0 evaluates every selected event.",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args(argv)

    if args.execution_horizon <= 0:
        parser.error("--execution-horizon must be positive")
    if args.minimum_plateau_frames < 2:
        parser.error("--minimum-plateau-frames must be at least 2")
    if not np.isfinite(args.strong_action_range) or args.strong_action_range < 0:
        parser.error("--strong-action-range must be finite and non-negative")
    if args.inference_batch_size <= 0:
        parser.error("--inference-batch-size must be positive")
    if args.denoising_steps <= 0:
        parser.error("--denoising-steps must be positive")
    if args.noise_repeats <= 0:
        parser.error("--noise-repeats must be positive")
    if args.bootstrap_replicates < 0:
        parser.error("--bootstrap-replicates must be non-negative")
    if args.max_events_per_split < 0:
        parser.error("--max-events-per-split must be non-negative")
    if any(offset <= 0 for offset in args.decision_offsets):
        parser.error("--decision-offsets must all be positive")
    args.decision_offsets = sorted(set(args.decision_offsets))
    if args.task:
        args.task = list(dict.fromkeys(task.strip() for task in args.task))
        if any(not task for task in args.task):
            parser.error("--task values must not be empty")

    split_names = [spec.split for spec in args.dataset]
    if len(split_names) != len(set(split_names)):
        parser.error("--dataset split labels must be unique")
    if args.output_dir.exists():
        parser.error(f"--output-dir already exists: {args.output_dir}; use a new result directory")
    return args


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match is None:
        raise ValueError(f"Not a checkpoint directory: {path}")
    return int(match.group(1))


def find_targets(
    run_dir: Path,
    selected_steps: list[int] | None,
    base_model_path: Path | None,
) -> list[EvaluationTarget]:
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Training run does not exist: {run_dir}")
    selected = set(selected_steps) if selected_steps else None
    if re.fullmatch(r"checkpoint-\d+", run_dir.name):
        checkpoints = [run_dir]
        processor_root = run_dir.parent
    else:
        checkpoints = sorted(
            (path for path in run_dir.glob("checkpoint-*") if path.is_dir()),
            key=checkpoint_step,
        )
        processor_root = run_dir
    targets = [
        EvaluationTarget(checkpoint_step(path), path)
        for path in checkpoints
        if selected is None or checkpoint_step(path) in selected
    ]

    if base_model_path is not None and (selected is None or 0 in selected):
        if not base_model_path.is_dir():
            raise FileNotFoundError(f"Base model does not exist: {base_model_path}")
        processor_path = processor_root / "processor"
        if not (processor_path / "processor_config.json").is_file():
            raise FileNotFoundError(
                f"Run processor is required for the base model: {processor_path}"
            )
        targets.append(EvaluationTarget(0, base_model_path, processor_path))

    targets.sort(key=lambda target: target.step)
    if selected is not None:
        missing = selected - {target.step for target in targets}
        if missing:
            raise FileNotFoundError(
                f"No model target found for checkpoint step(s): {sorted(missing)}"
            )
    if not targets:
        raise FileNotFoundError(f"No checkpoints found under {run_dir}")
    return targets


def _stack_columns(frame: pd.DataFrame, prefix: str, keys: Iterable[str]) -> np.ndarray:
    arrays = []
    for key in keys:
        column = f"{prefix}.{key}"
        if column not in frame:
            raise ValueError(f"Episode is missing required column {column}")
        values = np.vstack(
            [np.asarray(value, dtype=np.float64).reshape(-1) for value in frame[column]]
        )
        arrays.append(values)
    result = np.concatenate(arrays, axis=1)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"Episode contains non-finite values in {prefix} keys {list(keys)}")
    return result


def _task_for_episode(metadata: dict[str, Any]) -> str:
    tasks = [str(task) for task in metadata.get("tasks", []) if str(task)]
    return " / ".join(dict.fromkeys(tasks))


def _max_action_range(actions: np.ndarray) -> float:
    if len(actions) == 0:
        return 0.0
    return float(np.max(np.ptp(actions, axis=0)))


def detect_plateaus(
    frame: pd.DataFrame,
    *,
    split: str,
    dataset_path: Path,
    loader_position: int,
    episode_index: int,
    task: str,
    fps: float,
    minimum_frames: int,
    strong_action_range: float,
    provenance: dict[str, Any] | None = None,
) -> list[PlateauEvent]:
    """Return maximal, bit-exact bilateral hand-state plateaus."""

    hand_state = _stack_columns(frame, "state", REQUIRED_HAND_KEYS)
    left_actions = _stack_columns(frame, "action", ("left_hand",))
    right_actions = _stack_columns(frame, "action", ("right_hand",))
    if len(hand_state) != len(left_actions) or len(hand_state) != len(right_actions):
        raise ValueError("State/action row counts differ")
    if len(hand_state) == 0:
        return []

    boundaries = np.flatnonzero(np.any(hand_state[1:] != hand_state[:-1], axis=1)) + 1
    starts = np.concatenate(([0], boundaries))
    stops = np.concatenate((boundaries, [len(hand_state)]))
    events = []
    output_index = 0
    for start, stop in zip(starts, stops, strict=True):
        plateau_frames = int(stop - start)
        if plateau_frames < minimum_frames:
            continue
        end = int(stop - 1)
        left_range = _max_action_range(left_actions[start:stop])
        right_range = _max_action_range(right_actions[start:stop])
        left_moving = left_range >= strong_action_range
        right_moving = right_range >= strong_action_range
        if left_moving and right_moving:
            moving_hand = "both"
        elif left_moving:
            moving_hand = "left"
        elif right_moving:
            moving_hand = "right"
        else:
            moving_hand = "neither"
        provenance = provenance or {}
        events.append(
            PlateauEvent(
                split=split,
                dataset_path=str(dataset_path),
                loader_position=loader_position,
                episode_index=episode_index,
                event_index=output_index,
                fresh_proxy_frame=int(start),
                plateau_end_frame=end,
                plateau_frames=plateau_frames,
                stale_candidate_frames=max(0, plateau_frames - 1),
                fps=fps,
                duration_s=(plateau_frames - 1) / fps,
                left_action_range=left_range,
                right_action_range=right_range,
                strong=left_moving or right_moving,
                moving_hand=moving_hand,
                task=task,
                source=provenance.get("source"),
                source_episode=provenance.get("source_episode"),
                source_data_json_sha256=provenance.get("data_json_sha256"),
            )
        )
        output_index += 1
    return events


def plan_decisions(
    event: PlateauEvent,
    *,
    episode_length: int,
    action_horizon: int,
    execution_horizon: int,
    offsets: Iterable[int],
) -> tuple[list[Decision], list[dict[str, Any]]]:
    decisions = []
    skipped = []
    for offset in offsets:
        reason = None
        anchor = event.fresh_proxy_frame + int(offset)
        target_end = anchor + execution_horizon - 1
        if event.fresh_proxy_frame == 0:
            reason = "plateau_starts_at_episode_start"
        elif anchor > event.plateau_end_frame:
            reason = "decision_anchor_after_plateau"
        elif target_end >= episode_length:
            reason = "expert_window_crosses_episode_end"
        elif offset + execution_horizon > action_horizon:
            reason = "previous_chunk_slice_exceeds_action_horizon"
        if reason is not None:
            skipped.append(
                {
                    "event_id": event.event_id,
                    "split": event.split,
                    "episode_index": event.episode_index,
                    "fresh_proxy_frame": event.fresh_proxy_frame,
                    "plateau_end_frame": event.plateau_end_frame,
                    "decision_offset": int(offset),
                    "reason": reason,
                }
            )
            continue
        decisions.append(
            Decision(
                event=event,
                offset=int(offset),
                anchor_frame=anchor,
                target_start_frame=anchor,
                target_end_frame=target_end,
                window_all_stale=target_end <= event.plateau_end_frame,
            )
        )
    return decisions, skipped


def build_strategy_arrays(
    *,
    stale_chunk: np.ndarray,
    fresh_chunk: np.ndarray,
    pre_dropout_command: np.ndarray,
    expert: np.ndarray,
    decision_offset: int,
    execution_horizon: int,
) -> dict[str, np.ndarray]:
    stale_chunk = np.asarray(stale_chunk, dtype=np.float64)
    fresh_chunk = np.asarray(fresh_chunk, dtype=np.float64)
    pre_dropout_command = np.asarray(pre_dropout_command, dtype=np.float64).reshape(-1)
    expert = np.asarray(expert, dtype=np.float64)
    if expert.ndim != 2:
        raise ValueError(f"Expert action window must be two-dimensional, got {expert.shape}")
    expected_shape = (execution_horizon, expert.shape[1])
    strategies = {
        "stale_state_inference": stale_chunk[:execution_horizon],
        "hold_pre_dropout": np.repeat(pre_dropout_command[None, :], execution_horizon, axis=0),
        "continue_previous_chunk": fresh_chunk[
            decision_offset : decision_offset + execution_horizon
        ],
    }
    for name, values in strategies.items():
        if values.shape != expected_shape:
            raise ValueError(f"{name} has shape {values.shape}, expected {expected_shape}")
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{name} contains non-finite actions")
    return strategies


def _provenance_map(dataset_path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    manifest_path = dataset_path.parent / "provenance" / "merge_manifest.json"
    if not manifest_path.is_file():
        LOGGER.warning("No merge provenance manifest at %s", manifest_path)
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    result = {}
    for record in manifest.get("episodes", []):
        key = (str(record["split"]), int(record["final_episode_index"]))
        if key in result:
            raise ValueError(f"Duplicate provenance entry for {key} in {manifest_path}")
        result[key] = record
    return result


def _verify_evaluation_provenance(
    specs: list[DatasetSpec],
    provenance: dict[tuple[str, int], dict[str, Any]],
) -> None:
    """Fail if a source episode is assigned to train and a requested held-out split."""

    evaluation_hashes: dict[str, str] = {}
    requested_splits = {spec.split for spec in specs}
    for (split, _episode_index), record in provenance.items():
        if split not in requested_splits:
            continue
        digest = record.get("data_json_sha256")
        if not digest:
            continue
        digest = str(digest)
        previous = evaluation_hashes.get(digest)
        if previous is not None:
            raise ValueError(
                "Held-out source episode appears in multiple evaluation splits: "
                f"{previous} and {split} share {digest}"
            )
        evaluation_hashes[digest] = split

    train_hashes = {
        str(record["data_json_sha256"])
        for (split, _episode_index), record in provenance.items()
        if split == "train" and record.get("data_json_sha256")
    }
    overlap = set(evaluation_hashes) & train_hashes
    if overlap:
        raise ValueError(
            "Train/held-out provenance overlap detected for source hashes: "
            + ", ".join(sorted(overlap))
        )


def _validate_input_contract(modality: dict[str, Any], execution_horizon: int) -> int:
    spec = PolicyHorizonSpec.from_modality_config(modality, n_action_steps=execution_horizon)
    for name, config in modality.items():
        if name == "action":
            continue
        positive = [int(delta) for delta in config.delta_indices if int(delta) > 0]
        if positive:
            raise ValueError(
                f"Input modality {name} uses future delta indices {positive}; "
                "counterfactual inference would leak future observations"
            )
    missing_hands = [
        key for key in REQUIRED_HAND_KEYS if key not in modality["state"].modality_keys
    ]
    if missing_hands:
        raise ValueError(f"State modality is missing hand groups: {missing_hands}")
    for key in REQUIRED_HAND_KEYS:
        if key not in modality["action"].modality_keys:
            raise ValueError(f"Action modality is missing required group {key}")
    return spec.action_horizon


def _modality_signature(modality: dict[str, Any]) -> dict[str, Any]:
    return {
        name: {
            "keys": [str(key) for key in config.modality_keys],
            "delta_indices": [int(index) for index in config.delta_indices],
        }
        for name, config in modality.items()
    }


def _visual_frame_indices(
    anchors: Iterable[int],
    modality: dict[str, Any],
    episode_length: int,
) -> list[int]:
    indices = set()
    for name in ("video", "mask"):
        config = modality.get(name)
        if config is None:
            continue
        for anchor in anchors:
            indices.update(anchor + int(delta) for delta in config.delta_indices)
    invalid = sorted(index for index in indices if index < 0 or index >= episode_length)
    if invalid:
        raise IndexError(f"Observation needs visual frames outside episode range: {invalid}")
    return sorted(indices)


def _parsed_observation(
    trajectory: pd.DataFrame,
    anchor: int,
    modality: dict[str, Any],
    embodiment_tag: EmbodimentTag,
) -> dict[str, Any]:
    input_modality = deepcopy(modality)
    input_modality.pop("action")
    point = extract_step_data(trajectory, anchor, input_modality, embodiment_tag)
    flat = {}
    for key, value in point.states.items():
        flat[f"state.{key}"] = value
    for key, value in point.images.items():
        flat[f"video.{key}"] = np.asarray(value)
    for language_key in modality["language"].modality_keys:
        flat[language_key] = point.text
    return parse_observation_gr00t(flat, modality)


def _stable_noise_seed(
    base_seed: int,
    *,
    split: str,
    episode_index: int,
    event_index: int,
    repeat: int,
) -> int:
    payload = f"{base_seed}:{split}:{episode_index}:{event_index}:{repeat}".encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**63)


def _chunk_array(
    chunks: dict[str, Any],
    *,
    batch_index: int,
    action_keys: list[str],
) -> np.ndarray:
    arrays = []
    horizon = None
    for key in action_keys:
        value = np.asarray(chunks[key][batch_index], dtype=np.float64)
        if value.ndim == 1:
            value = value[:, None]
        if value.ndim != 2:
            raise ValueError(f"Policy action {key} has unexpected shape {value.shape}")
        horizon = len(value) if horizon is None else horizon
        if len(value) != horizon:
            raise ValueError("Policy action groups have different horizons")
        arrays.append(value)
    result = np.concatenate(arrays, axis=1)
    if not np.all(np.isfinite(result)):
        raise ValueError("Policy returned non-finite physical actions")
    return result


def _infer_requests(
    policy: Gr00tPolicy,
    requests: list[dict[str, Any]],
    *,
    batch_size: int,
    action_keys: list[str],
) -> dict[tuple[Any, ...], np.ndarray]:
    results = {}
    for batch_start in range(0, len(requests), batch_size):
        batch = requests[batch_start : batch_start + batch_size]
        observations = batch_policy_observations([item["observation"] for item in batch])
        chunks, _ = policy.get_action(
            observations,
            options={
                "inference_mode": "synchronous",
                "noise_seeds": [int(item["noise_seed"]) for item in batch],
            },
        )
        for batch_index, item in enumerate(batch):
            key = item["request_key"]
            if key in results:
                raise RuntimeError(f"Duplicate inference request key: {key}")
            results[key] = _chunk_array(chunks, batch_index=batch_index, action_keys=action_keys)
    return results


def _action_schema(
    trajectory: pd.DataFrame,
    action_keys: list[str],
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray], list[tuple[str, int]]]:
    actions = _stack_columns(trajectory, "action", action_keys)
    labels = []
    dimensions = []
    slices = {}
    cursor = 0
    for key in action_keys:
        width = np.asarray(trajectory[f"action.{key}"].iloc[0]).reshape(-1).size
        slices[key] = np.arange(cursor, cursor + width, dtype=np.int64)
        labels.extend(f"{key}[{index}]" for index in range(width))
        dimensions.extend((key, index) for index in range(width))
        cursor += width
    groups: dict[str, np.ndarray] = {
        "all": np.arange(cursor, dtype=np.int64),
        **slices,
    }
    arm_indices = [slices[key] for key in action_keys if "arm" in key]
    hand_indices = [slices[key] for key in action_keys if "hand" in key]
    left_indices = [slices[key] for key in action_keys if key.startswith("left_")]
    right_indices = [slices[key] for key in action_keys if key.startswith("right_")]
    if arm_indices:
        groups["arms"] = np.concatenate(arm_indices)
    if hand_indices:
        groups["hands"] = np.concatenate(hand_indices)
    if left_indices:
        groups["left_side"] = np.concatenate(left_indices)
    if right_indices:
        groups["right_side"] = np.concatenate(right_indices)
    return actions, labels, groups, dimensions


def _metric_row(
    *,
    values: np.ndarray,
    expert: np.ndarray,
    fresh_proxy_command: np.ndarray,
    strategy_previous_command: np.ndarray,
    group_indices: np.ndarray,
) -> dict[str, float | int]:
    prediction = values[:, group_indices]
    target = expert[:, group_indices]
    error = prediction - target
    within_window_transitions = np.diff(prediction, axis=0)
    if within_window_transitions.size:
        within_window_mean_slew = float(np.mean(np.abs(within_window_transitions)))
        within_window_max_slew = float(np.max(np.abs(within_window_transitions)))
    else:
        within_window_mean_slew = 0.0
        within_window_max_slew = 0.0
    squared_error = np.square(error)
    return {
        "sample_count": int(error.size),
        "sum_absolute_error": float(np.sum(np.abs(error))),
        "sum_squared_error": float(np.sum(squared_error)),
        "mae": float(np.mean(np.abs(error))),
        "mse": float(np.mean(squared_error)),
        "rmse": float(np.sqrt(np.mean(squared_error))),
        "bias": float(np.mean(error)),
        "max_absolute_error": float(np.max(np.abs(error))),
        "jump_from_fresh_proxy_command_mae": float(
            np.mean(np.abs(prediction[0] - fresh_proxy_command[group_indices]))
        ),
        "entry_jump_from_strategy_history_mae": float(
            np.mean(np.abs(prediction[0] - strategy_previous_command[group_indices]))
        ),
        "within_window_mean_absolute_slew": within_window_mean_slew,
        "within_window_max_absolute_slew": within_window_max_slew,
    }


def _base_result_fields(
    *,
    checkpoint_step_value: int,
    decision: Decision,
    repeat: int,
    noise_seed: int,
) -> dict[str, Any]:
    event = decision.event
    return {
        "checkpoint_step": checkpoint_step_value,
        "split": event.split,
        "event_id": event.event_id,
        "loader_position": event.loader_position,
        "episode_index": event.episode_index,
        "event_index": event.event_index,
        "task": event.task,
        "source": event.source,
        "source_episode": event.source_episode,
        "source_data_json_sha256": event.source_data_json_sha256,
        "fresh_proxy_frame": event.fresh_proxy_frame,
        "plateau_end_frame": event.plateau_end_frame,
        "plateau_frames": event.plateau_frames,
        "fps": event.fps,
        "decision_offset": decision.offset,
        "decision_offset_s": decision.offset / event.fps,
        "anchor_frame": decision.anchor_frame,
        "target_start_frame": decision.target_start_frame,
        "target_end_frame": decision.target_end_frame,
        "window_all_stale": decision.window_all_stale,
        "strong": event.strong,
        "event_class": ("strong_action_change" if event.strong else "ambiguous_stationary"),
        "moving_hand": event.moving_hand,
        "left_action_range": event.left_action_range,
        "right_action_range": event.right_action_range,
        "noise_repeat": repeat,
        "noise_seed": noise_seed,
    }


def evaluate_split(
    *,
    policy: Gr00tPolicy,
    loader: LeRobotEpisodeLoader,
    events: list[PlateauEvent],
    checkpoint_step_value: int,
    embodiment_tag: EmbodimentTag,
    action_horizon: int,
    execution_horizon: int,
    decision_offsets: list[int],
    inference_batch_size: int,
    inference_seed: int,
    noise_repeats: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    event_metrics = []
    prediction_rows = []
    skipped_rows = []
    action_keys = [str(key) for key in loader.modality_configs["action"].modality_keys]
    events_by_position: dict[int, list[PlateauEvent]] = defaultdict(list)
    for event in events:
        events_by_position[event.loader_position].append(event)

    for episode_number, (loader_position, episode_events) in enumerate(
        sorted(events_by_position.items()), start=1
    ):
        episode_length = loader.get_episode_length(loader_position)
        decisions = []
        for event in episode_events:
            valid, skipped = plan_decisions(
                event,
                episode_length=episode_length,
                action_horizon=action_horizon,
                execution_horizon=execution_horizon,
                offsets=decision_offsets,
            )
            decisions.extend(valid)
            skipped_rows.extend(
                {"checkpoint_step": checkpoint_step_value, **row} for row in skipped
            )
        if not decisions:
            continue

        anchors = {decision.event.fresh_proxy_frame for decision in decisions} | {
            decision.anchor_frame for decision in decisions
        }
        visual_indices = _visual_frame_indices(anchors, loader.modality_configs, episode_length)
        random.seed(
            _stable_noise_seed(
                inference_seed,
                split=episode_events[0].split,
                episode_index=episode_events[0].episode_index,
                event_index=0,
                repeat=0,
            )
        )
        trajectory = loader.load_episode(loader_position, frame_indices=visual_indices)
        actions, labels, groups, dimensions = _action_schema(trajectory, action_keys)
        parsed = {
            anchor: _parsed_observation(trajectory, anchor, loader.modality_configs, embodiment_tag)
            for anchor in sorted(anchors)
        }

        requests = []
        decisions_by_event: dict[str, list[Decision]] = defaultdict(list)
        for decision in decisions:
            decisions_by_event[decision.event.event_id].append(decision)
        for event_id, event_decisions in decisions_by_event.items():
            event = event_decisions[0].event
            for repeat in range(noise_repeats):
                noise_seed = _stable_noise_seed(
                    inference_seed,
                    split=event.split,
                    episode_index=event.episode_index,
                    event_index=event.event_index,
                    repeat=repeat,
                )
                requests.append(
                    {
                        "request_key": (event_id, repeat, "fresh"),
                        "observation": parsed[event.fresh_proxy_frame],
                        "noise_seed": noise_seed,
                    }
                )
                for decision in event_decisions:
                    requests.append(
                        {
                            "request_key": (
                                event_id,
                                repeat,
                                "stale",
                                decision.offset,
                            ),
                            "observation": parsed[decision.anchor_frame],
                            "noise_seed": noise_seed,
                        }
                    )
        chunks = _infer_requests(
            policy,
            requests,
            batch_size=inference_batch_size,
            action_keys=action_keys,
        )

        for decision in decisions:
            event = decision.event
            expert = actions[decision.target_start_frame : decision.target_end_frame + 1]
            previous_command = actions[event.fresh_proxy_frame]
            for repeat in range(noise_repeats):
                noise_seed = _stable_noise_seed(
                    inference_seed,
                    split=event.split,
                    episode_index=event.episode_index,
                    event_index=event.event_index,
                    repeat=repeat,
                )
                fresh_chunk = chunks[(event.event_id, repeat, "fresh")]
                stale_chunk = chunks[(event.event_id, repeat, "stale", decision.offset)]
                strategies = build_strategy_arrays(
                    stale_chunk=stale_chunk,
                    fresh_chunk=fresh_chunk,
                    pre_dropout_command=previous_command,
                    expert=expert,
                    decision_offset=decision.offset,
                    execution_horizon=execution_horizon,
                )
                base = _base_result_fields(
                    checkpoint_step_value=checkpoint_step_value,
                    decision=decision,
                    repeat=repeat,
                    noise_seed=noise_seed,
                )
                for strategy, values in strategies.items():
                    strategy_previous_command = (
                        fresh_chunk[decision.offset - 1]
                        if strategy == "continue_previous_chunk"
                        else previous_command
                    )
                    for group_name, group_indices in groups.items():
                        event_metrics.append(
                            {
                                **base,
                                "strategy": strategy,
                                "group": group_name,
                                **_metric_row(
                                    values=values,
                                    expert=expert,
                                    fresh_proxy_command=previous_command,
                                    strategy_previous_command=strategy_previous_command,
                                    group_indices=group_indices,
                                ),
                            }
                        )
                    for horizon_position in range(execution_horizon):
                        target_frame = decision.target_start_frame + horizon_position
                        for joint_index, (action_key, key_joint_index) in enumerate(dimensions):
                            prediction = float(values[horizon_position, joint_index])
                            target = float(expert[horizon_position, joint_index])
                            error = prediction - target
                            prediction_rows.append(
                                {
                                    **base,
                                    "strategy": strategy,
                                    "horizon_position": horizon_position,
                                    "target_frame": target_frame,
                                    "joint_index": joint_index,
                                    "action_key": action_key,
                                    "key_joint_index": key_joint_index,
                                    "joint": labels[joint_index],
                                    "prediction": prediction,
                                    "expert_action": target,
                                    "error": error,
                                    "absolute_error": abs(error),
                                    "squared_error": error * error,
                                }
                            )
        LOGGER.info(
            "checkpoint %d | %s: episode %d/%d (episode_index=%d, %d event(s))",
            checkpoint_step_value,
            episode_events[0].split,
            episode_number,
            len(events_by_position),
            episode_events[0].episode_index,
            len(episode_events),
        )
    return event_metrics, prediction_rows, skipped_rows


def _stable_summary_seed(base_seed: int, row: dict[str, Any], scope: str) -> int:
    payload = json.dumps(
        {"base": base_seed, "scope": scope, **row},
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**63)


def _episode_cluster_bootstrap_ci(
    frame: pd.DataFrame,
    *,
    value_column: str,
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    if replicates == 0 or frame.empty:
        return float("nan"), float("nan")
    clusters = [
        group[value_column].to_numpy(dtype=np.float64)
        for _episode, group in frame.groupby("episode_index", sort=True)
    ]
    if not clusters:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = np.empty(replicates, dtype=np.float64)
    for index in range(replicates):
        selected = rng.integers(0, len(clusters), size=len(clusters))
        samples[index] = float(np.mean(np.concatenate([clusters[item] for item in selected])))
    return tuple(float(value) for value in np.quantile(samples, [0.025, 0.975]))


def summarize_metrics(
    event_metrics: pd.DataFrame,
    *,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if event_metrics.empty:
        return pd.DataFrame()
    summaries = []
    scopes = [("all_windows", event_metrics)]
    full_stale = event_metrics[event_metrics["window_all_stale"].astype(bool)]
    if not full_stale.empty:
        scopes.append(("full_stale_window", full_stale))

    group_columns = [
        "checkpoint_step",
        "split",
        "decision_offset",
        "strategy",
        "group",
    ]
    paired_keys = [
        "checkpoint_step",
        "split",
        "event_id",
        "decision_offset",
        "noise_repeat",
        "group",
    ]
    for scope_name, scoped in scopes:
        hold = scoped[scoped["strategy"] == "hold_pre_dropout"][paired_keys + ["mse"]].rename(
            columns={"mse": "hold_mse"}
        )
        with_hold = scoped.merge(hold, on=paired_keys, how="left", validate="many_to_one")
        with_hold["mse_delta_vs_hold"] = with_hold["mse"] - with_hold["hold_mse"]

        for keys, values in with_hold.groupby(group_columns, sort=True, dropna=False):
            row = dict(zip(group_columns, keys, strict=True))
            sample_count = int(values["sample_count"].sum())
            row.update(
                {
                    "window_scope": scope_name,
                    "episodes": int(values["episode_index"].nunique()),
                    "events": int(values["event_id"].nunique()),
                    "noise_samples": int(len(values)),
                    "scalar_predictions": sample_count,
                    "micro_mae": float(values["sum_absolute_error"].sum() / sample_count),
                    "micro_mse": float(values["sum_squared_error"].sum() / sample_count),
                    "micro_rmse": float(np.sqrt(values["sum_squared_error"].sum() / sample_count)),
                    "mean_event_mae": float(values["mae"].mean()),
                    "median_event_mae": float(values["mae"].median()),
                    "mean_event_mse": float(values["mse"].mean()),
                    "median_event_mse": float(values["mse"].median()),
                    "mean_jump_from_fresh_proxy_command_mae": float(
                        values["jump_from_fresh_proxy_command_mae"].mean()
                    ),
                    "mean_entry_jump_from_strategy_history_mae": float(
                        values["entry_jump_from_strategy_history_mae"].mean()
                    ),
                    "within_window_mean_absolute_slew": float(
                        values["within_window_mean_absolute_slew"].mean()
                    ),
                    "mse_delta_vs_hold": float(values["mse_delta_vs_hold"].mean()),
                    "event_win_rate_vs_hold": (
                        float("nan")
                        if row["strategy"] == "hold_pre_dropout"
                        else float(np.mean(values["mse"] < values["hold_mse"]))
                    ),
                }
            )
            low, high = _episode_cluster_bootstrap_ci(
                values,
                value_column="mse_delta_vs_hold",
                replicates=bootstrap_replicates,
                seed=_stable_summary_seed(bootstrap_seed, row, scope_name),
            )
            row["mse_delta_vs_hold_ci95_low"] = low
            row["mse_delta_vs_hold_ci95_high"] = high
            summaries.append(row)
    return pd.DataFrame(summaries).sort_values(
        [
            "checkpoint_step",
            "split",
            "window_scope",
            "group",
            "decision_offset",
            "strategy",
        ]
    )


def summarize_metrics_by_stratum(
    event_metrics: pd.DataFrame,
    *,
    stratum_column: str,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    if event_metrics.empty:
        return pd.DataFrame()
    summaries = []
    for stratum, values in event_metrics.groupby(stratum_column, sort=True, dropna=False):
        summary = summarize_metrics(
            values,
            bootstrap_replicates=bootstrap_replicates,
            bootstrap_seed=bootstrap_seed,
        )
        if summary.empty:
            continue
        summary.insert(2, stratum_column, stratum)
        summaries.append(summary)
    return pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()


def _plot_summary(summary: pd.DataFrame, output_path: Path) -> None:
    if summary.empty:
        return
    if summary["split"].nunique() != 1 or summary["checkpoint_step"].nunique() != 1:
        raise ValueError("Each summary plot must contain exactly one split and checkpoint")
    preferred = summary[
        (summary["window_scope"] == "full_stale_window") & (summary["group"].isin(["all", "hands"]))
    ]
    if preferred.empty:
        preferred = summary[
            (summary["window_scope"] == "all_windows") & (summary["group"].isin(["all", "hands"]))
        ]
    groups = [group for group in ("all", "hands") if group in set(preferred["group"])]
    if not groups:
        return
    figure, axes = plt.subplots(len(groups), 1, figsize=(10, 4.5 * len(groups)), squeeze=False)
    colors = {
        "stale_state_inference": "tab:orange",
        "hold_pre_dropout": "tab:gray",
        "continue_previous_chunk": "tab:blue",
    }
    for row_index, group_name in enumerate(groups):
        axis = axes[row_index, 0]
        group = preferred[preferred["group"] == group_name]
        offsets = sorted(group["decision_offset"].unique())
        width = 0.24
        x_values = np.arange(len(offsets), dtype=np.float64)
        for strategy_index, strategy in enumerate(STRATEGIES):
            strategy_rows = group[group["strategy"] == strategy].set_index("decision_offset")
            values = [
                strategy_rows.loc[offset, "micro_mse"] if offset in strategy_rows.index else np.nan
                for offset in offsets
            ]
            axis.bar(
                x_values + (strategy_index - 1) * width,
                values,
                width,
                label=strategy.replace("_", " "),
                color=colors[strategy],
            )
        axis.set_xticks(x_values, [str(offset) for offset in offsets])
        axis.set_xlabel("Decision offset after fresh proxy (frames)")
        axis.set_ylabel("Unnormalised action MSE")
        axis.set_title(f"{group_name}: identical expert windows")
        axis.grid(axis="y", alpha=0.2)
        axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model_artifact_fingerprints(model_path: Path) -> dict[str, dict[str, Any]]:
    """Fingerprint the exact inference weights and processor contract."""

    candidates = {
        model_path / "config.json",
        model_path / "model.safetensors",
        model_path / "model.safetensors.index.json",
        model_path / "processor_config.json",
        model_path / "statistics.json",
    }
    index_path = model_path / "model.safetensors.index.json"
    if index_path.is_file():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        candidates.update(
            model_path / str(name) for name in set(index.get("weight_map", {}).values())
        )
    result = {}
    for path in sorted(candidates):
        if path.is_file():
            result[path.name] = {
                "size_bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
    return result


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
        dirty = bool(
            subprocess.run(
                ["git", "-C", root, "status", "--porcelain"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        )
        return {"root": root, "commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"root": None, "commit": None, "dirty": None}


def _write_csv(path: Path, rows: list[dict[str, Any]] | pd.DataFrame) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _write_predictions(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as stream:
        pd.DataFrame(rows).to_csv(stream, index=False)


def _scan_events(
    *,
    specs: list[DatasetSpec],
    modality: dict[str, Any],
    minimum_frames: int,
    strong_action_range: float,
    event_filter: str,
    tasks: list[str] | None,
    max_events_per_split: int,
    provenance: dict[tuple[str, int], dict[str, Any]],
) -> tuple[dict[str, list[PlateauEvent]], list[dict[str, Any]]]:
    selected_by_split = {}
    manifest_rows = []
    task_filter = set(tasks or [])
    for spec in specs:
        if not (spec.path / "meta" / "info.json").is_file():
            raise FileNotFoundError(f"Dataset is not ready: {spec.path}")
        loader = LeRobotEpisodeLoader(spec.path, modality_configs=modality)
        events = []
        excluded_task_episodes = 0
        for loader_position, metadata in enumerate(loader.episodes_metadata):
            episode_index = int(metadata["episode_index"])
            task = _task_for_episode(metadata)
            if task_filter and task not in task_filter:
                excluded_task_episodes += 1
                continue
            numeric = loader._load_parquet_data(episode_index)
            events.extend(
                detect_plateaus(
                    numeric,
                    split=spec.split,
                    dataset_path=spec.path,
                    loader_position=loader_position,
                    episode_index=episode_index,
                    task=task,
                    fps=float(loader.fps),
                    minimum_frames=minimum_frames,
                    strong_action_range=strong_action_range,
                    provenance=provenance.get((spec.split, episode_index)),
                )
            )
        eligible = [event for event in events if event_filter == "all" or event.strong]
        if max_events_per_split:
            eligible = eligible[:max_events_per_split]
        selected_ids = {event.event_id for event in eligible}
        for event in events:
            if event.event_id in selected_ids:
                selection = "selected"
            elif event_filter == "strong" and not event.strong:
                selection = "not_strong"
            else:
                selection = "smoke_cap"
            manifest_rows.append(
                {
                    **asdict(event),
                    "event_id": event.event_id,
                    "selected": event.event_id in selected_ids,
                    "selection": selection,
                }
            )
        selected_by_split[spec.split] = eligible
        if not eligible:
            requested = sorted(task_filter) if task_filter else ["all tasks"]
            raise ValueError(
                f"{spec.split}: no plateau events selected for task filter {requested}"
            )
        LOGGER.info(
            "%s: detected %d plateau candidate(s), %d strong, selected %d "
            "(%d off-task episode(s) skipped)",
            spec.split,
            len(events),
            sum(event.strong for event in events),
            len(eligible),
            excluded_task_episodes,
        )
        del loader
    return selected_by_split, manifest_rows


def _run_metadata(
    *,
    args: argparse.Namespace,
    targets: list[dict[str, Any]],
    modality_signature: dict[str, Any],
    elapsed_seconds: float,
) -> dict[str, Any]:
    arguments = {}
    for key, value in vars(args).items():
        if key == "dataset":
            arguments[key] = [{"split": item.split, "path": str(item.path)} for item in value]
        elif isinstance(value, Path):
            arguments[key] = str(value)
        else:
            arguments[key] = value
    return {
        "version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "offline_only": True,
        "interpretation": (
            "Open-loop agreement with recorded human command targets; not closed-loop "
            "robot safety or task success."
        ),
        "fresh_proxy_assumption": (
            "For q[f] == ... == q[e], f is treated as the last fresh/cache-fill proxy; "
            "historical recordings contain no DDS receipt timestamps."
        ),
        "arguments": arguments,
        "targets": targets,
        "modality_signature": modality_signature,
        "datasets": [
            {
                "split": spec.split,
                "path": str(spec.path.resolve()),
                "info_sha256": _sha256(spec.path / "meta" / "info.json"),
                "episodes_sha256": _sha256(spec.path / "meta" / "episodes.jsonl"),
                "tasks_sha256": _sha256(spec.path / "meta" / "tasks.jsonl"),
                "modality_sha256": _sha256(spec.path / "meta" / "modality.json"),
                "merge_manifest_sha256": _sha256(
                    spec.path.parent / "provenance" / "merge_manifest.json"
                ),
            }
            for spec in args.dataset
        ],
        "analysis_files": {
            str(Path(__file__).resolve()): _sha256(Path(__file__).resolve()),
            str(
                Path(__file__).resolve().with_name("multi_stale_state_counterfactual_evaluation.sh")
            ): _sha256(
                Path(__file__).resolve().with_name("multi_stale_state_counterfactual_evaluation.sh")
            ),
        },
        "git": {
            "isaac_groot": _git_state(Path.cwd()),
            "analysis_scripts": _git_state(Path(__file__).resolve().parent),
        },
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    targets = find_targets(args.run_dir, args.checkpoint_steps, args.base_model_path)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    running_marker = args.output_dir / "INCOMPLETE"
    running_marker.write_text(
        "Evaluation is incomplete until this marker is removed.\n", encoding="utf-8"
    )
    started = time.perf_counter()
    embodiment_tag = EmbodimentTag.resolve(args.embodiment_tag)
    provenance = {}
    for spec in args.dataset:
        candidate = _provenance_map(spec.path)
        for key, value in candidate.items():
            if key in provenance and provenance[key] != value:
                raise ValueError(f"Conflicting provenance for {key}")
            provenance[key] = value
    _verify_evaluation_provenance(args.dataset, provenance)

    canonical_signature = None
    selected_events = None
    all_metrics = []
    metadata_targets = []

    try:
        for target_number, target in enumerate(targets, start=1):
            LOGGER.info(
                "Loading target %d/%d: checkpoint step %d from %s",
                target_number,
                len(targets),
                target.step,
                target.model_path,
            )
            policy = Gr00tPolicy(
                embodiment_tag=embodiment_tag,
                model_path=str(target.model_path),
                device=args.device,
                processor_path=(
                    str(target.processor_path) if target.processor_path is not None else None
                ),
            )
            policy.model.action_head.num_inference_timesteps = args.denoising_steps
            modality = policy.get_modality_config()
            action_horizon = _validate_input_contract(modality, args.execution_horizon)
            invalid_offsets = [
                offset
                for offset in args.decision_offsets
                if offset + args.execution_horizon > action_horizon
            ]
            if invalid_offsets:
                raise ValueError(
                    f"Decision offsets {invalid_offsets} plus execution horizon "
                    f"{args.execution_horizon} exceed action horizon {action_horizon}"
                )
            signature = _modality_signature(modality)
            if canonical_signature is None:
                canonical_signature = signature
                selected_events, event_manifest_rows = _scan_events(
                    specs=args.dataset,
                    modality=modality,
                    minimum_frames=args.minimum_plateau_frames,
                    strong_action_range=args.strong_action_range,
                    event_filter=args.event_filter,
                    tasks=args.task,
                    max_events_per_split=args.max_events_per_split,
                    provenance=provenance,
                )
                _write_csv(args.output_dir / "events.csv", event_manifest_rows)
            elif signature != canonical_signature:
                raise ValueError(
                    "Checkpoint modality contract changed within one run; refusing to "
                    "compare different observation/action schemas"
                )

            target_dir = args.output_dir / f"checkpoint-{target.step}"
            target_dir.mkdir(parents=True, exist_ok=False)
            target_metrics = []
            target_predictions = []
            target_skipped = []
            for spec in args.dataset:
                loader = LeRobotEpisodeLoader(spec.path, modality_configs=modality)
                metrics, predictions, skipped = evaluate_split(
                    policy=policy,
                    loader=loader,
                    events=selected_events[spec.split],
                    checkpoint_step_value=target.step,
                    embodiment_tag=embodiment_tag,
                    action_horizon=action_horizon,
                    execution_horizon=args.execution_horizon,
                    decision_offsets=args.decision_offsets,
                    inference_batch_size=args.inference_batch_size,
                    inference_seed=args.inference_seed,
                    noise_repeats=args.noise_repeats,
                )
                target_metrics.extend(metrics)
                target_predictions.extend(predictions)
                target_skipped.extend(skipped)
                if not metrics:
                    raise ValueError(
                        f"{spec.split}: selected plateau events produced no valid decision windows"
                    )
                del loader

            metrics_frame = pd.DataFrame(target_metrics)
            summary = summarize_metrics(
                metrics_frame,
                bootstrap_replicates=args.bootstrap_replicates,
                bootstrap_seed=args.inference_seed,
            )
            _write_csv(target_dir / "event_metrics.csv", metrics_frame)
            _write_csv(target_dir / "strategy_summary.csv", summary)
            for stratum in ("event_class", "moving_hand", "source"):
                _write_csv(
                    target_dir / f"strategy_summary_by_{stratum}.csv",
                    summarize_metrics_by_stratum(
                        metrics_frame,
                        stratum_column=stratum,
                        bootstrap_replicates=args.bootstrap_replicates,
                        bootstrap_seed=args.inference_seed,
                    ),
                )
            _write_csv(target_dir / "skipped_decisions.csv", target_skipped)
            _write_predictions(target_dir / "frame_joint_predictions.csv.gz", target_predictions)
            for dataset_spec in args.dataset:
                _plot_summary(
                    summary[summary["split"] == dataset_spec.split],
                    target_dir / f"strategy_mse_by_offset_{dataset_spec.split}.png",
                )
            all_metrics.extend(target_metrics)
            metadata_targets.append(
                {
                    "step": target.step,
                    "model_path": str(target.model_path.resolve()),
                    "processor_path": (
                        str(target.processor_path.resolve())
                        if target.processor_path is not None
                        else None
                    ),
                    "config_sha256": _sha256(target.model_path / "config.json"),
                    "artifact_fingerprints": _model_artifact_fingerprints(target.model_path),
                    "processor_artifact_fingerprints": (
                        _model_artifact_fingerprints(target.processor_path)
                        if target.processor_path is not None
                        else None
                    ),
                }
            )
            del policy
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        all_metrics_frame = pd.DataFrame(all_metrics)
        combined_summary = summarize_metrics(
            all_metrics_frame,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_seed=args.inference_seed,
        )
        _write_csv(
            args.output_dir / "event_metrics_all_checkpoints.csv",
            all_metrics_frame,
        )
        _write_csv(
            args.output_dir / "strategy_summary_all_checkpoints.csv",
            combined_summary,
        )
        for stratum in ("event_class", "moving_hand", "source"):
            _write_csv(
                args.output_dir / f"strategy_summary_by_{stratum}_all_checkpoints.csv",
                summarize_metrics_by_stratum(
                    all_metrics_frame,
                    stratum_column=stratum,
                    bootstrap_replicates=args.bootstrap_replicates,
                    bootstrap_seed=args.inference_seed,
                ),
            )
        metadata = _run_metadata(
            args=args,
            targets=metadata_targets,
            modality_signature=canonical_signature,
            elapsed_seconds=time.perf_counter() - started,
        )
        (args.output_dir / "run_metadata.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        running_marker.unlink()
        LOGGER.info(
            "Finished %d checkpoint(s) in %.1f minutes. Results: %s",
            len(targets),
            (time.perf_counter() - started) / 60.0,
            args.output_dir,
        )
        return 0
    except BaseException:
        LOGGER.exception("Counterfactual evaluation failed; INCOMPLETE marker retained")
        raise


if __name__ == "__main__":
    raise SystemExit(main())
