#!/usr/bin/env python3
"""Safely prune non-best early checkpoints from the five Inspire 655 runs.

This utility is intentionally narrow.  It knows the exact five model directory
names involved in the 655-episode RGB / normals / RGB-D comparison and will
only ever consider checkpoint-5000, checkpoint-10000, and checkpoint-15000 for
deletion.  checkpoint-20000, checkpoint-25000, and every canonical
validation-MSE best checkpoint are immutable from this script's perspective.

The active RGB-D run is also the completion gate: both its training and
evaluation rows must be PASS/0 before either a check or an apply can succeed.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import sys
from typing import Any, Iterable
import uuid


DEFAULT_MODELS_ROOT = Path("/home/alex/Development/Models/inspire")
DEFAULT_RGBD_STATUS_FILE = Path(
    "/home/alex/Development/logs/groot/training/"
    "multi_finetune_evaluation_25k_rgbd_pre_"
    "all_tasks_655eps_20260916_normals_range_mask_v2_20260918_evaluation_status.tsv"
)
DEFAULT_AUDIT_DIR = Path("/home/alex/Development/logs/groot/checkpoint_cleanup")

TARGET_STEPS = (5000, 10000, 15000, 20000, 25000)
DELETABLE_STEPS = frozenset((5000, 10000, 15000))
ALWAYS_KEEP_STEPS = frozenset((20000, 25000))
METRICS_RELATIVE_PATH = Path("normalized_action_metrics_exec_hor_8/checkpoint_metric_summary.csv")
CANONICAL_VALIDATION_COUNTS = {
    "episodes": 65,
    "frames": 22719,
    "dimensions": 26,
    "samples": 590694,
}

MODEL_NAMES = {
    "rgb_655": (
        "inspire_c_rgb_patch_tuned_bf16_batch_32_acc_1_25k_"
        "all_tasks_655eps_20260916_normals_range_mask_v2_20260917"
    ),
    "normals_early_655": (
        "inspire_c_normals_6ch_early_fusion_patch_tuned_normals_init_rgb_mean_"
        "bf16_batch_32_acc_1_25k_all_tasks_655eps_20260916_normals_range_mask_v2"
    ),
    "normals_late_pre_655": (
        "inspire_c_rgb_surface_normals_late_fusion_pre_adapter_"
        "4x_linear_rgb50_geo50_patch_frozen_bf16_batch_32_acc_1_25k_"
        "all_tasks_655eps_20260916_normals_range_mask_v2_20260918"
    ),
    "normals_late_post_655": (
        "inspire_c_rgb_surface_normals_late_fusion_post_adapter_"
        "4x_linear_rgb50_geo50_patch_frozen_bf16_batch_32_acc_1_25k_"
        "all_tasks_655eps_20260916_normals_range_mask_v2_20260918"
    ),
    "rgbd_late_pre_655": (
        "inspire_c_rgbd_late_fusion_pre_adapter_"
        "4x_linear_rgb50_geo50_patch_frozen_bf16_batch_32_acc_1_25k_"
        "all_tasks_655eps_20260916_normals_range_mask_v2_20260918"
    ),
}
RGBD_LABEL = "rgbd_late_pre_655"


class SafetyError(RuntimeError):
    """A fail-closed precondition or postcondition failed."""


@dataclass(frozen=True)
class CheckpointRecord:
    model_label: str
    model_path: str
    checkpoint_step: int
    checkpoint_path: str
    mse: str
    is_best: bool
    existed_before: bool
    bytes_before: int
    files_before: int
    action: str
    reason: str
    device_before: int | None
    inode_before: int | None


@dataclass(frozen=True)
class ModelRecord:
    label: str
    path: str
    metrics_path: str
    metrics_sha256: str
    best_mse: str
    best_steps: tuple[int, ...]
    checkpoints: tuple[CheckpointRecord, ...]


@dataclass(frozen=True)
class Plan:
    mode: str
    models_root: str
    rgbd_status_file: str
    rgbd_status_sha256: str
    models: tuple[ModelRecord, ...]

    @property
    def deletion_targets(self) -> tuple[CheckpointRecord, ...]:
        return tuple(
            checkpoint
            for model in self.models
            for checkpoint in model.checkpoints
            if checkpoint.action == "delete"
        )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _assert_plain_directory(path: Path, description: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise SafetyError(f"missing {description}: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise SafetyError(f"refusing symlink {description}: {path}")
    if not stat.S_ISDIR(info.st_mode):
        raise SafetyError(f"{description} is not a directory: {path}")
    if path.resolve(strict=True) != path.absolute():
        raise SafetyError(f"{description} resolves outside its literal path: {path}")


def _assert_plain_file(path: Path, description: str) -> None:
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise SafetyError(f"missing {description}: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        raise SafetyError(f"refusing symlink {description}: {path}")
    if not stat.S_ISREG(info.st_mode):
        raise SafetyError(f"{description} is not a regular file: {path}")
    if path.resolve(strict=True) != path.absolute():
        raise SafetyError(f"{description} resolves outside its literal path: {path}")


def _assert_direct_child(parent: Path, child: Path, description: str) -> None:
    if child.parent != parent or child.name not in MODEL_NAMES.values():
        raise SafetyError(f"unexpected {description} outside the exact allowlist: {child}")
    if child.resolve(strict=True).parent != parent.resolve(strict=True):
        raise SafetyError(f"{description} escapes its allowed parent: {child}")


def _decode_mountinfo_path(value: str) -> str:
    """Decode the octal escapes used for paths in /proc/self/mountinfo."""

    for encoded, decoded in (
        (r"\040", " "),
        (r"\011", "\t"),
        (r"\012", "\n"),
        (r"\134", "\\"),
    ):
        value = value.replace(encoded, decoded)
    return value


def _mount_points() -> frozenset[Path]:
    """Return visible mount points, including same-device bind mounts."""

    mountinfo_path = Path("/proc/self/mountinfo")
    try:
        lines = mountinfo_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise SafetyError(f"cannot inspect mount boundaries via {mountinfo_path}: {exc}") from exc
    points: set[Path] = set()
    for line_number, line in enumerate(lines, start=1):
        fields = line.split()
        if len(fields) < 10 or "-" not in fields:
            raise SafetyError(f"malformed {mountinfo_path} line {line_number}")
        mount_point = Path(_decode_mountinfo_path(fields[4]))
        if not mount_point.is_absolute():
            raise SafetyError(
                f"non-absolute mount point in {mountinfo_path} line {line_number}: {mount_point}"
            )
        points.add(mount_point)
    return frozenset(points)


def _checkpoint_tree_stats(path: Path) -> tuple[int, int, int, int]:
    """Return bytes, regular-file count, device, inode while rejecting links/specials."""

    _assert_plain_directory(path, "checkpoint directory")
    root_info = path.stat(follow_symlinks=False)
    mount_points = _mount_points()
    if path.absolute() in mount_points or os.path.ismount(path):
        raise SafetyError(f"refusing checkpoint that is itself a mount point: {path}")
    total_bytes = 0
    file_count = 0
    for current_root, directory_names, file_names in os.walk(path, followlinks=False):
        current = Path(current_root)
        for name in directory_names:
            entry = current / name
            entry_info = entry.lstat()
            if stat.S_ISLNK(entry_info.st_mode):
                raise SafetyError(f"refusing checkpoint containing directory symlink: {entry}")
            if not stat.S_ISDIR(entry_info.st_mode):
                raise SafetyError(f"refusing non-directory checkpoint entry: {entry}")
            if entry_info.st_dev != root_info.st_dev:
                raise SafetyError(f"refusing cross-device checkpoint directory: {entry}")
            if entry.absolute() in mount_points or os.path.ismount(entry):
                raise SafetyError(f"refusing mounted checkpoint directory: {entry}")
        for name in file_names:
            entry = current / name
            entry_info = entry.lstat()
            if stat.S_ISLNK(entry_info.st_mode):
                raise SafetyError(f"refusing checkpoint containing file symlink: {entry}")
            if not stat.S_ISREG(entry_info.st_mode):
                raise SafetyError(f"refusing special checkpoint entry: {entry}")
            if entry_info.st_dev != root_info.st_dev:
                raise SafetyError(f"refusing cross-device checkpoint file: {entry}")
            if entry.absolute() in mount_points or os.path.ismount(entry):
                raise SafetyError(f"refusing mounted checkpoint file: {entry}")
            total_bytes += entry_info.st_size
            file_count += 1
    if file_count == 0:
        raise SafetyError(f"refusing empty/incomplete checkpoint directory: {path}")
    return total_bytes, file_count, root_info.st_dev, root_info.st_ino


def _parse_nonnegative_decimal(value: str, field: str, location: str) -> Decimal:
    try:
        parsed = Decimal(value)
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise SafetyError(f"invalid {field} at {location}: {value!r}") from exc
    if not parsed.is_finite() or parsed < 0:
        raise SafetyError(f"non-finite or negative {field} at {location}: {value!r}")
    return parsed


def _parse_positive_int(value: str, field: str, location: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise SafetyError(f"invalid {field} at {location}: {value!r}") from exc
    if parsed <= 0:
        raise SafetyError(f"non-positive {field} at {location}: {value!r}")
    return parsed


def _read_validation_mse(metrics_path: Path) -> tuple[dict[int, Decimal], str]:
    _assert_plain_file(metrics_path, "canonical normalized metric summary")
    required_columns = {
        "split",
        "checkpoint_step",
        "episodes",
        "frames",
        "mae",
        "mse",
        "dimensions",
        "samples",
    }
    try:
        with metrics_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames is None:
                raise SafetyError(f"metrics file has no header: {metrics_path}")
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise SafetyError(f"metrics file has duplicate column names: {metrics_path}")
            missing_columns = required_columns.difference(reader.fieldnames)
            if missing_columns:
                raise SafetyError(
                    f"metrics file missing required columns {sorted(missing_columns)}: {metrics_path}"
                )
            rows = list(reader)
    except UnicodeError as exc:
        raise SafetyError(f"metrics file is not valid UTF-8: {metrics_path}") from exc

    seen_keys: set[tuple[str, int]] = set()
    required_values: dict[int, Decimal] = {}
    for row_number, row in enumerate(rows, start=2):
        split = (row.get("split") or "").strip()
        raw_step = (row.get("checkpoint_step") or "").strip()
        try:
            step = int(raw_step)
        except ValueError as exc:
            raise SafetyError(
                f"invalid checkpoint_step at {metrics_path}:{row_number}: {raw_step!r}"
            ) from exc
        key = (split, step)
        if key in seen_keys:
            raise SafetyError(
                f"duplicate metric row for split={split!r}, checkpoint_step={step}: {metrics_path}"
            )
        seen_keys.add(key)
        if split != "validation" or step not in TARGET_STEPS:
            continue
        location = f"{metrics_path}:{row_number}"
        observed_counts = {
            field: _parse_positive_int(row[field], field, location)
            for field in CANONICAL_VALIDATION_COUNTS
        }
        if observed_counts["frames"] * observed_counts["dimensions"] != observed_counts["samples"]:
            raise SafetyError(
                f"inconsistent validation sample count at {location}: "
                f"frames*dimensions={observed_counts['frames'] * observed_counts['dimensions']} "
                f"but samples={observed_counts['samples']}"
            )
        if observed_counts != CANONICAL_VALIDATION_COUNTS:
            raise SafetyError(
                f"non-canonical validation cohort at {location}: "
                f"observed={observed_counts}, expected={CANONICAL_VALIDATION_COUNTS}"
            )
        _parse_nonnegative_decimal(row["mae"], "mae", location)
        required_values[step] = _parse_nonnegative_decimal(row["mse"], "mse", location)

    missing_steps = set(TARGET_STEPS).difference(required_values)
    if missing_steps:
        raise SafetyError(
            f"canonical validation metrics are incomplete; missing checkpoints "
            f"{sorted(missing_steps)} in {metrics_path}"
        )
    return required_values, _sha256(metrics_path)


def _validate_rgbd_pass_gate(status_path: Path, expected_model_path: Path) -> str:
    _assert_plain_file(status_path, "RGB-D training/evaluation status file")
    try:
        with status_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            expected_header = ["stage", "model", "status", "exit_code"]
            if reader.fieldnames != expected_header:
                raise SafetyError(
                    f"unexpected RGB-D status header {reader.fieldnames!r}; "
                    f"expected {expected_header!r}: {status_path}"
                )
            rows = list(reader)
    except UnicodeError as exc:
        raise SafetyError(f"RGB-D status file is not valid UTF-8: {status_path}") from exc

    if len(rows) != 2:
        raise SafetyError(
            f"RGB-D status gate requires exactly two rows (training and evaluation), "
            f"found {len(rows)}: {status_path}"
        )
    by_stage: dict[str, dict[str, str]] = {}
    expected_model = str(expected_model_path)
    for row in rows:
        stage = row["stage"]
        if stage in by_stage:
            raise SafetyError(f"duplicate RGB-D status row for stage {stage!r}: {status_path}")
        if stage not in {"training", "evaluation"}:
            raise SafetyError(f"unexpected RGB-D status stage {stage!r}: {status_path}")
        if row["model"] != expected_model:
            raise SafetyError(
                f"RGB-D status row belongs to unexpected model {row['model']!r}; "
                f"expected {expected_model!r}"
            )
        if row["status"] != "PASS" or row["exit_code"] != "0":
            raise SafetyError(
                f"RGB-D {stage} has not passed cleanly: "
                f"status={row['status']!r}, exit_code={row['exit_code']!r}"
            )
        by_stage[stage] = row
    if set(by_stage) != {"training", "evaluation"}:
        raise SafetyError(f"RGB-D status gate is incomplete: {status_path}")
    return _sha256(status_path)


def _build_plan(mode: str, models_root: Path, status_path: Path) -> Plan:
    models_root = models_root.absolute()
    _assert_plain_directory(models_root, "models root")

    model_paths = {label: models_root / name for label, name in MODEL_NAMES.items()}
    rgbd_status_sha256 = _validate_rgbd_pass_gate(status_path, model_paths[RGBD_LABEL])

    model_records: list[ModelRecord] = []
    for label, model_path in model_paths.items():
        _assert_plain_directory(model_path, f"{label} model directory")
        _assert_direct_child(models_root, model_path, f"{label} model directory")
        metrics_path = model_path / METRICS_RELATIVE_PATH
        mse_by_step, metrics_sha256 = _read_validation_mse(metrics_path)
        best_mse = min(mse_by_step.values())
        best_steps = tuple(step for step in TARGET_STEPS if mse_by_step[step] == best_mse)

        checkpoint_records: list[CheckpointRecord] = []
        for step in TARGET_STEPS:
            checkpoint_path = model_path / f"checkpoint-{step}"
            exists = checkpoint_path.exists() or checkpoint_path.is_symlink()
            bytes_before = 0
            files_before = 0
            device_before: int | None = None
            inode_before: int | None = None
            if exists:
                (
                    bytes_before,
                    files_before,
                    device_before,
                    inode_before,
                ) = _checkpoint_tree_stats(checkpoint_path)

            is_best = step in best_steps
            if step in ALWAYS_KEEP_STEPS:
                if not exists:
                    raise SafetyError(
                        f"required always-kept checkpoint-{step} is missing for {label}: "
                        f"{checkpoint_path}"
                    )
                action = "keep"
                reason = "always_keep_and_best" if is_best else "always_keep"
            elif is_best:
                if not exists:
                    raise SafetyError(
                        f"canonical best checkpoint-{step} is missing for {label}: {checkpoint_path}"
                    )
                action = "keep"
                reason = "canonical_validation_mse_best"
            elif exists:
                action = "delete"
                reason = "early_non_best"
            else:
                action = "absent"
                reason = "early_non_best_already_absent"

            checkpoint_records.append(
                CheckpointRecord(
                    model_label=label,
                    model_path=str(model_path),
                    checkpoint_step=step,
                    checkpoint_path=str(checkpoint_path),
                    mse=str(mse_by_step[step]),
                    is_best=is_best,
                    existed_before=exists,
                    bytes_before=bytes_before,
                    files_before=files_before,
                    action=action,
                    reason=reason,
                    device_before=device_before,
                    inode_before=inode_before,
                )
            )

        model_records.append(
            ModelRecord(
                label=label,
                path=str(model_path),
                metrics_path=str(metrics_path),
                metrics_sha256=metrics_sha256,
                best_mse=str(best_mse),
                best_steps=best_steps,
                checkpoints=tuple(checkpoint_records),
            )
        )

    return Plan(
        mode=mode,
        models_root=str(models_root),
        rgbd_status_file=str(status_path.absolute()),
        rgbd_status_sha256=rgbd_status_sha256,
        models=tuple(model_records),
    )


def _plan_fingerprint(plan: Plan) -> dict[str, Any]:
    return {
        "rgbd_status_sha256": plan.rgbd_status_sha256,
        "models": [
            {
                "label": model.label,
                "metrics_sha256": model.metrics_sha256,
                "best_steps": model.best_steps,
                "checkpoints": [
                    {
                        "step": checkpoint.checkpoint_step,
                        "exists": checkpoint.existed_before,
                        "action": checkpoint.action,
                        "bytes": checkpoint.bytes_before,
                        "files": checkpoint.files_before,
                        "device": checkpoint.device_before,
                        "inode": checkpoint.inode_before,
                    }
                    for checkpoint in model.checkpoints
                ],
            }
            for model in plan.models
        ],
    }


def _json_payload(
    plan: Plan, run_id: str, phase: str, result: str, error: str | None
) -> dict[str, Any]:
    deletion_bytes = sum(checkpoint.bytes_before for checkpoint in plan.deletion_targets)
    models_payload: list[dict[str, Any]] = []
    for model in plan.models:
        model_payload = asdict(model)
        for checkpoint_payload, checkpoint in zip(
            model_payload["checkpoints"], model.checkpoints, strict=True
        ):
            checkpoint_path = Path(checkpoint.checkpoint_path)
            checkpoint_payload["exists_at_report"] = (
                checkpoint_path.exists() or checkpoint_path.is_symlink()
            )
        models_payload.append(model_payload)
    return {
        "schema_version": 1,
        "run_id": run_id,
        "phase": phase,
        "generated_at_utc": _utc_now(),
        "mode": plan.mode,
        "result": result,
        "error": error,
        "safety_contract": {
            "target_steps": list(TARGET_STEPS),
            "deletable_steps": sorted(DELETABLE_STEPS),
            "always_keep_steps": sorted(ALWAYS_KEEP_STEPS),
            "best_metric": "normalized validation overall MSE",
            "metric_source_relative_path": str(METRICS_RELATIVE_PATH),
            "canonical_validation_counts": CANONICAL_VALIDATION_COUNTS,
            "tie_rule": "exact finite Decimal equality",
        },
        "models_root": plan.models_root,
        "rgbd_completion_gate": {
            "status_file": plan.rgbd_status_file,
            "sha256": plan.rgbd_status_sha256,
            "required_rows": ["training:PASS:0", "evaluation:PASS:0"],
        },
        "planned_deletion_count": len(plan.deletion_targets),
        "planned_deletion_bytes": deletion_bytes,
        "models": models_payload,
    }


def _tsv_text(plan: Plan, run_id: str, phase: str, result: str) -> str:
    output = io.StringIO(newline="")
    fieldnames = [
        "run_id",
        "phase",
        "mode",
        "result",
        "model_label",
        "checkpoint_step",
        "mse",
        "is_best",
        "existed_before",
        "action",
        "reason",
        "bytes_before",
        "files_before",
        "exists_at_report",
        "checkpoint_path",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
    writer.writeheader()
    for model in plan.models:
        for checkpoint in model.checkpoints:
            checkpoint_path = Path(checkpoint.checkpoint_path)
            writer.writerow(
                {
                    "run_id": run_id,
                    "phase": phase,
                    "mode": plan.mode,
                    "result": result,
                    "model_label": checkpoint.model_label,
                    "checkpoint_step": checkpoint.checkpoint_step,
                    "mse": checkpoint.mse,
                    "is_best": str(checkpoint.is_best).lower(),
                    "existed_before": str(checkpoint.existed_before).lower(),
                    "action": checkpoint.action,
                    "reason": checkpoint.reason,
                    "bytes_before": checkpoint.bytes_before,
                    "files_before": checkpoint.files_before,
                    "exists_at_report": str(
                        checkpoint_path.exists() or checkpoint_path.is_symlink()
                    ).lower(),
                    "checkpoint_path": checkpoint.checkpoint_path,
                }
            )
    return output.getvalue()


def _prepare_audit_dir(audit_dir: Path, model_paths: Iterable[Path]) -> Path:
    audit_dir = audit_dir.absolute()
    if audit_dir.exists() or audit_dir.is_symlink():
        _assert_plain_directory(audit_dir, "audit directory")
    else:
        audit_dir.mkdir(parents=True, exist_ok=False)
        _assert_plain_directory(audit_dir, "audit directory")
    resolved_audit = audit_dir.resolve(strict=True)
    for model_path in model_paths:
        resolved_model = model_path.resolve(strict=True)
        if resolved_audit == resolved_model or resolved_model in resolved_audit.parents:
            raise SafetyError(f"audit directory must not be inside a model directory: {audit_dir}")
    return audit_dir


def _durable_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_reports(
    plan: Plan,
    audit_dir: Path,
    run_id: str,
    phase: str,
    result: str,
    error: str | None = None,
) -> tuple[Path, Path]:
    json_path = audit_dir / f"{run_id}.{phase}.json"
    tsv_path = audit_dir / f"{run_id}.{phase}.tsv"
    _durable_write(
        json_path,
        json.dumps(_json_payload(plan, run_id, phase, result, error), indent=2, sort_keys=True)
        + "\n",
    )
    _durable_write(tsv_path, _tsv_text(plan, run_id, phase, result))
    return json_path, tsv_path


def _delete_planned_checkpoints(plan: Plan) -> None:
    for checkpoint in plan.deletion_targets:
        checkpoint_path = Path(checkpoint.checkpoint_path)
        if checkpoint.checkpoint_step not in DELETABLE_STEPS:
            raise SafetyError(f"internal safety violation: non-deletable step: {checkpoint_path}")
        model_path = Path(checkpoint.model_path)
        expected = model_path / f"checkpoint-{checkpoint.checkpoint_step}"
        if checkpoint_path != expected:
            raise SafetyError(
                f"internal safety violation: unexpected deletion path: {checkpoint_path}"
            )
        _assert_plain_directory(model_path, "model directory immediately before deletion")
        _assert_plain_directory(checkpoint_path, "checkpoint immediately before deletion")
        size, files, device, inode = _checkpoint_tree_stats(checkpoint_path)
        if (
            size != checkpoint.bytes_before
            or files != checkpoint.files_before
            or device != checkpoint.device_before
            or inode != checkpoint.inode_before
        ):
            raise SafetyError(
                f"checkpoint changed after audit manifest was written: {checkpoint_path}"
            )
        shutil.rmtree(checkpoint_path)
        if checkpoint_path.exists() or checkpoint_path.is_symlink():
            raise SafetyError(f"checkpoint still exists after deletion: {checkpoint_path}")


def _verify_postconditions(plan: Plan) -> None:
    for model in plan.models:
        for checkpoint in model.checkpoints:
            path = Path(checkpoint.checkpoint_path)
            exists = path.exists() or path.is_symlink()
            if checkpoint.action == "delete":
                if plan.mode == "apply" and exists:
                    raise SafetyError(f"planned deletion remains present: {path}")
                if plan.mode == "check" and not exists:
                    raise SafetyError(f"dry-run target unexpectedly disappeared: {path}")
            if checkpoint.action == "keep" and not exists:
                raise SafetyError(f"protected checkpoint disappeared: {path}")
            if exists:
                _checkpoint_tree_stats(path)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--check",
        "--dry-run",
        dest="mode",
        action="store_const",
        const="check",
        help="validate everything and write an audit plan without deleting",
    )
    mode_group.add_argument(
        "--apply",
        dest="mode",
        action="store_const",
        const="apply",
        help="write the pre-delete audit, then delete only eligible checkpoints",
    )
    parser.add_argument("--models-root", type=Path, default=DEFAULT_MODELS_ROOT)
    parser.add_argument("--rgbd-status-file", type=Path, default=DEFAULT_RGBD_STATUS_FILE)
    parser.add_argument("--audit-dir", type=Path, default=DEFAULT_AUDIT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    plan: Plan | None = None
    audit_dir: Path | None = None
    run_id: str | None = None
    try:
        plan = _build_plan(args.mode, args.models_root, args.rgbd_status_file)
        audit_dir = _prepare_audit_dir(args.audit_dir, (Path(model.path) for model in plan.models))
        run_id = (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ") + f"-{args.mode}-{os.getpid()}"
        )
        pre_json, pre_tsv = _write_reports(plan, audit_dir, run_id, "pre", "validated")
        print(f"Pre-delete audit JSON: {pre_json}")
        print(f"Pre-delete audit TSV:  {pre_tsv}")
        print(
            f"Eligible early checkpoints: {len(plan.deletion_targets)}; "
            f"bytes: {sum(item.bytes_before for item in plan.deletion_targets)}"
        )

        # Close the audit/delete race window as much as practical: rebuild every
        # gate, hash, metric decision, path identity, and size after the durable
        # pre-delete reports exist and before the first deletion.
        confirmation = _build_plan(args.mode, args.models_root, args.rgbd_status_file)
        if _plan_fingerprint(confirmation) != _plan_fingerprint(plan):
            raise SafetyError("inputs or checkpoint trees changed after pre-delete audit")

        if args.mode == "apply":
            _delete_planned_checkpoints(plan)
        _verify_postconditions(plan)
        result = "applied" if args.mode == "apply" else "dry_run_no_deletion"
        post_json, post_tsv = _write_reports(plan, audit_dir, run_id, "post", result)
        print(f"Post-check JSON:       {post_json}")
        print(f"Post-check TSV:        {post_tsv}")
        if args.mode == "check":
            print("CHECK ONLY: no checkpoints were deleted")
        else:
            print(f"APPLIED: deleted {len(plan.deletion_targets)} non-best early checkpoints")
        return 0
    except (OSError, SafetyError) as exc:
        if plan is not None and audit_dir is not None and run_id is not None:
            try:
                failed_json, failed_tsv = _write_reports(
                    plan, audit_dir, run_id, "post", "failed", str(exc)
                )
                print(f"Failure report JSON:   {failed_json}", file=sys.stderr)
                print(f"Failure report TSV:    {failed_tsv}", file=sys.stderr)
            except OSError as report_exc:
                print(
                    f"SECONDARY FAILURE: could not write post-failure report: {report_exc}",
                    file=sys.stderr,
                )
        print(f"SAFETY FAILURE: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
