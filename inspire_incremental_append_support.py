#!/usr/bin/env python3
"""Safety checks for the incremental Inspire LeRobot append pipeline.

The shell coordinator deliberately uses hard links only while its build is
hidden.  This helper proves the base is unchanged, detaches every surviving
shared inode before publication, and validates split/provenance invariants.
"""

from __future__ import annotations

import argparse
from collections import Counter
import ctypes
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any
import uuid

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import append_lerobot2 as append  # noqa: E402


SPLITS = ("train", "validation", "test")
PAYLOAD_DIRS = {"data", "videos", "raw_depths", "aligned_depths"}
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class AppendSafetyError(RuntimeError):
    pass


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise AppendSafetyError(f"Cannot read {path}: {exc}") from exc


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise AppendSafetyError(f"Cannot read {path}: {exc}") from exc


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        temporary = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _regular_files(root: Path) -> list[Path]:
    if not root.is_dir() or root.is_symlink():
        raise AppendSafetyError(f"Expected a real directory: {root}")
    files: list[Path] = []
    for directory, directories, filenames in os.walk(root, followlinks=False):
        directory_path = Path(directory)
        for name in directories:
            candidate = directory_path / name
            if candidate.is_symlink():
                raise AppendSafetyError(f"Directory symlinks are forbidden: {candidate}")
        for name in filenames:
            candidate = directory_path / name
            if candidate.is_symlink() or not candidate.is_file():
                raise AppendSafetyError(f"Only regular files are allowed: {candidate}")
            files.append(candidate)
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _summary_update(digest: Any, relative: str, size: int, file_digest: str) -> None:
    digest.update(relative.encode("utf-8"))
    digest.update(b"\0")
    digest.update(str(size).encode("ascii"))
    digest.update(b"\0")
    digest.update(bytes.fromhex(file_digest))


def tree_summary(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    combined = hashlib.sha256()
    file_count = 0
    total_bytes = 0
    for path in _regular_files(root):
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        _summary_update(combined, relative, size, _file_sha256(path))
        file_count += 1
        total_bytes += size
    return {
        "version": 1,
        "algorithm": "sha256(path\\0size\\0sha256(content))",
        "root": str(root),
        "file_count": file_count,
        "total_bytes": total_bytes,
        "tree_sha256": combined.hexdigest(),
    }


def snapshot_tree(root: Path, output: Path) -> None:
    summary = tree_summary(root)
    _atomic_json(output, summary)
    print(
        f"Tree snapshot: {summary['file_count']} files, {summary['total_bytes']} bytes, "
        f"{summary['tree_sha256']}"
    )


def verify_tree_snapshot(root: Path, snapshot_path: Path) -> None:
    expected = _read_json(snapshot_path)
    actual = tree_summary(root)
    for key in ("file_count", "total_bytes", "tree_sha256"):
        if actual[key] != expected.get(key):
            raise AppendSafetyError(
                f"Tree snapshot mismatch for {root}: {key}={actual[key]!r}, "
                f"expected {expected.get(key)!r}"
            )
    print(f"Tree snapshot verified: {root} ({actual['tree_sha256']})")


CHECKPOINT_STATE = "component_checkpoint.json"
CHECKPOINT_VERSION = 1
CHECKPOINT_PHASE = "component_ready"


def _safe_component_name_from_environment() -> str:
    value = os.environ.get(
        "INSPIRE_APPEND_CHECKPOINT_COMPONENT", "stack_red_cups_09_15"
    )
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value) is None:
        raise AppendSafetyError(
            "INSPIRE_APPEND_CHECKPOINT_COMPONENT must be one safe directory name"
        )
    return value


def _safe_provenance_relative_from_environment() -> Path:
    value = os.environ.get(
        "INSPIRE_APPEND_PROVENANCE_RELATIVE", "provenance/incremental_append"
    )
    relative = Path(value)
    if (
        relative.is_absolute()
        or len(relative.parts) < 2
        or relative.parts[0] != "provenance"
        or any(
            part in {"", ".", ".."}
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", part) is None
            for part in relative.parts
        )
    ):
        raise AppendSafetyError(
            "INSPIRE_APPEND_PROVENANCE_RELATIVE must be a safe relative path below provenance/"
        )
    return relative


CHECKPOINT_COMPONENT = _safe_component_name_from_environment()
APPEND_PROVENANCE_RELATIVE = _safe_provenance_relative_from_environment()
CHECKPOINT_BUILD = "final_build"
CHECKPOINT_SNAPSHOTS = {
    "base": "provenance/base_tree_snapshot.json",
    "source": "provenance/source_tree_snapshot.json",
    "component": "provenance/component_tree_snapshot.json",
}
BUILD_CHECKPOINT_SNAPSHOTS = {
    "base": "provenance/base_tree_snapshot.json",
    "source": "provenance/source_tree_snapshot.json",
    "build": "provenance/build_tree_snapshot.json",
}
PUBLISHED_TREE_SNAPSHOT = "provenance/published_tree_snapshot.json"


def _validated_snapshot_summary(path: Path) -> dict[str, Any]:
    value = _read_json(path)
    required = {
        "version",
        "algorithm",
        "root",
        "file_count",
        "total_bytes",
        "tree_sha256",
    }
    if set(value) != required:
        raise AppendSafetyError(f"Invalid tree snapshot fields: {path}")
    if value["version"] != 1 or value["algorithm"] != "sha256(path\\0size\\0sha256(content))":
        raise AppendSafetyError(f"Unsupported tree snapshot contract: {path}")
    if (
        not isinstance(value["root"], str)
        or not isinstance(value["file_count"], int)
        or value["file_count"] < 0
        or not isinstance(value["total_bytes"], int)
        or value["total_bytes"] < 0
        or not isinstance(value["tree_sha256"], str)
        or SHA256.fullmatch(value["tree_sha256"]) is None
    ):
        raise AppendSafetyError(f"Invalid tree snapshot values: {path}")
    return {
        "file_count": value["file_count"],
        "total_bytes": value["total_bytes"],
        "tree_sha256": value["tree_sha256"],
    }


def write_component_checkpoint(root: Path, base: Path, source: Path) -> None:
    """Seal one converted-component checkpoint before its atomic directory rename."""

    root = root.expanduser().resolve()
    base = base.expanduser().resolve()
    source = source.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise AppendSafetyError(f"Checkpoint staging root is not a real directory: {root}")
    component = root / CHECKPOINT_COMPONENT
    if not component.is_dir() or component.is_symlink():
        raise AppendSafetyError(f"Checkpoint component is missing: {component}")
    snapshots: dict[str, dict[str, Any]] = {}
    for name, relative in CHECKPOINT_SNAPSHOTS.items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise AppendSafetyError(f"Checkpoint snapshot is missing: {path}")
        snapshots[name] = _validated_snapshot_summary(path)
    state = {
        "version": CHECKPOINT_VERSION,
        "phase": CHECKPOINT_PHASE,
        "base_path": str(base),
        "source_path": str(source),
        "component_relative_path": CHECKPOINT_COMPONENT,
        "snapshots": snapshots,
    }
    _atomic_json(root / CHECKPOINT_STATE, state)
    print(
        "Sealed converted-component checkpoint: "
        f"{snapshots['component']['file_count']} files, "
        f"{snapshots['component']['total_bytes']} bytes, "
        f"{snapshots['component']['tree_sha256']}"
    )


def validate_component_checkpoint(
    root: Path,
    base: Path,
    source: Path,
    *,
    allow_orphan_final_build: bool = False,
) -> None:
    """Fail closed unless the retained component checkpoint is byte-for-byte intact."""

    root = root.expanduser().resolve()
    base = base.expanduser().resolve()
    source = source.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise AppendSafetyError(f"Checkpoint root is not a real directory: {root}")
    state_path = root / CHECKPOINT_STATE
    if not state_path.is_file() or state_path.is_symlink():
        raise AppendSafetyError(f"Checkpoint state is missing: {state_path}")
    state = _read_json(state_path)
    expected_keys = {
        "version",
        "phase",
        "base_path",
        "source_path",
        "component_relative_path",
        "snapshots",
    }
    if set(state) != expected_keys:
        raise AppendSafetyError("Checkpoint state has unexpected fields")
    if state["version"] != CHECKPOINT_VERSION or state["phase"] != CHECKPOINT_PHASE:
        raise AppendSafetyError("Checkpoint state has an unsupported version or phase")
    if state["base_path"] != str(base) or state["source_path"] != str(source):
        raise AppendSafetyError("Checkpoint belongs to a different base or source tree")
    if state["component_relative_path"] != CHECKPOINT_COMPONENT:
        raise AppendSafetyError("Checkpoint component path is invalid")
    if set(state["snapshots"]) != set(CHECKPOINT_SNAPSHOTS):
        raise AppendSafetyError("Checkpoint snapshot set is invalid")
    expected_entries = {
        CHECKPOINT_COMPONENT,
        "provenance",
        CHECKPOINT_STATE,
    }
    if allow_orphan_final_build:
        expected_entries.add(CHECKPOINT_BUILD)
        build = root / CHECKPOINT_BUILD
        if not build.is_dir() or build.is_symlink():
            raise AppendSafetyError(f"Orphan final build is not a real directory: {build}")
    if {path.name for path in root.iterdir()} != expected_entries:
        raise AppendSafetyError("Checkpoint root contains unexpected entries")

    roots = {
        "base": base,
        "source": source,
        "component": root / CHECKPOINT_COMPONENT,
    }
    for name, relative in CHECKPOINT_SNAPSHOTS.items():
        snapshot_path = root / relative
        summary = _validated_snapshot_summary(snapshot_path)
        if summary != state["snapshots"][name]:
            raise AppendSafetyError(f"Checkpoint {name} snapshot does not match sealed state")
        verify_tree_snapshot(roots[name], snapshot_path)
    label = (
        "converted-component checkpoint with pending final-build adoption"
        if allow_orphan_final_build
        else "reusable converted-component checkpoint"
    )
    print(f"Validated {label}: {root}")


def _checkpoint_state(root: Path) -> dict[str, Any]:
    root = root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise AppendSafetyError(f"Checkpoint root is not a real directory: {root}")
    state_path = root / CHECKPOINT_STATE
    if not state_path.is_file() or state_path.is_symlink():
        raise AppendSafetyError(f"Checkpoint state is missing: {state_path}")
    state = _read_json(state_path)
    if state.get("version") != CHECKPOINT_VERSION:
        raise AppendSafetyError("Checkpoint state has an unsupported version")
    phase = state.get("phase")
    if phase not in {CHECKPOINT_PHASE, "final_build_ready", "publish_ready"}:
        raise AppendSafetyError(f"Checkpoint state has an unsupported phase: {phase!r}")
    common = {"version", "phase", "base_path", "source_path", "snapshots"}
    if phase == CHECKPOINT_PHASE:
        expected = common | {"component_relative_path"}
        if (
            set(state) != expected
            or state.get("component_relative_path") != CHECKPOINT_COMPONENT
            or not isinstance(state.get("snapshots"), dict)
            or set(state["snapshots"]) != set(CHECKPOINT_SNAPSHOTS)
        ):
            raise AppendSafetyError("Component checkpoint state is malformed")
    elif phase == "final_build_ready":
        expected = common | {"build_relative_path"}
        if (
            set(state) != expected
            or state.get("build_relative_path") != CHECKPOINT_BUILD
            or not isinstance(state.get("snapshots"), dict)
            or set(state["snapshots"]) != set(BUILD_CHECKPOINT_SNAPSHOTS)
        ):
            raise AppendSafetyError("Build checkpoint state is malformed")
    else:
        expected = common | {"build_relative_path", "published_snapshot"}
        if (
            set(state) != expected
            or state.get("build_relative_path") != CHECKPOINT_BUILD
            or not isinstance(state.get("snapshots"), dict)
            or set(state["snapshots"]) != set(BUILD_CHECKPOINT_SNAPSHOTS)
            or not isinstance(state.get("published_snapshot"), dict)
        ):
            raise AppendSafetyError("Publish-ready checkpoint state is malformed")
    if not isinstance(state.get("base_path"), str) or not isinstance(
        state.get("source_path"), str
    ):
        raise AppendSafetyError("Checkpoint input paths are malformed")
    return state


def checkpoint_phase(root: Path) -> None:
    print(_checkpoint_state(root)["phase"])


def normalize_checkpoint(root: Path) -> None:
    """Remove only retry-safe detach scratch; never discard a payload phase."""

    root = root.expanduser().resolve()
    state = _checkpoint_state(root)
    phase = state["phase"]
    build = root / CHECKPOINT_BUILD
    if phase == CHECKPOINT_PHASE:
        allowed = {CHECKPOINT_COMPONENT, CHECKPOINT_BUILD, "provenance", CHECKPOINT_STATE}
    elif phase == "final_build_ready":
        detach_report = build / APPEND_PROVENANCE_RELATIVE / "detach_report.json"
        if detach_report.is_symlink():
            raise AppendSafetyError(f"Detach report may not be a symlink: {detach_report}")
        if detach_report.exists():
            detach_report.unlink()
        if build.is_dir() and not build.is_symlink():
            pattern = re.compile(r"^\.(?P<name>.+)\.detach-[0-9a-f]{32}$")
            for candidate in sorted(build.rglob("*")):
                if not candidate.is_file() or candidate.is_symlink():
                    continue
                match = pattern.fullmatch(candidate.name)
                if match is None:
                    continue
                destination = candidate.with_name(match.group("name"))
                if not destination.is_file() or destination.is_symlink():
                    raise AppendSafetyError(
                        f"Refusing orphan detach scratch without destination: {candidate}"
                    )
                candidate.unlink()
        allowed = {
            CHECKPOINT_COMPONENT,
            CHECKPOINT_BUILD,
            "provenance",
            CHECKPOINT_STATE,
        }
    else:
        # publish_ready is already byte-sealed. A retry must not remove its
        # detach report or any other file covered by the published snapshot.
        allowed = {CHECKPOINT_BUILD, "provenance", CHECKPOINT_STATE}
    actual = {path.name for path in root.iterdir()}
    required = (
        {CHECKPOINT_COMPONENT, "provenance", CHECKPOINT_STATE}
        if phase == CHECKPOINT_PHASE
        else {CHECKPOINT_BUILD, "provenance", CHECKPOINT_STATE}
    )
    if not required.issubset(actual) or not actual.issubset(allowed):
        raise AppendSafetyError(
            f"Checkpoint root contains unexpected entries for {phase}: {sorted(actual - allowed)}"
        )
    print(f"Normalized retry-safe scratch at phase {phase}: {root}")


def write_publish_ready_checkpoint(root: Path, base: Path, source: Path) -> None:
    """Seal the exact detached tree immediately before its no-replace publication."""

    root = root.expanduser().resolve()
    base = base.expanduser().resolve()
    source = source.expanduser().resolve()
    prior = _checkpoint_state(root)
    if prior["phase"] != "final_build_ready":
        raise AppendSafetyError(
            "Only a final-build-ready checkpoint may advance to publish-ready"
        )
    validate_retired_build_checkpoint(root, base, source)
    build = root / CHECKPOINT_BUILD
    published_path = root / PUBLISHED_TREE_SNAPSHOT
    published = _validated_snapshot_summary(published_path)
    actual = tree_summary(build)
    actual_summary = {
        key: actual[key] for key in ("file_count", "total_bytes", "tree_sha256")
    }
    if published != actual_summary:
        raise AppendSafetyError("Published-candidate snapshot does not match final build")
    state = {
        **prior,
        "phase": "publish_ready",
        "published_snapshot": published,
    }
    _atomic_json(root / CHECKPOINT_STATE, state)
    print(f"Sealed publish-ready checkpoint: {root}")


def _validate_publish_ready_common(
    root: Path,
    base: Path,
    source: Path,
    published_tree: Path,
    *,
    build_present: bool,
) -> None:
    root = root.expanduser().resolve()
    base = base.expanduser().resolve()
    source = source.expanduser().resolve()
    published_tree = published_tree.expanduser().resolve()
    state = _checkpoint_state(root)
    if state["phase"] != "publish_ready":
        raise AppendSafetyError("Checkpoint is not sealed publish-ready")
    if state.get("base_path") != str(base) or state.get("source_path") != str(source):
        raise AppendSafetyError("Publish-ready checkpoint belongs to different inputs")
    if (root / CHECKPOINT_COMPONENT).exists() or (root / CHECKPOINT_COMPONENT).is_symlink():
        raise AppendSafetyError("Publish-ready checkpoint still contains its component")
    expected_entries = {"provenance", CHECKPOINT_STATE}
    if build_present:
        expected_entries.add(CHECKPOINT_BUILD)
    if {path.name for path in root.iterdir()} != expected_entries:
        raise AppendSafetyError("Publish-ready checkpoint contains unexpected entries")
    for name in ("base", "source"):
        snapshot_path = root / BUILD_CHECKPOINT_SNAPSHOTS[name]
        summary = _validated_snapshot_summary(snapshot_path)
        if summary != state["snapshots"][name]:
            raise AppendSafetyError(
                f"Publish-ready {name} snapshot does not match sealed state"
            )
        verify_tree_snapshot({"base": base, "source": source}[name], snapshot_path)
    build_snapshot = _validated_snapshot_summary(
        root / BUILD_CHECKPOINT_SNAPSHOTS["build"]
    )
    if build_snapshot != state["snapshots"]["build"]:
        raise AppendSafetyError("Pre-detach build snapshot does not match sealed state")
    published_snapshot = _validated_snapshot_summary(root / PUBLISHED_TREE_SNAPSHOT)
    if published_snapshot != state["published_snapshot"]:
        raise AppendSafetyError("Published snapshot does not match sealed state")
    actual = tree_summary(published_tree)
    for key in ("file_count", "total_bytes", "tree_sha256"):
        if actual[key] != published_snapshot[key]:
            raise AppendSafetyError(
                f"Published tree mismatch: {key}={actual[key]!r}, "
                f"expected {published_snapshot[key]!r}"
            )


def validate_publish_ready_checkpoint(root: Path, base: Path, source: Path) -> None:
    build = root.expanduser().resolve() / CHECKPOINT_BUILD
    _validate_publish_ready_common(
        root, base, source, build, build_present=True
    )
    print(f"Validated publish-ready checkpoint: {root}")


def validate_published_target_checkpoint(
    root: Path, base: Path, source: Path, target: Path
) -> None:
    if not target.expanduser().resolve().is_dir() or target.is_symlink():
        raise AppendSafetyError(f"Published target is not a real directory: {target}")
    _validate_publish_ready_common(
        root, base, source, target, build_present=False
    )
    print(f"Validated published target left before checkpoint cleanup: {target}")


def write_build_checkpoint(root: Path, base: Path, source: Path) -> None:
    """Atomically advance the one checkpoint from component to validated final build."""

    root = root.expanduser().resolve()
    base = base.expanduser().resolve()
    source = source.expanduser().resolve()
    prior = _checkpoint_state(root)
    if prior["phase"] != CHECKPOINT_PHASE:
        raise AppendSafetyError("Only a component-ready checkpoint may advance to final build")
    if prior.get("base_path") != str(base) or prior.get("source_path") != str(source):
        raise AppendSafetyError("Checkpoint belongs to a different base or source tree")
    build = root / CHECKPOINT_BUILD
    if not build.is_dir() or build.is_symlink():
        raise AppendSafetyError(f"Final build checkpoint is missing: {build}")
    snapshots: dict[str, dict[str, Any]] = {}
    for name, relative in BUILD_CHECKPOINT_SNAPSHOTS.items():
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise AppendSafetyError(f"Build checkpoint snapshot is missing: {path}")
        snapshots[name] = _validated_snapshot_summary(path)
    state = {
        "version": CHECKPOINT_VERSION,
        "phase": "final_build_ready",
        "base_path": str(base),
        "source_path": str(source),
        "build_relative_path": CHECKPOINT_BUILD,
        "snapshots": snapshots,
    }
    _atomic_json(root / CHECKPOINT_STATE, state)
    print(
        "Advanced checkpoint to validated final build: "
        f"{snapshots['build']['file_count']} files, "
        f"{snapshots['build']['total_bytes']} bytes, "
        f"{snapshots['build']['tree_sha256']}"
    )


def validate_build_checkpoint(root: Path, base: Path, source: Path) -> None:
    """Fail closed unless the reusable final build and both inputs are unchanged."""

    root = root.expanduser().resolve()
    base = base.expanduser().resolve()
    source = source.expanduser().resolve()
    state = _checkpoint_state(root)
    expected_keys = {
        "version",
        "phase",
        "base_path",
        "source_path",
        "build_relative_path",
        "snapshots",
    }
    if set(state) != expected_keys or state["phase"] != "final_build_ready":
        raise AppendSafetyError("Checkpoint is not a sealed final-build checkpoint")
    if state["base_path"] != str(base) or state["source_path"] != str(source):
        raise AppendSafetyError("Checkpoint belongs to a different base or source tree")
    if state["build_relative_path"] != CHECKPOINT_BUILD:
        raise AppendSafetyError("Checkpoint build path is invalid")
    if set(state["snapshots"]) != set(BUILD_CHECKPOINT_SNAPSHOTS):
        raise AppendSafetyError("Build checkpoint snapshot set is invalid")
    # The sealed component must still be present for this pre-retirement
    # validator.  Once component media have been adopted by the sealed build,
    # retire_component_after_build records that transition durably.
    if {path.name for path in root.iterdir()} != {
        CHECKPOINT_COMPONENT,
        CHECKPOINT_BUILD,
        "provenance",
        CHECKPOINT_STATE,
    }:
        raise AppendSafetyError("Build checkpoint root contains unexpected entries")
    roots = {"base": base, "source": source, "build": root / CHECKPOINT_BUILD}
    for name, relative in BUILD_CHECKPOINT_SNAPSHOTS.items():
        snapshot_path = root / relative
        summary = _validated_snapshot_summary(snapshot_path)
        if summary != state["snapshots"][name]:
            raise AppendSafetyError(f"Checkpoint {name} snapshot does not match sealed state")
        verify_tree_snapshot(roots[name], snapshot_path)
    verify_tree_snapshot(
        root / CHECKPOINT_COMPONENT,
        root / CHECKPOINT_SNAPSHOTS["component"],
    )
    print(f"Validated reusable final-build checkpoint: {root}")


def validate_retired_build_checkpoint(root: Path, base: Path, source: Path) -> None:
    """Validate a final-build checkpoint after its adopted component was retired."""

    root = root.expanduser().resolve()
    base = base.expanduser().resolve()
    source = source.expanduser().resolve()
    state = _checkpoint_state(root)
    if state["phase"] != "final_build_ready":
        raise AppendSafetyError("Checkpoint is not a sealed final-build checkpoint")
    if state.get("base_path") != str(base) or state.get("source_path") != str(source):
        raise AppendSafetyError("Checkpoint belongs to a different base or source tree")
    if (root / CHECKPOINT_COMPONENT).exists() or (root / CHECKPOINT_COMPONENT).is_symlink():
        raise AppendSafetyError("Converted component was not fully retired")
    if {path.name for path in root.iterdir()} != {
        CHECKPOINT_BUILD,
        "provenance",
        CHECKPOINT_STATE,
    }:
        raise AppendSafetyError("Retired build checkpoint contains unexpected entries")

    roots = {"base": base, "source": source, "build": root / CHECKPOINT_BUILD}
    for name, relative in BUILD_CHECKPOINT_SNAPSHOTS.items():
        snapshot_path = root / relative
        summary = _validated_snapshot_summary(snapshot_path)
        if summary != state["snapshots"][name]:
            raise AppendSafetyError(f"Checkpoint {name} snapshot does not match sealed state")
        verify_tree_snapshot(roots[name], snapshot_path)
    print(f"Validated retired, reusable final-build checkpoint: {root}")


def retire_component_after_build(root: Path, base: Path, source: Path) -> None:
    """Adopt hard-linked component media only after the final build is sealed.

    A durable retirement marker is written after the intact component and final
    build have both validated.  If removal is interrupted, a retry may then
    finish deleting only the exact checkpoint component without distrusting the
    already sealed final build.
    """

    root = root.expanduser().resolve()
    component = root / CHECKPOINT_COMPONENT
    marker_path = root / "provenance" / "component_retirement.json"
    expected_marker = {
        "version": 1,
        "component_relative_path": CHECKPOINT_COMPONENT,
        "validated_component_snapshot": _validated_snapshot_summary(
            root / CHECKPOINT_SNAPSHOTS["component"]
        ),
        "reason": "media hardlinks adopted by sealed final build",
    }
    if not marker_path.exists():
        validate_build_checkpoint(root, base, source)
        _atomic_json(marker_path, expected_marker)
        parent_fd = os.open(marker_path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    elif marker_path.is_symlink() or _read_json(marker_path) != expected_marker:
        raise AppendSafetyError(f"Invalid component-retirement marker: {marker_path}")

    if component.is_symlink():
        raise AppendSafetyError(f"Checkpoint component may not be a symlink: {component}")
    if component.exists():
        if not component.is_dir():
            raise AppendSafetyError(f"Checkpoint component is not a directory: {component}")
        shutil.rmtree(component)
        root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(root_fd)
        finally:
            os.close(root_fd)
    validate_retired_build_checkpoint(root, base, source)
    print("Retired converted component after sealed final-build validation")


def _split_paths(root: Path) -> dict[str, Path]:
    result = append.split_dirs(root)
    if set(result) != set(SPLITS):
        raise AppendSafetyError(
            f"{root} must contain exactly train/validation/test LeRobot datasets; "
            f"found {sorted(result)}"
        )
    return result


def _calibration_fingerprint(split_root: Path) -> str:
    info = append.read_json(split_root / "meta" / "info.json")
    calibration = append.camera_calibration(info)
    if calibration is None:
        raise AppendSafetyError(f"Missing calibrated camera identity: {split_root}")
    return str(calibration["fingerprint"])


def _expected_map(values: list[int]) -> dict[str, int]:
    if len(values) != len(SPLITS):
        raise AppendSafetyError("Expected exactly three split values")
    return dict(zip(SPLITS, values, strict=True))


def validate_base(
    root: Path,
    expected_episodes: dict[str, int],
    expected_frames: dict[str, int],
) -> None:
    paths = _split_paths(root)
    append.validate_split_root_contract(root)
    fingerprints = set()
    for split in SPLITS:
        info = append.validate_dataset(paths[split])
        actual = (int(info["total_episodes"]), int(info["total_frames"]))
        expected = (expected_episodes[split], expected_frames[split])
        if actual != expected:
            raise AppendSafetyError(
                f"Base {split} population is {actual}; expected {expected}"
            )
        fingerprints.add(_calibration_fingerprint(paths[split]))
    if len(fingerprints) != 1:
        raise AppendSafetyError("Base splits do not have one exact calibration")
    if not (root / "split_manifest.json").is_file():
        raise AppendSafetyError(f"Base split provenance is missing: {root}")
    print(
        "Base validated: "
        + ", ".join(
            f"{split}={expected_episodes[split]}/{expected_frames[split]}"
            for split in SPLITS
        )
        + f"; calibration={next(iter(fingerprints))}"
    )


def validate_source(
    root: Path,
    expected_episodes: int,
    expected_frames: int,
    expected_goal_values: list[str],
) -> None:
    root = root.expanduser().resolve()
    episodes = sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and not path.is_symlink() and path.name.startswith("episode_")
    )
    if len(episodes) != expected_episodes:
        raise AppendSafetyError(
            f"Source has {len(episodes)} direct episodes; expected {expected_episodes}"
        )
    frame_count = 0
    hashes: set[str] = set()
    observed_goals: set[str] = set()
    for episode in episodes:
        data_path = episode / "data.json"
        payload = _read_json(data_path)
        frames = payload.get("data")
        if not isinstance(frames, list) or not frames:
            raise AppendSafetyError(f"Episode has no frame list: {data_path}")
        frame_count += len(frames)
        text = payload.get("text")
        if not isinstance(text, dict) or not isinstance(text.get("goal"), str):
            raise AppendSafetyError(f"Episode has no goal text: {data_path}")
        observed_goals.add(text["goal"])
        digest = _file_sha256(data_path)
        if digest in hashes:
            raise AppendSafetyError(f"Duplicate data.json content: {data_path}")
        hashes.add(digest)
    if frame_count != expected_frames:
        raise AppendSafetyError(
            f"Source has {frame_count} frames; expected {expected_frames}"
        )
    expected_goals = set(expected_goal_values)
    if len(expected_goals) != len(expected_goal_values) or not expected_goals:
        raise AppendSafetyError("Expected source goals must be non-empty and unique")
    if observed_goals != expected_goals:
        raise AppendSafetyError(
            f"Source goals are {sorted(observed_goals)!r}; "
            f"expected exactly {sorted(expected_goals)!r}"
        )
    print(
        f"Source validated: {len(episodes)} episodes, {frame_count} frames, "
        f"goals={sorted(expected_goals)!r}"
    )


def _manifest_records(root: Path) -> list[dict[str, Any]]:
    manifest_path = root / "split_manifest.json"
    manifest = _read_json(manifest_path)
    records = manifest.get("episodes")
    if not isinstance(records, list) or any(not isinstance(item, dict) for item in records):
        raise AppendSafetyError(f"Invalid episode list in {manifest_path}")
    return records


def validate_raw_split(
    root: Path,
    expected_episodes: dict[str, int],
    expected_frames: dict[str, int],
) -> None:
    records = _manifest_records(root)
    counts = Counter(str(record.get("split")) for record in records)
    frames = Counter()
    hashes: set[str] = set()
    for record in records:
        split = str(record.get("split"))
        frames[split] += int(record.get("frame_count", -1))
        digest = record.get("data_json_sha256")
        if not isinstance(digest, str) or len(digest) != 64:
            raise AppendSafetyError("Split provenance lacks a valid data.json SHA-256")
        if digest in hashes:
            raise AppendSafetyError(f"Duplicate source episode hash: {digest}")
        hashes.add(digest)
    for split in SPLITS:
        if counts[split] != expected_episodes[split] or frames[split] != expected_frames[split]:
            raise AppendSafetyError(
                f"Raw {split} population is {counts[split]}/{frames[split]}; expected "
                f"{expected_episodes[split]}/{expected_frames[split]}"
            )
    if len(records) != sum(expected_episodes.values()):
        raise AppendSafetyError("Raw split manifest has the wrong total population")
    print(
        "Raw component split validated: "
        + ", ".join(
            f"{split}={counts[split]}/{frames[split]}" for split in SPLITS
        )
    )


def validate_collection_order(root: Path, expected_components: list[str]) -> None:
    """Prove each split keeps the requested component concatenation order."""

    if not expected_components or len(set(expected_components)) != len(expected_components):
        raise AppendSafetyError("Expected component order must be non-empty and unique")
    manifest = _read_json(root / "split_manifest.json")
    components = manifest.get("components")
    if not isinstance(components, list) or any(
        not isinstance(component, dict) for component in components
    ):
        raise AppendSafetyError("Collection split manifest has no component metadata")
    actual_components = [component.get("dataset") for component in components]
    if actual_components != expected_components:
        raise AppendSafetyError(
            f"Collection component order is {actual_components!r}; "
            f"expected {expected_components!r}"
        )
    records = manifest.get("episodes")
    if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
        raise AppendSafetyError("Collection split manifest has an invalid episode list")
    for split in SPLITS:
        split_records = sorted(
            (record for record in records if record.get("split") == split),
            key=lambda record: str(record.get("split_episode")),
        )
        observed = [str(record.get("source_dataset")) for record in split_records]
        expected_rank = {name: rank for rank, name in enumerate(expected_components)}
        if any(name not in expected_rank for name in observed):
            raise AppendSafetyError(f"Collection {split} contains an unknown component")
        ranks = [expected_rank[name] for name in observed]
        if ranks != sorted(ranks) or set(observed) != set(expected_components):
            raise AppendSafetyError(
                f"Collection {split} does not preserve component order "
                f"{expected_components!r}"
            )
    print(f"Collection order validated in every split: {expected_components!r}")


def validate_component(
    root: Path,
    base: Path,
    expected_episodes: dict[str, int],
    expected_frames: dict[str, int],
) -> None:
    validate_raw_split(root, expected_episodes, expected_frames)
    paths = _split_paths(root)
    base_paths = _split_paths(base)
    append.validate_split_root_contract(root)
    for split in SPLITS:
        info = append.validate_dataset(paths[split])
        actual = (int(info["total_episodes"]), int(info["total_frames"]))
        expected = (expected_episodes[split], expected_frames[split])
        if actual != expected:
            raise AppendSafetyError(
                f"Converted component {split} is {actual}; expected {expected}"
            )
        if _calibration_fingerprint(paths[split]) != _calibration_fingerprint(
            base_paths[split]
        ):
            raise AppendSafetyError(
                f"Converted component {split} calibration differs from the base"
            )
    print("Converted component and exact camera calibration validated")


def record_provenance(
    base: Path,
    component: Path,
    output: Path,
    source_snapshot: Path,
    base_snapshot: Path,
    source_component_snapshots: list[tuple[str, Path]],
) -> None:
    destination = output / APPEND_PROVENANCE_RELATIVE
    if destination.exists() or destination.is_symlink():
        raise AppendSafetyError(f"Incremental provenance already exists: {destination}")
    if not source_component_snapshots:
        raise AppendSafetyError("At least one source-component snapshot is required")
    source_names = [name for name, _ in source_component_snapshots]
    if len(set(source_names)) != len(source_names):
        raise AppendSafetyError("Source-component snapshot names must be unique")
    destination.mkdir(parents=True)
    shutil.copy2(base / "split_manifest.json", destination / "base_split_manifest.json")
    shutil.copy2(
        component / "split_manifest.json",
        destination / "component_split_manifest.json",
    )
    shutil.copy2(source_snapshot, destination / "source_tree_snapshot.json")
    shutil.copy2(base_snapshot, destination / "base_tree_snapshot.json")
    source_components = {}
    for name, snapshot in source_component_snapshots:
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None:
            raise AppendSafetyError(f"Unsafe source-component name: {name!r}")
        target_name = f"{name}_source_tree_snapshot.json"
        shutil.copy2(snapshot, destination / target_name)
        source_components[name] = {
            **_validated_snapshot_summary(snapshot),
            "snapshot": target_name,
        }
    component_metadata = destination / "component_metadata"
    for split, split_root in _split_paths(component).items():
        target = component_metadata / split
        target.mkdir(parents=True)
        for filename in ("info.json", "tasks.jsonl", "episodes.jsonl", "modality.json"):
            shutil.copy2(split_root / "meta" / filename, target / filename)
    manifest = {
        "version": 1,
        "operation": "incremental-split-preserving-append",
        "base": str(base.resolve()),
        "component": str(component.resolve()),
        "base_tree_sha256": _read_json(base_snapshot)["tree_sha256"],
        "source_tree_sha256": _read_json(source_snapshot)["tree_sha256"],
        "source_components": source_components,
        "component_split_manifest": "component_split_manifest.json",
        "base_split_manifest": "base_split_manifest.json",
        "base_reuse": "hidden-hardlink-clone-detached-before-publish",
        "nontrain_stats_sha256": {
            split: _file_sha256(output / split / "meta" / "stats.json")
            for split in ("validation", "test")
        },
    }
    _atomic_json(destination / "append_manifest.json", manifest)
    print(f"Recorded self-contained append provenance: {destination}")


def clear_training_stats(train_root: Path) -> None:
    train_root = train_root.expanduser().resolve()
    if train_root.name != "train" or not (train_root / "meta" / "info.json").is_file():
        raise AppendSafetyError(f"Expected a converted train split: {train_root}")
    removed = []
    for filename in ("stats.json", "relative_stats.json"):
        path = train_root / "meta" / filename
        if path.is_symlink():
            raise AppendSafetyError(f"Statistics file may not be a symlink: {path}")
        if path.exists():
            path.unlink()
            removed.append(filename)
    print(f"Cleared derived train statistics for exact regeneration: {removed}")


def _schema_fingerprint(feature_name: str, feature_meta: dict[str, Any]) -> str:
    payload = {
        "feature": feature_name,
        "dtype": feature_meta.get("dtype"),
        "shape": feature_meta.get("shape"),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _relative_fingerprint(action_key: str) -> str:
    payload = {
        "embodiment_tag": "new_embodiment",
        "action_key": action_key,
        "action_delta_indices": list(range(32)),
        "state_delta_indices": [0],
        "rep": "RELATIVE",
        "type": "NON_EEF",
        "format": "DEFAULT",
        "state_key": None,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _validate_stat_entry(
    entry: Any,
    *,
    label: str,
    expected_shape: tuple[int, ...],
) -> None:
    if not isinstance(entry, dict):
        raise AppendSafetyError(f"Missing statistic entry: {label}")
    required = ("mean", "std", "min", "max", "q01", "q99")
    arrays: dict[str, np.ndarray] = {}
    for field in required:
        if field not in entry:
            raise AppendSafetyError(f"{label} lacks {field}")
        try:
            array = np.asarray(entry[field], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise AppendSafetyError(f"{label}.{field} is not numeric") from exc
        if array.shape != expected_shape:
            raise AppendSafetyError(
                f"{label}.{field} shape is {array.shape}; expected {expected_shape}"
            )
        if not np.all(np.isfinite(array)):
            raise AppendSafetyError(f"{label}.{field} contains non-finite values")
        arrays[field] = array
    if np.any(arrays["std"] < 0):
        raise AppendSafetyError(f"{label}.std contains negative values")
    if (
        np.any(arrays["min"] > arrays["q01"])
        or np.any(arrays["q01"] > arrays["q99"])
        or np.any(arrays["q99"] > arrays["max"])
    ):
        raise AppendSafetyError(f"{label} violates min <= q01 <= q99 <= max")


def validate_training_stats(
    dataset_root: Path,
    expected_frames: int,
    report: Path,
    *,
    read_only: bool = False,
) -> None:
    dataset_root = dataset_root.expanduser().resolve()
    train_root = dataset_root / "train"
    info = _read_json(train_root / "meta" / "info.json")
    episodes = _read_jsonl(train_root / "meta" / "episodes.jsonl")
    total_frames = int(info.get("total_frames", -1))
    episode_frames = sum(int(episode.get("length", -1)) for episode in episodes)
    if total_frames != expected_frames or episode_frames != expected_frames:
        raise AppendSafetyError(
            f"Train statistic count source is {total_frames}/{episode_frames}; "
            f"expected {expected_frames}"
        )
    features = info.get("features")
    if not isinstance(features, dict):
        raise AppendSafetyError("Train info.json has no feature schema")
    for key in ("observation.state", "action"):
        feature = features.get(key)
        if not isinstance(feature, dict) or feature.get("shape") != [26]:
            raise AppendSafetyError(f"Train {key} is not exactly 26D")

    stats_path = train_root / "meta" / "stats.json"
    stats = _read_json(stats_path)
    lowdim = {
        key: feature
        for key, feature in features.items()
        if isinstance(feature, dict) and "float" in str(feature.get("dtype", ""))
    }
    fingerprints = stats.get("__fingerprints__")
    if not isinstance(fingerprints, dict):
        raise AppendSafetyError("Exact train stats have no schema fingerprints")
    for key, feature in lowdim.items():
        shape = tuple(int(value) for value in feature.get("shape", []))
        _validate_stat_entry(stats.get(key), label=f"stats.{key}", expected_shape=shape)
        expected_fingerprint = _schema_fingerprint(key, feature)
        if fingerprints.get(key) != expected_fingerprint:
            raise AppendSafetyError(f"Train stats fingerprint is stale for {key}")
    if set(fingerprints) != set(lowdim):
        raise AppendSafetyError("Train stats fingerprints do not exactly match float features")

    relative_path = train_root / "meta" / "relative_stats.json"
    relative = _read_json(relative_path)
    relative_fingerprints = relative.get("__fingerprints__")
    expected_relative_keys = {"left_arm", "right_arm"}
    if not isinstance(relative_fingerprints, dict) or set(relative_fingerprints) != (
        expected_relative_keys
    ):
        raise AppendSafetyError("Relative stats must contain exactly the two relative arms")
    if set(relative) != {*expected_relative_keys, "__fingerprints__"}:
        raise AppendSafetyError("Relative stats contain unexpected or missing action groups")
    for key in sorted(expected_relative_keys):
        _validate_stat_entry(
            relative.get(key),
            label=f"relative_stats.{key}",
            expected_shape=(32, 7),
        )
        if relative_fingerprints.get(key) != _relative_fingerprint(key):
            raise AppendSafetyError(f"Relative stats fingerprint is stale for {key}")

    report = report.expanduser().resolve()
    try:
        report.relative_to(dataset_root)
    except ValueError as exc:
        raise AppendSafetyError(
            f"Statistics report must be inside the dataset root: {report}"
        ) from exc
    append_manifest = _read_json(report.parent / "append_manifest.json")
    expected_nontrain_hashes = append_manifest.get("nontrain_stats_sha256")
    actual_nontrain_hashes = {
        split: _file_sha256(dataset_root / split / "meta" / "stats.json")
        for split in ("validation", "test")
    }
    if actual_nontrain_hashes != expected_nontrain_hashes:
        raise AppendSafetyError("Statistics generation changed validation/test; train-only required")

    relative_trajectory_count = sum(
        max(0, int(episode["length"]) - 31) for episode in episodes
    )
    result = {
        "version": 1,
        "generator": "python -m gr00t.data.stats",
        "scope": "train-only",
        "dataset_statistics_sample_count": expected_frames,
        "sample_count_source": "train/meta/info.json total_frames cross-checked against episodes.jsonl",
        "relative_trajectory_count_per_arm": relative_trajectory_count,
        "relative_count_source": "sum(max(0, episode_length - 31)) for action delta_indices 0..31",
        "state_action_dimension": 26,
        "relative_groups": ["left_arm", "right_arm"],
        "stats_sha256": _file_sha256(stats_path),
        "relative_stats_sha256": _file_sha256(relative_path),
        "nontrain_stats_sha256": actual_nontrain_hashes,
    }
    if read_only:
        if _read_json(report) != result:
            raise AppendSafetyError(
                f"Existing statistics finalization report is stale or invalid: {report}"
            )
    else:
        _atomic_json(report, result)
    print(
        f"Exact train statistics validated: {expected_frames} rows, "
        f"{relative_trajectory_count} relative trajectories/arm, 26D, finite q01/q99 bounds"
    )


def _same_inode(left: Path, right: Path) -> bool:
    left_stat = left.stat()
    right_stat = right.stat()
    return left_stat.st_dev == right_stat.st_dev and left_stat.st_ino == right_stat.st_ino


def _is_payload(relative: Path) -> bool:
    return len(relative.parts) >= 3 and relative.parts[0] in SPLITS and relative.parts[1] in PAYLOAD_DIRS


def _is_regenerated_payload_metadata(relative: Path) -> bool:
    """Return whether append intentionally regenerates this payload-adjacent file.

    ``append_lerobot2.refresh_lz4_metadata`` atomically replaces the surface-normal
    manifest after appending episodes.  The manifest lives below ``videos`` but is
    metadata for the combined corpus, not immutable base media.  Keep the exception
    exact so every Parquet, image, LZ4 chunk, and video inherited from the base must
    still match byte-for-byte.
    """

    parts = relative.parts
    return (
        len(parts) >= 5
        and parts[0] in SPLITS
        and parts[1] == "videos"
        and parts[-3].startswith("chunk-")
        and parts[-2] == "observation.images.surface_normals_view"
        and parts[-1] == "manifest.json"
    )


def _copy_detached(source: Path, destination: Path) -> tuple[int, str]:
    temporary = destination.parent / f".{destination.name}.detach-{uuid.uuid4().hex}"
    digest = hashlib.sha256()
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            while chunk := reader.read(8 * 1024 * 1024):
                digest.update(chunk)
                writer.write(chunk)
        shutil.copystat(source, temporary, follow_symlinks=False)
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination.stat().st_size, digest.hexdigest()


def detach_and_verify(base: Path, output: Path, snapshot_path: Path, report: Path) -> None:
    base = base.resolve()
    output = output.resolve()
    if base == output or base in output.parents or output in base.parents:
        raise AppendSafetyError("Base and output paths may not overlap")
    expected = _read_json(snapshot_path)
    combined = hashlib.sha256()
    detached_files = 0
    detached_bytes = 0
    verified_payload_files = 0
    base_files = _regular_files(base)
    for source in base_files:
        relative = source.relative_to(base)
        destination = output / relative
        source_size = source.stat().st_size
        source_digest: str
        intentionally_regenerated = (
            len(relative.parts) >= 3
            and relative.parts[0] in SPLITS
            and relative.parts[-2:] == ("meta", "relative_stats.json")
        )
        if not destination.exists() and intentionally_regenerated:
            # append_lerobot2 deliberately removes stale relative-action statistics;
            # This append pipeline regenerates them exactly from the final train split.
            source_digest = _file_sha256(source)
        elif not destination.is_file() or destination.is_symlink():
            raise AppendSafetyError(f"Output does not retain base file: {relative}")
        elif _same_inode(source, destination):
            copied_size, source_digest = _copy_detached(source, destination)
            if copied_size != source_size:
                raise AppendSafetyError(f"Detached size mismatch: {relative}")
            destination_digest = _file_sha256(destination)
            if destination_digest != source_digest:
                raise AppendSafetyError(f"Detached byte mismatch: {relative}")
            detached_files += 1
            detached_bytes += source_size
        else:
            source_digest = _file_sha256(source)
            if _is_payload(relative) and not _is_regenerated_payload_metadata(relative):
                if destination.stat().st_size != source_size:
                    raise AppendSafetyError(f"Base payload size changed: {relative}")
                if _file_sha256(destination) != source_digest:
                    raise AppendSafetyError(f"Base payload bytes changed: {relative}")
                verified_payload_files += 1
        _summary_update(combined, relative.as_posix(), source_size, source_digest)

    actual_summary = {
        "file_count": len(base_files),
        "total_bytes": sum(path.stat().st_size for path in base_files),
        "tree_sha256": combined.hexdigest(),
    }
    for key, value in actual_summary.items():
        if value != expected.get(key):
            raise AppendSafetyError(
                f"Base changed during append: {key}={value!r}, expected {expected.get(key)!r}"
            )

    still_shared = [
        path.relative_to(base).as_posix()
        for path in base_files
        if (output / path.relative_to(base)).is_file()
        and _same_inode(path, output / path.relative_to(base))
    ]
    if still_shared:
        raise AppendSafetyError(
            f"Output still shares {len(still_shared)} regular files with base; "
            f"first={still_shared[0]}"
        )
    result = {
        "version": 1,
        "base_tree_sha256": combined.hexdigest(),
        "base_file_count": len(base_files),
        "detached_file_count": detached_files,
        "detached_bytes": detached_bytes,
        "independently_verified_payload_files": verified_payload_files,
        "remaining_shared_inodes": 0,
    }
    _atomic_json(report, result)
    print(
        f"Detached {detached_files} files ({detached_bytes} bytes); "
        "base digest unchanged; remaining shared inodes=0"
    )


def publish_no_replace(build: Path, target: Path) -> None:
    """Atomically rename a hidden build without ever replacing a target."""

    build = build.expanduser().resolve()
    target_parent = target.expanduser().absolute().parent.resolve()
    target = target_parent / target.name
    if not build.is_dir() or build.is_symlink():
        raise AppendSafetyError(f"Published build must be a real directory: {build}")
    if build == target or build in target.parents or target in build.parents:
        raise AppendSafetyError("Published build and target paths may not overlap")
    if build.stat().st_dev != target_parent.stat().st_dev:
        raise AppendSafetyError(
            f"Atomic publication requires one filesystem: {build.parent} != {target_parent}"
        )

    try:
        renameat2 = ctypes.CDLL(None, use_errno=True).renameat2
    except AttributeError as exc:  # pragma: no cover - current Linux/glibc exposes renameat2
        raise AppendSafetyError(
            "Atomic no-replace publication requires Linux renameat2"
        ) from exc
    renameat2.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    renameat2.restype = ctypes.c_int
    at_fdcwd = -100
    rename_noreplace = 1
    result = renameat2(
        at_fdcwd,
        os.fsencode(build),
        at_fdcwd,
        os.fsencode(target),
        rename_noreplace,
    )
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise AppendSafetyError(
                f"Final target already exists; hidden build was not published: {target}"
            )
        raise AppendSafetyError(
            f"Atomic publication failed for {build} -> {target}: "
            f"[{error_number}] {os.strerror(error_number)}"
        )
    for parent in {build.parent, target_parent}:
        parent_fd = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent_fd)
        finally:
            os.close(parent_fd)
    if build.exists() or build.is_symlink() or not target.is_dir() or target.is_symlink():
        raise AppendSafetyError("Atomic publication did not produce the exact target root")
    print(f"Atomically published without replacement: {target}")


def _validate_unique_hashes(records: list[dict[str, Any]], label: str) -> None:
    seen: dict[str, str] = {}
    for record in records:
        split = str(record.get("split"))
        if split not in SPLITS:
            raise AppendSafetyError(f"{label} has invalid split {split!r}")
        digest = record.get("data_json_sha256")
        if not isinstance(digest, str) or append.SHA256.fullmatch(digest) is None:
            raise AppendSafetyError(f"{label} has an invalid source hash: {digest!r}")
        previous = seen.get(digest)
        if previous is not None:
            raise AppendSafetyError(
                f"{label} repeats source hash {digest} in {previous} and {split}"
            )
        seen[digest] = split


def _ordered_source_records(
    records: list[dict[str, Any]], split: str, label: str
) -> list[dict[str, Any]]:
    candidates = [record for record in records if record.get("split") == split]
    numbered = []
    for record in candidates:
        match = append.EPISODE_NAME.fullmatch(str(record.get("split_episode", "")))
        if match is None:
            raise AppendSafetyError(f"{label} has invalid split_episode")
        numbered.append((int(match.group(1)), record))
    if len({number for number, _ in numbered}) != len(numbered):
        raise AppendSafetyError(f"{label} has duplicate {split} split_episode indices")
    return [record for _, record in sorted(numbered, key=lambda item: item[0])]


def _ordered_merged_records(
    records: list[dict[str, Any]], split: str
) -> list[dict[str, Any]]:
    candidates = [record for record in records if record.get("split") == split]
    if any(not isinstance(record.get("final_episode_index"), int) for record in candidates):
        raise AppendSafetyError(f"Merged {split} provenance lacks final episode indices")
    ordered = sorted(candidates, key=lambda record: int(record["final_episode_index"]))
    if [int(record["final_episode_index"]) for record in ordered] != list(
        range(len(ordered))
    ):
        raise AppendSafetyError(f"Merged {split} provenance indices are not contiguous")
    return ordered


def _validate_provenance_merge(
    base_records: list[dict[str, Any]],
    component_records: list[dict[str, Any]],
    merged_records: list[dict[str, Any]],
    output_frames: dict[str, int],
) -> None:
    _validate_unique_hashes(base_records, "base provenance")
    _validate_unique_hashes(component_records, "component provenance")
    _validate_unique_hashes(merged_records, "merged provenance")
    _validate_unique_hashes(
        [*base_records, *component_records],
        "combined base/component provenance",
    )

    for split in SPLITS:
        expected_records = [
            *_ordered_source_records(base_records, split, "base provenance"),
            *_ordered_source_records(component_records, split, "component provenance"),
        ]
        actual_records = _ordered_merged_records(merged_records, split)
        if len(actual_records) != len(expected_records):
            raise AppendSafetyError(f"Merged {split} provenance has the wrong population")
        expected_index_start = 0
        for final_episode_index, (expected_record, actual_record) in enumerate(
            zip(expected_records, actual_records, strict=True)
        ):
            for key in ("data_json_sha256", "frame_count"):
                if actual_record.get(key) != expected_record.get(key):
                    raise AppendSafetyError(
                        f"Merged {split} provenance changes {key} at episode "
                        f"{final_episode_index}"
                    )
            frame_count = int(expected_record["frame_count"])
            if (
                actual_record.get("final_index_start") != expected_index_start
                or actual_record.get("final_index_end_exclusive")
                != expected_index_start + frame_count
            ):
                raise AppendSafetyError(
                    f"Merged {split} frame ranges are not contiguous at episode "
                    f"{final_episode_index}"
                )
            expected_index_start += frame_count
        if expected_index_start != output_frames[split]:
            raise AppendSafetyError(
                f"Merged {split} provenance ends at {expected_index_start}; "
                f"dataset has {output_frames[split]} frames"
            )


def validate_final(
    base: Path,
    output: Path,
    component: Path | None,
    expected_component_episodes: dict[str, int],
    expected_component_frames: dict[str, int],
    require_independent: bool,
) -> None:
    base_paths = _split_paths(base)
    output_paths = _split_paths(output)
    component_paths = _split_paths(component) if component is not None else None
    append.validate_split_root_contract(output)
    for split in SPLITS:
        base_info = append.validate_dataset(base_paths[split])
        output_info = append.validate_dataset(output_paths[split])
        component_episode_count = expected_component_episodes[split]
        component_frame_count = expected_component_frames[split]
        if component_paths is not None:
            component_info = append.validate_dataset(component_paths[split])
            if (
                int(component_info["total_episodes"]) != component_episode_count
                or int(component_info["total_frames"]) != component_frame_count
            ):
                raise AppendSafetyError(f"Component {split} changed before final validation")
        expected = (
            int(base_info["total_episodes"]) + component_episode_count,
            int(base_info["total_frames"]) + component_frame_count,
        )
        actual = (int(output_info["total_episodes"]), int(output_info["total_frames"]))
        if actual != expected:
            raise AppendSafetyError(f"Final {split} population is {actual}; expected {expected}")
        base_episodes = _read_jsonl(base_paths[split] / "meta" / "episodes.jsonl")
        output_episodes = _read_jsonl(output_paths[split] / "meta" / "episodes.jsonl")
        if output_episodes[: len(base_episodes)] != base_episodes:
            raise AppendSafetyError(f"Final {split} does not preserve base episode order")
        embedded_component_episodes = _read_jsonl(
            output
            / APPEND_PROVENANCE_RELATIVE
            / "component_metadata"
            / split
            / "episodes.jsonl"
        )
        expected_suffix = []
        for offset, episode in enumerate(embedded_component_episodes):
            reindexed = dict(episode)
            reindexed["episode_index"] = len(base_episodes) + offset
            expected_suffix.append(reindexed)
        if output_episodes[len(base_episodes) :] != expected_suffix:
            raise AppendSafetyError(
                f"Final {split} does not preserve incoming episode order"
            )
        base_tasks = _read_jsonl(base_paths[split] / "meta" / "tasks.jsonl")
        output_tasks = _read_jsonl(output_paths[split] / "meta" / "tasks.jsonl")
        if output_tasks[: len(base_tasks)] != base_tasks:
            raise AppendSafetyError(f"Final {split} does not preserve base task indices")
        if _calibration_fingerprint(base_paths[split]) != _calibration_fingerprint(
            output_paths[split]
        ):
            raise AppendSafetyError(f"Final {split} calibration differs from base")

    provenance = output / APPEND_PROVENANCE_RELATIVE
    if (provenance / "base_split_manifest.json").read_bytes() != (
        base / "split_manifest.json"
    ).read_bytes():
        raise AppendSafetyError("Embedded base split provenance is not byte-exact")
    if component is not None and (
        provenance / "component_split_manifest.json"
    ).read_bytes() != (component / "split_manifest.json").read_bytes():
        raise AppendSafetyError("Embedded component split provenance is not byte-exact")
    component_manifest = _read_json(provenance / "component_split_manifest.json")
    component_records = component_manifest.get("episodes", [])
    if not isinstance(component_records, list) or any(
        not isinstance(record, dict) for record in component_records
    ):
        raise AppendSafetyError("Embedded component provenance has an invalid episode list")
    if len(component_records) != sum(expected_component_episodes.values()):
        raise AppendSafetyError("Embedded component provenance has the wrong population")
    component_counts = Counter(str(record.get("split")) for record in component_records)
    component_frames = Counter()
    for record in component_records:
        component_frames[str(record.get("split"))] += int(record.get("frame_count", -1))
    for split in SPLITS:
        if (
            component_counts[split] != expected_component_episodes[split]
            or component_frames[split] != expected_component_frames[split]
        ):
            raise AppendSafetyError(
                f"Embedded component {split} provenance has population "
                f"{component_counts[split]}/{component_frames[split]}"
            )
    merged = _read_json(output / "provenance" / "merge_manifest.json")
    merged_records = merged.get("episodes", [])
    if not isinstance(merged_records, list) or any(
        not isinstance(record, dict) for record in merged_records
    ):
        raise AppendSafetyError("Merged provenance has an invalid episode list")
    if len(merged_records) != sum(
        int(append.read_json(path / "meta" / "info.json")["total_episodes"])
        for path in output_paths.values()
    ):
        raise AppendSafetyError("Merged provenance does not cover every final episode")
    base_records = _read_json(base / "split_manifest.json").get("episodes", [])
    if not isinstance(base_records, list) or any(
        not isinstance(record, dict) for record in base_records
    ):
        raise AppendSafetyError("Base provenance has an invalid episode list")
    _validate_provenance_merge(
        base_records,
        component_records,
        merged_records,
        {
            split: int(
                append.read_json(output_paths[split] / "meta" / "info.json")[
                    "total_frames"
                ]
            )
            for split in SPLITS
        },
    )
    if require_independent:
        shared = []
        for source in _regular_files(base):
            destination = output / source.relative_to(base)
            if destination.is_file() and _same_inode(source, destination):
                shared.append(source.relative_to(base).as_posix())
        if shared:
            raise AppendSafetyError(
                f"Published candidate still shares {len(shared)} base files; first={shared[0]}"
            )
    print(
        "Final append validated: "
        + ", ".join(
            f"{split}={int(append.read_json(output_paths[split] / 'meta' / 'info.json')['total_episodes'])}/"
            f"{int(append.read_json(output_paths[split] / 'meta' / 'info.json')['total_frames'])}"
            for split in SPLITS
        )
        + ("; independent from base" if require_independent else "; hidden linked build")
    )


def _add_expected(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--episodes", type=int, nargs=3, required=True, metavar=("TRAIN", "VALIDATION", "TEST"))
    parser.add_argument("--frames", type=int, nargs=3, required=True, metavar=("TRAIN", "VALIDATION", "TEST"))


def _source_component_snapshots(values: list[str]) -> list[tuple[str, Path]]:
    result: list[tuple[str, Path]] = []
    for value in values:
        name, separator, snapshot_text = value.partition("=")
        if (
            not separator
            or not snapshot_text
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name) is None
        ):
            raise AppendSafetyError(
                "--source-component must use SAFE_NAME=/path/to/tree_snapshot.json"
            )
        result.append((name, Path(snapshot_text)))
    if len({name for name, _ in result}) != len(result):
        raise AppendSafetyError("Duplicate --source-component name")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    snapshot = commands.add_parser("snapshot-tree")
    snapshot.add_argument("--root", type=Path, required=True)
    snapshot.add_argument("--output", type=Path, required=True)

    verify = commands.add_parser("verify-tree-snapshot")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--snapshot", type=Path, required=True)

    write_checkpoint = commands.add_parser("write-component-checkpoint")
    write_checkpoint.add_argument("--root", type=Path, required=True)
    write_checkpoint.add_argument("--base", type=Path, required=True)
    write_checkpoint.add_argument("--source", type=Path, required=True)

    validate_checkpoint = commands.add_parser("validate-component-checkpoint")
    validate_checkpoint.add_argument("--root", type=Path, required=True)
    validate_checkpoint.add_argument("--base", type=Path, required=True)
    validate_checkpoint.add_argument("--source", type=Path, required=True)

    validate_orphan = commands.add_parser("validate-orphan-final-build-checkpoint")
    validate_orphan.add_argument("--root", type=Path, required=True)
    validate_orphan.add_argument("--base", type=Path, required=True)
    validate_orphan.add_argument("--source", type=Path, required=True)

    phase = commands.add_parser("checkpoint-phase")
    phase.add_argument("--root", type=Path, required=True)

    normalize = commands.add_parser("normalize-checkpoint")
    normalize.add_argument("--root", type=Path, required=True)

    write_build_checkpoint_parser = commands.add_parser("write-build-checkpoint")
    write_build_checkpoint_parser.add_argument("--root", type=Path, required=True)
    write_build_checkpoint_parser.add_argument("--base", type=Path, required=True)
    write_build_checkpoint_parser.add_argument("--source", type=Path, required=True)

    validate_build_checkpoint_parser = commands.add_parser("validate-build-checkpoint")
    validate_build_checkpoint_parser.add_argument("--root", type=Path, required=True)
    validate_build_checkpoint_parser.add_argument("--base", type=Path, required=True)
    validate_build_checkpoint_parser.add_argument("--source", type=Path, required=True)

    write_publish_ready = commands.add_parser("write-publish-ready-checkpoint")
    write_publish_ready.add_argument("--root", type=Path, required=True)
    write_publish_ready.add_argument("--base", type=Path, required=True)
    write_publish_ready.add_argument("--source", type=Path, required=True)

    validate_publish_ready = commands.add_parser("validate-publish-ready-checkpoint")
    validate_publish_ready.add_argument("--root", type=Path, required=True)
    validate_publish_ready.add_argument("--base", type=Path, required=True)
    validate_publish_ready.add_argument("--source", type=Path, required=True)

    validate_published = commands.add_parser("validate-published-target-checkpoint")
    validate_published.add_argument("--root", type=Path, required=True)
    validate_published.add_argument("--base", type=Path, required=True)
    validate_published.add_argument("--source", type=Path, required=True)
    validate_published.add_argument("--target", type=Path, required=True)

    retire_component = commands.add_parser("retire-component-after-build")
    retire_component.add_argument("--root", type=Path, required=True)
    retire_component.add_argument("--base", type=Path, required=True)
    retire_component.add_argument("--source", type=Path, required=True)

    validate_retired = commands.add_parser("validate-retired-build-checkpoint")
    validate_retired.add_argument("--root", type=Path, required=True)
    validate_retired.add_argument("--base", type=Path, required=True)
    validate_retired.add_argument("--source", type=Path, required=True)

    base = commands.add_parser("validate-base")
    base.add_argument("--root", type=Path, required=True)
    _add_expected(base)

    source = commands.add_parser("validate-source")
    source.add_argument("--root", type=Path, required=True)
    source.add_argument("--episodes", type=int, required=True)
    source.add_argument("--frames", type=int, required=True)
    source.add_argument(
        "--goal",
        action="append",
        required=True,
        help="Exact allowed source goal; repeat for multi-goal components",
    )

    raw = commands.add_parser("validate-raw-split")
    raw.add_argument("--root", type=Path, required=True)
    _add_expected(raw)

    collection_order = commands.add_parser("validate-collection-order")
    collection_order.add_argument("--root", type=Path, required=True)
    collection_order.add_argument("--component", action="append", required=True)

    component = commands.add_parser("validate-component")
    component.add_argument("--root", type=Path, required=True)
    component.add_argument("--base", type=Path, required=True)
    _add_expected(component)

    provenance = commands.add_parser("record-provenance")
    provenance.add_argument("--base", type=Path, required=True)
    provenance.add_argument("--component", type=Path, required=True)
    provenance.add_argument("--output", type=Path, required=True)
    provenance.add_argument("--source-snapshot", type=Path, required=True)
    provenance.add_argument("--base-snapshot", type=Path, required=True)
    provenance.add_argument(
        "--source-component",
        action="append",
        required=True,
        metavar="SAFE_NAME=SNAPSHOT",
    )

    clear_stats = commands.add_parser("clear-training-stats")
    clear_stats.add_argument("--train-root", type=Path, required=True)

    validate_stats = commands.add_parser("validate-training-stats")
    validate_stats.add_argument("--dataset-root", type=Path, required=True)
    validate_stats.add_argument("--expected-frames", type=int, required=True)
    validate_stats.add_argument("--report", type=Path, required=True)
    validate_stats.add_argument("--read-only", action="store_true")

    detach = commands.add_parser("detach-and-verify")
    detach.add_argument("--base", type=Path, required=True)
    detach.add_argument("--output", type=Path, required=True)
    detach.add_argument("--snapshot", type=Path, required=True)
    detach.add_argument("--report", type=Path, required=True)

    publish = commands.add_parser("publish-no-replace")
    publish.add_argument("--build", type=Path, required=True)
    publish.add_argument("--target", type=Path, required=True)

    final = commands.add_parser("validate-final")
    final.add_argument("--base", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    final.add_argument("--component", type=Path)
    final.add_argument("--require-independent", action="store_true")
    _add_expected(final)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "snapshot-tree":
        snapshot_tree(args.root, args.output)
    elif args.command == "verify-tree-snapshot":
        verify_tree_snapshot(args.root, args.snapshot)
    elif args.command == "write-component-checkpoint":
        write_component_checkpoint(args.root, args.base, args.source)
    elif args.command == "validate-component-checkpoint":
        validate_component_checkpoint(args.root, args.base, args.source)
    elif args.command == "validate-orphan-final-build-checkpoint":
        validate_component_checkpoint(
            args.root,
            args.base,
            args.source,
            allow_orphan_final_build=True,
        )
    elif args.command == "checkpoint-phase":
        checkpoint_phase(args.root)
    elif args.command == "normalize-checkpoint":
        normalize_checkpoint(args.root)
    elif args.command == "write-build-checkpoint":
        write_build_checkpoint(args.root, args.base, args.source)
    elif args.command == "validate-build-checkpoint":
        validate_build_checkpoint(args.root, args.base, args.source)
    elif args.command == "write-publish-ready-checkpoint":
        write_publish_ready_checkpoint(args.root, args.base, args.source)
    elif args.command == "validate-publish-ready-checkpoint":
        validate_publish_ready_checkpoint(args.root, args.base, args.source)
    elif args.command == "validate-published-target-checkpoint":
        validate_published_target_checkpoint(
            args.root, args.base, args.source, args.target
        )
    elif args.command == "retire-component-after-build":
        retire_component_after_build(args.root, args.base, args.source)
    elif args.command == "validate-retired-build-checkpoint":
        validate_retired_build_checkpoint(args.root, args.base, args.source)
    elif args.command == "validate-base":
        validate_base(args.root, _expected_map(args.episodes), _expected_map(args.frames))
    elif args.command == "validate-source":
        validate_source(args.root, args.episodes, args.frames, args.goal)
    elif args.command == "validate-raw-split":
        validate_raw_split(args.root, _expected_map(args.episodes), _expected_map(args.frames))
    elif args.command == "validate-collection-order":
        validate_collection_order(args.root, args.component)
    elif args.command == "validate-component":
        validate_component(
            args.root,
            args.base,
            _expected_map(args.episodes),
            _expected_map(args.frames),
        )
    elif args.command == "record-provenance":
        record_provenance(
            args.base,
            args.component,
            args.output,
            args.source_snapshot,
            args.base_snapshot,
            _source_component_snapshots(args.source_component),
        )
    elif args.command == "clear-training-stats":
        clear_training_stats(args.train_root)
    elif args.command == "validate-training-stats":
        validate_training_stats(
            args.dataset_root,
            args.expected_frames,
            args.report,
            read_only=args.read_only,
        )
    elif args.command == "detach-and-verify":
        detach_and_verify(args.base, args.output, args.snapshot, args.report)
    elif args.command == "publish-no-replace":
        publish_no_replace(args.build, args.target)
    elif args.command == "validate-final":
        validate_final(
            args.base,
            args.output,
            args.component,
            _expected_map(args.episodes),
            _expected_map(args.frames),
            args.require_independent,
        )
    else:  # pragma: no cover
        raise AssertionError(args.command)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AppendSafetyError, append.MergeError, OSError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
