from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = SCRIPTS_ROOT / "prune_latest_inspire_655_early_checkpoints.py"

MODEL_NAMES = (
    "inspire_c_rgb_patch_tuned_bf16_batch_32_acc_1_25k_"
    "all_tasks_655eps_20260916_normals_range_mask_v2_20260917",
    "inspire_c_normals_6ch_early_fusion_patch_tuned_normals_init_rgb_mean_"
    "bf16_batch_32_acc_1_25k_all_tasks_655eps_20260916_normals_range_mask_v2",
    "inspire_c_rgb_surface_normals_late_fusion_pre_adapter_"
    "4x_linear_rgb50_geo50_patch_frozen_bf16_batch_32_acc_1_25k_"
    "all_tasks_655eps_20260916_normals_range_mask_v2_20260918",
    "inspire_c_rgb_surface_normals_late_fusion_post_adapter_"
    "4x_linear_rgb50_geo50_patch_frozen_bf16_batch_32_acc_1_25k_"
    "all_tasks_655eps_20260916_normals_range_mask_v2_20260918",
    "inspire_c_rgbd_late_fusion_pre_adapter_"
    "4x_linear_rgb50_geo50_patch_frozen_bf16_batch_32_acc_1_25k_"
    "all_tasks_655eps_20260916_normals_range_mask_v2_20260918",
)
RGBD_NAME = MODEL_NAMES[-1]
STEPS = (5000, 10000, 15000, 20000, 25000)
HEADER = "split,checkpoint_step,episodes,frames,mae,mse,dimensions,samples\n"


def _metric_text(values: dict[int, str] | None = None) -> str:
    values = values or {
        5000: "0.019",
        10000: "0.011",
        15000: "0.009",
        20000: "0.0087",
        25000: "0.0086",
    }
    return HEADER + "".join(
        f"validation,{step},65,22719,0.05,{values[step]},26,590694\n" for step in STEPS
    )


def _make_fixture(
    tmp_path: Path,
    *,
    best_early_model_index: int | None = None,
    tied_best_model_index: int | None = None,
) -> tuple[Path, Path, Path]:
    models_root = tmp_path / "Models" / "inspire"
    audit_dir = tmp_path / "audit"
    status_path = tmp_path / "rgbd_status.tsv"
    models_root.mkdir(parents=True)

    for index, name in enumerate(MODEL_NAMES):
        model = models_root / name
        metrics = model / "normalized_action_metrics_exec_hor_8" / "checkpoint_metric_summary.csv"
        metrics.parent.mkdir(parents=True)
        values = None
        if index == best_early_model_index:
            values = {
                5000: "0.019",
                10000: "0.011",
                15000: "0.007",
                20000: "0.0087",
                25000: "0.0086",
            }
        if index == tied_best_model_index:
            values = {
                5000: "0.019",
                10000: "0.0086",
                15000: "0.0086",
                20000: "0.0087",
                25000: "0.0086",
            }
        metrics.write_text(_metric_text(values), encoding="utf-8")
        for step in STEPS:
            checkpoint = model / f"checkpoint-{step}"
            checkpoint.mkdir()
            (checkpoint / "model.safetensors").write_bytes(f"{name}:{step}".encode())

    status_path.write_text(
        "stage\tmodel\tstatus\texit_code\n"
        f"training\t{models_root / RGBD_NAME}\tPASS\t0\n"
        f"evaluation\t{models_root / RGBD_NAME}\tPASS\t0\n",
        encoding="utf-8",
    )
    return models_root, status_path, audit_dir


def _run(
    mode: str, models_root: Path, status_path: Path, audit_dir: Path
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            mode,
            "--models-root",
            str(models_root),
            "--rgbd-status-file",
            str(status_path),
            "--audit-dir",
            str(audit_dir),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_check_writes_pre_and_post_audits_without_deleting(tmp_path: Path) -> None:
    models_root, status_path, audit_dir = _make_fixture(tmp_path)

    result = _run("--check", models_root, status_path, audit_dir)

    assert result.returncode == 0, result.stderr
    assert "CHECK ONLY" in result.stdout
    assert len(list(audit_dir.glob("*.pre.json"))) == 1
    assert len(list(audit_dir.glob("*.pre.tsv"))) == 1
    assert len(list(audit_dir.glob("*.post.json"))) == 1
    assert len(list(audit_dir.glob("*.post.tsv"))) == 1
    pre = json.loads(next(audit_dir.glob("*.pre.json")).read_text(encoding="utf-8"))
    assert pre["planned_deletion_count"] == len(MODEL_NAMES) * 3
    assert pre["rgbd_completion_gate"]["required_rows"] == [
        "training:PASS:0",
        "evaluation:PASS:0",
    ]
    for name in MODEL_NAMES:
        for step in STEPS:
            assert (models_root / name / f"checkpoint-{step}").is_dir()


def test_apply_deletes_only_non_best_early_and_keeps_all_tied_best(tmp_path: Path) -> None:
    models_root, status_path, audit_dir = _make_fixture(
        tmp_path, best_early_model_index=0, tied_best_model_index=1
    )

    result = _run("--apply", models_root, status_path, audit_dir)

    assert result.returncode == 0, result.stderr
    assert "APPLIED" in result.stdout
    for model_index, name in enumerate(MODEL_NAMES):
        model = models_root / name
        for step in (20000, 25000):
            assert (model / f"checkpoint-{step}").is_dir()
        if model_index == 0:
            assert (model / "checkpoint-15000").is_dir()
            assert not (model / "checkpoint-5000").exists()
            assert not (model / "checkpoint-10000").exists()
        elif model_index == 1:
            assert not (model / "checkpoint-5000").exists()
            assert (model / "checkpoint-10000").is_dir()
            assert (model / "checkpoint-15000").is_dir()
        else:
            for step in (5000, 10000, 15000):
                assert not (model / f"checkpoint-{step}").exists()

    post = json.loads(next(audit_dir.glob("*.post.json")).read_text(encoding="utf-8"))
    assert post["result"] == "applied"
    first_model_checkpoints = {
        row["checkpoint_step"]: row for row in post["models"][0]["checkpoints"]
    }
    assert first_model_checkpoints[5000]["exists_at_report"] is False
    assert first_model_checkpoints[15000]["exists_at_report"] is True
    assert first_model_checkpoints[25000]["exists_at_report"] is True


def test_refuses_until_exact_rgbd_training_and_evaluation_pass(tmp_path: Path) -> None:
    models_root, status_path, audit_dir = _make_fixture(tmp_path)
    status_path.write_text(
        f"stage\tmodel\tstatus\texit_code\ntraining\t{models_root / RGBD_NAME}\tPASS\t0\n",
        encoding="utf-8",
    )

    result = _run("--apply", models_root, status_path, audit_dir)

    assert result.returncode == 2
    assert "exactly two rows" in result.stderr
    assert not audit_dir.exists()
    assert (models_root / MODEL_NAMES[0] / "checkpoint-5000").is_dir()


@pytest.mark.parametrize("corruption", ["missing", "duplicate", "nonfinite"])
def test_refuses_incomplete_duplicate_or_nonfinite_metrics(tmp_path: Path, corruption: str) -> None:
    models_root, status_path, audit_dir = _make_fixture(tmp_path)
    metrics = (
        models_root
        / MODEL_NAMES[0]
        / "normalized_action_metrics_exec_hor_8"
        / "checkpoint_metric_summary.csv"
    )
    text = metrics.read_text(encoding="utf-8")
    if corruption == "missing":
        text = (
            "\n".join(
                line for line in text.splitlines() if not line.startswith("validation,15000,")
            )
            + "\n"
        )
    elif corruption == "duplicate":
        text += "validation,15000,65,22719,0.05,0.009,26,590694\n"
    else:
        text = text.replace(
            "validation,15000,65,22719,0.05,0.009,", "validation,15000,65,22719,0.05,nan,"
        )
    metrics.write_text(text, encoding="utf-8")

    result = _run("--apply", models_root, status_path, audit_dir)

    assert result.returncode == 2
    assert not audit_dir.exists()
    assert (models_root / MODEL_NAMES[0] / "checkpoint-5000").is_dir()


def test_refuses_checkpoint_symlink(tmp_path: Path) -> None:
    models_root, status_path, audit_dir = _make_fixture(tmp_path)
    checkpoint = models_root / MODEL_NAMES[0] / "checkpoint-5000"
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "do_not_delete").write_text("sentinel", encoding="utf-8")
    for item in checkpoint.iterdir():
        item.unlink()
    checkpoint.rmdir()
    checkpoint.symlink_to(outside, target_is_directory=True)

    result = _run("--apply", models_root, status_path, audit_dir)

    assert result.returncode == 2
    assert "symlink" in result.stderr
    assert (outside / "do_not_delete").read_text(encoding="utf-8") == "sentinel"
    assert not audit_dir.exists()


def test_refuses_missing_protected_checkpoint(tmp_path: Path) -> None:
    models_root, status_path, audit_dir = _make_fixture(tmp_path)
    protected = models_root / MODEL_NAMES[2] / "checkpoint-20000"
    for item in protected.iterdir():
        item.unlink()
    protected.rmdir()

    result = _run("--apply", models_root, status_path, audit_dir)

    assert result.returncode == 2
    assert "always-kept" in result.stderr
    assert not audit_dir.exists()


def test_refuses_subset_metrics_even_when_all_five_steps_are_present(tmp_path: Path) -> None:
    models_root, status_path, audit_dir = _make_fixture(tmp_path)
    metrics = (
        models_root
        / MODEL_NAMES[0]
        / "normalized_action_metrics_exec_hor_8"
        / "checkpoint_metric_summary.csv"
    )
    metrics.write_text(
        metrics.read_text(encoding="utf-8").replace(
            "validation,15000,65,22719,0.05,0.009,26,590694",
            "validation,15000,10,3000,0.05,0.001,26,78000",
        ),
        encoding="utf-8",
    )

    result = _run("--apply", models_root, status_path, audit_dir)

    assert result.returncode == 2
    assert "non-canonical validation cohort" in result.stderr
    assert not audit_dir.exists()
    assert (models_root / MODEL_NAMES[0] / "checkpoint-5000").is_dir()
