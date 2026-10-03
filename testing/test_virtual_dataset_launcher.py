"""Opt-in dataset composition must not change or silently bypass legacy paths."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
HELPER = SCRIPTS_ROOT / "resolve_dataset_view.sh"
SELECTORS = (
    "DATASET_SOURCES",
    "VIRTUAL_DATASET_ROOT",
    "DATASET_ROOT",
    "TRAIN_DATASET",
    "VALIDATION_DATASET",
    "TEST_DATASET",
)


def run_helper(tmp_path: Path, values: dict[str, str], exit_code: int = 0):
    interpreter = tmp_path / ".venv/bin/python"
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text(
        '#!/usr/bin/env bash\nprintf "%s\\n" "$@" > "$DATASET_VIEW_TEST_ARGS"\n'
        f"exit {exit_code}\n",
        encoding="utf-8",
    )
    interpreter.chmod(0o755)
    env = {
        key: value for key, value in os.environ.items()
        if key not in (*SELECTORS, "EVALUATION_SPLIT", "ALLOW_TEST_EVALUATION")
    }
    env.update(values, DATASET_VIEW_TEST_ARGS=str(tmp_path / "args"))
    result = subprocess.run(
        [
            "bash",
            "-uc",
            'source "$1" || exit 17; printf "%s\\n" '
            '"${DATASET_ROOT-}" "${TRAIN_DATASET-}" "${VALIDATION_DATASET-}" "${TEST_DATASET-}"',
            "test",
            str(HELPER),
        ],
        cwd=tmp_path,
        env=env,
        text=True,
        capture_output=True,
    )
    args_path = tmp_path / "args"
    return result, args_path.read_text().splitlines() if args_path.exists() else None


def test_default_path_does_not_invoke_builder(tmp_path):
    result, args = run_helper(
        tmp_path, {"DATASET_ROOT": "/existing", "TRAIN_DATASET": "/custom/train"}
    )
    assert result.returncode == 0
    assert args is None
    assert result.stdout.splitlines()[:2] == ["/existing", "/custom/train"]


def test_view_passes_quoted_roots_and_selects_train_validation_only(tmp_path):
    result, args = run_helper(
        tmp_path,
        {
            "DATASET_SOURCES": "/source one:/source two",
            "VIRTUAL_DATASET_ROOT": "/view output",
        },
    )
    assert result.returncode == 0, result.stderr
    assert args == [
        "-m",
        "gr00t.data.virtual_dataset",
        "--sources",
        "/source one",
        "/source two",
        "--output",
        "/view output",
        "--reuse",
        "--splits",
        "train",
        "validation",
    ]
    assert result.stdout.splitlines() == [
        "/view output",
        "/view output/train",
        "/view output/validation",
        "",
    ]


def test_training_only_view_does_not_require_validation_or_test(tmp_path):
    result, args = run_helper(tmp_path, {
        "DATASET_SOURCES": "/a:/b", "VIRTUAL_DATASET_ROOT": "/v",
        "EVALUATION_SPLIT": "none",
    })
    assert result.returncode == 0, result.stderr
    assert args[-2:] == ["--splits", "train"]
    assert result.stdout.splitlines() == ["/v", "/v/train", "", ""]


def test_test_split_is_never_prepared_without_separate_enablement(tmp_path):
    result, args = run_helper(tmp_path, {
        "DATASET_SOURCES": "/a:/b", "VIRTUAL_DATASET_ROOT": "/v",
        "EVALUATION_SPLIT": "test",
    })
    assert result.returncode == 17
    assert args is None
    assert "test evaluation is deferred" in result.stderr


def test_explicit_future_test_view_requires_all_splits(tmp_path):
    result, args = run_helper(tmp_path, {
        "DATASET_SOURCES": "/a:/b", "VIRTUAL_DATASET_ROOT": "/v",
        "EVALUATION_SPLIT": "test", "ALLOW_TEST_EVALUATION": "1",
    })
    assert result.returncode == 0, result.stderr
    assert args[-4:] == ["--splits", "train", "validation", "test"]
    assert result.stdout.splitlines() == ["/v", "/v/train", "/v/validation", "/v/test"]


def test_builder_error_stops_runner(tmp_path):
    result, args = run_helper(
        tmp_path, {"DATASET_SOURCES": "/a:/b", "VIRTUAL_DATASET_ROOT": "/v"}, 3
    )
    assert result.returncode == 17
    assert args is not None
    assert "will not start" in result.stderr
    assert not result.stdout


def test_conflicting_or_partial_selection_fails_before_builder(tmp_path):
    for values in (
        {"DATASET_SOURCES": "/a:/b"},
        {"VIRTUAL_DATASET_ROOT": "/v"},
        {"DATASET_SOURCES": "/a::/b", "VIRTUAL_DATASET_ROOT": "/v"},
        {"DATASET_SOURCES": "/a:", "VIRTUAL_DATASET_ROOT": "/v"},
        {"DATASET_SOURCES": ":/a", "VIRTUAL_DATASET_ROOT": "/v"},
        *(
            {
                "DATASET_SOURCES": "/a:/b",
                "VIRTUAL_DATASET_ROOT": "/v",
                selector: "/conflict",
            }
            for selector in SELECTORS[2:]
        ),
    ):
        result, args = run_helper(tmp_path, values)
        assert result.returncode == 17, values
        assert args is None


def test_both_runners_resolve_before_applying_defaults():
    for name in (
        "multi_finetune_evaluation.sh",
        "partial_multi_finetune_evaluation.sh",
    ):
        text = (SCRIPTS_ROOT / name).read_text(encoding="utf-8")
        assert text.index(
            'source "$SCRIPT_DIR/resolve_dataset_view.sh" || exit 1'
        ) < text.index('TRAIN_DATASET="')
        subprocess.run(["bash", "-n", str(SCRIPTS_ROOT / name)], check=True)
    subprocess.run(["bash", "-n", str(HELPER)], check=True)
