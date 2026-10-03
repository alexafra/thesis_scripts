"""Final-fit launcher contracts without invoking real conversion, training or GPUs."""

import os
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
MULTI = SCRIPTS_ROOT / "multi_finetune_evaluation.sh"
FINAL = SCRIPTS_ROOT / "final_train_inspire.sh"


def _settings_script():
    source = MULTI.read_text()
    script = source[source.index("# Test-set evaluation is deliberately disabled") : source.index('source "$SCRIPT_DIR/resolve_dataset_view.sh"')]
    script += source[source.index('if [[ "$DRY_RUN" == "1" ]]') : source.index('EVALUATION_STATUS_FILE="')]
    return script


def _settings(**overrides):
    script = _settings_script()
    script += '\nprintf "SET:%s:%s:%s:%s:%s:%s\\n" "$MAX_STEPS" "$SAVE_STEPS" "$RUN_LABEL" "$EVALUATION_DATASET" "$EVALUATION_DIR_NAME" "$NORMALIZED_DIR_NAME"\n'
    script += 'printf "CHECKPOINT:%s\\n" "${EVAL_CHECKPOINT_ARGS[@]}"\n'
    script += 'printf "PROBE:%s\\n" "${EVAL_SELECTION_ARGS[@]}"\n'
    script += 'printf "REQUIRED:%s\\n" "${REQUIRED_DATASETS[@]}"\n'
    env = {k: v for k, v in os.environ.items() if k not in {"MAX_STEPS", "SAVE_STEPS", "RUN_LABEL", "EVALUATION_SPLIT", "ALLOW_TEST_EVALUATION"}}
    env.update(DRY_RUN="0", TRAIN_DATASET="/train", VALIDATION_DATASET="/validation", TEST_DATASET="/heldout-test", EXECUTION_HORIZON="8", **overrides)
    return subprocess.run(["bash", "-euc", script], env=env, text=True, capture_output=True)


def test_legacy_budget_and_validation_defaults_unchanged():
    result = _settings()
    assert result.returncode == 0, result.stderr
    assert "SET:25000:5000:25k:/validation:evaluation_exec_hor_8:normalized_action_metrics_exec_hor_8" in result.stdout
    assert "--checkpoint-steps" not in result.stdout
    assert "PROBE:--train-probe-episodes-per-goal\nPROBE:5" in result.stdout
    assert "REQUIRED:/train\nREQUIRED:/validation\n" in result.stdout
    assert "REQUIRED:/heldout-test" not in result.stdout


@pytest.mark.parametrize("steps,label", [("30000", "30k"), ("35000", "35k"), ("40000", "40k"), ("32500", "32500steps")])
def test_explicit_future_test_evaluates_only_fixed_final_checkpoint(steps, label):
    result = _settings(MAX_STEPS=steps, SAVE_STEPS="2500", EVALUATION_SPLIT="test", ALLOW_TEST_EVALUATION="1")
    assert result.returncode == 0, result.stderr
    assert f"SET:{steps}:2500:{label}:/heldout-test:test_evaluation_exec_hor_8:test_normalized_action_metrics_exec_hor_8" in result.stdout
    assert f"CHECKPOINT:--checkpoint-steps\nCHECKPOINT:{steps}" in result.stdout
    assert "REQUIRED:/train\nREQUIRED:/heldout-test\n" in result.stdout
    assert "REQUIRED:/validation" not in result.stdout


@pytest.mark.parametrize("allow", ["", "0", "true", "yes"])
def test_test_evaluation_requires_explicit_enablement(allow):
    result = _settings(EVALUATION_SPLIT="test", ALLOW_TEST_EVALUATION=allow)
    assert result.returncode != 0
    assert "test evaluation is deferred" in result.stderr


def test_training_only_mode_has_no_evaluation_dataset_or_checkpoint_selection():
    result = _settings(EVALUATION_SPLIT="none", MAX_STEPS="35000")
    assert result.returncode == 0, result.stderr
    assert "SET:35000:5000:35k:::" in result.stdout
    assert "REQUIRED:/train\n" in result.stdout
    assert "REQUIRED:/validation" not in result.stdout
    assert "REQUIRED:/heldout-test" not in result.stdout
    assert "--checkpoint-steps" not in result.stdout


@pytest.mark.parametrize("split", ["validation", "none"])
def test_preflight_does_not_require_a_test_dataset(tmp_path, split):
    train = tmp_path / "train"
    validation = tmp_path / "validation"
    datasets = [train, validation] if split == "validation" else [train]
    for dataset in datasets:
        (dataset / "meta").mkdir(parents=True)
        (dataset / "meta/info.json").write_text(json.dumps({
            "robot_type": "Unitree_G1_Inspire_HeadOnly",
            "features": {"observation.images.ego_view": {}},
        }))
    python = tmp_path / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    source = MULTI.read_text()
    start = source.index('for dataset in "${REQUIRED_DATASETS[@]}"; do')
    end = source.index('\nfor i in "${SELECTED_INDICES[@]}"; do', start)
    script = _settings_script()
    script += '\nREQUIRED_FEATURES=(observation.images.ego_view)\n'
    script += source[start:end]
    env = {k: v for k, v in os.environ.items() if k not in {"MAX_STEPS", "SAVE_STEPS", "ALLOW_TEST_EVALUATION"}}
    env.update(DRY_RUN="0", EVALUATION_SPLIT=split, TRAIN_DATASET=str(train),
               VALIDATION_DATASET=str(validation), TEST_DATASET=str(tmp_path / "absent-test"),
               EXECUTION_HORIZON="8", DATASET_ROBOT_TYPE="Unitree_G1_Inspire_HeadOnly")
    result = subprocess.run(["bash", "-euc", script], cwd=tmp_path, env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert not (tmp_path / "absent-test").exists()


def test_training_only_skips_evaluation_before_protocol_and_inference(tmp_path):
    source = MULTI.read_text()
    start = source.index('    if [[ "$EVALUATION_SPLIT" == "none" ]]; then', source.index('echo "Finished training:'))
    end = source.index('    # The legacy evaluator calls its held-out series', start)
    script = 'for experiment in fixture; do\n' + source[start:end]
    script += 'echo EVALUATION_REACHED\ndone\n'
    status = tmp_path / "status.tsv"
    result = subprocess.run(["bash", "-euc", script], env={**os.environ,
                            "EVALUATION_SPLIT": "none", "MODEL_DIR": "/fixture/model",
                            "EVALUATION_STATUS_FILE": str(status)}, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert "EVALUATION_REACHED" not in result.stdout
    assert status.read_text() == "evaluation\t/fixture/model\tSKIP\tNA\n"


@pytest.mark.parametrize("settings", [
    {"MAX_STEPS": "0"}, {"MAX_STEPS": "-1"}, {"MAX_STEPS": "035000"},
    {"MAX_STEPS": "35k"}, {"SAVE_STEPS": "0"}, {"MAX_STEPS": "32000"},
    {"EVALUATION_SPLIT": "train"},
])
def test_invalid_steps_and_split_rejected(settings):
    result = _settings(**settings)
    assert result.returncode != 0
    assert "Error:" in result.stderr


def test_evaluation_invocation_consumes_split_and_checkpoint_arguments():
    source = MULTI.read_text()
    evaluation = source[source.index("python -m scripts.analysis_tools.evaluate_checkpoints") :]
    assert '--dataset-path "$EVALUATION_DATASET"' in evaluation
    assert '"${EVAL_CHECKPOINT_ARGS[@]}"' in evaluation
    assert '--output-dir "$MODEL_DIR/$EVALUATION_DIR_NAME"' in evaluation
    assert '--evaluation-dir "$MODEL_DIR/$EVALUATION_DIR_NAME"' in evaluation
    assert '--output-dir "$MODEL_DIR/$NORMALIZED_DIR_NAME"' in evaluation
    assert '"${INCLUDE_BASE_MODEL[$i]}" == "1" && "$EVALUATION_SPLIT" == "validation"' in source
    subprocess.run(["bash", "-n", str(MULTI)], check=True)


@pytest.mark.parametrize("split", ["validation", "test"])
def test_protocol_manifest_records_true_split_and_fixed_checkpoint(tmp_path, split):
    source = MULTI.read_text()
    protocol = source[source.index('    # The legacy evaluator calls its held-out series') :]
    code = protocol.split("<<'PY'\n", 1)[1].split("\nPY\n", 1)[0]
    output = tmp_path / "protocol.json"
    result = subprocess.run(
        [sys.executable, "-", str(output), split, "/eval", "/trainval", "35000", "8", "output"],
        input=code, text=True, capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    metadata = json.loads(output.read_text())
    assert metadata["evaluation_split"] == split
    assert metadata["internal_evaluator_split_label"] == "validation"
    assert metadata["checkpoint_steps"] == ([35000] if split == "test" else None)
    assert metadata["checkpoint_selection_permitted"] == (split != "test")


def _wrapper(tmp_path, mode, **overrides):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copyfile(FINAL, scripts / FINAL.name)
    (scripts / MULTI.name).write_text(
        '#!/usr/bin/env bash\n'
        'printf "RUN:%s:%s:%s:%s:%s:%s\\n" "$TRAIN_DATASET" "${TEST_DATASET:-unset}" "$EVALUATION_SPLIT" "$EXPERIMENTS" "$MAX_STEPS" "$PRECHECK_ONLY" >> "$TEST_CALLS"\n'
    )
    groot = tmp_path / "groot"
    python = groot / ".venv/bin/python"
    python.parent.mkdir(parents=True)
    python.write_text('#!/usr/bin/env bash\nprintf "BUILD:%s\\n" "$*" >> "$TEST_CALLS"\n')
    python.chmod(0o755)
    sources = [tmp_path / "source one", tmp_path / "source two"]
    for source in sources:
        source.mkdir()
    view = tmp_path / "final view"
    keys = {"FINAL_DATASET_SOURCES", "FINAL_VIEW_ROOT", "DATASET_SOURCES", "VIRTUAL_DATASET_ROOT", "DATASET_ROOT", "TRAIN_DATASET", "VALIDATION_DATASET", "TEST_DATASET", "MAX_STEPS", "SAVE_STEPS", "EXPERIMENTS", "PRECHECK_ONLY", "EVALUATION_SPLIT", "DRY_RUN"}
    env = {k: v for k, v in os.environ.items() if k not in keys}
    env.update(FINAL_DATASET_SOURCES=":".join(map(str, sources)), FINAL_VIEW_ROOT=str(view), GROOT_ROOT=str(groot), TEST_CALLS=str(tmp_path / "calls"), **overrides)
    result = subprocess.run(["bash", str(scripts / FINAL.name), mode], env=env, text=True, capture_output=True)
    calls = (tmp_path / "calls").read_text() if (tmp_path / "calls").exists() else ""
    return result, calls, view


def test_prepare_builds_split_view_then_trainval_and_never_starts_training(tmp_path):
    result, calls, view = _wrapper(tmp_path, "prepare")
    assert result.returncode == 0, result.stderr
    lines = calls.splitlines()
    assert len(lines) == 2
    assert f"--output {view}/combined --reuse --splits train validation" in lines[0]
    assert f"--sources {view}/combined/train {view}/combined/validation --output {view}/trainval --reuse" in lines[1]
    assert "RUN:" not in calls


@pytest.mark.parametrize("mode,precheck", [("check", "1"), ("run", "0")])
def test_final_wrapper_selects_trainval_without_any_evaluation(mode, precheck, tmp_path):
    result, calls, view = _wrapper(tmp_path, mode, MAX_STEPS="35000")
    assert result.returncode == 0, result.stderr
    assert f"RUN:{view}/trainval:unset:none:rgb,rgbd_turbo_late_fusion_pre_adapter:35000:{precheck}" in calls
    assert f"{view}/combined/test" not in calls
    assert len(calls.splitlines()) == 3


@pytest.mark.parametrize("mode,settings", [
    ("run", {}), ("check", {}), ("prepare", {"TRAIN_DATASET": "/conflict"}),
    ("run", {"MAX_STEPS": "35000", "EVALUATION_SPLIT": "validation"}),
    ("run", {"MAX_STEPS": "35000", "EVALUATION_SPLIT": "test", "ALLOW_TEST_EVALUATION": "1"}),
    ("run", {"MAX_STEPS": "35000", "DRY_RUN": "1"}),
    ("run", {"MAX_STEPS": "32000"}), ("invalid", {}),
])
def test_wrapper_rejects_ambiguous_requests_before_building(mode, settings, tmp_path):
    result, calls, _ = _wrapper(tmp_path, mode, **settings)
    assert result.returncode != 0
    assert calls == ""
