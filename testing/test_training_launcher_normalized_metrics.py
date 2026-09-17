from pathlib import Path
import re

import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "script_name",
    ["multi_finetune_evaluation.sh", "partial_multi_finetune_evaluation.sh"],
)
def test_normalized_metrics_are_required_before_evaluation_pass(script_name: str) -> None:
    source = (SCRIPTS_ROOT / script_name).read_text(encoding="utf-8")

    evaluator = source.index("-m scripts.analysis_tools.evaluate_checkpoints")
    normalizer = source.index("-m scripts.analysis_tools.normalized_action_metrics", evaluator)
    evaluation_pass = source.index("printf 'evaluation\\t%s\\tPASS\\t0\\n'", normalizer)

    assert evaluator < normalizer < evaluation_pass
    normalization_command = source[normalizer:evaluation_pass]
    assert "--run-dir" in normalization_command
    assert "--evaluation-dir" in normalization_command
    assert "--output-dir" in normalization_command
    assert "normalized_action_metrics_exec_hor_" in normalization_command
    assert "--statistics-path" in normalization_command
    assert "experiment_cfg/dataset_statistics.json" in normalization_command
    assert "--embodiment-tag NEW_EMBODIMENT" in normalization_command
    assert "--execution-horizon" in normalization_command
    assert "validation_frame_predictions" not in normalization_command


def test_multi_training_skips_duplicate_root_model_save() -> None:
    source = (SCRIPTS_ROOT / "multi_finetune_evaluation.sh").read_text(encoding="utf-8")

    training = source.index("-m gr00t.experiment.launch_finetune")
    evaluation = source.index("-m scripts.analysis_tools.evaluate_checkpoints", training)

    assert "--skip-final-model-save" in source[training:evaluation]


def test_multi_training_selects_robot_scoped_model_root_with_override() -> None:
    source = (SCRIPTS_ROOT / "multi_finetune_evaluation.sh").read_text(encoding="utf-8")

    inspire_start = source.index("Unitree_G1_Inspire_HeadOnly)")
    dex3_start = source.index("Unitree_G1_Dex3_HeadOnly)", inspire_start)
    case_end = source.index("*)", dex3_start)
    inspire_branch = source[inspire_start:dex3_start]
    dex3_branch = source[dex3_start:case_end]
    model_dirs = source[
        source.index("MODEL_DIRS=(") : source.index("PATCH_EMBED_FLAGS=(")
    ]

    assert 'DEFAULT_MODEL_ROOT="$HOME/Development/Models/inspire"' in inspire_branch
    assert 'DEFAULT_MODEL_PREFIX="inspire_"' in inspire_branch
    assert 'DEFAULT_MODEL_ROOT="$HOME/Development/Models/dex3"' in dex3_branch
    assert 'DEFAULT_MODEL_PREFIX=""' in dex3_branch
    assert 'MODEL_ROOT="${MODEL_ROOT:-$DEFAULT_MODEL_ROOT}"' in source
    assert '$MODEL_ROOT/${MODEL_PREFIX}c_rgb_' in model_dirs
    assert '$MODEL_ROOT/${MODEL_PREFIX}c_d1_' in model_dirs
    assert '$MODEL_ROOT/${MODEL_PREFIX}c_normals_' in model_dirs


def test_multi_training_always_retains_the_final_step_checkpoint() -> None:
    source = (SCRIPTS_ROOT / "multi_finetune_evaluation.sh").read_text(encoding="utf-8")

    step_pairs = {
        (int(max_steps), int(save_steps))
        for max_steps, save_steps in re.findall(
            r"MAX_STEPS=(\d+)\s+SAVE_STEPS=(\d+)", source
        )
    }
    assert {(1, 1), (25_000, 5_000)} <= step_pairs
    assert all(max_steps % save_steps == 0 for max_steps, save_steps in step_pairs)

    checkpoint_guard = re.search(
        r"\(\(\s*MAX_STEPS\s*%\s*SAVE_STEPS\s*!=\s*0\s*\)\)", source
    )
    assert checkpoint_guard is not None
    assert checkpoint_guard.start() < source.index("-m gr00t.experiment.launch_finetune")
