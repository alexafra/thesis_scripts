"""Execute only launcher argument/recipe construction, never training or prechecks."""

import os
from pathlib import Path
import subprocess

import pytest


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
MULTI = SCRIPTS_ROOT / "multi_finetune_evaluation.sh"
PARTIAL = SCRIPTS_ROOT / "partial_multi_finetune_evaluation.sh"


def _recipe_script():
    source = MULTI.read_text()
    return (
        source[source.index('MODALITY_DROPOUT="') : source.index('LOG_ROOT="')]
        + source[
            source.index("MODALITY_CONFIGS=(") : source.index('for dataset in "$TRAIN_DATASET"')
        ]
        + '\nfor i in "${SELECTED_INDICES[@]}"; do\n'
        + source[
            source.index("    MODALITY_DROPOUT_ARGS=()") : source.index(
                "    BACKBONE_STORAGE_ARGS=()"
            )
        ]
        + 'printf "ENTRY:%s:%s:%s\\n" "${EXPERIMENT_NAMES[$i]}" "${MODEL_DIRS[$i]}" "${MODALITY_CONFIGS[$i]}"\n'
        + 'printf "ARG:%s\\n" "${MODALITY_DROPOUT_ARGS[@]}"\n'
        + 'done\nprintf "FEATURE:%s\\n" "${REQUIRED_FEATURES[@]}"\n'
    )


def _run_recipe(
    dropout="0", experiments="rgb,depth,rgbd_gray_separate_views", policy="independent"
):
    env = dict(os.environ)
    env.update(
        MODALITY_DROPOUT=dropout,
        MODALITY_DROPOUT_STATE_POLICY=policy,
        EXPERIMENTS=experiments,
        CONFIG_PREFIX="g1_inspire",
        MODEL_ROOT="/unused/models",
        MODEL_PREFIX="inspire_",
        RUN_LABEL="25k",
        RUN_SUFFIX="test",
        DATASET_ROBOT_TYPE="Unitree_G1_Inspire_HeadOnly",
    )
    if dropout is None:
        env.pop("MODALITY_DROPOUT", None)
    return subprocess.run(
        ["bash", "-euc", _recipe_script()], env=env, capture_output=True, text=True
    )


def test_zero_dropout_retains_legacy_arguments_and_names():
    result = _run_recipe()
    assert result.returncode == 0, result.stderr
    assert "--vision-modality-dropout" not in result.stdout
    assert "moddrop05each" not in result.stdout
    assert "c_rgb_patch_tuned_bf16_batch_32_acc_1_25k_test:" in result.stdout


def test_unset_dropout_defaults_on_for_geometry_but_not_rgb():
    result = _run_recipe(None)
    assert result.returncode == 0, result.stderr
    entries = result.stdout.split("ENTRY:")[1:]
    assert "--vision-modality-dropout" not in entries[0]
    assert "moddrop05each" not in entries[0]
    for entry in entries[1:]:
        assert "_moddrop05each_independent:" in entry
        assert "ARG:--vision-modality-dropout-rgb-prob\nARG:0.05" in entry
        assert "ARG:--vision-modality-dropout-geometry-prob\nARG:0.05" in entry


@pytest.mark.parametrize("policy", ["state_first", "independent"])
def test_opt_in_leaves_rgb_untouched_and_labels_every_geometry_recipe(policy):
    experiments = (
        "rgb,depth,normals,rgbd_late_fusion_pre_adapter,rgbd_turbo_late_fusion_pre_adapter,"
        "rgbd_late_fusion_post_adapter,normals_late_fusion_pre_adapter,"
        "normals_late_fusion_post_adapter,rgbd_turbo_separate_views,normals_separate_views,"
        "rgbd_gray_separate_views"
    )
    result = _run_recipe("1", experiments, policy)
    assert result.returncode == 0, result.stderr
    entries = result.stdout.split("ENTRY:")[1:]
    assert len(entries) == 11
    assert "--vision-modality-dropout" not in entries[0]
    assert "moddrop05each" not in entries[0]
    for entry in entries[1:]:
        assert f"_moddrop05each_{policy}:" in entry
        assert "ARG:--vision-modality-dropout-rgb-prob\nARG:0.05" in entry
        assert "ARG:--vision-modality-dropout-geometry-prob\nARG:0.05" in entry
        assert f"ARG:--vision-modality-dropout-state-policy\nARG:{policy}" in entry


def test_separate_gray_recipe_requires_depth_and_uses_existing_config():
    result = _run_recipe("1", "rgbd_gray_separate_views")
    assert result.returncode == 0, result.stderr
    assert "g1_inspire_head_3_channel_gray_depth_config.py" in result.stdout
    assert "FEATURE:observation.images.depth_gray_view" in result.stdout


@pytest.mark.parametrize("dropout,policy", [("2", "state_first"), ("1", "wrong")])
def test_invalid_dropout_switches_fail_before_any_execution(dropout, policy):
    result = _run_recipe(dropout, policy=policy)
    assert result.returncode != 0
    assert "Error:" in result.stderr


def test_generic_wrapper_exposes_and_forwards_all_three_dropout_flags():
    # This source-based check complements actual execution of the multi-runner's
    # argument selection without substituting a real Python training process.
    default_repo = SCRIPTS_ROOT.parent / "Isaac-GR00T"
    if not default_repo.is_dir():
        default_repo = SCRIPTS_ROOT.parent
    repo = Path(os.environ.get("GROOT_MODALITY_DROPOUT_TEST_REPO", default_repo))
    source = (repo / "examples/finetune.sh").read_text()
    for name in ("rgb-prob", "geometry-prob", "state-policy"):
        assert f"--vision-modality-dropout-{name})" in source
        assert f"LAUNCH_CMD+=(--vision-modality-dropout-{name}" in source
    subprocess.run(["bash", "-n", str(repo / "examples/finetune.sh")], check=True)
    subprocess.run(["bash", "-n", str(MULTI)], check=True)


def _run_partial_recipe(dropout="0", experiments="missed_normals", policy="independent"):
    source = PARTIAL.read_text()
    script = (
        source[source.index('MODALITY_DROPOUT="') : source.index('LOG_ROOT="')]
        + source[source.index("MODEL_DIRS=()") : source.index("printf 'stage\\tmodel")]
        + '\nprintf "MODEL:%s\\n" "${MODEL_DIRS[@]}"\n'
    )
    env = dict(os.environ)
    env.update(
        MODALITY_DROPOUT=dropout,
        MODALITY_DROPOUT_STATE_POLICY=policy,
        EXPERIMENTS=experiments,
        MODEL_ROOT="/unused/models",
        MODEL_PREFIX="inspire_",
        RUN_LABEL="25k",
        RUN_SUFFIX="test",
        DATASET_ROBOT_TYPE="Unitree_G1_Inspire_HeadOnly",
    )
    if dropout is None:
        env.pop("MODALITY_DROPOUT", None)
    return subprocess.run(["bash", "-euc", script], env=env, text=True, capture_output=True)


def test_partial_unset_dropout_matches_new_training_default():
    result = _run_partial_recipe(None, "rgb,rgbd_gray_separate_views")
    assert result.returncode == 0, result.stderr
    models = [line for line in result.stdout.splitlines() if line.startswith("MODEL:")]
    assert "moddrop05each" not in models[0]
    assert models[1].endswith("_moddrop05each_independent")


def test_partial_historical_selection_stays_no_dropout_when_unset():
    result = _run_partial_recipe(None)
    assert result.returncode == 0, result.stderr
    assert "moddrop05each" not in result.stdout
    assert "20k_1808_1" in result.stdout


@pytest.mark.parametrize("dropout", ["0", "1"])
@pytest.mark.parametrize("policy", ["independent", "state_first"])
def test_partial_directory_selection_exactly_matches_all_multi_recipes(dropout, policy):
    experiments = (
        "rgb,depth,normals,rgbd_late_fusion_pre_adapter,rgbd_turbo_late_fusion_pre_adapter,"
        "rgbd_late_fusion_post_adapter,normals_late_fusion_pre_adapter,"
        "normals_late_fusion_post_adapter,rgbd_turbo_separate_views,normals_separate_views,"
        "rgbd_gray_separate_views"
    )
    multi = _run_recipe(dropout, experiments, policy)
    partial = _run_partial_recipe(dropout, experiments, policy)
    assert multi.returncode == 0, multi.stderr
    assert partial.returncode == 0, partial.stderr
    expected = [
        line.split(":")[2] for line in multi.stdout.splitlines() if line.startswith("ENTRY:")
    ]
    actual = [
        line.removeprefix("MODEL:")
        for line in partial.stdout.splitlines()
        if line.startswith("MODEL:")
    ]
    assert actual == expected
    assert "moddrop05each" not in actual[0]


def test_partial_preserves_fixed_legacy_default_and_does_not_enable_eval_dropout():
    result = _run_partial_recipe()
    assert result.returncode == 0, result.stderr
    paths = [line for line in result.stdout.splitlines() if line.startswith("MODEL:")]
    assert len(paths) == 2
    assert all(path.endswith("20k_1808_1") for path in paths)
    assert "moddrop05each" not in result.stdout
    source = PARTIAL.read_text()
    assert "--vision-modality-dropout" not in source
    assert "gr00t.experiment.launch_finetune" not in source
    subprocess.run(["bash", "-n", str(PARTIAL)], check=True)


@pytest.mark.parametrize(
    "dropout,experiments,policy",
    [
        ("1", "missed_normals", "independent"),
        ("2", "rgb", "independent"),
        ("1", "normals", "bad-policy"),
    ],
)
def test_partial_rejects_ambiguous_legacy_dropout_or_invalid_switches(dropout, experiments, policy):
    result = _run_partial_recipe(dropout, experiments, policy)
    assert result.returncode != 0
    assert "Error:" in result.stderr
