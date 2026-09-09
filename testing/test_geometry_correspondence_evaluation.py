from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).parents[1] / "geometry_correspondence_evaluation.py"
SPEC = spec_from_file_location("geometry_correspondence_evaluation", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
geometry = module_from_spec(SPEC)
sys.modules[SPEC.name] = geometry
SPEC.loader.exec_module(geometry)


def _record(position: int, task: str, *, length: int = 101):
    return geometry.EpisodeRecord(
        loader_position=position,
        episode_index=position + 10,
        length=length,
        tasks=(task,),
        task=task,
        cohort="base",
    )


def test_donor_assignment_is_deterministic_same_task_and_never_self():
    records = [
        _record(0, "task a"),
        _record(1, "task a"),
        _record(2, "task a"),
        _record(3, "task b"),
        _record(4, "task b"),
    ]
    first = geometry.build_donor_assignments(records, shuffle_seed=42, shuffle_repeat=0)
    second = geometry.build_donor_assignments(records, shuffle_seed=42, shuffle_repeat=0)

    assert first == second
    by_position = {record.loader_position: record for record in records}
    for position, assignment in first.items():
        recipient = by_position[position]
        donor = by_position[assignment.donor_loader_position]
        assert donor.episode_index != recipient.episode_index
        assert donor.tasks == recipient.tasks


def test_donor_assignment_fails_closed_for_singleton_task():
    records = [_record(0, "task a"), _record(1, "task a"), _record(2, "singleton")]

    try:
        geometry.build_donor_assignments(records, shuffle_seed=42, shuffle_repeat=0)
    except ValueError as exc:
        assert "requires at least two" in str(exc)
        assert "singleton" in str(exc)
    else:
        raise AssertionError("singleton task should not fall back to a different task")


def test_phase_mapping_matches_endpoints_and_normalized_progress():
    assert geometry.phase_matched_frame(0, 101, 51) == 0
    assert geometry.phase_matched_frame(50, 101, 51) == 25
    assert geometry.phase_matched_frame(100, 101, 51) == 50
    assert geometry.phase_matched_frame(0, 1, 51) == 0


def test_out_of_phase_mapping_is_shifted_by_half_an_episode():
    assert geometry.donor_frame_for_phase(0, 101, 51, donor_phase="out_of_phase") == 25
    assert geometry.donor_frame_for_phase(25, 101, 51, donor_phase="out_of_phase") == 38
    assert geometry.donor_frame_for_phase(50, 101, 51, donor_phase="out_of_phase") == 0
    assert geometry.donor_frame_for_phase(100, 101, 51, donor_phase="out_of_phase") == 25


def test_training_split_cannot_be_mislabeled_as_evaluation():
    try:
        geometry.validate_heldout_dataset_path(Path("/dataset/train"), "validation")
    except ValueError as exc:
        assert "held-out split" in str(exc)
        assert "validation" in str(exc)
    else:
        raise AssertionError("training path must never be accepted as validation")


def test_numpy_label_encoding_does_not_truncate_task_or_cohort():
    task = "stack the three red cups."
    cohort = "right_only_1408"
    encoded_task = geometry._repeat_label(task, 3)
    encoded_cohort = geometry._repeat_label(cohort, 3)

    assert encoded_task.tolist() == [task, task, task]
    assert encoded_cohort.tolist() == [cohort, cohort, cohort]


def test_geometry_replacement_changes_only_requested_evaluation_view():
    rgb = np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3)
    depth = np.zeros((1, 2, 2, 3), dtype=np.uint8)
    state = np.arange(4, dtype=np.float32).reshape(1, 4)
    flat = {
        "video.ego_view": rgb,
        "video.depth_gray_view": depth,
        "state.left_arm": state,
        "annotation.human.task_description": "stack the three red cups.",
    }
    donor = np.full((2, 2, 3), 17, dtype=np.uint8)

    shuffled = geometry.replace_geometry(flat, "depth_gray_view", donor)

    assert shuffled is not flat
    assert shuffled["video.ego_view"] is flat["video.ego_view"]
    assert shuffled["state.left_arm"] is flat["state.left_arm"]
    assert (
        shuffled["annotation.human.task_description"] == flat["annotation.human.task_description"]
    )
    np.testing.assert_array_equal(shuffled["video.depth_gray_view"], donor[None])
    np.testing.assert_array_equal(flat["video.depth_gray_view"], depth)


def test_zero_geometry_preserves_shape_dtype_and_supports_depth_and_normals():
    for geometry_key in ("depth_gray_view", "surface_normals_view"):
        original = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        zeroed = geometry.zero_geometry_frame(original)
        flat = {
            "video.ego_view": np.full((1, 2, 2, 3), 99, dtype=np.uint8),
            f"video.{geometry_key}": original[None],
        }
        replaced = geometry.replace_geometry(flat, geometry_key, zeroed)

        assert zeroed.shape == original.shape
        assert zeroed.dtype == original.dtype
        assert not np.any(zeroed)
        assert np.any(original)
        assert not np.any(replaced[f"video.{geometry_key}"])
        assert replaced["video.ego_view"] is flat["video.ego_view"]

    assert "zero_geometry" in geometry.INTERVENTIONS


def test_paired_summary_reports_sensitivity_and_positive_counterfactual_error_delta():
    rows = []
    for episode, task, delta in (
        (1, "task a", 0.2),
        (2, "task a", 0.4),
        (3, "task b", 0.6),
    ):
        rows.append(
            {
                "checkpoint_step": 25000,
                "split": "validation",
                "intervention": "out_of_phase",
                "shuffle_repeat": 0,
                "episode_index": episode,
                "task": task,
                "cohort": "base",
                "group": "all",
                "sample_count": 10,
                "intact_sum_absolute_error": 1.0,
                "counterfactual_sum_absolute_error": 1.0 + 10 * delta,
                "intact_sum_squared_error": 0.5,
                "counterfactual_sum_squared_error": 0.5 + 10 * delta,
                "intact_mae": 0.1,
                "counterfactual_mae": 0.1 + delta,
                "delta_mae_counterfactual_minus_intact": delta,
                "intact_mse": 0.05,
                "counterfactual_mse": 0.05 + delta,
                "delta_mse_counterfactual_minus_intact": delta,
                "sum_absolute_prediction_change": 5.0,
                "mean_absolute_prediction_change": 0.5,
                "max_absolute_prediction_change": 1.0,
            }
        )
    summary = geometry.summarize_paired_effects(
        pd.DataFrame(rows),
        checkpoint_step_value=25000,
        split="validation",
        bootstrap_replicates=20,
        bootstrap_seed=42,
    )
    primary = summary[(summary["scope"] == "all_tasks") & (summary["group"] == "all")].iloc[0]

    assert np.isclose(primary["micro_delta_mae_counterfactual_minus_intact"], 0.4)
    assert np.isclose(primary["episode_macro_mean_delta_mae"], 0.4)
    assert np.isclose(primary["task_balanced_mean_delta_mae"], 0.45)
    assert primary["episode_win_rate_counterfactual_worse"] == 1.0
    assert primary["episode_macro_prediction_change_mae"] == 0.5


def test_runner_is_evaluation_only_and_runs_three_interventions_for_both_models():
    runner = SCRIPT_PATH.with_name("multi_geometry_correspondence_evaluation.sh")
    text = runner.read_text(encoding="utf-8")

    for required in (
        'MODE="${MODE:-both}"',
        'GEOMETRY_KEYS=("depth_gray_view" "surface_normals_view")',
        'INTERVENTIONS=("phase_matched" "out_of_phase" "zero_geometry")',
        'DATASET_SCOPE="validation"',
        'DATASET_PATH="$DATASET_ROOT/validation"',
        "--geometry-key",
        "--intervention",
        "--shuffle-seed 42",
        "HF_HUB_OFFLINE=1",
        "PRECHECK_ONLY",
        "validate_checkpoint_artifacts",
        "model.safetensors.index.json",
        "nvidia-smi failed; refusing to assume the GPU is available",
        "[/]geometry_correspondence_evaluation.py",
    ):
        assert required in text
    for forbidden in (
        "python -m gr00t.experiment.launch_finetune",
        "systemd-run",
        '--dataset-path "$DATASET_ROOT/train"',
        'DATASET_PATH="${DATASET_PATH:-',
        'DATASET_SCOPE="${DATASET_SCOPE:-',
    ):
        assert forbidden not in text


def test_zero_geometry_frame_mapping_has_no_donor():
    record = _record(0, "task a", length=17)
    rows = geometry._frame_mapping(
        [record],
        {0: {}},
        execution_horizon=8,
        intervention="zero_geometry",
    )

    assert [row["recipient_frame"] for row in rows] == [0, 8, 16]
    assert {row["replacement_source"] for row in rows} == {"all_zeros"}
    assert all(row["donor_episode_index"] is None for row in rows)
    assert all(row["donor_frame"] is None for row in rows)
