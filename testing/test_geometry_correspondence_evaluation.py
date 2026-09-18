from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

from gr00t.data.types import VideoChannelSource
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
    second = geometry.build_donor_assignments(
        records, shuffle_seed=42, shuffle_repeat=0
    )

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
    assert (
        geometry.donor_frame_for_phase(100, 101, 51, donor_phase="out_of_phase") == 25
    )


def test_same_episode_offsets_shift_by_requested_fraction_and_wrap():
    assert (
        geometry.same_episode_frame_for_offset(0, 101, intervention="offset_10pct")
        == 10
    )
    assert (
        geometry.same_episode_frame_for_offset(25, 101, intervention="offset_10pct")
        == 35
    )
    assert (
        geometry.same_episode_frame_for_offset(90, 101, intervention="offset_10pct")
        == 100
    )
    assert (
        geometry.same_episode_frame_for_offset(100, 101, intervention="offset_10pct")
        == 9
    )
    assert (
        geometry.same_episode_frame_for_offset(0, 101, intervention="offset_50pct")
        == 50
    )
    assert (
        geometry.same_episode_frame_for_offset(50, 101, intervention="offset_50pct")
        == 100
    )
    assert (
        geometry.same_episode_frame_for_offset(100, 101, intervention="offset_50pct")
        == 49
    )
    assert (
        geometry.same_episode_frame_for_offset(0, 1, intervention="offset_50pct") == 0
    )


def test_training_split_cannot_be_mislabeled_as_evaluation():
    try:
        geometry.validate_heldout_dataset_path(Path("/dataset/train"), "validation")
    except ValueError as exc:
        assert "held-out split" in str(exc)
        assert "validation" in str(exc)
    else:
        raise AssertionError("training path must never be accepted as validation")


def test_zero_image_is_rgb_only_and_zero_geometry_remains_geometry_only(tmp_path):
    common = [
        "--run-dir",
        "/model",
        "--dataset-path",
        "/dataset/validation",
        "--output-dir",
        str(tmp_path / "output"),
    ]
    args = geometry.parse_args(
        [
            *common,
            "--view-key",
            "ego_view",
            "--rgb-contract",
            "rgb_only",
            "--intervention",
            "zero_image",
        ]
    )
    assert args.geometry_key == "ego_view"
    assert args.intervention == "zero_image"

    for view_key, intervention in (
        ("ego_view", "zero_geometry"),
        ("surface_normals_view", "zero_image"),
    ):
        try:
            geometry.parse_args(
                [
                    *common,
                    "--view-key",
                    view_key,
                    *(["--rgb-contract", "rgb_only"] if view_key == "ego_view" else []),
                    "--intervention",
                    intervention,
                ]
            )
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(f"{intervention} should be rejected for {view_key}")


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
        shuffled["annotation.human.task_description"]
        == flat["annotation.human.task_description"]
    )
    np.testing.assert_array_equal(shuffled["video.depth_gray_view"], donor[None])
    np.testing.assert_array_equal(flat["video.depth_gray_view"], depth)


def test_rgb_replacement_changes_only_ego_view_and_zero_image_is_black():
    rgb = np.arange(12, dtype=np.uint8).reshape(1, 2, 2, 3)
    normals = np.full((1, 2, 2, 3), 127, dtype=np.uint8)
    state = np.arange(4, dtype=np.float32).reshape(1, 4)
    flat = {
        "video.ego_view": rgb,
        "video.surface_normals_view": normals,
        "state.left_arm": state,
        "annotation.human.task_description": "stack the three red cups.",
    }
    black = geometry.zero_geometry_frame(rgb[0])
    replaced = geometry.replace_geometry(flat, "ego_view", black)

    assert black.shape == rgb[0].shape
    assert black.dtype == rgb.dtype
    assert not np.any(black)
    assert replaced["video.surface_normals_view"] is normals
    assert replaced["state.left_arm"] is state
    assert (
        replaced["annotation.human.task_description"]
        == flat["annotation.human.task_description"]
    )
    assert not np.any(replaced["video.ego_view"])
    np.testing.assert_array_equal(flat["video.ego_view"], rgb)
    assert "zero_image" in geometry.INTERVENTIONS


def test_brightness_scaling_is_uint8_half_up_nonmutating_and_preserves_normals():
    rgb = np.array([0, 1, 2, 3, 254, 255], dtype=np.uint8).reshape(1, 2, 3)
    original = rgb.copy()
    dimmed = geometry.scale_rgb_brightness(rgb, 0.5)

    assert dimmed.shape == rgb.shape
    assert dimmed.dtype == np.uint8
    assert dimmed.flags.c_contiguous
    assert not np.shares_memory(dimmed, rgb)
    np.testing.assert_array_equal(dimmed.reshape(-1), [0, 1, 1, 2, 127, 128])
    np.testing.assert_array_equal(rgb, original)

    normals = np.full((1, 1, 2, 3), 127, dtype=np.uint8)
    flat = {"video.ego_view": rgb[None], "video.surface_normals_view": normals}
    replaced = geometry.replace_geometry(flat, "ego_view", dimmed)
    assert replaced["video.surface_normals_view"] is normals
    assert replaced["video.ego_view"] is not flat["video.ego_view"]
    np.testing.assert_array_equal(flat["video.ego_view"], original[None])

    for invalid_scale in (0.0, 1.0, -0.1, float("inf"), float("nan")):
        try:
            geometry.scale_rgb_brightness(rgb, invalid_scale)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid brightness scale accepted: {invalid_scale}")
    try:
        geometry.scale_rgb_brightness(rgb.astype(np.float32), 0.5)
    except ValueError as exc:
        assert "uint8" in str(exc)
    else:
        raise AssertionError("non-uint8 brightness input must fail closed")


def test_brightness_cli_requires_explicit_valid_scale_and_rgb_contract(tmp_path):
    common = [
        "--run-dir",
        "/model",
        "--dataset-path",
        "/dataset/validation",
        "--output-dir",
        str(tmp_path / "output"),
        "--view-key",
        "ego_view",
        "--rgb-contract",
        "rgb_only",
    ]
    args = geometry.parse_args(
        [
            *common,
            "--intervention",
            "brightness_scale",
            "--brightness-scale",
            "0.5",
        ]
    )
    assert args.brightness_scale == 0.5
    assert args.rgb_contract == "rgb_only"

    invalid_argvs = (
        [*common, "--intervention", "brightness_scale"],
        [
            *common,
            "--intervention",
            "brightness_scale",
            "--brightness-scale",
            "1",
        ],
        [*common, "--intervention", "zero_image", "--brightness-scale", "0.5"],
        [
            "--run-dir",
            "/model",
            "--dataset-path",
            "/dataset/validation",
            "--output-dir",
            str(tmp_path / "other"),
            "--view-key",
            "ego_view",
            "--intervention",
            "brightness_scale",
            "--brightness-scale",
            "0.5",
        ],
    )
    for argv in invalid_argvs:
        try:
            geometry.parse_args(argv)
        except SystemExit as exc:
            assert exc.code == 2
        else:
            raise AssertionError(f"invalid brightness CLI accepted: {argv}")


def test_rgb_and_early_fusion_visual_contracts_remain_distinct():
    shared = {
        "state": geometry.ModalityConfig(delta_indices=[0], modality_keys=["state"]),
        "action": geometry.ModalityConfig(
            delta_indices=list(range(8)), modality_keys=["action"]
        ),
        "language": geometry.ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    }
    for companion_key, channels, intervention_keys in (
        ("depth_gray_view", (0,), ("depth_gray_view",)),
        (
            "surface_normals_view",
            (0, 1, 2),
            ("ego_view", "surface_normals_view"),
        ),
    ):
        fused = {
            **shared,
            "video": geometry.ModalityConfig(
                delta_indices=[0],
                modality_keys=["ego_view", companion_key],
                channel_fusion=[
                    VideoChannelSource("ego_view", (0, 1, 2)),
                    VideoChannelSource(companion_key, channels),
                ],
            ),
        }
        for intervention_key in intervention_keys:
            rgb_contract = (
                "rgb_normals_early_fusion" if intervention_key == "ego_view" else None
            )
            assert (
                geometry._validate_contract(
                    fused,
                    geometry_key=intervention_key,
                    execution_horizon=8,
                    rgb_contract=rgb_contract,
                )
                == 8
            )

    rgb_only = {
        **shared,
        "video": geometry.ModalityConfig(delta_indices=[0], modality_keys=["ego_view"]),
    }
    assert (
        geometry._validate_contract(
            rgb_only,
            geometry_key="ego_view",
            execution_horizon=8,
            rgb_contract="rgb_only",
        )
        == 8
    )
    for modality, contract in (
        (rgb_only, "rgb_normals_early_fusion"),
        (fused, "rgb_only"),
    ):
        try:
            geometry._validate_contract(
                modality,
                geometry_key="ego_view",
                execution_horizon=8,
                rgb_contract=contract,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"mismatched RGB contract accepted: {contract}")


def test_pre_adapter_late_fusion_contract_supports_independent_rgb_and_normals():
    shared = {
        "state": geometry.ModalityConfig(delta_indices=[0], modality_keys=["state"]),
        "action": geometry.ModalityConfig(
            delta_indices=list(range(8)), modality_keys=["action"]
        ),
        "language": geometry.ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    }
    late_fusion = {
        **shared,
        "video": geometry.ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view", "surface_normals_view"],
            post_vision_fusion=True,
            post_vision_fusion_stage="pre_vision_language_adapter",
        ),
    }

    assert (
        geometry._validate_contract(
            late_fusion,
            geometry_key="ego_view",
            execution_horizon=8,
            rgb_contract="rgb_normals_late_fusion_pre_adapter",
        )
        == 8
    )
    assert (
        geometry._validate_contract(
            late_fusion,
            geometry_key="surface_normals_view",
            execution_horizon=8,
        )
        == 8
    )


def test_late_fusion_ablation_rejects_post_adapter_stage():
    modality = {
        "video": geometry.ModalityConfig(
            delta_indices=[0],
            modality_keys=["ego_view", "surface_normals_view"],
            post_vision_fusion=True,
            post_vision_fusion_stage="post_vision_language_adapter",
        ),
        "state": geometry.ModalityConfig(delta_indices=[0], modality_keys=["state"]),
        "action": geometry.ModalityConfig(
            delta_indices=list(range(8)), modality_keys=["action"]
        ),
        "language": geometry.ModalityConfig(
            delta_indices=[0], modality_keys=["annotation.human.task_description"]
        ),
    }

    for geometry_key, rgb_contract in (
        ("ego_view", "rgb_normals_late_fusion_pre_adapter"),
        ("surface_normals_view", None),
    ):
        try:
            geometry._validate_contract(
                modality,
                geometry_key=geometry_key,
                execution_horizon=8,
                rgb_contract=rgb_contract,
            )
        except ValueError as exc:
            assert "pre-adapter late fusion" in str(exc)
        else:
            raise AssertionError("post-adapter late fusion must fail the pre-adapter contract")


def test_rgb_metadata_declares_only_rgb_changed():
    assert geometry.unchanged_inputs_for_visual_intervention(
        "ego_view", ["ego_view", "surface_normals_view"]
    ) == [
        "surface_normals_view",
        "state",
        "language",
        "expert_action",
    ]
    assert geometry.unchanged_inputs_for_visual_intervention(
        "surface_normals_view", ["ego_view", "surface_normals_view"]
    ) == [
        "ego_view",
        "state",
        "language",
        "expert_action",
    ]
    assert geometry.unchanged_inputs_for_visual_intervention(
        "ego_view", ["ego_view"]
    ) == ["state", "language", "expert_action"]


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
    primary = summary[
        (summary["scope"] == "all_tasks") & (summary["group"] == "all")
    ].iloc[0]

    assert np.isclose(primary["micro_delta_mae_counterfactual_minus_intact"], 0.4)
    assert np.isclose(primary["episode_macro_mean_delta_mae"], 0.4)
    assert np.isclose(primary["task_balanced_mean_delta_mae"], 0.45)
    assert primary["episode_win_rate_counterfactual_worse"] == 1.0
    assert primary["episode_macro_prediction_change_mae"] == 0.5


def test_runner_is_evaluation_only_and_supports_selected_interventions_for_all_models():
    runner = SCRIPT_PATH.with_name("multi_geometry_correspondence_evaluation.sh")
    text = runner.read_text(encoding="utf-8")

    for required in (
        'MODE="${MODE:-both}"',
        'RGB_MODEL="${RGB_MODEL:-',
        'GEOMETRY_KEYS=("depth_gray_view" "surface_normals_view")',
        'GEOMETRY_KEYS=("ego_view")',
        'REQUIRED_VIDEO_KEYS=("ego_view" "surface_normals_view")',
        'RGB_CONTRACTS=("rgb_only")',
        'RGB_CONTRACTS=("rgb_normals_early_fusion")',
        'RGB_CONTRACTS=("rgb_normals_late_fusion_pre_adapter")',
        'VISION_CONTRACTS=("late_fusion_pre_adapter")',
        'INTERVENTIONS_CSV="${INTERVENTIONS_CSV:-phase_matched,out_of_phase,zero_geometry}"',
        'BRIGHTNESS_SCALE="${BRIGHTNESS_SCALE:-}"',
        "offset_10pct | offset_50pct | zero_geometry | zero_image",
        "brightness_scale)",
        'NORMALS_MODEL="${NORMALS_MODEL:-',
        'LATE_NORMALS_MODEL="${LATE_NORMALS_MODEL:-',
        "rgb)",
        "rgb_in_normals)",
        "late_normals)",
        "rgb_in_late_normals)",
        'DATASET_SCOPE="validation"',
        'DATASET_PATH="$DATASET_ROOT/validation"',
        "--view-key",
        "--intervention",
        "--rgb-contract",
        "--brightness-scale",
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

    rgb_rows = geometry._frame_mapping(
        [record],
        {0: {}},
        execution_horizon=8,
        intervention="zero_image",
    )
    assert rgb_rows == [{**row, "intervention": "zero_image"} for row in rows]


def test_same_episode_offset_mapping_uses_recipient_episode_without_donor_assignment():
    record = _record(0, "task a", length=101)
    rows = geometry._frame_mapping(
        [record],
        {0: {}},
        execution_horizon=25,
        intervention="offset_10pct",
    )

    assert [row["recipient_frame"] for row in rows] == [0, 25, 50, 75, 100]
    assert [row["donor_frame"] for row in rows] == [10, 35, 60, 85, 9]
    assert {row["replacement_source"] for row in rows} == {"same_episode"}
    assert {row["donor_episode_index"] for row in rows} == {record.episode_index}
    manifest = geometry._donor_manifest({0: {}}, intervention="offset_10pct")
    assert manifest == [
        {
            "intervention": "offset_10pct",
            "donor_required": False,
            "replacement": "same-episode visual input with a circular integer-frame shift",
            "phase_offset_fraction": 0.1,
        }
    ]


def test_brightness_manifest_and_mapping_use_same_frame_without_any_donor():
    record = _record(0, "task a", length=17)
    rows = geometry._frame_mapping(
        [record],
        {0: {}},
        execution_horizon=8,
        intervention="brightness_scale",
        brightness_scale=0.5,
    )

    assert [row["recipient_frame"] for row in rows] == [0, 8, 16]
    assert {row["replacement_source"] for row in rows} == {
        "same_frame_brightness_scale"
    }
    assert all(row["donor_progress"] is None for row in rows)
    assert all(row["donor_episode_index"] is None for row in rows)
    assert all(row["donor_length"] is None for row in rows)
    assert all(row["donor_frame"] is None for row in rows)
    assert {row["brightness_scale"] for row in rows} == {0.5}
    assert {row["rounding_rule"] for row in rows} == {geometry.BRIGHTNESS_ROUNDING_RULE}
    assert "brightness_scale" in geometry.CURRENT_FRAME_INTERVENTIONS
    assert "brightness_scale" not in geometry.CROSS_EPISODE_INTERVENTIONS

    manifest = geometry._donor_manifest(
        {0: {}}, intervention="brightness_scale", brightness_scale=0.5
    )
    assert manifest == [
        {
            "intervention": "brightness_scale",
            "donor_required": False,
            "replacement": "same-frame ego_view digital sRGB-byte intensity scaling",
            "brightness_scale": 0.5,
            "rounding_rule": geometry.BRIGHTNESS_ROUNDING_RULE,
        }
    ]


def test_brightness_metadata_is_explicit_about_transform_and_unchanged_normals(
    tmp_path,
):
    args = geometry.parse_args(
        [
            "--run-dir",
            "/model",
            "--dataset-path",
            "/dataset/validation",
            "--output-dir",
            str(tmp_path / "output"),
            "--view-key",
            "ego_view",
            "--rgb-contract",
            "rgb_normals_early_fusion",
            "--intervention",
            "brightness_scale",
            "--brightness-scale",
            "0.5",
        ]
    )
    metadata = geometry._run_metadata(
        args=args,
        target_metadata=[],
        modality_signature={
            "video": {
                "keys": ["ego_view", "surface_normals_view"],
                "delta_indices": [0],
                "vision_channel_layout": None,
            }
        },
        elapsed_seconds=1.0,
    )
    intervention = metadata["intervention"]
    assert metadata["version"] == 5
    assert intervention["brightness_scale"] == 0.5
    assert intervention["rounding_rule"] == geometry.BRIGHTNESS_ROUNDING_RULE
    assert intervention["changed_input"] == "ego_view"
    assert intervention["unchanged_inputs"] == [
        "surface_normals_view",
        "state",
        "language",
        "expert_action",
    ]
    assert "sRGB-byte" in intervention["colour_space_semantics"]
    assert "auto-exposure" in intervention["caveat"]
    assert "dark-condition degradation" in intervention["caveat"]
