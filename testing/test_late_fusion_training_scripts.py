from __future__ import annotations

from pathlib import Path


SCRIPTS_ROOT = Path(__file__).resolve().parents[1]
MULTI = SCRIPTS_ROOT / "multi_finetune_evaluation.sh"
PARTIAL = SCRIPTS_ROOT / "partial_multi_finetune_evaluation.sh"

LATE_EXPERIMENTS = (
    "rgbd_late_fusion_pre_adapter",
    "rgbd_late_fusion_post_adapter",
    "normals_late_fusion_pre_adapter",
    "normals_late_fusion_post_adapter",
)


def test_multi_late_fusion_is_opt_in_and_uses_the_four_existing_configs() -> None:
    source = MULTI.read_text(encoding="utf-8")

    assert 'EXPERIMENTS="${EXPERIMENTS:-rgb,normals,depth}"' in source
    expected_configs = (
        "g1_inspire_head_rgbd_late_fusion_pre_adapter_config.py",
        "g1_inspire_head_rgbd_late_fusion_post_adapter_config.py",
        "g1_inspire_head_rgb_surface_normals_late_fusion_pre_adapter_config.py",
        "g1_inspire_head_rgb_surface_normals_late_fusion_post_adapter_config.py",
    )
    for experiment in LATE_EXPERIMENTS:
        assert f"{experiment}) index=" in source
    for config in expected_configs:
        assert source.count(config) == 1


def test_multi_late_fusion_names_encode_the_frozen_four_adapter_contract() -> None:
    source = MULTI.read_text(encoding="utf-8")
    model_dirs = source[source.index("MODEL_DIRS=(") : source.index("PATCH_EMBED_FLAGS=(")]
    patch_flags = source[
        source.index("PATCH_EMBED_FLAGS=(") : source.index("LOAD_BF16_FLAGS=(")
    ]

    for experiment in LATE_EXPERIMENTS:
        assert experiment in model_dirs
    assert model_dirs.count("4x_linear_rgb50_geo50_patch_frozen") == 4
    assert patch_flags.count('"--no-tune-vision-patch-embed"') == 4
    assert "--no-tune-llm" in source
    assert "--no-tune-visual" in source
    assert "--tune-projector" in source


def test_late_fusion_requires_the_geometry_feature_used_by_each_pair() -> None:
    source = MULTI.read_text(encoding="utf-8")

    depth_guard = source[
        source.index('SELECTED_EXPERIMENT_SET[depth]') : source.index(
            "REQUIRED_FEATURES+=(observation.images.depth_gray_view)"
        )
    ]
    normals_guard = source[
        source.index('SELECTED_EXPERIMENT_SET[normals]') : source.index(
            "REQUIRED_FEATURES+=(observation.images.surface_normals_view)"
        )
    ]
    assert "rgbd_late_fusion_pre_adapter" in depth_guard
    assert "rgbd_late_fusion_post_adapter" in depth_guard
    assert "normals_late_fusion_pre_adapter" in normals_guard
    assert "normals_late_fusion_post_adapter" in normals_guard


def test_partial_uses_the_same_late_fusion_model_names_without_training() -> None:
    multi_source = MULTI.read_text(encoding="utf-8")
    partial_source = PARTIAL.read_text(encoding="utf-8")

    for experiment in LATE_EXPERIMENTS:
        assert experiment in multi_source
        assert experiment in partial_source
    assert partial_source.count("4x_linear_rgb50_geo50_patch_frozen") == 4
    assert "gr00t.experiment.launch_finetune" not in partial_source
    assert "scripts.analysis_tools.evaluate_checkpoints" in partial_source
    assert 'EXPERIMENTS="${EXPERIMENTS:-missed_normals}"' in partial_source
