from pathlib import Path


SCRIPT = Path(__file__).parents[1] / "geometry_only_finetune_evaluation.sh"


def test_geometry_only_runner_has_depth_and_normals_without_fusion_flags():
    text = SCRIPT.read_text(encoding="utf-8")

    assert 'VISUAL_MODE="${VISUAL_MODE:-depth}"' in text
    assert "g1_dex3_head_depth_gray_only_config.py" in text
    assert "g1_dex3_head_surface_normals_only_config.py" in text
    assert 'VIEW_KEY="depth_gray_view"' in text
    assert 'VIEW_KEY="surface_normals_view"' in text
    assert "--vision-patch-embed-init" not in text
    assert "4_channel_gray_depth_fusion" not in text
    assert "6_channel_surface_normals_fusion" not in text


def test_geometry_only_runner_matches_control_protocol_and_stays_offline():
    text = SCRIPT.read_text(encoding="utf-8")

    for required in (
        "MAX_STEPS=30000",
        "SAVE_STEPS=5000",
        "--global-batch-size 32",
        "--gradient-accumulation-steps 1",
        "--load-bf16",
        "--execution-horizon",
        "--denoising-steps 4",
        "--inference-seed 42",
        "--base-model-path \"$BASE_MODEL_PATH\"",
        "HF_HUB_OFFLINE=1",
        "TRANSFORMERS_OFFLINE=1",
        "unset HF_TOKEN HUGGING_FACE_HUB_TOKEN",
        "nvidia-smi failed; refusing to assume the GPU is available",
        "PRECHECK_ONLY",
    ):
        assert required in text
