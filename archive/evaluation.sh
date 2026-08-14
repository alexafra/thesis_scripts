cd "$HOME/Development/Isaac-GR00T"

RUN_DIR="$HOME/Development/Models/gr00t_colour_only_batch_32_acc_1_0803_0930"
EVAL_DATASET="/home/alex/Development/Datasets/lerobot2/toothepaste_3107_test"
TRAIN_DATASET="/home/alex/Development/Datasets/lerobot2/toothepaste_3107_train"

CUDA_VISIBLE_DEVICES=0 \
NO_ALBUMENTATIONS_UPDATE=1 \
uv run --no-sync python -m scripts.analysis_tools.evaluate_checkpoints \
    --run-dir "$RUN_DIR" \
    --dataset-path "$EVAL_DATASET" \
    --train-dataset-path "$TRAIN_DATASET" \
    --output-dir "$RUN_DIR/evaluation" \
    --steps 0 \
    --execution-horizon 16 \
    --denoising-steps 4 \
    --modality-keys left_arm right_arm left_hand right_hand \
    --train-probe-episodes 4 \
    --train-probe-seed 42 \


uv run --no-sync python scripts/analysis_tools/plot_training_history.py \
    --run-dir "$RUN_DIR" \
    --smooth-window 20

RUN_DIR="$HOME/Development/Models/gr00t_gray_scale_batch_32_acc_1_0803_0930"

CUDA_VISIBLE_DEVICES=0 \
NO_ALBUMENTATIONS_UPDATE=1 \
uv run --no-sync python -m scripts.analysis_tools.evaluate_checkpoints \
    --run-dir "$RUN_DIR" \
    --dataset-path "$EVAL_DATASET" \
    --train-dataset-path "$TRAIN_DATASET" \
    --output-dir "$RUN_DIR/evaluation" \
    --steps 0 \
    --execution-horizon 16 \
    --denoising-steps 4 \
    --modality-keys left_arm right_arm left_hand right_hand \
    --train-probe-episodes 4 \
    --train-probe-seed 42 \


uv run --no-sync python scripts/analysis_tools/plot_training_history.py \
    --run-dir "$RUN_DIR" \
    --smooth-window 20