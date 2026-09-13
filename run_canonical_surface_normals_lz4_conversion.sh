#!/usr/bin/env bash
set -euo pipefail

PYTHON="/home/alex/miniconda3/envs/unitree_lerobot/bin/python"
SCRIPT="/home/alex/Development/scripts/convert_canonical_surface_normals_to_lz4.py"
LOG_ROOT="/home/alex/Development/logs/data_pipeline"
mkdir -p "$LOG_ROOT"
LOG="$LOG_ROOT/surface_normals_lz4_conversion.log"

exec "$PYTHON" -u "$SCRIPT" \
    --dataset /home/alex/Development/Datasets/lerobot2/atomic_combined_09_08_And_10_08 \
    --splits train validation test \
    --chunk-frames 32 \
    >>"$LOG" 2>&1
