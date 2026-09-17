#!/usr/bin/env bash
set -Eeuo pipefail

# Evaluation-only visual correspondence ablation. This script never trains,
# mutates a dataset, queues itself, or opens a robot/DDS connection.
export PATH="/home/alex/.local/bin:/usr/local/bin:/usr/bin:/bin:${PATH:-}"

GROOT_DIR="/home/alex/Development/Isaac-GR00T"
GROOT_PYTHON="$GROOT_DIR/.venv/bin/python"
EVALUATOR="/home/alex/Development/scripts/geometry_correspondence_evaluation.py"
DATASET_ROOT="${DATASET_ROOT:-/home/alex/Development/Datasets/lerobot2/atomic_combined_09_08_And_10_08_plus_pick_three_cups_right_only_1408_plus_stack_cups_09_08}"
DATASET_SCOPE="validation"
DATASET_PATH="$DATASET_ROOT/validation"
MODEL_ROOT="${MODEL_ROOT:-/home/alex/Development/Models/dex3}"
LOG_ROOT="${LOG_ROOT:-/home/alex/Development/logs/groot/evaluation}"
RUN_SUFFIX="${RUN_SUFFIX:-rightonly_1408_stack_0908_2026-09-09T021219+0800}"
ANALYSIS_ID="${ANALYSIS_ID:-$(date -u +%Y%m%d)}"
MODE="${MODE:-both}"
PRECHECK_ONLY="${PRECHECK_ONLY:-0}"
DRY_RUN="${DRY_RUN:-0}"
CHECKPOINT_STEPS="${CHECKPOINT_STEPS:-best}"
EXECUTION_HORIZON="${EXECUTION_HORIZON:-8}"
PAIR_BATCH_SIZE="${PAIR_BATCH_SIZE:-4}"
SHUFFLE_REPEATS="${SHUFFLE_REPEATS:-1}"
BOOTSTRAP_REPLICATES="${BOOTSTRAP_REPLICATES:-10000}"
MAX_EPISODES="${MAX_EPISODES:-0}"
MIN_FREE_GIB="${MIN_FREE_GIB:-5}"
INTERVENTIONS_CSV="${INTERVENTIONS_CSV:-phase_matched,out_of_phase,zero_geometry}"
IFS=',' read -r -a INTERVENTIONS <<< "$INTERVENTIONS_CSV"
[[ ${#INTERVENTIONS[@]} -gt 0 ]] || {
    echo "ERROR: INTERVENTIONS_CSV must select at least one intervention" >&2
    exit 1
}
declare -A SEEN_INTERVENTIONS=()
REQUIRES_CROSS_EPISODE_DONORS=0
for intervention in "${INTERVENTIONS[@]}"; do
    case "$intervention" in
        phase_matched | out_of_phase)
            REQUIRES_CROSS_EPISODE_DONORS=1
            ;;
        offset_10pct | offset_50pct | zero_geometry | zero_image) ;;
        *)
            echo "ERROR: unsupported INTERVENTIONS_CSV entry: $intervention" >&2
            exit 1
            ;;
    esac
    [[ -z "${SEEN_INTERVENTIONS[$intervention]:-}" ]] || {
        echo "ERROR: duplicate INTERVENTIONS_CSV entry: $intervention" >&2
        exit 1
    }
    SEEN_INTERVENTIONS[$intervention]=1
done

RGBD_MODEL="${RGBD_MODEL:-$MODEL_ROOT/c_d1_4ch_early_fusion_patch_tuned_depth_init_rgb_mean_bf16_batch_32_acc_1_30k_${RUN_SUFFIX}}"
NORMALS_MODEL="${NORMALS_MODEL:-$MODEL_ROOT/c_normals_6ch_early_fusion_patch_tuned_normals_init_rgb_mean_bf16_batch_32_acc_1_30k_${RUN_SUFFIX}}"

INTERVENES_ON_RGB=0
case "$MODE" in
    rgb_in_normals)
        MODEL_DIRS=("$NORMALS_MODEL")
        GEOMETRY_KEYS=("ego_view")
        REQUIRED_VIDEO_KEYS=("ego_view" "surface_normals_view")
        LABELS=("rgb_in_rgb_normals")
        INTERVENES_ON_RGB=1
        ;;
    depth)
        MODEL_DIRS=("$RGBD_MODEL")
        GEOMETRY_KEYS=("depth_gray_view")
        REQUIRED_VIDEO_KEYS=("ego_view" "depth_gray_view")
        LABELS=("rgbd")
        ;;
    normals)
        MODEL_DIRS=("$NORMALS_MODEL")
        GEOMETRY_KEYS=("surface_normals_view")
        REQUIRED_VIDEO_KEYS=("ego_view" "surface_normals_view")
        LABELS=("rgb_normals")
        ;;
    both)
        MODEL_DIRS=("$RGBD_MODEL" "$NORMALS_MODEL")
        GEOMETRY_KEYS=("depth_gray_view" "surface_normals_view")
        REQUIRED_VIDEO_KEYS=("ego_view" "depth_gray_view" "surface_normals_view")
        LABELS=("rgbd" "rgb_normals")
        ;;
    *)
        echo "ERROR: MODE must be rgb_in_normals, depth, normals, or both; got $MODE" >&2
        exit 1
        ;;
esac

for intervention in "${INTERVENTIONS[@]}"; do
    if [[ "$INTERVENES_ON_RGB" == "1" && "$intervention" == "zero_geometry" ]]; then
        echo "ERROR: MODE=rgb_in_normals uses zero_image, not zero_geometry" >&2
        exit 1
    fi
    if [[ "$INTERVENES_ON_RGB" != "1" && "$intervention" == "zero_image" ]]; then
        echo "ERROR: zero_image is only valid with MODE=rgb_in_normals" >&2
        exit 1
    fi
done

case "$PRECHECK_ONLY" in
    0 | 1) ;;
    *) echo "ERROR: PRECHECK_ONLY must be 0 or 1" >&2; exit 1 ;;
esac
case "$DRY_RUN" in
    0) ;;
    1)
        MAX_EPISODES="${DRY_MAX_EPISODES:-1}"
        BOOTSTRAP_REPLICATES="${DRY_BOOTSTRAP_REPLICATES:-50}"
        ;;
    *) echo "ERROR: DRY_RUN must be 0 or 1" >&2; exit 1 ;;
esac

mkdir -p "$LOG_ROOT"
LOG_FILE="${LOG_FILE:-$LOG_ROOT/geometry_correspondence_${MODE}_${DATASET_SCOPE}_${ANALYSIS_ID}.log}"
STATUS_FILE="${STATUS_FILE:-$LOG_ROOT/geometry_correspondence_${MODE}_${DATASET_SCOPE}_${ANALYSIS_ID}_status.tsv}"
MPLCONFIGDIR="${MPLCONFIGDIR:-$LOG_ROOT/matplotlib-cache}"
mkdir -p "$MPLCONFIGDIR"

export HF_HOME="/home/alex/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export GROOT_HF_LOCAL_FIRST=1
export GROOT_PATCH_MISTRAL=1
export NO_ALBUMENTATIONS_UPDATE=1
export MPLCONFIGDIR
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN

echo "Geometry correspondence evaluation preflight"
echo "MODE=$MODE"
echo "DATASET_PATH=$DATASET_PATH (held-out $DATASET_SCOPE only)"
echo "CHECKPOINT_STEPS=$CHECKPOINT_STEPS"
echo "INTERVENTIONS=${INTERVENTIONS[*]} (in memory only)"
echo "CROSS_EPISODE_DONORS_REQUIRED=$REQUIRES_CROSS_EPISODE_DONORS"
echo "TEMPORAL_OFFSETS=same episode, discrete circular frame shift"
echo "RAW_DATA_MUTATION=none"

[[ -x "$GROOT_PYTHON" ]] || { echo "ERROR: missing Python: $GROOT_PYTHON" >&2; exit 1; }
[[ -f "$EVALUATOR" ]] || { echo "ERROR: missing evaluator: $EVALUATOR" >&2; exit 1; }
[[ -f "$DATASET_PATH/meta/info.json" ]] || {
    echo "ERROR: held-out dataset is incomplete: $DATASET_PATH" >&2
    exit 1
}
[[ -f "$DATASET_PATH/meta/episodes.jsonl" ]] || {
    echo "ERROR: held-out episodes metadata is missing: $DATASET_PATH" >&2
    exit 1
}
[[ -f "$DATASET_ROOT/provenance/merge_manifest.json" ]] || {
    echo "ERROR: merge provenance is missing: $DATASET_ROOT/provenance/merge_manifest.json" >&2
    exit 1
}

available_kib=$(df -Pk "$MODEL_ROOT" | awk 'NR == 2 {print $4}')
required_kib=$((MIN_FREE_GIB * 1024 * 1024))
if (( available_kib < required_kib )); then
    echo "ERROR: analysis requires ${MIN_FREE_GIB} GiB free; only $((available_kib / 1024 / 1024)) GiB is available" >&2
    exit 1
fi

select_best_checkpoint() {
    local summary=$1
    "$GROOT_PYTHON" - "$summary" <<'PY'
import csv
from pathlib import Path
import sys

path = Path(sys.argv[1])
rows = [
    row for row in csv.DictReader(path.open(newline="", encoding="utf-8"))
    if row["split"] == "validation" and int(row["checkpoint_step"]) > 0
]
if not rows:
    raise SystemExit(f"No finetuned validation rows in {path}")
best = min(rows, key=lambda row: (float(row["mae"]), int(row["checkpoint_step"])))
print(int(best["checkpoint_step"]))
PY
}

validate_checkpoint_artifacts() {
    local checkpoint=$1
    "$GROOT_PYTHON" - "$checkpoint" <<'PY'
import json
from pathlib import Path
import sys

checkpoint = Path(sys.argv[1])
required = (
    "config.json",
    "embodiment_id.json",
    "processor_config.json",
    "statistics.json",
    "model.safetensors.index.json",
)
missing = [
    name for name in required
    if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0
]
if missing:
    raise SystemExit(f"Checkpoint is missing required nonempty artifacts: {missing}")
index = json.loads((checkpoint / "model.safetensors.index.json").read_text(encoding="utf-8"))
shards = sorted(set(index.get("weight_map", {}).values()))
if not shards:
    raise SystemExit("Checkpoint model index has no weight-map shards")
bad_shards = [
    name for name in shards
    if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0
]
if bad_shards:
    raise SystemExit(f"Checkpoint model index references missing/empty shards: {bad_shards}")
print(f"Checkpoint artifacts complete: {checkpoint.name} ({len(shards)} shards)")
PY
}

READY=1
SELECTED_STEPS=()
for i in "${!MODEL_DIRS[@]}"; do
    model_dir="${MODEL_DIRS[$i]}"
    geometry_key="${GEOMETRY_KEYS[$i]}"
    summary="$model_dir/evaluation_exec_hor_8/checkpoint_metric_summary.csv"
    if [[ ! -d "$model_dir" ]]; then
        echo "WAITING: ${LABELS[$i]} model directory is not ready: $model_dir" >&2
        READY=0
        SELECTED_STEPS+=("")
        continue
    fi
    if [[ ! -f "$summary" ]]; then
        echo "WAITING: ${LABELS[$i]} intact validation summary is not ready: $summary" >&2
        READY=0
        SELECTED_STEPS+=("")
        continue
    fi
    if [[ "$CHECKPOINT_STEPS" == "best" ]]; then
        step=$(select_best_checkpoint "$summary")
    else
        step="$CHECKPOINT_STEPS"
    fi
    for selected in $step; do
        [[ -f "$model_dir/checkpoint-$selected/config.json" ]] || {
            echo "ERROR: selected checkpoint is missing: $model_dir/checkpoint-$selected" >&2
            exit 1
        }
        validate_checkpoint_artifacts "$model_dir/checkpoint-$selected"
    done
    SELECTED_STEPS+=("$step")
    echo "READY: ${LABELS[$i]} view=$geometry_key checkpoint(s)=$step"
done

"$GROOT_PYTHON" - "$DATASET_PATH" "$REQUIRES_CROSS_EPISODE_DONORS" "${REQUIRED_VIDEO_KEYS[@]}" <<'PY'
import json
from pathlib import Path
import sys

dataset = Path(sys.argv[1])
requires_cross_episode_donors = bool(int(sys.argv[2]))
geometry_keys = sys.argv[3:]
info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
features = info.get("features", {})
required = ["observation.images.ego_view", *(f"observation.images.{key}" for key in geometry_keys)]
missing = [key for key in required if key not in features]
if missing:
    raise SystemExit(f"Held-out dataset is missing required visual features: {missing}")
episodes = [json.loads(line) for line in (dataset / "meta" / "episodes.jsonl").read_text().splitlines()]
counts = {}
for episode in episodes:
    tasks = tuple(episode.get("tasks", []))
    counts[tasks] = counts.get(tasks, 0) + 1
singletons = {" / ".join(task): count for task, count in counts.items() if count < 2}
if requires_cross_episode_donors and singletons:
    raise SystemExit(f"Same-task donor derangement is impossible for: {singletons}")
if requires_cross_episode_donors:
    print(
        f"Held-out donor pool: {len(episodes)} episodes across {len(counts)} tasks; "
        "no singleton tasks"
    )
else:
    print(
        f"Held-out evaluation pool: {len(episodes)} episodes across {len(counts)} tasks; "
        "cross-episode donors not requested"
    )
PY

if [[ "$PRECHECK_ONLY" == "1" ]]; then
    if [[ "$READY" == "1" ]]; then
        echo "PRECHECK_ONLY complete: all requested models are ready for ${#INTERVENTIONS[@]} intervention(s) each; nothing was launched."
    else
        echo "PRECHECK_ONLY complete: at least one requested model is still pending; nothing was launched."
    fi
    exit 0
fi
if [[ "$READY" != "1" ]]; then
    echo "ERROR: requested trained model/evaluation artifacts are not all ready." >&2
    exit 1
fi

if pgrep -af '[g]r00t.experiment.launch_finetune|[e]valuate_checkpoints|[r]un_gr00t_server|[/]geometry_correspondence_evaluation.py' >/dev/null; then
    echo "ERROR: GR00T training, evaluation, or policy serving is active; refusing GPU contention." >&2
    pgrep -af '[g]r00t.experiment.launch_finetune|[e]valuate_checkpoints|[r]un_gr00t_server|[/]geometry_correspondence_evaluation.py' >&2 || true
    exit 1
fi
if ! gpu_compute_apps=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader,nounits 2>&1); then
    echo "ERROR: nvidia-smi failed; refusing to assume the GPU is available." >&2
    echo "$gpu_compute_apps" >&2
    exit 1
fi
if grep -q '[0-9]' <<< "$gpu_compute_apps"; then
    echo "ERROR: a GPU compute process is active; refusing GPU contention." >&2
    echo "$gpu_compute_apps" >&2
    exit 1
fi

for i in "${!MODEL_DIRS[@]}"; do
    for intervention in "${INTERVENTIONS[@]}"; do
        candidate="${MODEL_DIRS[$i]}/geometry_correspondence_${intervention}_${DATASET_SCOPE}_exec_hor_${EXECUTION_HORIZON}_${ANALYSIS_ID}"
        [[ ! -e "$candidate" ]] || {
            echo "ERROR: output already exists: $candidate" >&2
            exit 1
        }
    done
done

printf 'model\tview_key\tintervention\tcheckpoint_steps\tstatus\texit_code\toutput_dir\n' > "$STATUS_FILE"
FAILURES=()
exec > >(tee -a "$LOG_FILE") 2>&1
cd "$GROOT_DIR"

TOTAL_ANALYSES=$((${#MODEL_DIRS[@]} * ${#INTERVENTIONS[@]}))
analysis_number=0
for i in "${!MODEL_DIRS[@]}"; do
    model_dir="${MODEL_DIRS[$i]}"
    geometry_key="${GEOMETRY_KEYS[$i]}"
    step="${SELECTED_STEPS[$i]}"
    read -r -a checkpoint_args <<< "$step"
    for intervention in "${INTERVENTIONS[@]}"; do
        analysis_number=$((analysis_number + 1))
        output_dir="$model_dir/geometry_correspondence_${intervention}_${DATASET_SCOPE}_exec_hor_${EXECUTION_HORIZON}_${ANALYSIS_ID}"
        echo "============================================================"
        echo "Analysis ${analysis_number}/${TOTAL_ANALYSES}: ${LABELS[$i]} / $intervention"
        echo "Model: $model_dir"
        echo "View key: $geometry_key only"
        echo "Intervention: $intervention"
        echo "Held-out split: $DATASET_PATH"
        echo "Checkpoint(s): $step"
        echo "Output: $output_dir"
        echo "============================================================"
        if CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
            "$GROOT_PYTHON" "$EVALUATOR" \
            --run-dir "$model_dir" \
            --dataset-path "$DATASET_PATH" \
            --output-dir "$output_dir" \
            --view-key "$geometry_key" \
            --intervention "$intervention" \
            --split "$DATASET_SCOPE" \
            --checkpoint-steps "${checkpoint_args[@]}" \
            --embodiment-tag NEW_EMBODIMENT \
            --execution-horizon "$EXECUTION_HORIZON" \
            --pair-batch-size "$PAIR_BATCH_SIZE" \
            --denoising-steps 4 \
            --inference-seed 42 \
            --shuffle-seed 42 \
            --shuffle-repeats "$SHUFFLE_REPEATS" \
            --bootstrap-replicates "$BOOTSTRAP_REPLICATES" \
            --max-episodes "$MAX_EPISODES" \
            --device cuda:0; then
            printf '%s\t%s\t%s\t%s\tPASS\t0\t%s\n' \
                "$model_dir" "$geometry_key" "$intervention" "$step" "$output_dir" >> "$STATUS_FILE"
        else
            exit_code=$?
            printf '%s\t%s\t%s\t%s\tFAIL\t%d\t%s\n' \
                "$model_dir" "$geometry_key" "$intervention" "$step" "$exit_code" "$output_dir" >> "$STATUS_FILE"
            FAILURES+=("${LABELS[$i]}/$intervention (exit $exit_code)")
        fi
    done
done

echo "Status: $STATUS_FILE"
echo "Log: $LOG_FILE"
if [[ ${#FAILURES[@]} -gt 0 ]]; then
    printf 'FAILED: %s\n' "${FAILURES[@]}" >&2
    exit 1
fi
echo "All requested geometry correspondence evaluations completed successfully."
