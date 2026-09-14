# Thesis scripts

## General Inspire dataset pipeline

`prepare_inspire_lerobot2.sh` is the reusable coordinator for either one complete,
already-curated `processed_raw` dataset or a parent containing several curated
dataset leaves. It creates deterministic train/validation/test splits with
`split_dataset.py`, then uses `convert_to_lerobot2.sh` once for the full RGB,
gray-depth, lossless-depth, and surface-normal LeRobot v2.1 conversion. It never
changes the processed-raw source. Single-dataset mode includes every direct
`episode_*`; collection mode can exclude only explicitly named child datasets.

The canonical dataset hierarchy is:

```text
/home/alex/Development/Datasets/
├── raw/{dex3,inspire}/<dataset>/
├── processed_raw/{dex3,inspire}/<dataset>/
└── lerobot2/{dex3,inspire}/<dataset>/
```

The Inspire coordinator can resolve its two working paths from one safe leaf
name. It starts at `processed_raw`; the matching `raw` directory is never read,
copied, or modified:

```bash
./prepare_inspire_lerobot2.sh check --dataset-name pick_place_red_cup_08_13
./prepare_inspire_lerobot2.sh convert --dataset-name pick_place_red_cup_08_13
```

Set `DATASETS_ROOT` or pass `--datasets-root` when the three stage directories
live somewhere else. Explicit `--source` and `--output` paths remain supported
for staging and compatibility.

No mode is implicit. Preflight a processed-raw dataset without writing anything:

```bash
./prepare_inspire_lerobot2.sh check \
  --source /path/to/processed_raw/inspire/task
```

Convert and publish one dataset:

```bash
./prepare_inspire_lerobot2.sh convert \
  --source /path/to/processed_raw/inspire/task \
  --output /path/to/lerobot2/inspire/task \
  --repo-id task \
  --split-strategy goal-stratified \
  --split-seed 42
```

Compose several leaves directly into one final LeRobot dataset with no retained
per-leaf converted intermediates:

```bash
./prepare_inspire_lerobot2.sh check \
  --collection-source /home/alex/Development/Datasets/processed_raw/inspire \
  --output /home/alex/Development/Datasets/lerobot2/inspire/all_tasks_452eps_20260915 \
  --exclude-dataset data_colour_only_test \
  --preserve-split pick_place_red_cup_08_13=/home/alex/Development/Datasets/lerobot2/inspire/pick_place_red_cup_08_13/split_manifest.json \
  --split-strategy goal-stratified \
  --split-seed 42
```

`--preserve-split` preserves that leaf's existing membership and within-split
order while refreshing its current goal, frame-count, and `data.json` hash
provenance. Every unpreserved leaf is split independently with the requested
strategy and seed before the component splits are appended train-to-train,
validation-to-validation, and test-to-test. The current six-leaf Inspire plan is
452 episodes: 364 train, 44 validation, and 44 test. The 137-episode red-cup
assignments remain 109/14/14; `data_colour_only_test` is excluded exactly.

For the deliberately authorized conversion-plus-training run, change `check` to
`all` and add `--run-suffix all_tasks_452eps_20260915`. The three 25,000-step
stages then run sequentially as RGB train/evaluate, surface-normal
train/evaluate, and gray-depth train/evaluate, saving at steps 5,000, 10,000,
15,000, 20,000, and 25,000. Use `convert` instead of `all` when training should
not start.

Collection mode rejects symlinks, nested dataset wrappers, duplicate episode
content/capture identifiers, an existing output or staging directory, and an
active `data_editor_EN_rgbd.py`. It copies rather than mutates sources, verifies
the source hash snapshot after staging, preserves the optional root curation
manifest under `provenance/`, and publishes only after the converted population
and modality contracts validate. Every preserved split manifest is also copied
under `provenance/` and hash-validated, so later validation does not depend on
the older per-task LeRobot directory remaining present.

The current 137-episode red-cup source therefore targets 109 train, 14
validation, and 14 test episodes. The pipeline verifies the exact raw episode
names and `data.json` hashes before publishing, as well as the 26D Inspire and
RGB/depth/normals contract in all three converted splits.

The production gray-depth model view remains on its fixed linear 0.25--1.0 m
contract. Choose different bounds only for a new, consistently converted
lineage; the environment values are forwarded through split conversion:

```bash
DEPTH_NEAR_M=0.3 DEPTH_FAR_M=3.0 \
  ./prepare_inspire_lerobot2.sh convert \
  --source /path/to/processed_raw/inspire/task \
  --output /path/to/lerobot2/inspire/task
```

`convert_to_lerobot2.sh` validates any recorded
`info.depth.scale_m_per_unit` by IEEE-754 float32 equality with 0.001 and writes
the exact JSON `0.001` into converted sidecar metadata without modifying the
source. This accepts the RealSense spelling `0.0010000000474974513` but rejects
a physically different scale.

New recordings carry `info.depth.calibration`; the converter uses that recorded
calibration automatically for surface-normal geometry. The converted
`meta/info.json` retains the full payload as `camera_calibration`, while
`surface_normals_encoding.camera_calibration` carries its compact identity for
checkpoint/live checks. The general `convert_to_lerobot2.sh` command remains
`legacy-untagged` by default for old Dex3/unknown-camera data. The Inspire
coordinator defaults legacy source episodes to the known replacement D435i,
serial `254322071415`, matching the Inspire collection lineage. Override it
explicitly only when preserving a genuinely untagged legacy lineage:

```bash
./prepare_inspire_lerobot2.sh convert \
  --camera-calibration-profile legacy-untagged \
  --source /path/to/processed_raw/inspire/legacy_task \
  --output /path/to/lerobot2/inspire/legacy_task
```

Append accepts legacy-with-legacy datasets and requires matching calibration
provenance for calibrated datasets. It rejects known-versus-unknown or differing
camera calibrations so an append cannot silently assign calibration to old data.

Check training readiness or deliberately start the three-model training and
validation run:

```bash
./prepare_inspire_lerobot2.sh training-check --dataset-name task
./prepare_inspire_lerobot2.sh train --dataset-name task
```

`multi_finetune_evaluation.sh` runs each selected model as a complete
train-then-validation-evaluate stage. Its default order is RGB, surface
normals, then gray depth so the depth stage can be stopped without affecting
the completed RGB and normals results. Override the ordered subset with, for
example, `EXPERIMENTS=rgb,normals`. Its Dex3 default prefers the canonical
`lerobot2/dex3/` dataset when that dataset exists, otherwise it retains the
legacy flat `lerobot2/` fallback during migration. An explicit `DATASET_ROOT`
always takes precedence.

Passing the original `--source` to either command additionally rechecks exact
raw-to-split provenance. `all` is the only mode that performs conversion and
then starts training, and it must be selected explicitly. Every option also has
an uppercase environment equivalent, such as `SOURCE_ROOT`, `DATASET_ROOT`,
`DATASETS_ROOT`, `DATASET_NAME`, `REPO_ID`, `SPLIT_STRATEGY`, and `SPLIT_SEED`.

There is no merge step when converting a single processed-raw population. To
append an independently converted Inspire FTP component to an existing LeRobot
root, use `append_lerobot2.py`; that transaction updates train,
validation, and test together. It remains separate from conversion/training so
an append cannot start a training run implicitly.

## Geometry-only GR00T ablations

`geometry_only_finetune_evaluation.sh` trains and evaluates one visual ablation at a
time on the merged right-only/stack dataset. It does not queue or launch itself.

Run a non-training preflight:

```bash
VISUAL_MODE=depth PRECHECK_ONLY=1 ./geometry_only_finetune_evaluation.sh
VISUAL_MODE=normals PRECHECK_ONLY=1 ./geometry_only_finetune_evaluation.sh
```

When the GPU and sufficient disk space are available, omit `PRECHECK_ONLY` to run
either the depth-only or surface-normals-only 30k experiment. Both modes retain
language and robot state, but load exactly one three-channel geometry view and no
RGB camera view. Model and Hugging Face backbone loading is local/offline only.

These configurations are supported by offline training and evaluation. The live
Unitree deployment adapter needs a separate contract change before either model is
used on the robot.

## Geometry correspondence evaluation

`multi_geometry_correspondence_evaluation.sh` measures whether the trained RGB-D
and RGB+surface-normal policies benefit from correctly aligned geometry. It is an
evaluation-only tool and never queues itself.

For every recipient episode in the held-out validation split, it runs three
geometry interventions for both RGB-D and RGB+surface-normals: a different
same-task donor at matched normalized progress, the same kind of donor shifted by
half an episode, and an all-zero geometry frame. RGB, robot state, language,
expert targets, execution horizon, and diffusion noise stay fixed. Replacement
happens only in memory; no training split or dataset file is changed.

Run the non-GPU readiness check:

```bash
MODE=both PRECHECK_ONLY=1 ./multi_geometry_correspondence_evaluation.sh
```

Once both standard model evaluations are complete and the GPU is free, omit
`PRECHECK_ONLY` to evaluate the best intact-validation-MAE checkpoint for each
model. `MODE=depth` and `MODE=normals` restrict the three interventions to one
model. Each intervention has its own paired intact control. The primary output is
task-balanced paired degradation (`counterfactual - intact`); positive values
mean the intact geometry improved open-loop command prediction. The tool also
reports ordinary frame-weighted errors, per-task results (including stack three
cups), per-joint and horizon results, prediction sensitivity, paired
episode-bootstrap intervals, and the exact donor/frame mapping. All-zero geometry
is intentionally a strong out-of-distribution ablation, while the two donor tests
preserve more of the held-out geometry distribution.
