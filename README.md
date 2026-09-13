# Thesis scripts

## General Inspire dataset pipeline

`prepare_inspire_lerobot2.sh` is the reusable coordinator for one complete,
already-curated `processed_raw` dataset. It stages every direct `episode_*`,
creates deterministic train/validation/test splits with `split_dataset.py`, and
uses `convert_to_lerobot2.sh` for the full RGB, gray-depth, lossless-depth, and
surface-normal LeRobot v2.1 conversion. It never changes the raw source and has
no episode filtering or exclusion behavior.

No mode is implicit. Preflight a raw dataset without writing anything:

```bash
./prepare_inspire_lerobot2.sh check \
  --source /path/to/processed_raw/task
```

Convert and publish one dataset:

```bash
./prepare_inspire_lerobot2.sh convert \
  --source /path/to/processed_raw/task \
  --output /path/to/lerobot2/task \
  --repo-id task \
  --split-strategy goal-stratified \
  --split-seed 42
```

The current 137-episode red-cup source therefore targets 109 train, 14
validation, and 14 test episodes. The pipeline verifies the exact raw episode
names and `data.json` hashes before publishing, as well as the 26D Inspire and
RGB/depth/normals contract in all three converted splits.

Check training readiness or deliberately start the three-model training and
validation run:

```bash
./prepare_inspire_lerobot2.sh training-check --output /path/to/lerobot2/task
./prepare_inspire_lerobot2.sh train --output /path/to/lerobot2/task
```

`multi_finetune_evaluation.sh` runs each selected model as a complete
train-then-validation-evaluate stage. Its default order is RGB, surface
normals, then gray depth so the depth stage can be stopped without affecting
the completed RGB and normals results. Override the ordered subset with, for
example, `EXPERIMENTS=rgb,normals`.

Passing the original `--source` to either command additionally rechecks exact
raw-to-split provenance. `all` is the only mode that performs conversion and
then starts training, and it must be selected explicitly. Every option also has
an uppercase environment equivalent, such as `SOURCE_ROOT`, `DATASET_ROOT`,
`REPO_ID`, `SPLIT_STRATEGY`, and `SPLIT_SEED`.

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
