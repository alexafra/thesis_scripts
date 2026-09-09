# Thesis scripts

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

For every recipient episode in the held-out validation split, it selects a
different donor episode with the exact same task. RGB, robot state, language,
expert targets, execution horizon, and diffusion noise stay fixed. Only the depth
or surface-normal frame is replaced in memory, using normalized episode progress
to choose the donor frame. No training split or dataset file is changed.

Run the non-GPU readiness check:

```bash
MODE=both PRECHECK_ONLY=1 ./multi_geometry_correspondence_evaluation.sh
```

Once both standard model evaluations are complete and the GPU is free, omit
`PRECHECK_ONLY` to evaluate the best intact-validation-MAE checkpoint for each
model. `MODE=depth` and `MODE=normals` run only one intervention. The primary
output is task-balanced paired degradation (`shuffled - intact`); positive values
mean correctly aligned geometry improved open-loop command prediction. The tool
also reports ordinary frame-weighted errors, per-task results (including stack
three cups), per-joint and horizon results, prediction sensitivity, paired
episode-bootstrap intervals, and the exact donor/frame mapping.
