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
