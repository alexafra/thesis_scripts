# Evaluator improvements

## Verdict

The existing evaluator is reasonable for offline screening and model ranking. It
evaluates the same complete held-out trajectories for every model and uses a
fixed inference seed. It is not a replacement for robot task-success testing,
and small MAE/MSE differences should not be treated as definitive.

## Current evaluation

- 42 complete validation episodes across eight tasks.
- 11,189 frames and 28 action joints.
- Execution horizon 8 and four denoising steps.
- Fixed inference seed 42 for paired model comparisons.
- Unnormalised physical action-command MAE and MSE.
- All checkpoint errors are pooled across frames and joints.
- A task-balanced train probe is requested as three episodes, but expands to
  eight episodes because the selector includes one episode per task.

## Important limitations

### Teacher-forced rather than closed-loop

The model receives the recorded demonstration observation again every eight
frames. Its predictions do not change the subsequent state. The test therefore
measures conditional imitation accuracy, but cannot measure compounding errors,
recovery, contact, grasp success, object movement, closed-loop stability, or the
complete RTC deployment behaviour.

### Correlated observations

The headline calculation contains 11,189 x 28 = 313,292 scalar residuals, but
frames and joints are highly correlated. The defensible statistical unit is
closer to 42 paired episodes, with only eight task types. Do not use a naive
frame-level significance test.

### Unequal task weighting

The pooled metric gives longer and more frequent tasks more weight. For example,
put-down toothpaste contributes 2,618 validation frames while put-down cereal
contributes 472. If all tasks are equally important, equal-task MAE should be the
primary score. Pooled MAE remains useful when the dataset frequency represents
the intended deployment frequency.

### Capture-session leakage

The dataset was goal-stratified episode-by-episode rather than grouped by
recording session. Validation and test captures are temporally adjacent to
training captures and often share the same continuous scene. No exact episode
duplication was found, but this makes generalisation estimates optimistic and
may particularly favour RGB scene cues. Relative comparisons remain controlled,
but a robust future benchmark should hold out complete recording sessions or
time blocks and then retrain on the revised split.

### One inference seed and one training replicate

The fixed seed is good for common-random-number comparisons, but it measures only
one flow-sampling sequence. A single training run also does not measure
run-to-run optimisation variance. Use extra inference seeds only for finalists,
and retain at least one repeated anchor training recipe when feasible.

## Recommended metrics

These can be calculated from the existing saved CSVs without GPU inference:

1. Pooled MAE: the existing headline score.
2. Equal-episode MAE.
3. Equal-task MAE: calculate each task's MAE, then average the eight tasks
   equally. Use this as the primary metric when tasks are equally important.
4. Per-task MAE.
5. Per-arm and per-hand MAE, including the existing right-side view.
6. Median and p95 absolute error; keep MSE/RMSE as secondary outlier-sensitive
   metrics.
7. Paired episode-cluster bootstrap 95% intervals and episode win counts.
8. Error by offset within the eight-step execution chunk to reveal horizon
   degradation.
9. Active-motion versus low-motion error, so long stationary portions cannot
   conceal errors during manipulation.
10. Predicted velocity and chunk-boundary discontinuity for finalist safety
    checks.

Treat a small difference as a tie when its paired episode interval overlaps zero.
As a practical heuristic, differences below roughly 3% should not drive a costly
decision unless they are consistent across tasks, checkpoints, and a repeated
training run.

## Compute-efficient evaluation plan

### Every candidate model

- Run the full 42-episode validation evaluation at checkpoints 16k and 20k.
- Skip early full evaluations, or evaluate 4k/8k/12k on one fixed validation
  episode per task.
- Do not reevaluate the same base model for every run.
- Run the train probe only at the final checkpoint unless diagnosing overfit.

### Finalists

- Take the top two models only.
- Evaluate 3-5 common inference seeds on a fixed eight-episode panel, one episode
  per task.
- Evaluate the selected final recipe once on the untouched test split.
- Use standardised robot task-success trials as the final metric.

This reduces full evaluation work by roughly two-thirds while yielding stronger
comparisons.

## Exact performance improvements

The current implementation has three major efficiency problems:

1. Each episode is fully decoded twice per checkpoint: once in the outer
   evaluator and again inside `evaluate_single_trajectory`.
2. Every video frame is decoded even though only frames 0, 8, 16, ... are used as
   model observations.
3. Inference runs at batch size one; there is no evaluation batch-size flag.

Recommended implementation order:

1. Eliminate the duplicate episode load/decode. This is the easiest exact win.
2. Decode only observation frames required by the execution horizon while still
   loading all action targets from parquet.
3. Batch independent offline observations for GPU inference while preserving a
   deterministic noise assignment across models.

These changes should preserve the metric definition. Verify equality against a
small reference evaluation before adopting them.

## Practical interpretation of the recent results

- Colour patch-tuned and four-channel depth at 20k are effectively tied on MAE.
- Most observed 1-4% differences do not clearly separate under paired
  episode-level resampling.
- None of the recent 20k MSE differences clearly separates after accounting for
  episode clustering.
- Some larger historical differences do survive the episode-level check, so the
  evaluator contains useful signal.
- Rankings vary by task and by pooled/equal-episode/equal-task weighting.

Use this evaluator to eliminate clearly weaker models and identify finalists.
Do not interpret a narrow offline win as proof of superior robot performance.
