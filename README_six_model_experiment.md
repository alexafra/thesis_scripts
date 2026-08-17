# Six-Model Experiment

**Design status:** locked on 2026-08-16.

This is the final design for the main modality experiment. It is separate from partial training runs used to explore recipes or prepare individual models.

## Research question

Compare RGB-only control policies with policies that add either grayscale depth or surface normals, using two practical integration strategies:

1. **Separate streams with a frozen vision patch embedding.** The added three-channel image stream reuses the pretrained, frozen vision front end. This tests an easy-to-integrate, "quasi-zero-shot" geometry representation. More precisely, it is **frozen vision-front-end modality integration** because the downstream policy is still fine-tuned.
2. **Early fusion with a tuned vision patch embedding.** RGB and geometry channels are concatenated before patch embedding. The expanded input layer must be tuned because its added channels do not have useful pretrained weights.

## Locked model matrix

| # | Input | Integration | Vision patch embedding |
|---|---|---|---|
| 1 | RGB | Single RGB stream | Frozen |
| 2 | RGB + grayscale depth | Two separate three-channel streams | Frozen |
| 3 | RGB + surface normals | Two separate three-channel streams | Frozen |
| 4 | RGB | Single RGB stream | Tuned |
| 5 | RGB + grayscale depth | One four-channel early-fusion stream | Tuned |
| 6 | RGB + surface normals | One six-channel early-fusion stream | Tuned |

## Settings held constant

- BF16 backbone storage for all six models.
- Microbatch 32, gradient accumulation 1, effective batch 32.
- AdaFactor with learning rate `1e-4`.
- Cosine learning-rate schedule with 5% warmup.
- Seed 42.
- Same training data, step budget, checkpoint schedule, augmentations, evaluation data, and evaluation procedure.
- No gradient checkpointing.

## Intended comparisons

Within the frozen-patch cohort:

- Model 2 versus model 1 measures the value of separate-stream depth.
- Model 3 versus model 1 measures the value of separate-stream normals.

Within the tuned-patch cohort:

- Model 5 versus model 4 measures the value of early-fusion depth.
- Model 6 versus model 4 measures the value of early-fusion normals.

For comparisons between integration strategies, compare each multimodal model's improvement relative to its matching RGB baseline. Raw comparisons between separate-stream and early-fusion models are comparisons between complete practical strategies, not a pure causal ablation of fusion location: patch-embedding trainability necessarily differs between the two strategies.

## Outside the locked experiment

FP32 controls, repeatability runs, optimizer searches, different microbatch/accumulation combinations, and tuned-patch separate-stream models are recipe exploration. They are not part of the locked six-model experiment unless this document is deliberately revised.

The partial run on 2026-08-16 is preparation and recipe exploration; it is not the execution of all six locked models.
