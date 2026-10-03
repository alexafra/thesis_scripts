# Thesis scripts

## Training evaluation probe: five episodes per goal

The checkpoint evaluator now defaults to `--train-probe-episodes-per-goal 5`.
It samples without replacement with seed 42, uses all available episodes when
a goal has fewer than five, and deduplicates multi-goal episodes. The current
Inspire training view yields 70 episodes / 21,603 frames across 14 goals.
Training data and full held-out validation are unchanged. Existing runner
arguments such as `--train-probe-episodes 3` are only a soft total minimum;
they inherit the new five-per-goal default. Use the per-goal flag with `1`
for legacy behavior, explicit `--train-traj-ids` for fixed IDs, or
`--train-probe-episodes 0` to evaluate the entire training split.

To redo only the training diagnostic, use `--train-probe-only` with
`--train-dataset-path` and a new explicit `--output-dir`, such as
`MODEL/training_probe_5_per_goal_exec_hor_8`. This does not rerun validation
or select a checkpoint. Keep original validation/test outputs and selection
provenance intact. Long training/evaluation jobs must run as independent
system services, not Codex-managed command sessions; do not run probe
evaluation alongside a training or live-policy GPU job.

## Default RGB/geometry modality dropout

The multi-runner now defaults to `MODALITY_DROPOUT=1`: **5% RGB dropout and 5%
geometry dropout overall**, with mutually exclusive masks: 90% retain both
views and neither case drops both. No extra flag is needed. Only selected
geometry experiments receive these flags; RGB-only training and its output
name remain unchanged. Set `MODALITY_DROPOUT=0` explicitly to use the original
no-modality-dropout pipeline.

The option covers RGB + grayscale depth, RGB + turbo depth, and RGB + surface
normals: early channel fusion, pre/post-adapter late fusion, and separate image
views. The separate grayscale recipe is selected with
`EXPERIMENTS=rgbd_gray_separate_views`; existing experiment names are unchanged.

The default state policy is `MODALITY_DROPOUT_STATE_POLICY=independent`: the
existing state-input and state-feature masks remain independent of modality
dropout. At the current 20% probability per state stage, measured state is
absent in approximately 36% of examples. With 10% total visual dropout,
approximately `0.36 * 0.10 = 3.6%` of examples lose state information and one
visual modality together, but never both visual modalities. The existing
state-dropout probabilities and mechanisms are not changed.

The optional `MODALITY_DROPOUT_STATE_POLICY=state_first` samples the existing
two state masks first. Vision can drop only when **both** state
stages retain the measured state. At the current 20% probability per state
stage, measured state is still absent in approximately 36% of examples, and
each visual modality is still dropped in 5% of **all** examples—not 5% of the
eligible subset. The conditional probabilities are `0.05 / 0.64 = 7.8125%`
each among the 64% retaining state. This preserves the state-dropout marginal
rates, rather than disabling state dropout on a separately selected visual
subset.
The total global visual-drop rate must not exceed `(1 - state_dropout_prob)^2`
with `state_first`.

Training with the default gets a `_moddrop05each_independent` (or
`_moddrop05each_state_first`) output/status suffix. It does not reuse or
overwrite old no-dropout output directories. Internal network dropout,
arm/hand normalization, action labels, and dataset conversion stay unchanged.
Masks apply consistently across all frames of a sample and every visual
feature/fusion path. There is no rescaling of the surviving modality. Modality
dropout is disabled during evaluation/deployment, where both inputs remain.
This augmentation does not automatically include the dark recordings in
training, and missing a modality is not equivalent to realistic low lighting.

Example readiness check **without training** (first-time virtual-dataset
creation, if requested separately, still writes its indexed view):

```bash
DATASET_ROOT=/path/to/normal_condition_dataset \
EXPERIMENTS=rgb,normals_late_fusion_pre_adapter,rgbd_gray_separate_views \
PRECHECK_ONLY=1 \
  /home/alex/Development/scripts/multi_finetune_evaluation.sh
```

The generic `examples/finetune.sh` wrapper and Python launcher expose
`--vision-modality-dropout-rgb-prob`,
`--vision-modality-dropout-geometry-prob`, and
`--vision-modality-dropout-state-policy`. Probabilities default to zero and
are saved in the checkpoint config together with the ordered source keys.
These low-level Python/model defaults remain zero for checkpoint compatibility;
the regular multi-runner automatically supplies 0.05/0.05 for geometry runs.
Active dropout rejects RGB-only or ambiguous view/channel layouts.

To re-run evaluation later, use the same state policy,
`RUN_SUFFIX`, and named `EXPERIMENTS` with
`partial_multi_finetune_evaluation.sh`. Named recipes there also default to
the new dropout-suffixed directories; set `MODALITY_DROPOUT=0` to evaluate
older no-dropout models. In that evaluator these switches only
select the matching training output directories; they do **not** enable
dropout during evaluation. RGB paths stay unchanged. The fixed historical
`missed_normals` selection supports only `MODALITY_DROPOUT=0`; choose `normals`
or another named geometry recipe for newer runs. Grayscale separate views are
available there as `rgbd_gray_separate_views` too.

## Separate converted datasets, one training/evaluation view

New LeRobot v2.1 components can stay in their own folders. The opt-in indexed
view gives the existing trainer and evaluator one ordinary logical dataset,
without copying/re-encoding RGB, depth, or normal media. It writes small numeric
Parquet indexes and metadata, and references the original media with symlinks.
**Keep the source folders in place and immutable while a view is in use.**

Create the view only (does not start training):

```bash
cd /home/alex/Development/Isaac-GR00T
.venv/bin/python -m gr00t.data.virtual_dataset \
  --sources /path/to/existing_split_root /path/to/new_split_root \
  --output /path/to/combined_view
```

Each source above contains `train/`, `validation/`, and `test/`. Existing split
membership is retained: train joins train, validation joins validation, and test
joins test. The standalone command also accepts direct dataset roots to form
one direct view; never include held-out data in training unless deliberately
changing that experiment. Source order fixes global episode order. Identical
goal text shares one task ID per split; it does **not** deduplicate episodes.
Do not supply both an older dataset and an appended superset of it.

The existing multi-runner and partial evaluation runner can build/reuse the view
automatically. For example, a preparation/readiness check without training:

```bash
DATASET_SOURCES=/path/to/existing_split_root:/path/to/new_split_root \
VIRTUAL_DATASET_ROOT=/path/to/combined_view \
PRECHECK_ONLY=1 \
  /home/alex/Development/scripts/multi_finetune_evaluation.sh
```

The first invocation writes the view even with `PRECHECK_ONLY=1`; subsequent
invocations reuse it only when its source signature matches. Remove
`PRECHECK_ONLY=1` only when ready to run training. Use the same two variables
with `partial_multi_finetune_evaluation.sh` to evaluate existing checkpoints;
that runner does not train. Do not combine these variables with `DATASET_ROOT`
or explicit split overrides. The shell form uses `:` as the separator; the
standalone command supports paths containing colons.
The runner prepares only train/validation splits by default; `EVALUATION_SPLIT=none`
prepares train alone. Test is neither required nor recreated. The standalone builder
also accepts `--splits train validation`; omitting `--splits` retains its legacy discovery.

Normalization is recomputed over the combined **training** population, using
the unchanged GR00T code, not averages of component percentiles. The existing
relative-arm/absolute-hand configuration and clipping remain unchanged;
relative-arm statistics are generated for the selected action horizon by the
normal launcher. Evaluation uses checkpoint-saved training normalization, not
validation normalization. A single view also preserves the current sharding,
episode boundaries, and evaluation aggregation instead of introducing mixture
weights. Equivalence means the same ordered examples and preprocessing as a
freshly normalized physical append of these same components, with the same
split membership, configuration, seed, and worker settings—not identical
results to a smaller old dataset, nor a guarantee of bit-identical GPU training.

Compatibility checks reject mismatched layouts, encoding/range/calibration
contracts and repeated source roots. Available episode provenance hashes are
also checked across all selected splits to catch duplicate recordings and split leakage.
Without provenance, disjoint recording membership remains the caller's
responsibility; shared goals are fine. New components require a new view output path rather
than overwriting a view used by a running job. Existing source statistics are
not edited or inherited. Old physical-append commands and ordinary
`DATASET_ROOT=/path/to/merged_dataset` usage remain unchanged as the fallback.

## Final train + validation fit, evaluation deferred

`final_train_inspire.sh` is an opt-in final-training wrapper. It first builds the
combined train/validation view (including cross-split provenance checks),
then creates a fresh training view containing **all train followed by all validation
episodes**. Its normalization is fitted to that full training population. It neither
requires nor builds a test view, and it does not run evaluation after training.
Validation is now training data, so evaluating on it would not be held-out validation.
No media is copied or re-encoded, and all source folders must remain available.

Prepare the views without starting training or evaluation:

```bash
FINAL_DATASET_SOURCES=/path/to/existing_split_root:/path/to/new_split_root \
FINAL_VIEW_ROOT=/path/to/new_final_view \
  bash /home/alex/Development/scripts/final_train_inspire.sh prepare
```

Then choose the training budget explicitly. `check` runs the existing readiness
checks; replace it with `run` to train RGB followed by turbo-depth pre-adapter late
fusion without running evaluation:

```bash
FINAL_DATASET_SOURCES=/path/to/existing_split_root:/path/to/new_split_root \
FINAL_VIEW_ROOT=/path/to/new_final_view \
MAX_STEPS=35000 \
  bash /home/alex/Development/scripts/final_train_inspire.sh check
```

The wrapper defaults to `EXPERIMENTS=rgb,rgbd_turbo_late_fusion_pre_adapter`.
Geometry modality dropout retains the standard default; RGB remains unchanged.
`MAX_STEPS` must be explicitly chosen for `check`/`run`, and divisible by
`SAVE_STEPS` (default 5000). In the general multi-runner, `MAX_STEPS` defaults to
25000 and `RUN_LABEL` is derived automatically (for example `30k` or `35k`).

The wrapper enforces `EVALUATION_SPLIT=none` and records evaluation as skipped in
the existing status table. Ordinary validation runs require train and validation
datasets only; a test dataset is not a readiness prerequisite.

Test evaluation remains deferred. A separate, deliberate general multi-runner
invocation must set both `EVALUATION_SPLIT=test` and `ALLOW_TEST_EVALUATION=1` before
it can touch a test dataset. That mode selects **only the predetermined final
checkpoint**, not the best checkpoint on test. Results go to
`test_evaluation_exec_hor_8` and `test_normalized_action_metrics_exec_hor_8`.
The existing evaluator still labels its held-out CSV/plot series `validation`
internally; in these specifically named test output folders, that series is the
test set, not the old validation episodes. `evaluation_protocol.json` at the model
root records the true split, training/test paths, fixed checkpoint, and the
prohibition on selecting checkpoints using test. Do not use test results to retune or select checkpoints and
then claim the same data as an untouched final test.

Ordinary `multi_finetune_evaluation.sh` usage retains `EVALUATION_SPLIT=validation`,
its original output directory names, and all-checkpoint validation comparison.
Its training probe uses five episodes per goal.
The physical-merge pipeline remains available and unchanged.

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

`check` remains read-only. A collection `convert` or `all` command must also
carry `--allow-full-reconversion`. This is an intentional guard: once a
converted corpus exists, adding another processed-raw leaf must use an append
pipeline and must not reconvert the old episodes. The authorization flag is for
an initial collection build or a deliberate from-scratch rebuild only.

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

## Incremental Inspire append (452 + both 2026-09-15 components)

`append_inspire_stack_0915.sh` is the dedicated incremental path for
adding 145 `stack_red_cups_09_15` episodes and 58 literal
`woorden_block_09_15` episodes to the migrated 452-episode corpus. It
goal-stratifies each source independently with seed 42, preserves stack-then-
woorden ordering within every split, and targets the exact sibling
`all_tasks_655eps_20260916_normals_range_mask_v2`.

The dataset-only production command is explicit and cannot start training:

```bash
/home/alex/Development/scripts/append_inspire_stack_0915.sh build
```

For a detached user service, use this exact dataset-only launch:

```bash
systemd-run --user \
  --unit=inspire-655-v2-append-20260916 \
  --collect \
  /usr/bin/bash \
  /home/alex/Development/scripts/append_inspire_stack_0915.sh build
```

There is no implicit mode and the wrapper has no training mode. `check`
performs read-only preflight; `build` stops after dataset publication. Training
must be launched separately only after a later explicit authorization.

The append is transactional and retains resumable hidden checkpoints. It makes
a zero-block hard-link view of both raw sources, composes one disposable raw
split, converts only the new 203-episode component, and uses the existing
`append_lerobot2.py` compatibility checks for features, 26D modality layout,
and exact camera-calibration equality. To bound peak disk use on ext4, the
existing base and immutable incoming media are hard-linked only into a hidden
build. After that build is sealed, the adopted component is retired to free
space; every surviving base hard link is then detached, the unchanged input
digests and zero-shared-base-inode condition are verified, and only then is the
build atomically renamed without replacement. The original base and both
processed-raw sources remain untouched.

If interruption occurs after the validated build has moved into the checkpoint
but before its phase record advances, the next `build` invocation revalidates
the component, final population/provenance, statistics, and source snapshots,
then rolls the checkpoint forward without deleting the build. An external
unsealed `.build-*` scratch remains deliberately fail-closed for manual
inspection because it can represent several earlier incomplete phases.

Before detachment and publication, the wrapper deletes the merged train
statistics cache and runs the same `python -m gr00t.data.stats` finalization
used by the training launcher against the hidden final `train/` split only. It
verifies fresh schema/config fingerprints, finite 26D state/action statistics,
32x7 relative-arm statistics, q01/q99 bounds, frame/trajectory count sources,
and byte-identical validation/test statistics. This is dataset finalization;
it does not load a model or start training.

Expected final populations are 525/65/65 episodes and
196,453/22,719/24,566 frames for train/validation/test. After the dataset build
publishes successfully, use the existing multi-runner rather than creating a
dataset-specific training wrapper:

```bash
DATASET_ROOT=/home/alex/Development/Datasets/lerobot2/inspire/all_tasks_655eps_20260916_normals_range_mask_v2 \
RUN_SUFFIX=all_tasks_655eps_20260916_normals_range_mask_v2 \
EXPERIMENTS=normals \
bash /home/alex/Development/scripts/multi_finetune_evaluation.sh
```

Use the same dataset/model variables with
`partial_multi_finetune_evaluation.sh` when only evaluation must be resumed.

`multi_finetune_evaluation.sh` runs each selected model as a complete
train-then-validation-evaluate stage. Its default order is RGB, surface
normals, then gray depth so the depth stage can be stopped without affecting
the completed RGB and normals results. Override the ordered subset with, for
example, `EXPERIMENTS=rgb,normals`. Its Dex3 default prefers the canonical
`lerobot2/dex3/` dataset when that dataset exists, otherwise it retains the
legacy flat `lerobot2/` fallback during migration. An explicit `DATASET_ROOT`
always takes precedence.

The same launcher also has four opt-in Inspire late-fusion experiments:
`rgbd_late_fusion_pre_adapter`, `rgbd_late_fusion_post_adapter`,
`normals_late_fusion_pre_adapter`, and `normals_late_fusion_post_adapter`.
They use the corresponding GR00T modality configs, keep the vision patch
embedding frozen, and train the four 50/50-initialized linear fusion adapters.
They are never included by the default `rgb,normals,depth` selection. The same
names can be passed to `partial_multi_finetune_evaluation.sh` to evaluate a
completed late-fusion training run without another dedicated runner.

Two additional opt-in frozen-patch, ordinary separate-view recipes are
`rgbd_turbo_separate_views` and `normals_separate_views`. The former applies
the pinned fixed-Turbo lookup to `depth_gray_view` in memory; neither recipe
uses early-channel or late-adapter fusion. Both names are also accepted by the
partial evaluator.

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
