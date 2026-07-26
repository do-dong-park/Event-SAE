# OpenVLA pipeline

Closed-loop interpretability pipeline for the openVLA backbone on the
LIBERO simulation suites.

## Pipeline sanity check (5-trial scale)

This is the public minimal sweep (LIBERO-Spatial, top-5 features
per ranking, α = 0, 5 trials per task, seed 0), not the paper's
full-scale experiment. It is a one-seed diagnostic reproduction:
the ranking order is recovered in this small configuration, but
paper-scale replication and generalization are
**confounded — 판정 보류**.

| Configuration                  | SAE       | Feature lists  | Baseline SR | Δ event-aligned | Δ window-mean | Δ task-mean | Δ random-alive |
|---|---|---|---:|---:|---:|---:|---:|
| Reference                      | original  | original       | 80.0%       | −28.4           | −7.2          | −8.0        | −6.8           |
| This codebase, both reused     | original  | original       | 82.0%       | −26.8           | −9.2          | −9.6        | −9.2           |
| This codebase, SAE only reused | original  | this codebase  | 80.0%       | −25.2           | −6.4          | −8.0        | −8.8           |

ΔSR is in percentage points relative to each row's own baseline run.

In the third row, only the SAE is held fixed, and every other step
runs through this repository. Each condition has 5 trials/task × 10
tasks = 50 suite episodes. The paper's activation collection uses 50
rollouts/task and its Hooked SR protocol uses 10 rollouts/task; the exact
per-feature rollout count behind Table 3 is not stated. This table has no
confidence interval, so differences must not be attributed to sampling or
runtime versions without a multi-seed comparison.

To make a result row reproducible, retain the Event-SAE commit and
dirty diff, external dependency commits, Hugging Face revision,
`config.json` + `ae.pt` hashes, `candidates.jsonl`, the task-level
`events.csv`, `stdout.log` with episode outcomes, seed, and the exact
aggregation command. The summary table alone is not a provenance artifact.

## Installation

### Step 1: Conda environment

```bash
conda env create -f environment-openvla.yml
conda activate event-sae-openvla
```

Optional flash-attn for faster inference (openVLA falls back to `sdpa`
if skipped):

```bash
pip install flash-attn==2.5.5 --no-build-isolation
```

### Step 2: External libraries

Three libraries installed editable into the conda env. Clone under
`external/` at the repo root (already gitignored):

```bash
export EVENT_SAE_ROOT=/path/to/event-sae   # adjust to this checkout
mkdir -p "$EVENT_SAE_ROOT/external"
cd "$EVENT_SAE_ROOT/external"
```

**(a) LIBERO** — sim benchmark:

```bash
git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git
cd LIBERO
git checkout 8f1084e3132a39270c3a13ebe37270a43ece2a01
touch libero/__init__.py libero/lifelong/models/modules/__init__.py   # missing in upstream
pip install -e .
pip install robosuite==1.4.0 bddl==1.0.1 robomimic==0.2.0 mujoco \
            gym==0.25.2 easydict==1.9 cloudpickle==2.1.0 future
cd ..
```

**(b) `dictionary_learning`** — SAE training library:

```bash
git clone https://github.com/saprmarks/dictionary_learning.git
git -C dictionary_learning checkout 60ec6bf5264944d64a4ca271f45a29ebfb9d4946
pip install -e ./dictionary_learning
```

The full commit above is the tested dependency revision.

**(c) AWE** — kinematic keyframe extraction (fork with packaging
fixes; see the fork's NOTICE):

```bash
git clone https://github.com/xc-j/awe.git
git -C awe checkout 7197bb86a20784666dabed90e6eabcf8bb1e9912
pip install -e ./awe
```

Return to the Event-SAE root before running the pipeline:

```bash
cd "$EVENT_SAE_ROOT"
```

## Usage

The pipeline has four stages: **(1)** SAE training, **(2)** kinematic
keyframe extraction, **(3)** event clustering with VLM annotation,
**(4)** closed-loop intervention. Between (3) and (4) a feature
ranking step picks candidate features to intervene on.

Every command below chains off a single rollout. Step (a) creates a
timestamped run directory under `logs/openvla/`; later commands
reference it via `$EVAL_RUN` (the export is shown after step (a)).
Later commands also reference `$SAE_CKPT` — set it in step (b) to
either a freshly-trained checkpoint or a pre-trained one.

## Phase 1 — SAE training

Roll out openVLA on LIBERO, save the residual-stream activations, then
train a BatchTopK SAE on them.

### (a) Collect openVLA activations during a LIBERO rollout

Edit `configs/examples/openvla/collect_libero_spatial.yaml` (or copy
and adapt). Requires GPU and LIBERO assets (`LIBERO_CONFIG_PATH`).

```bash
python scripts/openvla/collect_activations.py \
    --config configs/examples/openvla/collect_libero_spatial.yaml
```

This creates a timestamped run directory. Export its name so later
steps can derive their paths:

```bash
export EVAL_RUN=EVAL-libero_spatial-openvla-DATE_TIME   # replace DATE_TIME
```

Outputs under `logs/openvla/$EVAL_RUN/sae_activations/`:
- dense `.pt` shards — input to step (b)
- `activation_index.jsonl` — input to step (h)

### (b) Train an SAE on collected shards

Edit `configs/examples/openvla/train_sae_layer31.yaml` and point
`data_dir` at the shard directory from step (a). Set `wandb_project`
to log to wandb, or leave empty to disable.

```bash
python scripts/train_sae.py \
    --config configs/examples/openvla/train_sae_layer31.yaml \
    --save-dir logs/openvla/sae/libero_spatial_layer31
```

The checked-in example uses the paper's OpenVLA settings
(`lr=5e-5`, 4,000 steps, batch size 40,000). Output: `ae.pt` +
`config.json` under
`logs/openvla/sae/libero_spatial_layer31/trainer_0/`.

Or skip step (b) and use the paper's four pre-trained SAEs (one per
LIBERO suite, each at openVLA layer 31, BatchTopK k=64) from the
[Hugging Face Hub](https://huggingface.co/mr-cabbage/event-sae-openvla-libero).
Set `$SAE_CKPT` to either the HF download or the local training
output:

```bash
# Pretrained, e.g. LIBERO-Spatial:
OPENVLA_SAE_REV=d77aec094f2799e11b352630803601f9d2db5956
OPENVLA_SAE_DIR="$EVENT_SAE_ROOT/external/checkpoints/openvla-$OPENVLA_SAE_REV"
hf download mr-cabbage/event-sae-openvla-libero \
    libero_spatial/ae.pt libero_spatial/config.json \
    --revision "$OPENVLA_SAE_REV" \
    --local-dir "$OPENVLA_SAE_DIR"
SAE_CKPT="$OPENVLA_SAE_DIR/libero_spatial/ae.pt"

# Or locally trained:
SAE_CKPT=logs/openvla/sae/libero_spatial_layer31/trainer_0/ae.pt
```

`ae.pt` and its sibling `config.json` are one checkpoint contract;
download both from the same immutable revision. All subsequent
commands in this doc reference `--sae-checkpoint $SAE_CKPT`.

## Phase 2 — Kinematic keyframe extraction

Pick a small number of waypoints per episode from the end-effector
trajectory. These waypoints anchor the events used in Phase 3 and are
independent of the SAE.

### (c) Extract AWE kinematic keyframes from rollout trajectories

CPU-only. Defaults (`pos_only`, error budget η = 0.05) are baked
into the CLI.

```bash
python scripts/extract_keyframes.py \
    --trajectory-records-path logs/openvla/$EVAL_RUN/trajectory_records.jsonl
```

Output: `waypoint_summary.json` under
`logs/openvla/keyframes/$EVAL_RUN/dp_pos_only_err0p05/`.

## Phase 3 — Event clustering with VLM annotation

Group waypoint windows into per-task event clusters, then ask Gemini
to label each cluster with a short phrase and one of six phase tags
(`pre_grasp`, `immobilization`, `contact`, `detach`, `post_grasp`,
`transition`).

### (d) Render 5-frame bundles around each keyframe

Save 5 PNG frames per waypoint at offsets `-4, -2, 0, 2, 4` plus a
short MP4 over the same window. Requires step (a) to have saved
rollout videos (`logging.save_video: true`).

```bash
python scripts/extract_keyframe_media.py \
    --waypoint-summary-path logs/openvla/keyframes/$EVAL_RUN/dp_pos_only_err0p05/waypoint_summary.json
```

Outputs under `logs/openvla/events/$EVAL_RUN/samples_5frames_stride2/`:
- per-sample PNG frames — input to step (e)
- per-sample MP4 clips — for human inspection
- `samples.jsonl` — sample manifest

### (e) Build vision embeddings + state vectors per sample

Encode each 5-frame bundle through a frozen vision encoder (default
SigLIP), L2-normalize, then concatenate the end-effector pose at the
waypoint. Requires GPU.

```bash
python scripts/build_event_features.py \
    --samples-path logs/openvla/events/$EVAL_RUN/samples_5frames_stride2/samples.jsonl
```

Output: `event_features.jsonl` next to `samples.jsonl`.

### (f) Task-local agglomerative clustering of event features

Cluster samples per task by cosine-distance agglomerative clustering
on the weighted [vision, state, progress] descriptor. CPU-only, runs
in seconds. Defaults: cosine threshold 0.18, weights 1.0 / 0.5 / 0.4,
5 exemplars per cluster.

```bash
python scripts/cluster_events.py \
    --event-features-path logs/openvla/events/$EVAL_RUN/samples_5frames_stride2/event_features.jsonl
```

Outputs under `clusters/` next to `event_features.jsonl`:
- `cluster_assignments.jsonl` — sample → cluster_id map
- `clusters.jsonl` — per-cluster members + exemplars
- `summary.json` — overall stats

### (g) Annotate clusters with Gemini

Send each cluster's representative 5-frame sequences to Gemini and
parse a `{phrase, phase}` JSON response. `phase` is one of the six
tags from the Phase 3 intro. Default model: `gemini-2.5-flash`
(override with `--model`). The paper used a stronger Gemini model;
this default keeps annotation cost low for reproduction. Parsed label
strings are not numerical score inputs. Annotation validity is nevertheless
an eligibility gate: missing, API-error, parse-error, or empty phrase/phase
rows are excluded and can therefore change the ranking population and
downstream intervention candidates.

```bash
export GEMINI_API_KEY=YOUR_GEMINI_API_KEY
python scripts/annotate_clusters.py \
    --clusters-path logs/openvla/events/$EVAL_RUN/samples_5frames_stride2/clusters/clusters.jsonl
```

Output: `gemini-2_5-flash_cluster_annotations.jsonl` next to
`clusters.jsonl`.

## Feature ranking (bridge between Phase 3 and Phase 4)

Encode the saved activations through the trained SAE, then score each
event cluster against the SAE features. The result is a ranked list of
candidate features for Phase 4.

### (h) Top-k SAE encoding (offline)

Apply the SAE to the dense shards from step (a) and write sparse
top-k shards. Encoding is decoupled from collection, so re-encoding
with a different SAE or layer needs no fresh rollout.

```bash
python scripts/extract_topk.py \
    --dense-dir logs/openvla/$EVAL_RUN/sae_activations/post_mlp_residual \
    --sae-checkpoint $SAE_CKPT \
    --layer-idx 31 \
    --output-dir logs/openvla/$EVAL_RUN/topk_activations
```

Output: top-k shards + `manifest.json` under
`logs/openvla/$EVAL_RUN/topk_activations/`.

To skip step (h) entirely, use the **online top-k mode** in step
(a): set `sae_collect.mode: "topk"` and `sae_collect.sae_checkpoint:
<path>` in the YAML and the rollout writes sparse shards directly.
This requires an SAE checkpoint already available (from a prior
training run or the Hugging Face Hub).

### (i) Event-feature score matrix

For each VLM-labeled cluster, score every SAE feature on how
strongly its activation lines up with that cluster's events. Inside
each `(cluster, episode)`, project every event window onto pulse,
step-up, and step-down templates; average events separately for
each template; then take the feature-wise maximum across those
three template means. Finally, average the resulting vector equally
across episodes in the cluster. This event-average → template-max →
episode-average order matches Appendix D and prevents long episodes
from receiving extra weight. CPU-only.

```bash
python scripts/score_cluster_features.py \
    --topk-run-dir logs/openvla/$EVAL_RUN/topk_activations \
    --prompt-records-path logs/openvla/$EVAL_RUN/prompt_records.jsonl \
    --event-features-path logs/openvla/events/$EVAL_RUN/samples_5frames_stride2/event_features.jsonl \
    --cluster-assignments-path logs/openvla/events/$EVAL_RUN/samples_5frames_stride2/clusters/cluster_assignments.jsonl \
    --cluster-annotations-path logs/openvla/events/$EVAL_RUN/samples_5frames_stride2/clusters/gemini-2_5-flash_cluster_annotations.jsonl \
    --output-path logs/openvla/scores/$EVAL_RUN/event_feature_scores.pt
```

Output: one `.pt` payload with `(num_clusters, dict_size)`
`matrix_raw`, `matrix_window_mean`, and `matrix_task_mean`, plus the
compatibility alias `matrix`, `row_keys`, `row_results`, `templates`,
`selection_counts`, `selected_events`, and `source`.

### (j) Build candidate feature lists

Surface the top-K features under four ranking strategies. CPU-only.

```bash
python scripts/build_feature_rankings.py \
    --scores-pt logs/openvla/scores/$EVAL_RUN/event_feature_scores.pt \
    --topk-run-dir logs/openvla/$EVAL_RUN/topk_activations \
    --output-dir logs/openvla/rankings/$EVAL_RUN \
    --top-k 5
```

The four rankings:

- **event-aligned** — mean of the score matrix across canonical cluster
  rows.
- **window-mean** — per-row window-mean vectors weighted by event count,
  restricted to the same canonical cluster rows.
- **task-mean** — per-task feature means weighted by per-task step count.
- **random-alive** — uniform sample over alive features, excluding any
  feature already chosen by the three informed rankings.

Outputs under `--output-dir`:

- per-ranking JSONL: `event_aligned.jsonl`, `window_mean.jsonl`,
  `task_mean.jsonl`, `random_alive.jsonl`
- `candidates.jsonl` — flat list of `4 × K` `(ranking, rank,
  feature_id, score)` rows that feeds step (k)

## Phase 4 — Closed-loop intervention

Edit one SAE feature at inference time and check how the policy's
success rate changes. For selected feature `i` and scaling factor
`α`:

    z'_i = α · z_i        # selected feature, scaled
    z'_j = z_j            # all other features unchanged (j ≠ i)
    x'   = x + Dec(z') − Dec(z)

`α = 0` zeros the feature out, `α = 1` leaves the hidden state
unchanged, intermediate values give partial suppression, `α > 1`
amplifies. The SAE reconstruction error on the un-edited code is
preserved.

### (k) Run a single-feature intervention on LIBERO

Run a closed-loop LIBERO eval with the residual-preserving hook
applied at the SAE's layer. Repeat the command once per `feature_id`
in `candidates.jsonl` — extract them with
`jq -r '.feature_id' candidates.jsonl`. Requires GPU.

```bash
FEATURE_ID=$(jq -r 'select(.ranking == "event_aligned") | .feature_id' \
    "logs/openvla/rankings/$EVAL_RUN/candidates.jsonl" \
    | head -n 1)

python scripts/openvla/intervene.py \
    --config configs/examples/openvla/collect_libero_spatial.yaml \
    --sae-checkpoint "$SAE_CKPT" \
    --layer-idx 31 \
    --feature-id "$FEATURE_ID" \
    --alpha 0.0
```

Add one no-hook baseline:

```bash
python scripts/openvla/collect_activations.py \
    --config configs/examples/openvla/collect_libero_spatial.yaml \
    --override sae_collect.enabled=false
```

For each ranking, take the mean of `SR_hook − SR_baseline` across
its K features — this is how much zeroing that ranking's features
hurts the policy. Use the same task set, trials/task, seed, model
checkpoint, and simulator settings for hook and no-hook runs, and
retain both the task-level `events.csv` and `stdout.log` episode outcomes
rather than only a copied aggregate table.

Each intervention run also writes
`intervene_feat<N>_alpha<A>_records.jsonl` (per-step feature
activation before / after the edit) for verifying the hook fired.

## Frozen environment snapshot

`environment-openvla.lock.yml` is a pinned record of the conda + pip
package versions on our working machine. It is **not a working
installer** — editable external libraries are not included. Use it
together with the explicit Step 2 commits and checkpoint revision to
cross-check a recreated environment.
