# OpenPI (π₀.₅) pipeline

Closed-loop interpretability pipeline for the π₀.₅ backbone (PaliGemma
vision-language prefix + action expert) on LIBERO simulation suites.

The default commands in this doc run a **5-trial public minimal
sweep** (10 tasks × 5 trials × 3 layers per capture target: AE
{0,5,17}, PG {0,11,16}). The paper's activation collection uses 50
rollouts/task and four layers per target (AE {0,5,11,17}, PG
{0,5,11,16}), while its Hooked SR protocol uses 10 rollouts/task.
The exact per-feature rollout count behind Table 3 is not stated.
The included sweep is therefore a runnable demonstration of the full
pipeline, not the full paper coverage or a paper-scale replication. PG layer 17 is
excluded from intervention sweeps because its post-MLP residual is
the final PaliGemma output;
no later action-expert block consumes that edited prefix state, so
these interventions are structural no-ops.

## Pipeline sanity check (5-trial scale)

This is the public minimal sweep (LIBERO-Spatial, 10 tasks × 5
trials = 50 suite episodes per condition, top-3 features per
ranking), not the paper's 50-rollout/task activation collection or
10-rollout/task Hooked SR protocol. Table 3 does not state its exact
per-feature rollout count. It is a pipeline/intervention smoke test. The broad
PaliGemma-versus-action-expert sensitivity regime and the AE l17
soft intervention are visible, but the hard-zero informed-ranking
order is not reproduced consistently. Paper-scale replication and
generalization are therefore **confounded — 판정 보류**.

### Hard zero-out (α_f = 0)

| Layer | source | Baseline | Δ event-aligned | Δ window-mean | Δ task-mean | Δ random-alive |
|---|---|---:|---:|---:|---:|---:|
| AE l00 | paper     | 96.4% | **−96.4** | **−96.4** | **−96.4** | −0.7  |
|        | this repo | 96.0% | **−96.0** | **−96.0** | **−96.0** | −30.0 |
| AE l05 | paper     | 96.4% | **−95.6** | **−96.2** | **−96.2** | −23.4 |
|        | this repo | 98.0% | **−98.0** | **−98.0** | **−98.0** | −78.7 |
| AE l17 | paper     | 96.4% | **−81.2** | **−96.4** | **−96.4** | −7.1  |
|        | this repo |100.0% | **−99.3** | **−100.0**| **−100.0**| −2.0  |
| PG l00 | paper     | 96.8% | −2.2 | −1.4 | −0.9 | −0.7 |
|        | this repo | 98.0% | −0.7 | −0.7 | +0.7 | +0.7 |
| PG l11 | paper     | 96.8% | −2.7 | −2.2 | −2.8 | −1.0 |
|        | this repo | 94.0% | +5.3 | +3.3 | +4.7 | +4.0 |
| PG l16 | paper     | 96.7% | −0.7 | +0.0 | −1.5 | −0.3 |
|        | this repo | 98.0% | +0.7 | −0.7 | −0.7 | +0.0 |

At 5 trials/task the AE-shallow alive pool contains only 75–84
features (≈8% of the dictionary). That limited pool and the
single-seed evaluation are consistent with sampling variability in
the random-alive rows, but they do not establish its cause:
**confounded — 판정 보류**.

### Soft feature sweep on AE l17 (α_f = 0.50)

| source    | Δ event-aligned | Δ window-mean | Δ task-mean | Δ random-alive |
|---|---:|---:|---:|---:|
| paper     | −54.6 | −88.6 | −89.3 | +0.0 |
| this repo | −55.3 | −86.7 | −88.7 | −2.0 |

The public table has no confidence intervals. For every reported row,
retain the Event-SAE commit and dirty diff, external dependency
commits, Hugging Face revision, `config.json` + `ae.pt` hashes,
`candidates.jsonl`, every raw `success.csv`, seed, and the exact
aggregation command. The summary table alone is not a provenance
artifact.

## Installation

openpi runs JAX inference in one process and a LIBERO sim policy in
another, communicating over an `openpi-client` websocket. The two
sides need different Python versions, so two envs are required.

Both envs use [uv](https://docs.astral.sh/uv/) (openpi's chosen
package manager — `pip` does not read `uv.lock`, so it would resolve
git-pinned deps like `lerobot` to incompatible PyPI latests). Install
once if it is not on `PATH`:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Define one env var pointing at this repo's checkout. All
installation and pipeline commands below reference `$EVENT_SAE_ROOT`:

```bash
export EVENT_SAE_ROOT=/path/to/event-sae   # adjust to the actual checkout
export OPENPI_ROOT="$EVENT_SAE_ROOT/external/openpi-event-sae"
mkdir -p "$EVENT_SAE_ROOT/external"
```

### Step 1: Clone the openpi fork

The openpi fork at `xc-j/openpi-event-sae` is vanilla openpi
plus the in-source SAE collection + intervention hooks needed by the
JAX path. Clone into `external/`:

```bash
cd "$EVENT_SAE_ROOT/external"
git clone https://github.com/xc-j/openpi-event-sae.git
cd "$OPENPI_ROOT"
git checkout 5ca3b6d95281f1a4c87ee445b11f60bc38adb920
git submodule update --init --recursive
```

### Step 2: Main env (py3.11, JAX + torch 2.7.1)

From the fork root, sync the locked dependencies and install the
openpi fork in editable mode:

```bash
cd "$OPENPI_ROOT"
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install \
    --python "$OPENPI_ROOT/.venv/bin/python" \
    -e "$OPENPI_ROOT"
```

Run main/JAX-side scripts with `uv run --project "$OPENPI_ROOT"` and
set `PYTHONPATH="$EVENT_SAE_ROOT"` for this non-packaged checkout.
This selects the main environment even if a LIBERO client venv was
active in another shell.

Sanity check:

```bash
PYTHONPATH="$EVENT_SAE_ROOT" uv run --project "$OPENPI_ROOT" \
    python -c "import jax, torch, lerobot.common; print('jax', jax.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
# expect: jax 0.5.3 torch 2.7.1 cuda True
```

### Step 3: Shared external libraries (same as openvla)

Two libraries from the OpenVLA runbook's Step 2 are also used here.
Clone under `external/` at the repo root (already gitignored), then
install into the openpi main env via `uv pip`. Skip the clone if
the openvla half already did it.

**(b) `dictionary_learning`** — SAE training library:

```bash
cd "$EVENT_SAE_ROOT/external"
git clone https://github.com/saprmarks/dictionary_learning.git
git -C dictionary_learning checkout 60ec6bf5264944d64a4ca271f45a29ebfb9d4946
uv pip install --python "$OPENPI_ROOT/.venv/bin/python" \
    -e "$EVENT_SAE_ROOT/external/dictionary_learning"
```

**(c) AWE** — kinematic keyframe extraction (fork with packaging
fixes; see the fork's NOTICE):

```bash
cd "$EVENT_SAE_ROOT/external"
git clone https://github.com/xc-j/awe.git
git -C awe checkout 7197bb86a20784666dabed90e6eabcf8bb1e9912
uv pip install --python "$OPENPI_ROOT/.venv/bin/python" \
    -e "$EVENT_SAE_ROOT/external/awe"

# AWE imports robosuite.utils.transform_utils for quaternion math; this
# pulls robosuite + mujoco (~500 MB on disk) but no GL is exercised.
uv pip install --python "$OPENPI_ROOT/.venv/bin/python" robosuite==1.4.0
```

**(d) Gemini SDK** — event-cluster annotation:

```bash
uv pip install --python "$OPENPI_ROOT/.venv/bin/python" google-genai
```

### Step 4: LIBERO sim env (py3.8, robosuite + torch 1.11+cu113)

The LIBERO sim has incompatible deps with the main env, so it gets a
separate venv. Recipe from `external/openpi-event-sae/examples/libero/README.md`:

```bash
cd "$OPENPI_ROOT"

uv venv --python 3.8 .venv-libero

uv pip sync \
    --python "$OPENPI_ROOT/.venv-libero/bin/python" \
    examples/libero/requirements.txt \
    third_party/libero/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cu113 \
    --index-strategy=unsafe-best-match

uv pip install --python "$OPENPI_ROOT/.venv-libero/bin/python" \
    -e packages/openpi-client
uv pip install --python "$OPENPI_ROOT/.venv-libero/bin/python" \
    -e third_party/libero

# Required when running LIBERO sim scripts:
source "$OPENPI_ROOT/.venv-libero/bin/activate"
export PYTHONPATH="$OPENPI_ROOT/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"
```

Sanity check:

```bash
python -c "import torch, robosuite, libero; print('torch', torch.__version__, 'robosuite', robosuite.__version__)"
# expect: torch 1.11.0+cu113 robosuite 1.4.1
```

Commands marked **openpi main env** use the explicit `uv run
--project "$OPENPI_ROOT"` form. Commands marked **LIBERO sim client**
run only after activating `$OPENPI_ROOT/.venv-libero`.

## Usage

The pipeline mirrors the [OpenVLA runbook](openvla.md) Phase 1–4 (collect →
train SAE → keyframes → events → score → rank → intervene), with
openpi-specific activation collection going through the JAX
`io_callback` hook in `external/openpi-event-sae/src/openpi/sae_collection/`.

Examples below assume the dense-collection run dir
`logs/openpi/sae_collection/$RUN` (under this repo) and the
LIBERO-Spatial / action-expert capture target. PaliGemma uses the
same flow with the companion run dir (capture target swap is
server-side).

Run the following snippets from the Event-SAE root unless a terminal
is explicitly labeled otherwise. `--project "$OPENPI_ROOT"` selects
the main JAX environment without changing where relative `logs/...`
paths resolve:

```bash
cd "$EVENT_SAE_ROOT"
```

## Phase 1 — SAE training

### (a) Collect openpi activations during a LIBERO rollout

The server (openpi main env, JAX) and the LIBERO sim client
(LIBERO sim env, Python 3.8) run in separate processes. The server
is suite-agnostic — the suite (`libero_spatial`,
`libero_object`, `libero_goal`, `libero_10`) is picked client-side
by the `--config` YAML. Action-expert and PaliGemma run as two
separate sessions, one capture target per server.

**Terminal 1 — server** (openpi main env):

```bash
PYTHONPATH="$EVENT_SAE_ROOT" uv run --project "$OPENPI_ROOT" \
    python "$EVENT_SAE_ROOT/scripts/openpi/serve_policy.py" \
    --env libero \
    --mode dense \
    --capture-target action_expert \
    --layer-indices 0,5,17 \
    --output-root "$EVENT_SAE_ROOT/logs/openpi/sae_collection" \
    --run-name libero_spatial_ae_l0_5_17 \
    --port 8000
```

For PaliGemma, swap `--capture-target paligemma --layer-indices
0,11,16` (and pick a different `--run-name`).

**Terminal 2 — LIBERO sim client** (LIBERO sim env). After the
server prints `Enabled SAE collection: ...`, activate the sim venv
and run the client from this repo root. The `--config` chooses the
suite:

```bash
cd "$EVENT_SAE_ROOT"
source "$OPENPI_ROOT/.venv-libero/bin/activate"
export PYTHONPATH="$OPENPI_ROOT/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"

python scripts/openpi/eval_libero.py \
    --config configs/examples/openpi/eval_libero_spatial.yaml \
    --override server.port=8000 \
    --override env.num_trials_per_task=5 \
    --override sae_collect.enabled=true \
    --override sae_collect.capture_target=action_expert \
    --override "sae_collect.layer_idxs=[0,5,17]" \
    --libero-root external/openpi-event-sae/third_party/libero
```

Output: `logs/openpi/sae_collection/<run_name>/` with
`sae_activations/post_mlp_residual{,__paligemma}/layer_NN_shard_*.pt`,
`activation_index.jsonl`, `trajectory_records.jsonl`, `videos/`, plus
`run_metadata.json` and `success.csv`.

### (b) Train one BatchTopK SAE per (target, layer)

One python invocation per `(capture_target, layer)` pair, e.g. for
action-expert layer 17:

```bash
AE_RUN=libero_spatial_ae_l0_5_17   # match Terminal 1 --run-name
PYTHONPATH="$EVENT_SAE_ROOT" uv run --project "$OPENPI_ROOT" \
    python "$EVENT_SAE_ROOT/scripts/train_sae.py" \
    --config configs/examples/openpi/train_sae_libero_spatial.yaml \
    --save-dir logs/openpi/sae/libero_spatial_action_expert_l17 \
    --override data_dir=logs/openpi/sae_collection/$AE_RUN/sae_activations/post_mlp_residual \
    --override layer_idx=17 \
    --override activation_dim=1024 \
    --override dict_size=1024 \
    --override submodule_name=post_mlp_residual \
    --override run_tag=libero_spatial_action_expert_l17
```

Repeat for the other five `(target, layer)` pairs (AE l00/l05/l17 +
PG l00/l11/l16). PaliGemma uses `activation_dim=dict_size=2048` and
`submodule_name=post_mlp_residual__paligemma`. Its activation input is
also target-specific; for example, set
`PG_RUN=libero_spatial_pg_l0_11_16` and pass
`data_dir=logs/openpi/sae_collection/$PG_RUN/sae_activations/post_mlp_residual__paligemma`.
Do not reuse the AE `data_dir`.

Output: `logs/openpi/sae/libero_spatial_<target>_l<NN>/trainer_0/ae.pt`
(+ `config.json` and intermediate checkpoints).

Or skip step (b) and use the six publicly released SAEs
(three AE layers + three PG layers, BatchTopK k=64) from the
[Hugging Face Hub](https://huggingface.co/mr-cabbage/event-sae-openpi-libero).
These six checkpoints are the public minimal subset, not all eight
layers in the paper's sweep.
For each `(target, layer)` pair you operate on, set `$SAE_CKPT` to
that pair's checkpoint — either the HF download or the local
training output:

```bash
# Pretrained, e.g. action_expert layer 17:
OPENPI_SAE_REV=ce210419aab908c47121e803f777633004efbaf1
OPENPI_SAE_DIR="$EVENT_SAE_ROOT/external/checkpoints/openpi-$OPENPI_SAE_REV"
PYTHONPATH="$EVENT_SAE_ROOT" uv run --project "$OPENPI_ROOT" \
    huggingface-cli download mr-cabbage/event-sae-openpi-libero \
    action_expert_l17/ae.pt action_expert_l17/config.json \
    --revision "$OPENPI_SAE_REV" \
    --local-dir "$OPENPI_SAE_DIR"
SAE_CKPT="$OPENPI_SAE_DIR/action_expert_l17/ae.pt"

# Or locally trained:
SAE_CKPT=logs/openpi/sae/libero_spatial_action_expert_l17/trainer_0/ae.pt
```

`ae.pt` and its sibling `config.json` are one checkpoint contract;
download both from the same immutable revision. Reset `$SAE_CKPT`
for each pair as you sweep through them. All subsequent commands in
this doc that need a checkpoint reference `--sae-checkpoint
$SAE_CKPT`. The intervention server fails closed unless the sibling
config's layer, activation width, and capture-target submodule match
the requested hook; it also validates tensor shapes and feature-id
bounds before policy creation.

## Phase 2 — Kinematic keyframe extraction

Keyframes are computed from the rollout trajectory only, not from
the SAE activations, so a single keyframe set is reused for scoring
features from every layer of every capture target. Use either the
AE or the PG run's `trajectory_records.jsonl` — by convention the
AE collection (the first stream collected) supplies the events
used downstream. Substitute `$RUN` below with that collection's
run name (the `--run-name` chosen in step (a)).

### (c) Extract AWE kinematic keyframes from rollout trajectories

CPU-only.

```bash
RUN=libero_spatial_ae_l0_5_17   # replace if step (a) used another run name
PYTHONPATH="$EVENT_SAE_ROOT" uv run --project "$OPENPI_ROOT" \
    python "$EVENT_SAE_ROOT/scripts/extract_keyframes.py" \
    --trajectory-records-path logs/openpi/sae_collection/$RUN/trajectory_records.jsonl
```

Output: `waypoint_summary.json` under
`logs/openpi/keyframes/$RUN/dp_pos_only_err0p05/`.

## Phase 3 — Event clustering with VLM annotation

### (d) Render 5-frame bundles around each keyframe

```bash
PYTHONPATH="$EVENT_SAE_ROOT" uv run --project "$OPENPI_ROOT" \
    python "$EVENT_SAE_ROOT/scripts/extract_keyframe_media.py" \
    --waypoint-summary-path logs/openpi/keyframes/$RUN/dp_pos_only_err0p05/waypoint_summary.json
```

Outputs under `logs/openpi/events/$RUN/samples_5frames_stride2/`: PNG
frames + clip MP4s + `samples.jsonl`.

### (e) Build vision embeddings + state vectors per sample

Requires GPU.

```bash
PYTHONPATH="$EVENT_SAE_ROOT" uv run --project "$OPENPI_ROOT" \
    python "$EVENT_SAE_ROOT/scripts/build_event_features.py" \
    --samples-path logs/openpi/events/$RUN/samples_5frames_stride2/samples.jsonl
```

Output: `event_features.jsonl` next to `samples.jsonl`.

### (f) Task-local agglomerative clustering

CPU-only.

```bash
PYTHONPATH="$EVENT_SAE_ROOT" uv run --project "$OPENPI_ROOT" \
    python "$EVENT_SAE_ROOT/scripts/cluster_events.py" \
    --event-features-path logs/openpi/events/$RUN/samples_5frames_stride2/event_features.jsonl
```

Outputs under `clusters/` next to `event_features.jsonl`:
`cluster_assignments.jsonl`, `clusters.jsonl`, `summary.json`.

### (g) Annotate clusters with Gemini

```bash
export GEMINI_API_KEY=YOUR_GEMINI_API_KEY
PYTHONPATH="$EVENT_SAE_ROOT" uv run --project "$OPENPI_ROOT" \
    python "$EVENT_SAE_ROOT/scripts/annotate_clusters.py" \
    --clusters-path logs/openpi/events/$RUN/samples_5frames_stride2/clusters/clusters.jsonl
```

Output: `gemini-2_5-flash_cluster_annotations.jsonl` next to
`clusters.jsonl`. Default model `gemini-2.5-flash` (override with
`--model`); the paper used a stronger Gemini model, and these
label strings are not numerical score inputs. Annotation validity is
nevertheless an eligibility gate: missing, API-error, parse-error, or
empty phrase/phase rows are excluded and can therefore change the
ranking population and downstream intervention candidates.

## Feature ranking (bridge between Phase 3 and Phase 4)

Encode the saved dense activations through each trained SAE, score each
event cluster against the resulting sparse features, and produce a
ranked list of candidate features for Phase 4. Run all three steps
once per (capture target, layer) pair — six pairs for libero_spatial
(`{action_expert} × {0, 5, 17}` ∪ `{paligemma} × {0, 11, 16}`). All
six are scored against the same AE-derived event clusters (see
Phase 2 note).

### (h) Top-k SAE encoding (offline)

Apply each trained SAE to its own dense shards and write sparse top-k
shards. CPU- or GPU-friendly (auto-detects).

```bash
TARGET=action_expert     # or paligemma
LAYER=17                 # AE: {0,5,17}; PG: {0,11,16}
DENSE_RUN=libero_spatial_ae_l0_5_17   # use the matching target's run
TAG=${TARGET}_l$(printf '%02d' ${LAYER})

PYTHONPATH="$EVENT_SAE_ROOT" uv run --project "$OPENPI_ROOT" \
    python "$EVENT_SAE_ROOT/scripts/extract_topk.py" \
    --dense-dir logs/openpi/sae_collection/${DENSE_RUN} \
    --sae-checkpoint $SAE_CKPT \
    --layer-idx ${LAYER} \
    --output-dir logs/openpi/topk/${TAG}
```

Output: per-pair `shard_*.pt` (sparse top-k rows) + `manifest.json`
under `logs/openpi/topk/<target>_l<NN>/`.

To skip step (h), use the **online top-k mode** in step (a):
re-launch the server with `--mode topk --sae-checkpoint $SAE_CKPT`
(and a single `--layer-indices NN`) and the rollout writes sparse
`token_topk_sparse_v1` shards directly into the run dir. Requires
an SAE checkpoint already available (from step (b) or the Hugging
Face Hub).

### (i) Event-feature score matrix

For each pair, score every SAE feature against the AE-derived event
clusters. Inside each `(cluster, episode)`, project every event
window onto pulse, step-up, and step-down templates; average events
separately for each template; then take the feature-wise maximum
across those three template means. Finally, average the resulting
vector equally across episodes in the cluster. This event-average →
template-max → episode-average order matches Appendix D and
prevents long episodes from receiving extra weight. CPU-only.

```bash
AE_RUN=libero_spatial_ae_l0_5_17   # event-source run from step (a)
EVT=logs/openpi/events/${AE_RUN}/samples_5frames_stride2

PYTHONPATH="$EVENT_SAE_ROOT" uv run --project "$OPENPI_ROOT" \
    python "$EVENT_SAE_ROOT/scripts/score_cluster_features.py" \
    --topk-run-dir logs/openpi/topk/${TAG} \
    --event-features-path ${EVT}/event_features.jsonl \
    --cluster-assignments-path ${EVT}/clusters/cluster_assignments.jsonl \
    --cluster-annotations-path ${EVT}/clusters/gemini-2_5-flash_cluster_annotations.jsonl \
    --prompt-records-path logs/openpi/sae_collection/${DENSE_RUN}/prompt_records.jsonl \
    --output-path logs/openpi/scores/${TAG}.pt
```

`--prompt-records-path` is required for paper-faithful
`matrix_task_mean` (per-task mean over **all** rollout timesteps in
the task, not only event-window timesteps). Omit it only if you
explicitly want the approximate event-only fallback.

Output: one `.pt` payload per pair — three matrices
(`matrix_raw`, `matrix_window_mean`, `matrix_task_mean`) plus
`row_keys`, `selected_events`, `templates`, `source`.

### (j) Build candidate feature lists

Surface the top-K features under four ranking strategies. CPU-only.
For π₀.₅ the paper uses K = 3. The script reads pre-computed matrices
from the score artifact; `--topk-run-dir` is only used by the
`random_alive` ranking (for the alive-feature scan).

```bash
PYTHONPATH="$EVENT_SAE_ROOT" uv run --project "$OPENPI_ROOT" \
    python "$EVENT_SAE_ROOT/scripts/build_feature_rankings.py" \
    --scores-pt logs/openpi/scores/${TAG}.pt \
    --topk-run-dir logs/openpi/topk/${TAG} \
    --output-dir logs/openpi/rankings/${TAG} \
    --top-k 3
```

Output: `event_aligned.jsonl`, `window_mean.jsonl`, `task_mean.jsonl`,
`random_alive.jsonl`, and `candidates.jsonl` (flat `4 × K` rows that
feed Phase 4 intervention).

## Phase 4 — Closed-loop intervention

Edit one SAE feature at inference time and check how the policy's
success rate changes. For selected feature `i`, feature-scaling
factor `α_f`, and reconstruction mix `α`:

    z'_i = α_f · z_i      # selected feature, scaled
    z'_j = z_j            # all other features unchanged (j ≠ i)
    x'   = x + α · (Dec(z') − Dec(z))

`α_f = 0` is the paper's hard zero-out (feature i is removed);
`α_f ∈ (0, 1)` is a soft suppression that the paper calls
"dose-response"; `α_f > 1` boosts. `α` scales only the decoded
feature-edit delta: `α = 1` applies the full intervention and `α = 0`
disables it. The unedited SAE reconstruction is subtracted, so the
base SAE reconstruction error is not injected into the policy
residual. The server-side hook lives in
`external/openpi-event-sae/src/openpi/sae_collection/reconstruction.py`
and is enabled by `scripts/openpi/serve_policy.py --mode intervene`.

### (k) Run a single-feature intervention on LIBERO

Repeat the server-then-client pair once per `feature_id` in
`candidates.jsonl` — extract them with
`jq -r '.feature_id' candidates.jsonl`.

First run one no-hook baseline with the same task set,
trials/task, seed, model checkpoint, and simulator settings used by
the interventions.

**Baseline terminal 1 — server** (openpi main env):

```bash
PYTHONPATH="$EVENT_SAE_ROOT" uv run --project "$OPENPI_ROOT" \
    python "$EVENT_SAE_ROOT/scripts/openpi/serve_policy.py" \
    --env libero \
    --mode baseline \
    --port 8000
```

**Baseline terminal 2 — client** (LIBERO sim env):

```bash
cd "$EVENT_SAE_ROOT"
source "$OPENPI_ROOT/.venv-libero/bin/activate"
export PYTHONPATH="$OPENPI_ROOT/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"

python scripts/openpi/eval_libero.py \
    --config configs/examples/openpi/eval_libero_spatial.yaml \
    --override server.port=8000 \
    --override env.num_trials_per_task=5 \
    --override sae_collect.enabled=false \
    --override "logging.root_dir=$EVENT_SAE_ROOT/logs/openpi/intervene/baseline" \
    --libero-root "$OPENPI_ROOT/third_party/libero"
```

Then start the server in intervention mode (openpi main env):

```bash
FEATURE_ID=$(jq -r 'select(.ranking == "event_aligned") | .feature_id' \
    "$EVENT_SAE_ROOT/logs/openpi/rankings/action_expert_l17/candidates.jsonl" \
    | head -n 1)

PYTHONPATH="$EVENT_SAE_ROOT" uv run --project "$OPENPI_ROOT" \
    python "$EVENT_SAE_ROOT/scripts/openpi/serve_policy.py" \
    --env libero \
    --mode intervene \
    --sae-checkpoint "$SAE_CKPT" \
    --capture-target action_expert \
    --layer-idx 17 \
    --feature-indices "$FEATURE_ID" \
    --feature-alpha 0.0 \
    --recon-alpha 1.0 \
    --port 8000
```

Run the LIBERO client (other terminal, LIBERO sim env). Disable
`sae_collect.enabled` so the client does no per-request context
plumbing — the server applies the intervention transparently:

```bash
cd "$EVENT_SAE_ROOT"
source "$OPENPI_ROOT/.venv-libero/bin/activate"
export PYTHONPATH="$OPENPI_ROOT/third_party/libero${PYTHONPATH:+:$PYTHONPATH}"

python scripts/openpi/eval_libero.py \
    --config configs/examples/openpi/eval_libero_spatial.yaml \
    --override server.port=8000 \
    --override env.num_trials_per_task=5 \
    --override sae_collect.enabled=false \
    --override "logging.root_dir=$EVENT_SAE_ROOT/logs/openpi/intervene/single_feature" \
    --libero-root "$OPENPI_ROOT/third_party/libero"
```

Output: `logs/openpi/intervene/single_feature/EVAL-*/success.csv`
with one row per (episode, success). For each ranking, take the
mean of `SR_hook − SR_baseline` across its K features — this is
how much zeroing that ranking's features hurts the policy. Repeat
the server-then-client invocation once per `feature_id` in
`candidates.jsonl` to fill the full per-pair ΔSR table, and retain
the per-episode CSVs rather than only the aggregate.

## Frozen environment snapshot

`environment-openpi.lock.yml` is a pinned record of the conda + pip
package versions on our working machine. It is **not a working
installer** — editable external repos (the openpi fork,
`dictionary_learning`, AWE) are not included. Use it together with
the explicit external commits and checkpoint revision above to
cross-check a recreated environment.
