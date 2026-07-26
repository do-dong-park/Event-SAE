# GR00T offline pipeline — SAE, waypoint, and event partition

- Updated: 2026-07-26
- Model: GR00T N1.5, action expert physical layer 15
- Environment: RoboCasa PQ3
- Scope: offline SAE training through instruction-local raw clustering
- Claim strength: **diagnostic evidence**

[GR00T status](../README.md) ·
[Annotation lineage](annotation_lineage.md) ·
[Phase-feature analysis](phase_feature_analysis.md) ·
[Historical controlled result](anchor_view_controlled_ablation_results.md)

This document owns the mechanical pipeline before semantic annotation. It
combines the former Stage 1, Stage 2, and Stage 3A reports without changing
historical artifact identifiers. Annotation transport and label provenance
belong to [Annotation lineage](annotation_lineage.md); feature discovery and
steering candidates belong to
[Phase-feature analysis](phase_feature_analysis.md).

## 1. End-to-end contract

The pipeline keeps one row-level identity from source rollout to event
partition:

```text
150 rollout files
→ 12,041 policy records
→ 770,624 layer-15 action-token activation rows
→ BatchTopK SAE checkpoints

150 trajectories
→ 967 absolute-position anchors + 379 gripper-close peaks
→ 1,278 merged events
→ synchronized LEFT/RIGHT/WRIST media and descriptors
→ 162 instruction-local raw clusters at distance 0.18
```

| Boundary | Input | Output | Invariant |
| --- | --- | --- | --- |
| SAE data | trusted rollout PKL | validated activation cache | 150 files, 770,624 rows |
| SAE training | activation cache | checkpoint and full-data audit | finite encode/decode, L0 near 64 |
| Waypoints | trajectory records | merged event indices | exact geometry, episode-local indices |
| Media | merged events | three synchronized views | 5 chronological frames per view |
| Descriptor | media + physical state | one vector per event | 1,278 exact joins |
| Clustering | descriptors | raw partition | instruction-local, no semantic labels |

Success, failure, simulator predicates, and Oracle phase labels are not used to
fit the SAE, propose events, or build raw clusters. They are evaluation
metadata only.

Historical path components such as `v9`, `v11`, and `v1` are immutable artifact
locators. They are not current API names and are not renamed retroactively.

## 2. Source inventory and activation rows

The source contains two task families and five instruction/scene cells:

| Task family | Cell | Episodes |
| --- | --- | ---: |
| `OpenDrawer` | left drawer | 30 |
| `OpenDrawer` | right drawer | 30 |
| `PickPlaceCounterToCabinet` | beer | 30 |
| `PickPlaceCounterToCabinet` | bread | 30 |
| `PickPlaceCounterToCabinet` | pizza cutter | 30 |
| **Total** | **5 cells** | **150** |

Each policy record stores activation shape
`[layers=7, denoise=4, tokens=49, width=1536]`. The SAE dataset selects physical
layer 15 and action tokens `[33,49)`, then keeps denoise step and token offset
as independent samples:

```text
[7, 4, 49, 1536]
→ layer 15
→ action tokens [33,49)
→ [4, 16, 1536]
→ [64, 1536] rows per policy record
```

Therefore:

```text
12,041 records × 4 denoise steps × 16 action tokens
= 770,624 activation rows
```

State/future tokens, other physical layers, and pooled PQ2 activations are not
part of this cache. Raw PKL remains remote; local training and audit consume
only the validated cache:

```text
logs/groot_n15/stage1_sae/activation_cache/l15_action_tokens.pt
```

## 3. SAE training and retained checkpoints

The shared BatchTopK SAE uses:

| Parameter | Value |
| --- | ---: |
| Input / dictionary width | 1,536 / 1,536 |
| Active budget | 64 |
| Learning rate | `1e-4` |
| Seed | 0 |
| Training dtype | float32 |
| Activation normalization | enabled |

For residual row `x`, the model computes an encoder preactivation, applies
BatchTopK, and decodes the sparse code. Average L0 64 corresponds to 4.17% of
the dictionary active per row.

### 3.1 Step sweep

All rows below use the same source cache. The full-data audit evaluates 770,560
rows because it keeps complete batches of 512.

| Run | Passes | MSE ↓ | FVE ↑ | L0 | Alive |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.2k | 6.38 | 10.2842 | 0.987958 | 63.979 | 1,284 / 1,536 |
| 4k | 21.26 | 7.3981 | 0.991338 | 63.908 | 262 / 1,536 |
| 10k | 53.15 | 5.8102 | 0.993197 | 63.987 | 233 / 1,536 |
| 20k | 106.30 | 5.3015 | 0.993793 | 63.997 | 192 / 1,536 |

Longer training improved reconstruction while concentrating activity in fewer
features. Alive count is dictionary usage on this source, not semantic
generality or behavioral importance.

### 3.2 Row-matched batch sensitivity

Thresholds were recalibrated on the same 8,192 rows to target average L0 64.

| Row budget | Batch / steps | MSE ↓ | FVE ↑ | Alive |
| --- | ---: | ---: | ---: | ---: |
| 4.92M | 4,096 / 1,200 | 10.2812 | 0.987962 | 1,316 |
| 4.92M | 8,192 / 600 | 13.1164 | 0.984642 | 1,511 |
| 4.92M | 16,384 / 300 | 83.8915 | 0.901814 | 1,535 |
| 40.96M | 4,096 / 10,000 | 5.8094 | 0.993198 | 238 |
| 40.96M | 8,192 / 5,000 | 6.6244 | 0.992244 | 738 |
| 40.96M | 16,384 / 2,500 | 8.0248 | 0.990604 | 1,008 |

Fixed row count changes both batch size and optimizer update count, so this is
training-dynamics sensitivity rather than a causal batch-size ablation.

Three checkpoints were retained for downstream sensitivity:

- `1.2k`: broad alive coverage;
- `10k`: strongest reconstruction among the two batch-4096 discovery choices;
- `bs8192/5k`: similar row budget with intermediate coverage.

The current directional phase analysis uses the **10k coordinate** so identical
integer feature IDs refer to one checkpoint. This does not establish 10k as the
best phase SAE. Closed-loop reconstruction-only Hooked SR has not been run.

## 4. Waypoint construction

One trajectory row is a policy inference record, not an environment substep.
The active event proposal combines exact absolute-position compression and
gripper-closing peaks.

### 4.1 Exact position path

For anchors `i<j`, an edge is valid only when every intermediate point is less
than `η=0.05` from the segment joining the anchors. The shortest valid path
from the first to final record gives the position waypoints.

| Frame | Threshold | Waypoints | Mean/episode | Maximum error | Violations |
| --- | ---: | ---: | ---: | ---: | ---: |
| relative | 0.05 | 913 | 6.09 | 0.049994 | 0/150 |
| absolute | 0.05 | 967 | 6.45 | 0.049964 | 0/150 |

The active condition uses absolute coordinates. The frame is a GR00T project
choice, not a paper constant.

### 4.2 Gripper-close proposal

Per instruction cell, aperture is normalized from the two gripper joints.
Closing peaks use:

| Parameter | Value |
| --- | ---: |
| Height | 0.08 |
| Prominence | 0.04 |
| Minimum distance | 3 records |
| Position deduplication | ±2 records |
| Opening peaks | excluded |

The merged inventory is:

| Source | Events |
| --- | ---: |
| position | 899 |
| gripper close | 311 |
| both | 68 |
| **Total** | **1,278** |

The arithmetic is `967 + 379 - 68 = 1,278`. A closing peak is a motion
candidate, not a grasp label. As a diagnostic, it falls within ±2 records of
91/117 pick-place grasp markers and 18/36 drawer-open starts; these are coverage
figures, not precision.

## 5. Event representation

Every merged event has five chronological frames from each synchronized view:

```text
LEFT | RIGHT | WRIST
```

Each frame is encoded with `google/siglip-base-patch16-224`; the five frames in
one view are mean-pooled and L2-normalized.

| Block | Width | Contents |
| --- | ---: | --- |
| LEFT vision | 768 | mean-5 SigLIP |
| RIGHT vision | 768 | mean-5 SigLIP |
| WRIST vision | 768 | mean-5 SigLIP |
| Fused vision | 2,304 | equal-view concatenation |
| Physical state | 5 | absolute xyz, aperture, aperture delta |
| Progress | 1 | episode-relative progress |

Balanced C0 normalizes blocks and applies weights
`vision/state/progress = 1.0/0.5/0.4`. Progress can correlate with phase and is
therefore a known confound, not a semantic target.

## 6. Instruction-local clustering

Agglomerative clustering uses cosine distance, average linkage, and never mixes
different `task_description` values.

| Option | Value |
| --- | --- |
| Distance sweep | 0.12, 0.15, 0.18, 0.21, 0.24 |
| Primary distance | 0.18 |
| Coverage views | 0.3, 0.4, 0.5 |
| Representatives | up to 5 |

Primary partition:

| Cell | Events | Raw clusters | Coverage ≥0.3 |
| --- | ---: | ---: | ---: |
| left drawer | 232 | 32 | 2 |
| right drawer | 231 | 33 | 3 |
| beer | 324 | 44 | 5 |
| bread | 257 | 29 | 4 |
| pizza cutter | 234 | 24 | 6 |
| **Total** | **1,278** | **162** | **20** |

Coverage is unique episodes containing at least one cluster member divided by
30 episodes in the cell. Thus 0.3/0.4/0.5 require at least 9/12/15 episodes.

| Coverage | Clusters | Events |
| ---: | ---: | ---: |
| 0.3 | 20 | 484 |
| 0.4 | 14 | 359 |
| 0.5 | 10 | 273 |

Changing coverage filters the fixed raw partition. It does not recluster or
call Gemini again, but phase groups and score matrices must be rebuilt from the
qualified source clusters.

### 6.1 Controlled E0–E4 partitions

| Condition | Event proposal / views | Raw | ≥0.3 | ≥0.4 | ≥0.5 |
| --- | --- | ---: | ---: | ---: | ---: |
| E0/E1 | relative position, LEFT clustering | 128 | 17 | 12 | 9 |
| E2 | relative position, 3-view clustering | 152 | 18 | 12 | 7 |
| E3 | relative + gripper, 3-view | 160 | 18 | 12 | 11 |
| E4 | absolute + gripper, 3-view | 162 | 20 | 14 | 10 |

E0 and E1 share one partition and differ only in annotation view. Pairwise
geometry results are preserved in the
[historical controlled snapshot](anchor_view_controlled_ablation_results.md).

## 7. Reproduction boundaries

Maintained entry points:

| Purpose | Entry point |
| --- | --- |
| Activation export and merge | `scripts/groot/export_pq3_activation_shards.py` |
| SAE training | `event_sae/groot/train_from_activation_cache.py` |
| SAE audit | `scripts/groot/audit_sae_checkpoint.py` |
| Waypoint extraction | `scripts/extract_keyframes.py` |
| Event media | `scripts/extract_keyframe_media.py` |
| Event descriptor | `scripts/build_event_features.py` |
| Clustering | `scripts/cluster_events.py` |

Relevant machine-readable profiles:

```text
environment-sae-dev.yml
configs/groot/abs_position_gripper_multiview_phase.json
configs/groot/anchor_view_controlled_ablation_v1.json
```

New runs use a disjoint output root. Existing summaries, partitions, TopK
activation shards, and score artifacts are read-only provenance.

## 8. Confound audit

| Gate | Status | Evidence |
| --- | --- | --- |
| Length | **FAIL** | Long episodes contribute more activation rows and event opportunities. |
| Task identity | PASS | Training inventory is explicit; clustering is instruction-local. |
| Instruction balance | N/A | No paraphrase-balanced evaluation exists. |
| In-sample rescue | N/A | No detector or intervention is evaluated here. |
| Rollout pooling | PASS | Activation and event records remain individually joinable. |
| Phase / dwell | **FAIL** | Progress and event opportunity are not dwell-matched. |
| Observation ≠ causation | PASS | Reconstruction and partitions are mechanical diagnostics. |
| Scene-local ≠ general | **FAIL** | Only five instruction/scene cells are present. |

The pipeline establishes reproducible offline reconstruction, event geometry,
and raw partition joins. Claims that a checkpoint preserves policy behavior,
a gripper peak is a grasp, or a raw cluster is a semantic phase are
**confounded — 판정 보류**.
