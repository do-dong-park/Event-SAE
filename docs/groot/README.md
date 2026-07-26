# GR00T N1.5 / RoboCasa PQ3 status

- Updated: 2026-07-26
- Model: GR00T N1.5, action expert physical layer 15
- Scope: five instruction/scene cells, 150 source episodes
- Scientific status: **offline diagnostic evidence**
- Closed-loop GR00T intervention: **not run**

This is the entry point for the GR00T extension. Current methods, annotation
lineage, and phase-feature results are separated into three non-overlapping
reports. Historical execution contracts remain read-only.

## 1. Current state

| Area | Status | Source |
| --- | --- | --- |
| SAE, waypoints, clustering | Complete offline; three SAE sensitivity checkpoints retained | [Pipeline methods](reports/pipeline_methods.md) |
| Annotation | V12 provisional 90/90; 89 automatic + 1 user override; human review 0/90 | [Annotation lineage](reports/annotation_lineage.md) |
| Directional phase discovery | Oracle + V12 E3/E4, 10k SAE, W5 primary | [Phase-feature analysis](reports/phase_feature_analysis.md) |
| Strict historical inference | Oracle 0/25, V12 0/112, centroid E4 0/11 after their stated corrections | [Strict appendix](reports/phase_feature_analysis.md#11-strict-and-historical-appendix) |
| Closed-loop steering | Not run | [Probe design](reports/phase_feature_analysis.md#10-hooked-sr-probe-design) |

Historical V11 scores remain under `stage4/`. Strict V12 E0–E4 scores remain
under `stage4_v12_phase_features/`. The current directional analysis has a
separate semantic root and overwrites neither.

## 2. Scientific answer

The latest directional W5 analysis:

- preserves pulse, step-up, and step-down matrices;
- reproduces all 12 corresponding legacy combined scores exactly;
- ranks 6,054 instruction-local phase candidates and 442 state-pair candidates;
- measures sparse persistence for 366 displayed state-pair candidates;
- finds 1 confirmed, 20 partial, 260 boundary-only, and 85
  insufficient-support traces;
- tests a shared coarse ontology `reach → grasp → transport → terminal`;
- keeps fine-label, family, and global denominators explicit.

The most useful held-out probe candidates are:

| Role | Feature | Evidence |
| --- | ---: | --- |
| reach ON | F892 | Oracle 5/5 step-up |
| grasp ON | F1155 | E3 5/5, E4 4/4 available |
| transport ON | F432 | E3 4/4, E4 3/3 available |
| transport state | F533 | Bread transport↑→terminal↓, partial in E3/E4 |
| terminal ON/OFF | F506 / F1451 | Object-family directional recurrence |

Right-drawer F1093, F402, and F1179 match the same
`grasp→transport` direction in Oracle, E3, and E4, but their sparse traces are
boundary-only.

These are discovery candidates. They do not establish semantic identity,
causal control, or success-rate improvement. Claim strength remains
**diagnostic evidence**.

## 3. Documentation map

### Current reader-facing reports

| Document | Responsibility |
| --- | --- |
| [Pipeline methods](reports/pipeline_methods.md) | Activation contract, SAE training, waypoint geometry, event descriptors, raw partitions |
| [Annotation lineage](reports/annotation_lineage.md) | Prompt/media contract, provider ledgers, consensus derivations, review gate |
| [Phase-feature analysis](reports/phase_feature_analysis.md) | Directional W5, state pairs, recurrence, persistence, steering shortlist, strict appendix |

[OpenVLA](../backends/openvla.md) and
[OpenPI](../backends/openpi.md) are separate paper-backend runbooks, not GR00T
result sources.

### Historical and compatibility documents

| Document | Role |
| --- | --- |
| [Frozen controlled protocol](protocols/anchor_view_controlled_ablation.md) | Ex-ante V11 Gate 0–6 contract; byte-preserved provenance |
| [V11 automatic snapshot](reports/anchor_view_controlled_ablation_results.md) | Mechanical Q1–Q4 and 45-cell artifact snapshot |
| [V12r3 locator](reports/v12r3_annotation_rerun.md) | Compatibility path into annotation lineage |
| [Batch locator](reports/batch_annotation_rerun.md) | Compatibility path into annotation lineage |

The two locators remain because the frozen protocol links to their paths.

## 4. Evidence boundaries

| Evidence layer | Can answer | Cannot answer |
| --- | --- | --- |
| Offline SAE audit | reconstruction, sparsity, source usage | policy preservation, semantic quality |
| Raw event partition | geometry, join, recurrence inventory | true phase, causal event |
| V12 annotation | automatic consensus and label sensitivity | human-reviewed ground truth |
| Directional W5 | phase-local modulation and ON/OFF candidates | detector accuracy, causality, ΔSR |
| Sparse persistence | lossless TopK ON–dwell–OFF diagnostic | dense activation magnitude or intervention effect |
| Strict appendix | corrected association under its test family | proof that no useful steering feature exists |
| Historical V11 grid | condition/coverage/checkpoint sensitivity | current directional result |

Oracle and V12 raw scores are not pooled. Oracle uses simulator transitions at
environment-step scale; V12 uses annotated waypoint anchors on another clock.

Artifact terms such as `frozen`, `finalized`, and `complete` describe byte or
transport state, not semantic correctness.

## 5. Artifact roots

Local artifacts are outside Git:

```text
logs/groot_n15/
├─ stage1_sae/
├─ stage2_waypoints/
├─ stage4_feature_ranking/                  retained sparse TopK
├─ oracle_phase_five_cell_v1/               full Oracle source
└─ experiments/anchor_view_controlled_ablation_v1/
   ├─ annotations/                          V11
   ├─ annotations_v12r3/                    retired partial
   ├─ annotations_batch_generate_content_v1/
   ├─ annotations_batch_singleton_normalized_v2/
   ├─ annotations_batch_centroid_nearest_five/
   ├─ annotations_v12_adaptive_plurality/
   ├─ annotations_v12_user_phase_override/  current provisional labels
   ├─ stage4/                               historical V11
   ├─ stage4_v12_phase_features/            strict V12 grid
   ├─ stage4_centroid_plurality_phase_features/
   └─ analysis/
      └─ oracle_five_cell_vs_v12_e3_e4_10k_directional_w5_v1/
```

Historical path tokens remain unchanged because they are provenance locators.
New code and documentation use semantic names.

## 6. Retired execution archives

All paths are relative to
`logs/groot_n15/experiments/anchor_view_controlled_ablation_v1/`.

| Archive | SHA-256 |
| --- | --- |
| `archives/v12r3_retired_runtime_20260725.tar` | `0753f74afac6c2e7df484f2c689914c6cf56b5e4ecd4d99d4ca1fc1c3bcaa1c7` |
| `archives/batch_annotation_runtime_20260725.tar` | `3a07e51c5d84846b6d5a7110d6f30f98259aaff6aef4b8b66bb8047e90d7a9ae` |
| `archives/annotation_lineage_source_20260725.tar` | `71d6ddf9395c5bb41ec3e1a0d455dc55fcdf925ade75e0ecb15ebb441411543d` |

V12r3 contract:

```text
SHA-256  7ec0d330943180bd496d5490f913996eb4757c2320827761dd3d4118be275c33
Git HEAD 4a3f40ee9aee5dd4d53bbdefb804bc1662fb9e81
```

The source freeze is lifted for active code, but historical catalogs, media,
responses, manifests, and derived results remain read-only.

The frozen protocol is also provenance. Its working-copy SHA differs from the
historical experiment manifest, so this cleanup does not edit it.

## 7. Read-only commands

Run repository tests:

```bash
conda run --no-capture-output -n event-sae-dev \
  python -m pytest -q tests
```

Serve the result explorer:

```bash
conda run -n event-sae-dev \
  python scripts/review_clusters.py results --host 127.0.0.1 --port 8766
```

Run blind review on its separate route and output:

```bash
conda run -n event-sae-dev \
  python scripts/review_clusters.py serve --projection pca --port 8765
```

The machine source of truth for current directional discovery is:

```text
logs/groot_n15/experiments/anchor_view_controlled_ablation_v1/
  analysis/oracle_five_cell_vs_v12_e3_e4_10k_directional_w5_v1/
  summary.json
```

## 8. Active-code boundary

Maintained responsibilities:

1. shared media, schema, parsing, and consensus under `event_sae.events`;
2. GR00T artifact materialization under `event_sae.groot`;
3. directional scoring and recurrence under `event_sae.scoring`;
4. blind review isolated from read-only analysis;
5. new runs use new manifests and disjoint output roots.

No completed historical annotation or score root is resumed by active code.
