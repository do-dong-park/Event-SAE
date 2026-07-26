# GR00T annotation contract and execution lineage

- Updated: 2026-07-26
- Model: GR00T N1.5, RoboCasa PQ3
- Current analysis source: `annotations_v12_user_phase_override/`
- Current source size: 90/90 rows
- Formal human review: 0/90
- Claim strength: **diagnostic evidence**

[GR00T status](../README.md) ·
[Pipeline methods](pipeline_methods.md) ·
[Phase-feature analysis](phase_feature_analysis.md) ·
[Historical controlled result](anchor_view_controlled_ablation_results.md)

This document owns annotation semantics, provider execution history, derived
consensus, and promotion rules. It does not own raw clustering or feature
ranking. Every annotation root is a separate evidence layer; rows are never
silently copied over another root.

## 1. Current semantic contract

The current analysis uses V12 centroid-nearest annotation with adaptive
plurality. It contains 89 automatic decisions and one explicit user-directed
phase override. `90/90 complete` means artifact completeness, not formal human
approval.

| Property | Contract |
| --- | --- |
| Unit | one representative event clip per request |
| E0 media | five chronological LEFT frames |
| E1–E4 media | synchronized LEFT/RIGHT/WRIST at five timestamps |
| Initial representatives | three |
| Expansion | up to five, then unique-episode pairs to ranks 7 and 9 |
| Output | phrase, phase, visibility |
| Model | `gemini-3.1-pro-preview` |
| Temperature | 0.0 |
| Semantic status | automatic provisional |

Each representative is judged independently. Consensus is computed after
parsing; the model does not see the other representatives' answers.

The current V12 source is a replacement experiment. It does not overwrite the
historical V11 grid, V12r3 partial run, provider Batch ledger, or centroid-only
derivations.

## 2. Input and selection

Annotation consumes the fixed instruction-local partitions described in
[Pipeline methods](pipeline_methods.md). Coverage ≥0.3 supplies the widest
annotation set needed by the 0.3/0.4/0.5 downstream sensitivity views.

| Condition | Clusters |
| --- | ---: |
| E0 | 17 |
| E1 | 17 |
| E2 | 18 |
| E3 | 18 |
| E4 | 20 |
| **Total** | **90** |

Coverage changes do not trigger new provider calls. Downstream code filters
these fixed rows by source-cluster coverage and rebuilds phase groups.

Hidden request information includes:

- success or failure;
- simulator Oracle event and predicates;
- raw gripper qpos;
- waypoint anchor source;
- representative relative progress;
- original source frame step.

Episode coverage remains visible because it is a cluster recurrence statistic
used by the paper-style annotation prompt. It is not the same as within-episode
progress.

## 3. Phase vocabularies

Fine labels remain task-family-specific:

| Family | Allowed labels |
| --- | --- |
| Drawer | `reach-to-handle`, `grasp-handle`, `pull`, `push-back`, `disengage`, `wrong-grasp`, `open-done` |
| Object placement | `reach-to-object`, `grasp`, `transport`, `place`, `insert-settle`, `terminal`, `wrong-grasp` |

Current-state labels need not be monotone. A failed attempt can return from a
grasp-like state to reach.

Directional analysis also derives a shared coarse view:

```text
reach → grasp → transport → terminal
```

This mapping is an analysis ontology, not a rewrite of the stored fine label.
Fine and coarse phase groups are regenerated separately and retain source
hashes.

## 4. Provider and parsing contract

Newly generated rows preserve:

- exact prompt text and SHA-256;
- model, prompt version, temperature, MIME, and JSON schema;
- allowed phase vocabulary and media layout;
- representative sample, clip, and frame paths;
- prompt input policy;
- raw provider response and parse status.

Credentials are environment-only. Source manifests identify media and catalogs,
but historical runs do not all contain per-image byte hashes; exact request
bytes must not be claimed when they were not recorded.

Parsing is fail-closed. The normalized Batch child permits only a top-level
object or a one-element list containing one otherwise valid object. It unwraps
once and then applies the frozen parser. Recursive repair, aliases, and type
coercion remain forbidden.

## 5. Execution lineage

| Run | Provider result | Human review | Downstream role |
| --- | --- | ---: | --- |
| V9 current condition | 20/20 schema-valid | 0/20 | Historical single-condition diagnostic |
| V11 controlled E0–E4 | 90/90 | 0/90 | Historical 45-cell score grid |
| V12r3 interactive | 41/90 finalized | 0/90 | Frozen partial view diagnostic |
| Source Batch v1 | 233/270 initial valid, 37 exhausted | 0/90 | Immutable fail-closed transport history |
| Singleton-normalized child v2 | 406 selected responses, 90/90 rows | 0/90 | Complete replacement, not current ranking source |
| Centroid-nearest-five | 450/450 responses, 90 rows | 0/90 | Representative-policy sensitivity |
| Unique plurality | 72/90 accepted | 0/90 | Historical E4 inferential sensitivity |
| V12 adaptive plurality | 89/90 accepted | 0/90 | Parent of current source |
| V12 user override | 90/90, one explicit override | 0/90 formal | Current provisional E0–E4 and directional analysis |

The table separates transport completion from semantic correctness. Only a
blind reviewer can promote an automatic row.

## 6. Historical V11

V11 annotates one cluster-level bundle and emits one phrase/phase pair. Its
historical current-condition input used up to five clips, each containing five
chronological triptych images.

V11 supplies the immutable `stage4/` 45-cell result. It remains useful for
condition, coverage, and checkpoint sensitivity, but does not supply the
current directional E3/E4 ranking.

The historical current-condition V9/V11-era artifact has 20/20 schema-valid
rows, no API or parse errors, and no human review. It predates some exact prompt
and generation fields, so its request-level provenance is incomplete.

No paid retry is required to interpret the completed V12 source. A new provider
run, if needed, must use a new manifest and output root rather than backfilling
metadata into V11.

## 7. Retired V12r3 interactive run

V12r3 started with three representatives and expanded to five when strict
consensus was absent.

| Condition | Target | Finalized | Strict consensus | Mixed |
| --- | ---: | ---: | ---: | ---: |
| E0 | 17 | 17 | 1 | 16 |
| E1 | 17 | 17 | 4 | 13 |
| E2 | 18 | 7 | 3 | 4 |
| E3 | 18 | 0 | — | — |
| E4 | 20 | 0 | — | — |
| **Total** | **90** | **41** | **8** | **33** |

The run stopped during E2 after quota exhaustion. E0/E1 phase agreement over
81 shared representative keys was 28/81. This is view sensitivity, not
accuracy.

```text
root
logs/groot_n15/experiments/anchor_view_controlled_ablation_v1/annotations_v12r3/

contract SHA-256
7ec0d330943180bd496d5490f913996eb4757c2320827761dd3d4118be275c33

recorded Git HEAD
4a3f40ee9aee5dd4d53bbdefb804bc1662fb9e81
```

The partial root, lock, inflight state, media, catalogs, and manifest are
read-only. Do not resume it. The runtime is preserved in
`archives/v12r3_retired_runtime_20260725.tar`.

## 8. Batch source and normalized child

The source Batch regenerated all 90 clusters independently. Ten provider jobs
completed, but 37 of 270 initial logical requests returned a one-element list
instead of the required object. The strict source correctly failed closed and
materialized no final 90-row set.

The normalized child imported all initial responses and generated only the
required expanded requests.

| Child result | Count |
| --- | ---: |
| Imported initial requests | 270/270 |
| Newly generated expanded requests | 136/136 |
| Unique selected responses | 406 |
| consensus 3-of-3 | 22 |
| consensus 4-of-5 | 11 |
| mixed | 57 |
| Materialized rows | 90/90 |

```text
source contract
31fb6ad3819b2ed49e899f65ebeb01273d905b2b85d0f18532c7a4ddca78f0dc

normalized-child contract
0b1f21df0eb39c5c9e9a73c7436ccd9bb04a56651fee1ba16dcd4d259d718dc6
```

Neither root supplies the current directional ranking.

## 9. Centroid-five and V12 completion

Centroid-five uses the five feature-space samples nearest each raw cluster
centroid. All 450 logical requests completed.

The first derivation accepted strong consensus, majority, or a unique 2-of-5
plurality:

| Condition | Rows | Accepted |
| --- | ---: | ---: |
| E0 | 17 | 12 |
| E1 | 17 | 14 |
| E2 | 18 | 14 |
| E3 | 18 | 13 |
| E4 | 20 | 19 |
| **Total** | **90** | **72** |

The confidence inventory was 26 strong, 36 majority, 10 unique pluralities,
and 18 unresolved.

V12 adaptive plurality starts from the 18 exact `2-2-1` ties. It adds complete
pairs of unique-episode centroid representatives and decides only after ranks
7 and 9. Thirteen ties resolved at rank 7 and four at rank 9.

One row remained tied:

```text
E3 beer cluster_10
grasp 4 / transport 4 / place 1
```

The workspace owner selected `grasp` for provisional analysis. The child
preserves all nine votes, changes only the derived top-level phrase/phase, and
records `user-directed-phase-override`. Two later completed responses and all
cancelled ranks are excluded.

```text
adaptive contract
20cbde0fcb3301ddb09b064e9db6566f26a19eaeada389b98d98ce23c2803b5b

user-override derivation contract
61f89ffa7062aba95a34103cc5e7467dae30eb03a38f35b1c5b1d2c16fbcc044
```

## 10. Artifact separation and archives

All paths below are relative to:

```text
logs/groot_n15/experiments/anchor_view_controlled_ablation_v1/
```

```text
annotations/                                  V11 controlled source
annotations_v12r3/                            retired interactive partial
annotations_batch_generate_content_v1/        immutable Batch source
annotations_batch_singleton_normalized_v2/    normalized child
annotations_batch_centroid_nearest_five/       centroid provider source
annotations_batch_centroid_nearest_five_unique_plurality/
annotations_v12_adaptive_plurality/
annotations_v12_user_phase_override/           current provisional source
```

Runtime archives:

| Archive | SHA-256 |
| --- | --- |
| `archives/v12r3_retired_runtime_20260725.tar` | `0753f74afac6c2e7df484f2c689914c6cf56b5e4ecd4d99d4ca1fc1c3bcaa1c7` |
| `archives/batch_annotation_runtime_20260725.tar` | `3a07e51c5d84846b6d5a7110d6f30f98259aaff6aef4b8b66bb8047e90d7a9ae` |
| `archives/annotation_lineage_source_20260725.tar` | `71d6ddf9395c5bb41ec3e1a0d455dc55fcdf925ade75e0ecb15ebb441411543d` |

Compatibility summaries remain at
[V12r3 locator](v12r3_annotation_rerun.md) and
[Batch locator](batch_annotation_rerun.md) because the frozen protocol links
to those paths.

## 11. Human-review promotion gate

Promotion requires:

1. blind assessment before automatic phrase/phase reveal;
2. independent phrase, phase, mixed, and visually-insufficient fields;
3. atomic blind save before adjudication;
4. `approved`, `corrected`, or `ambiguous` verdict for every row;
5. explicit exclusion denominator for ambiguous rows;
6. a new phase-group and score root;
7. exact source, media, annotation, review, and output hashes.

The current user override is not a substitute for this process.

## 12. Evidence boundary

| Gate | Status | Evidence |
| --- | --- | --- |
| Length | **FAIL** | Fixed clips inherit unequal upstream event opportunities. |
| Task identity | PASS | Closed vocabularies and groups remain instruction-local. |
| Instruction balance | N/A | No instruction paraphrase evaluation exists. |
| In-sample rescue | N/A | This is not detector or intervention evaluation. |
| Rollout pooling | PASS | Representative clips are judged separately. |
| Phase / dwell | **FAIL** | Labels are automatic and anchor timing is not dwell-matched. |
| Observation ≠ causation | PASS | Labels are metadata, not causal feature identities. |
| Scene-local ≠ general | **FAIL** | Only five instruction/scene cells are covered. |

Schema validity, provider completion, and vote consensus do not establish phase
correctness. The semantic correctness of the 90 provisional rows is
**confounded — 판정 보류**.
