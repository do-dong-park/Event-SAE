# GR00T directional phase-feature analysis

- Updated: 2026-07-26
- Model: GR00T N1.5, RoboCasa PQ3, action expert layer 15
- Primary checkpoint coordinate: batch 4,096, step 10,000 (`10k`)
- Primary window: W5
- Sources: full five-cell Oracle, V12 E3, V12 E4
- Claim strength: **diagnostic evidence**

[GR00T status](../README.md) ·
[Pipeline methods](pipeline_methods.md) ·
[Annotation lineage](annotation_lineage.md) ·
[Historical controlled result](anchor_view_controlled_ablation_results.md)

This document asks which SAE features align with a phase boundary or ON→OFF
state interval and deserve a held-out steering probe. The answer is candidate
discovery, not causal identification. Strict tests remain an appendix.

## 1. Current answer

Directional candidates exist for reach, grasp, transport, and terminal. The
strongest reusable probes are concentrated in grasp and transport; reach is
mostly Oracle-led, and terminal is often clearer as the OFF side of an earlier
state.

| Phase role | Candidate | Direction | Main support | Probe use |
| --- | ---: | --- | --- | --- |
| reach ON | F892 | step-up | Oracle 5/5 global-strict | exploratory reach injection/zero-out |
| grasp ON | F1155 | step-up | E3 5/5; E4 4/4 available | primary grasp probe |
| grasp ON | F64 | step-up | E3 4/5; E4 4/4; Oracle partial | cross-source sensitivity |
| grasp OFF | F1389 | step-down | Oracle/E3/E4 global-relaxed | matched OFF-direction probe |
| transport ON | F432 | step-up | E3 4/4; E4 3/3 available | primary transport boundary probe |
| transport state | F533 | transport↑→terminal↓ | Bread E3/E4 sparse persistence partial | primary state-interval probe |
| terminal ON | F506 | step-up | E4 object 3/3 | terminal-entry sensitivity |
| terminal OFF | F1451 | step-down | E3/E4 object 3/3 | terminal transition control |

These priorities do not mean the features encode the named semantics. They
mean the features satisfy the stated ranking contract on the available source.

The best exact three-source state-pair matches are right-drawer
`grasp→transport`:

| Feature | Oracle rank | E3 rank | E4 rank | Oracle coverage | Family-strict sources |
| ---: | ---: | ---: | ---: | ---: | ---: |
| F1093 | 2 | 5 | 6 | 0.8 | 1 |
| F402 | 3 | 12 | 4 | 0.8 | 0 |
| F1179 | 4 | 19 | 19 | 0.8 | 0 |

Their sparse traces are boundary-only, so they are cross-source boundary
candidates rather than confirmed persistent states.

## 2. Scope and immutable inputs

The analysis reuses existing TopK activation and annotation artifacts. It does
not retrain the SAE, recollect activations, rerun Gemini, or modify prior score
roots.

| Source | Fine view | Coarse view | Coverage role |
| --- | --- | --- | --- |
| Oracle full | simulator fine phases | exact coarse4 regroup | no coverage gate; low coverage marked |
| V12 E3 | original automatic labels | exact coarse4 regroup | coverage 0.3 |
| V12 E4 | original automatic labels | exact coarse4 regroup | coverage 0.3 |

For each of 3 sources × 2 views, W4 and W5 are rescored, yielding 12 new score
artifacts. Oracle and V12 scores are never averaged or added.

Fine labels retain their source vocabulary. State-pair ranking is defined only
for the shared coarse order:

```text
reach → grasp → transport → terminal
```

All six earlier→later pairs are considered. No grasp/place pair is hard-coded.

## 3. Directional W5 score

For every episode group and phase row, the scorer preserves separate
`pulse`, `step_up`, and `step_down` matrices. The legacy combined score remains:

```text
episode_group_matrix_raw = max(pulse, step_up, step_down)
```

`matrix_raw` remains the mean of the episode-group combined score for exact
legacy compatibility. Because mean and max do not commute,
`matrix_template_max = max(matrix_pulse, matrix_step_up, matrix_step_down)` is
stored separately and is not substituted for legacy `matrix_raw`.

All 12 regenerated artifacts reproduce the corresponding immutable legacy
`matrix_raw` exactly.

Implementation:

- [Directional scorer](../../../event_sae/scoring/score_matrix.py)
- [Task and recurrence ranking](../../../event_sae/scoring/task_phase_ranking.py)
- [Cross-source report](../../../event_sae/scoring/feature_activation_grid.py)

## 4. Instruction-local discovery

For instruction `i`, phase `p`, feature `f`, and template `q`:

```text
Δ(i,p,f,q)
= score(i,p,f,q)
  - max score(i,other observed phase,f,q)
```

Candidate membership requires only:

```text
W5 Δ > 0
```

W4 is sensitivity metadata. Window-mean and task-mean ranks are nuisance flags.
None is an exclusion gate. Every positive candidate is serialized; Top-10 is a
display limit only.

Each candidate records raw W5 score, margin rank/percentile, episode
support/coverage, W4 direction consistency, control overlap, eligibility, and
source/view identity.

`N/A` means the phase or contrast is unavailable. `—` means the contrast exists
but no positive candidate satisfies the cell definition.

## 5. State-transition pairs

For an earlier ON phase `a` and later OFF phase `b`, a same-feature pair uses:

```text
pair_score(i,a→b,f)
= min(
    Δ(i,a,f,step_up),
    Δ(i,b,f,step_down)
  )
```

Both W5 component margins must be positive. Same-episode support is the
intersection of episodes with comparable positive ON and OFF evidence.
Pair-support fraction is recorded but is not a discovery gate.

Pulse candidates remain separate because a transient response does not imply a
persistent state.

## 6. Family and global recurrence

Families are:

- Drawer: left and right;
- Object placement: beer, bread, and pizza cutter.

Raw margins are not averaged across tasks. Recurrence uses support counts and
within-task percentiles.

| Level | Strict | Relaxed |
| --- | --- | --- |
| Drawer | 2/2 | 2/2 |
| Object | 3/3 | 2/3 |
| Global | 5/5 same phase/direction | at least 3/5 and both families |

Every row separates `support/eligible`, `eligible/expected`, and
`support/expected`; `2/2` observed support is therefore not presented as a
three-task result. Global percentile balances the two family medians. Missing
phases produce `N/A`, not failure.

## 7. Sparse persistence

Displayed coarse pairs are rescanned once per source over lossless sparse TopK.
The check measures pre/post-ON activation, interval prevalence, post-OFF
activation, and episode-level full-pattern repetition.

The status vocabulary is:

| Status | Meaning |
| --- | --- |
| confirmed | full ON–dwell–OFF pattern repeats above the configured threshold |
| partial | some full patterns or two-of-three components recur |
| boundary-only | directional boundary score exists without trace persistence |
| insufficient support | ordered anchors or comparable episodes are inadequate |

Persistence is diagnostic support, not a candidate-membership gate. Only
selected features need dense recollection after this sparse screen.

## 8. Result inventory

| Item | Count |
| --- | ---: |
| Sources / views | 3 / 6 |
| New directional score artifacts | 12 |
| Exact legacy reproductions | 12/12 |
| All phase candidates | 6,054 |
| Displayed phase candidates | 2,755 |
| All state-pair candidates | 442 |
| Trace-measured state-pair candidates | 366 |
| Candidate JSONL records | 13,975 |
| Cross-source probe candidates | 2,842 |

Sparse status:

| Status | Candidates |
| --- | ---: |
| confirmed | 1 |
| partial | 20 |
| boundary-only | 260 |
| insufficient support | 85 |

The only confirmed displayed pair is E3 pizza-cutter F875
`transport→terminal`, with a full pattern in 6/11 anchored episodes. It is
source-local.

Important partial traces:

| Source / cell | Pair | Feature | Full pattern |
| --- | --- | ---: | ---: |
| E3 bread | transport→terminal | F533 | 4/12 |
| E4 bread | transport→terminal | F533 | 5/11 |
| E3 bread | transport→terminal | F685 | 4/12 |
| Oracle right drawer | reach→transport | F892 | 3/10 |
| Oracle beer | reach→grasp | F1326† | 2/5 |

`†` marks Oracle phase coverage below 0.3.

## 9. Recurrence findings

### 9.1 Phase direction

- Oracle global-strict: reach pulse F875, reach step-up F892, and reach
  step-down F1109.
- E3 global-strict: grasp step-up F1155 and grasp step-down F703.
- E4 has no 5/5 strict row because relevant phases are available in fewer
  cells; this is availability, not a negative result.
- Grasp step-down F1389 and transport step-up F432 are useful relaxed,
  control-clean directions across the V12 views.

### 9.2 State pairs

- Oracle has no global-strict or global-relaxed state pair.
- E3 has global-relaxed `grasp→transport` candidates F909, F1120, F64, and
  F1397; availability is 4/5 and support is 3/4.
- E4 has no global pair because drawer transitions are incomplete.
- E3/E4 share 77 displayed state-pair directions; 15 also match Oracle exactly,
  and 7 are exact across all three sources. Only 3 of those have Oracle
  coverage at least 0.3.

These recurrence counts are discovery summaries. They are not independent
replications because the sources reuse episodes, checkpoint coordinates, and
related annotation inputs.

## 10. Hooked-SR probe design

Recommended order:

1. F1155 grasp step-up, with terminal step-down as an OFF-side check;
2. F533 bread transport-state pair;
3. F64 grasp step-up as cross-source sensitivity;
4. F506 terminal step-up and F1451 terminal step-down;
5. F892 reach step-up as an Oracle-led exploratory probe.

For each feature:

- zero-out only in the hypothesized phase;
- sweep positive and negative scaling;
- inject during a matched OFF phase;
- include matched window/task-mean nuisance controls;
- include a random alive control;
- separate candidate-selection episodes from held-out evaluation seeds.

The sparse screen should be followed by dense trace collection only for the
shortlist. Behavioral claims require baseline versus reconstruction-hooked and
steered success rates.

## 11. Strict and historical appendix

The directional ranking is deliberately a discovery analysis. Existing strict
tests answer a different question and remain valid as an appendix:

| Analysis | Strict family | Corrected support |
| --- | --- | ---: |
| Oracle five-cell, 3-SAE decoder matched | 25 task×phase hypotheses | 0/25 |
| V12 E0–E4, 3-SAE decoder matched | 112 task×phase hypotheses | 0/112 |
| V12 checkpoint-local sensitivity | 336 tests | 0/336 |
| Centroid-plurality E4 | 11 task×phase hypotheses | 0/11 |
| Historical V11 45-cell | no max-T/outer correction | N/A |

These tests used W4/W5 conjunction, task-stratified permutation, feature max-T,
decoder matching where applicable, and outer correction. They should not be
used as an exclusion gate for a held-out causal probe.

Historical V11 remains a condition/coverage/checkpoint sensitivity source.
Detailed 45-cell counts and Q1–Q4 geometry are preserved in the
[historical controlled result](anchor_view_controlled_ablation_results.md).

The 10k checkpoint has strong reconstruction, but neither strict inference nor
directional recurrence proves it is the best phase SAE.

## 12. Artifacts and integrity

Primary output root:

```text
logs/groot_n15/experiments/anchor_view_controlled_ablation_v1/
  analysis/oracle_five_cell_vs_v12_e3_e4_10k_directional_w5_v1/
```

It contains `summary.json`, `candidates.jsonl`, generated `report.md`, and 12
non-overwriting score artifacts under `scores/`.

The run verified:

- legacy combined W4/W5 score reproduction 12/12;
- input hash audit 37/37;
- implementation hash audit 4/4;
- checkpoint and TopK manifest hashes unchanged;
- Oracle and V12 raw scores not pooled;
- missing phase `N/A` distinct from no candidate `—`.

Previous V11, strict V12, Oracle, centroid, SAE, TopK, and annotation artifacts
remain read-only.

## 13. Confound audit

| Gate | Status | Evidence |
| --- | --- | --- |
| Length | **FAIL** | W5 is fixed, but episode length and phase dwell are unmatched. |
| Task identity | PASS | Every contrast is within one exact instruction. |
| Instruction balance | PASS | Ranking never pools instructions. |
| In-sample rescue | N/A | No detector or intervention has been evaluated. |
| Rollout pooling | PASS | Events are averaged within episode-phase before equal episode weighting. |
| Phase / dwell | **FAIL** | Retry, dwell, progress, and transition position are unmatched. |
| Oracle–V12 clock | **FAIL** | Oracle uses environment-step entries; V12 uses waypoint anchors at another scale. |
| Observation ≠ causation | PASS | Results are candidate rankings only. |
| Scene-local ≠ general | **FAIL** | There is no held-out scene or rollout replication. |

Claim strength: **diagnostic evidence**.

- A positive directional margin proves phase semantics: **confounded — 판정 보류**.
- Sparse ON–dwell–OFF evidence proves causal control: **confounded — 판정 보류**.
- Oracle–V12 rank differences prove annotation superiority:
  **confounded — 판정 보류**.
- A feature improves Hooked-SR before a held-out intervention:
  **confounded — 판정 보류**.
