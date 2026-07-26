# GR00T Event-Grounded SAE — Anchor/view 통제 실험 실행 규약

[GR00T 현황](../README.md) ·
[Automatic 결과 snapshot](../reports/anchor_view_controlled_ablation_results.md) ·
[V12r3 interactive snapshot](../reports/v12r3_annotation_rerun.md) ·
[별도 full Batch rerun](../reports/batch_annotation_rerun.md)

> 이 실행 규약은 V11 cluster-level annotation으로 생성된 기존 Gate 0–6 artifact
> 계약이다. Annotation unit과 schema가 달라진 V12r3 interactive snapshot과
> 별도 full Batch rerun은 각자 분리된 실행 기록에서 관리하며 기존 protocol과
> 결과를 덮어쓰지 않는다.

- 작성일: 2026-07-24
- 갱신일: 2026-07-25
- 대상: GR00T N1.5, RoboCasa PQ3, physical layer 15
- 상태: **Gate 6 automatic artifact·final mechanical audit 완료,
  human review와 semantic 판정 대기**
- 범위: 기존 150 episodes를 이용한 waypoint·clustering·annotation ablation
- 실행 profile:
  [`configs/groot/anchor_view_controlled_ablation_v1.json`](../../../configs/groot/anchor_view_controlled_ablation_v1.json)

이 문서는 과거 artifact의 시간 순서별 비교를 반복하는 계획이 아니다. 논문과
일치시키기로 한 기본 descriptor와 ranking 조건은 고정하고, 이 프로젝트에서
변경한 네 가지 선택만 한 번에 하나씩 비교한다.

1. Annotation image: LEFT vs synchronized 3-view
2. Clustering vision: LEFT vs synchronized 3-view
3. Event proposal: position-only vs position + gripper-closing peak
4. AWE position frame: relative vs absolute EEF

Primary 설정은 실행 전에 고정한다. AWE threshold와 clustering distance를 새
실험의 sweep 축으로 다시 사용하지 않는다. Coverage만 `0.3/0.4/0.5`에서
downstream sensitivity를 확인한다.

## 0. 현재 구현 연결 상태

현재 코드로 다섯 semantic condition의 waypoint, media, descriptor, clustering,
annotation, review와
Stage 4를 실행할 수 있다. 단, profile을 한 번에 소비하는 orchestration
command는 아직 없으므로 각 단계는 아래 script에 profile의 고정값을 명시해
실행한다. Script 기본값이 실험 계약과 다른 곳이 있어 생략 가능한 인자는 없다.

| Gate | 구현 진입점 | 실행 계약 |
| --- | --- | --- |
| 1 | `scripts/extract_keyframes.py` | `pos_only`/`pos_gripper_close`, `rel`/`abs`, exact DP `η=0.05` |
| 2 | `scripts/extract_keyframe_media.py`, `scripts/build_event_features.py` | gripper supplement 조립, LEFT/3-view feature 생성 |
| 3 | `scripts/cluster_events.py cluster` | `balanced`, `d=0.18`, coverage `0.3`, exemplar 5; `sweep` 사용 금지 |
| 4–5 | `scripts/annotate_clusters.py`, `scripts/review_clusters.py` | V11, task-family phase, view별 layout, blind two-stage review |
| 6 | `scripts/review_clusters.py phase-groups`, `scripts/score_cluster_features.py`, `scripts/build_feature_rankings.py` | coverage별 phase group·score·ranking 재생성; reviewed가 계획 계약, provisional automatic이 현재 실행 |

실행 root는
`logs/groot_n15/experiments/anchor_view_controlled_ablation_v1/`로 고정한다.
사전 Gate 0 계획에는 다음 두 결정이 필요했다.

1. 보존된 SAE top-k 세 후보 중 Stage 4에 사용할 **한 개**를 선택해 profile의
   `stage4.selected_topk_run_dir`에 기록한다.
2. Annotation pilot/retry shard를 중복 없이 합치는 비파괴 merge 절차를
   구현하거나 독립 audit 가능한 명령으로 고정한다.

현재 계획 config는 이 단일 checkpoint 계약과
`selected_topk_run_dir=null`을 그대로 보존한다. 실제 automatic 실행에서는
사용자가 세 checkpoint를 sensitivity 축으로 승인했고, annotation merge와
exact-join audit 후 human review를 의도적으로 미룬 채 Gate 6까지 진행했다.
따라서 실제 read-only 결과의 scope와 deviation은 profile이 아니라
`experiment_manifest.json`과 `audits/final_audit.json`을 source of truth로
사용한다. 현재 결과는 core human-reviewed 완료가 아니다.

R-PG의 1,217/1,227 multi-source 재사용은 현재 CLI가 직접 지원하지 않는 선택적
최적화다. 실험의 실행 가능성을 막지 않으며, 정식 경로는 R-P 913개를 재사용하고
새 gripper-only 314개를 materialize하는 것이다.

## 1. 연구 질문

이 실험은 다음 네 질문에만 답한다.

| ID | 질문 |
| --- | --- |
| Q1 | 같은 cluster와 exemplar에서 synchronized 3-view가 LEFT-only보다 visual annotation을 더 일관되게 만드는가? |
| Q2 | annotation 조건을 3-view로 고정했을 때, 3-view clustering이 LEFT clustering과 다른 recurring event partition을 만드는가? |
| Q3 | relative EEF AWE에서 gripper-closing peak를 추가하면 position-only가 놓친 interaction event가 recurring cluster에 포함되는가? |
| Q4 | gripper-closing peak를 포함한 상태에서 relative와 absolute EEF AWE가 어떤 event·cluster 차이를 만드는가? |

Q1은 annotation 효과, Q2는 clustering representation 효과, Q3은 event proposal
효과, Q4는 AWE position-frame 효과다. 전체 조합의 interaction이나 어느
설정이 다른 scene에서도 일반적으로 우월한지는 이 실험의 질문이 아니다.

## 2. 의미 기반 실험 행렬

| 조건 ID | AWE anchor | Clustering view | Annotation view | 직접 비교 |
| --- | --- | --- | --- | --- |
| `rel_pos__cluster_left__label_left` | rel position-only | LEFT | LEFT | 기준 |
| `rel_pos__cluster_left__label_multiview` | rel position-only | LEFT | 3-view | annotation view |
| `rel_pos__cluster_multiview__label_multiview` | rel position-only | 3-view | 3-view | clustering view |
| `rel_pos_gripper__cluster_multiview__label_multiview` | rel position + gripper close | 3-view | 3-view | gripper anchor |
| `abs_pos_gripper__cluster_multiview__label_multiview` | abs position + gripper close | 3-view | 3-view | rel↔abs AWE |

비교는 표에서 인접한 두 조건 사이에서만 해석한다. 첫 조건과 마지막 조건을 직접 비교해
3-view나 gripper의 효과라고 부르지 않는다.

실제 output directory와 UI payload는 다음 alias를 사용한다.

| Alias | Output condition ID |
| --- | --- |
| E0 | `e0_rel_pos_cluster_left_label_left` |
| E1 | `e1_rel_pos_cluster_left_label_multiview` |
| E2 | `e2_rel_pos_cluster_multiview_label_multiview` |
| E3 | `e3_rel_pos_gripper_cluster_multiview_label_multiview` |
| E4 | `e4_abs_pos_gripper_cluster_multiview_label_multiview` |

### 2.1 고유 계산 단위

P0의 두 annotation view는 clustering partition과 representative sample ID·순서를
정확히 공유한다. 따라서 필요한 고유 단위는 다음과 같다.

- Anchor set: 3개
- Clustering partition: 4개
- Annotation condition: 5개

```text
Anchor sets
├─ R-P  : rel position-only
├─ R-PG : rel position + gripper close
└─ A-PG : abs position + gripper close

Partitions
├─ P0 : R-P  + LEFT clustering      → LEFT label, multiview label
├─ P1 : R-P  + 3-view clustering    → multiview label
├─ P2 : R-PG + 3-view clustering    → multiview label
└─ P3 : A-PG + 3-view clustering    → multiview label
```

## 3. 입력 데이터와 예상 규모

모든 조건은 같은 source rollout을 사용한다.

| 항목 | 값 |
| --- | ---: |
| Task families | 2 |
| Instruction/scene cells | 5 |
| Episodes/cell | 30 |
| Total episodes | 150 |
| Policy records | 12,041 |

Success/failure, oracle event, simulator predicate는 waypoint 생성, clustering,
cluster 선택과 Gemini prompt에 사용하지 않는다.

### 3.1 Canonical 입력 locator

아래 경로는 현재 정리된 `logs/groot_n15/` 구조의 source of truth다. 새
experiment root에는 원본을 복사하지 않고 path와 SHA-256을
`inputs/source_manifest.json`에 기록한다.

| 입력 | Canonical path |
| --- | --- |
| Relative trajectory | `stage2_waypoints/relative_position/trajectory_records.jsonl` |
| Absolute trajectory | `stage2_waypoints/absolute_position/trajectory_records.jsonl` |
| R-P `η=0.05` summary | `stage2_waypoints/relative_position/dp_pos_only_err0p05/waypoint_summary.json` |
| Historical R-P features | `stage3_event_descriptors/relative_position/siglip_base_patch16_224_mean5_v1/` |
| A-PG current run | `experiments/v9_abs_position_gripper_3view_action_phase_v1/` |
| Preserved Stage 4 top-k | `stage4_feature_ranking/{l15_sae1p2k_exec5_mean4_top96_v1,l15_sae10k_exec5_mean4_top96_v1,l15_sae10k_bs8192_exec5_mean4_top96_v1}/topk/` |

표의 상대 경로 기준점은 `logs/groot_n15/`이다. 구 legacy 이름은 relocation
resolver의 호환 입력일 뿐 새 manifest에 쓰지 않는다.

### 3.2 Anchor 수 사전 계약

기존 `η=0.05` exact-DP artifact와 동일한 closing detector를 결합하면 다음
event 수를 재현해야 한다.

| Anchor set | Position | Raw closing peaks | Position과 중복 | Dedup 후 events |
| --- | ---: | ---: | ---: | ---: |
| R-P | 913 | — | — | **913** |
| R-PG | 913 | 379 | 65 | **1,227** |
| A-PG | 967 | 379 | 68 | **1,278** |

R-PG의 `1,227 = 913 + 379 - 65`는 기존 relative position indices와 현재 조건의
closing indices를 record 거리 `±2`로 deterministic merge해 계산한 preflight
기대값이다. 새 R-PG summary가 이 수와 다르면 clustering으로 진행하지 않는다.

### 3.3 Reuse-first 계약

새 experiment root는 기존 artifact를 복사해 새 결과인 것처럼 취급하지 않는다.
입력 identity와 변환 계약이 같은 artifact는 source hash로 참조하고, 달라진
단계부터만 새로 materialize한다.

| 단위 | 기존 artifact | 재사용 범위 | 새 작업 |
| --- | --- | --- | --- |
| R-P waypoint | rel `η=0.05`, 913 events | summary와 anchor indices 재사용 | source/hash audit |
| R-P media/vision | LEFT와 synchronized 3-view, 913 events | frame bundle과 frozen SigLIP embedding 재사용 | 5D state를 붙인 새 descriptor record |
| R-PG waypoint | 없음 | rel position 913개와 기존 closing indices 재사용 | deterministic union summary 생성 |
| R-PG media/vision | R-P 913 events | **913개 exact episode-step** 재사용 | gripper-only **314개** frame bundle·embedding 생성 |
| A-PG Stage 2–3A | current condition, 1,278 events, 3-view C0 `d=0.18` | profile/hash 일치 시 waypoint·media·feature·P3 partition 재사용 | 새 annotation/review |
| P0/P1 partition | historical relative-position conditions | vision embedding만 재사용 | 통제된 5D ABS state로 재clustering |
| P2 partition | 없음 | upstream media/vision 대부분 재사용 | descriptor·clustering 생성 |
| Stage 4 sparse activation | 보존된 top-k shard 3종 | 계획: 선택한 한 checkpoint의 shard 재사용 | partition·coverage별 reviewed phase-group score와 ranking 재생성 |
| Existing annotation | historical prompts | historical provenance로만 보존 | 최종 prompt/config로 semantic conditions 재호출 |

실제 automatic 실행은 위 단일 checkpoint 계획에서 벗어나 세 shard를 모두
sensitivity 축으로 사용했다. 각 coverage×checkpoint의 score와 ranking은 별도
artifact이며 이 deviation은 manifest에 기록돼 있다.

아래 overlap은 계산량 상한을 확인한 진단값이며 현재 실행 계약은 아니다.
`reuse-vision` provider는 단일 source feature set의 모든 row가 정확히 한 번
소비되기를 요구하므로 R-P와 A-PG를 동시에 sparse join할 수 없다.

```text
R-PG ∩ (R-P ∪ A-PG) = 1,217
R-PG ∖ (R-P ∪ A-PG) = 10
```

향후 multi-source sparse reuse utility와 독립 audit을 추가한 뒤에만
1,217개 재사용/10개 신규 경로로 바꿀 수 있다. 그 전에는 913개 재사용/314개
신규를 사용한다.

같은 episode-step이라도 sample ID 문자열만 보고 재사용하지 않는다. Source
trajectory hash, selected frame indices, decoded image identity, vision model/revision,
frame pooling이 모두 일치해야 vision embedding을 재사용한다. State vector,
anchor source와 progress 같은 record metadata는 새 experiment contract에 맞춰
다시 작성한다.

## 4. 전 조건 공통 고정값

### 4.1 Waypoint와 gripper

| 항목 | 고정값 |
| --- | --- |
| AWE implementation | `exact_pos_only` |
| AWE error threshold | `η=0.05` |
| Gripper aperture | `sum(abs(gripper_qpos))` |
| Normalization | instruction/scene cell 전체 source episode min–max |
| Closing height | `0.08` |
| Closing prominence | `0.04` |
| Minimum peak distance | 3 policy records |
| Position/closing dedup | `±2` policy records |
| Opening peaks | 제외 |

Relative/absolute는 AWE가 position waypoint를 고를 때 사용하는 좌표만
변경한다. 공정한 Q4 비교를 위해 downstream physical-state descriptor는 모든
조건에서 동일한 absolute representation을 사용한다.

### 4.2 Event descriptor와 clustering

| 항목 | 고정값 |
| --- | --- |
| Vision encoder | frozen `google/siglip-base-patch16-224` |
| Frames | anchor 중심 5 frames |
| Per-view pooling | mean-5 후 L2 normalization |
| 3-view fusion | LEFT/RIGHT/WRIST equal-view concat |
| State | `[abs xyz, normalized_aperture, aperture_delta]` |
| Progress | episode-relative scalar |
| Descriptor weights | vision/state/progress = `1.0/0.5/0.4` |
| Block normalization | `balanced` |
| Scope | exact `task_description` local |
| Algorithm | agglomerative, cosine, average linkage |
| Distance | **`d=0.18` 고정** |
| Exemplars | cluster당 최대 5, centroid-nearest deterministic selection |

P0의 LEFT partition에도 5D absolute state와 progress를 동일하게 넣는다.
P1–P3와 달라지는 것은 표에 명시한 anchor set 또는 vision block뿐이다.

### 4.3 Coverage와 downstream ranking

| 항목 | 계약 |
| --- | --- |
| Annotation superset | `episode_coverage≥0.3` |
| Coverage sensitivity | `0.3/0.4/0.5` |
| Re-clustering per coverage | 하지 않음 |
| Gemini recall per coverage | 하지 않음 |
| Primary Stage 4 window | `W=5` 고정 |
| Candidate size | top-5 고정 |

Coverage `0.3/0.4/0.5`는 각 partition의 같은 raw cluster와 annotation
superset을 filter한다. AWE `η`, clustering `d`, descriptor weight, Stage 4
window는 이번 실험의 sweep 대상이 아니다.

Coverage `c`의 exact contract는 다음과 같다.

```text
S(c) = {raw cluster ID | episode_coverage >= c}
     = filtered_finalized_annotations의 cluster ID
     = phase group source_cluster_ids의 union

S(0.5) ⊆ S(0.4) ⊆ S(0.3)
```

Threshold마다 `S(c)`의 annotation을 `(task_description, phase)`로 다시 병합한다.
그 phase-group assignment를 입력으로 coverage×checkpoint별 score와 ranking을
다시 계산한다. 단순히 이전 coverage의 score-matrix row만 거르는 절차가 아니다.

### 4.4 CLI에서 반드시 명시할 값

현재 CLI 기본값과 이 계획의 계약은 완전히 같지 않다. 실행 manifest에는
명령문과 아래 인자를 그대로 저장한다.

| 단계 | 필수 명시값 |
| --- | --- |
| Clustering | `cluster --block-normalization balanced --distance-threshold 0.18 --min-coverage 0.3 --vision-weight 1.0 --state-weight 0.5 --progress-weight 0.4 --num-exemplars 5 --expected-samples {913,1227,1278}` |
| Annotation | `--phase-scheme robocasa_action --model gemini-3.1-pro-preview --temperature 0.0 --min-episode-coverage 0.3 --media-layout <condition layout>` |
| Review | condition마다 input/output path와 `--condition-id`, `--awe-anchor`, `--clustering-view`, `--annotation-view`, `--media-layout`, `--block-normalization balanced`를 명시 |
| Score | `--window-size 5 --event-step-scale 5`; GR00T policy-record index를 executed env step으로 변환 |
| Ranking | coverage별 phase group을 이미 filter했으므로 `--top-k 5 --min-coverage 0.0` |

`review_clusters.py audit`의 기본 expected count는 현재 canonical condition
전용이므로 나머지 네 조건에서는 각 partition의 실제 event/annotation 수를
반드시 넘긴다. 현재 canonical profile과 historical artifact 경로가 기본값인
review/triptych 명령도 이 실험에서는 기본값으로 실행하지 않는다.

## 5. Annotation prompt와 generation 계약

### 5.1 Task-family prompt

Prompt는 shared template 위에 task-family별 vocabulary·decision rule을
주입한다.

| Family | 적용 instruction | Allowed phase |
| --- | --- | --- |
| Pick-place | beer, bread, pizza cutter | `reach-to-object`, `grasp`, `transport`, `place`, `insert-settle`, `terminal`, `wrong-grasp` |
| Drawer | left drawer, right drawer | `reach-to-handle`, `grasp-handle`, `pull`, `push-back`, `disengage`, `wrong-grasp`, `open-done` |

Instruction마다 별도 taxonomy를 만들지는 않는다. 실제 instruction 문장은
prompt에 그대로 넣고, family resolver가 선택한 phase만 structured-output
schema에서 허용한다. 매칭되지 않는 task는 generic taxonomy로 fallback하지 않고
실패해야 한다.

### 5.2 LEFT/3-view prompt parity

P0의 LEFT-label/multiview-label 조건은 동일 cluster에 대한 paired annotation
실험이다. 두 prompt는 다음
항목만 달라야 한다.

- LEFT: 각 image가 한 시점의 `robot0_agentview_left`
- 3-view: 각 image가 같은 시점의 `LEFT | RIGHT | WRIST` triptych
- 3-view의 세 panel은 시간 순서가 아니라 동시 관측이라는 설명

Contact, held state, support, target motion, drawer direction과 proximity에 관한
판단 규칙은 shared section에서 byte-identical하게 유지한다. Layout마다 서로
다른 semantic decision rule을 넣지 않는다. 5-frame clip에서는 frame 3이
label 대상 temporal center임을 명시한다.

현재 V11 text를 변경한다면 같은 version 이름을 재사용하지 않는다. 최종 prompt
text가 고정된 뒤 새 version, exact text와 SHA-256을 manifest에 기록한다.

### 5.3 Generation

| 항목 | 고정값 |
| --- | --- |
| Model | `gemini-3.1-pro-preview` |
| Temperature | `0.0` |
| Response MIME | `application/json` |
| Output | exact `{phrase, phase}` object |
| Hidden inputs | progress, source step, gripper qpos, anchor source, success/failure, oracle/simulator predicate |

모든 row에 model, prompt version/text/hash, generation config, JSON schema,
media layout, representative sample/frame paths와 raw response를 저장한다. API
error나 parse error는 기존 row를 덮어쓰지 않고 별도 retry attempt에 기록한다.
첫 full request 이후에는 조건 사이에서 model이나 generation config를 바꾸지
않는다.

### 5.4 Annotation 호출량

`coverage≥0.3`에서 partition별 selected cluster 수를 각각
`N(P0),…,N(P3)`라 하면 full annotation 호출 수는 다음과 같다.

```text
2×N(P0) + N(P1) + N(P2) + N(P3)
```

P0는 같은 cluster를 LEFT와 3-view로 각각 한 번씩 호출한다. 과거 partition의
26/26/20 selected-cluster 규모와 아직 실행하지 않은 P2를 고려하면 약
120 calls 전후가 예상되지만, 새 통제 descriptor의 Gate 3 결과를 얻기 전에는
정확한 budget으로 확정하지 않는다. Retry는 이 수에 포함하지 않고 별도로
보고한다.

## 6. 실행 순서

### Gate 0 — Profile과 prompt freeze 계획

1. 다섯 semantic condition profile과 공통 source hash를 기록한다.
2. task-family별 LEFT/3-view rendered prompt를 검토한다.
3. P0의 두 annotation view에서 shared semantic section이 동일한지 검사한다.
4. 최종 prompt version과 generation config를 동결한다.
5. Stage 4 top-k 세 후보 중 하나를 선택해 `selected_topk_run_dir`에 고정한다.
6. Annotation attempt merge 명령과 중복·누락 audit을 고정한다.
7. 기존 historical artifact를 overwrite하지 않는 새 experiment root를 만든다.

현재 실행은 5번을 단일 checkpoint 선택 대신 승인된 3-checkpoint sensitivity로
대체했고 pilot과 human review를 미뤘다. 이는 계획을 소급 수정하지 않고
manifest의 operational deviation으로 보존한다.

### Gate 1 — Waypoint materialization

1. R-P를 기존 exact relative summary에서 검증한다.
2. R-PG를 relative position과 closing peak union으로 생성한다.
3. A-PG를 현재 조건의 기존 source와 hash 검증 후 재사용하거나 새 root에 참조한다.
4. 각각 913/1,227/1,278 events와 `η=0.05` geometric contract를 audit한다.

### Gate 2 — Media와 descriptor

1. 기존 R-P/A-PG media와 vision embedding의 reuse eligibility를 hash로 검사한다.
2. R-PG는 R-P 913개 media/vision을 단일 source로 exact reuse하고,
   gripper-only 314개를 frame packaging·vision encoding한다.
3. 각 anchor에 LEFT/RIGHT/WRIST의 동일 sample·frame metadata를 exact join한다.
4. P0용 LEFT vision과 P1–P3용 synchronized 3-view vision을 구성한다.
5. 모든 partition에 같은 5D absolute state와 progress를 새 descriptor
   metadata로 결합한다.
6. P0의 LEFT frames와 triptych frames가 같은 sample ID와 frame index를
   가리키는지 검사한다.

### Gate 3 — Primary clustering

1. P0–P3를 `d=0.18`, balanced C0로 한 번씩 clustering한다.
2. sample assignment가 913/913/1,227/1,278 events를 빠짐없이 덮는지 검사한다.
3. `coverage≥0.3` cluster와 deterministic exemplar를 materialize한다.
4. P0의 두 annotation view가 같은 cluster/exemplar manifest를 참조하는지 hash로 확인한다.

### Gate 4 — Annotation pilot

1. 각 task family에서 deterministic하게 고른 cluster로 schema·media ordering을
   확인한다.
2. Pilot은 최종 prompt/config를 사용하며 cluster ID shard로 저장한다.
3. Prompt를 바꾸지 않았다면 나머지 cluster만 실행하고, Gate 0에서 고정한
   비파괴 merge/audit으로 하나의 frozen annotation superset을 만든다.
4. Prompt를 바꿨다면 pilot은 provisional로 보존하고 전 조건을 새 version으로
   다시 시작한다.

### Gate 5 — Full annotation과 human review

1. 다섯 semantic condition의 `coverage≥0.3` cluster를 모두 annotate한다.
2. Human reviewer는 Gemini 결과를 보기 전에 independent phase를 기록한다.
3. Human reference는 task instruction과 synchronized 3-view exemplar를 사용하고
   hidden simulator state를 보지 않는다.
4. 다음 필드를 cluster마다 기록한다.

```text
human_phrase
human_phase
mixed_cluster
visually_insufficient
phrase_phase_consistent
phase_corrected
notes
adjudication_notes
reviewer
blind_reviewed_at
reviewed_at
review_stage
verdict
```

P0의 두 annotation view는 같은 human reference에 대해 paired 평가한다.
Mixed 또는 visually
insufficient cluster는 phase exact-match 분모와 분리해 별도 비율로 보고한다.

Review UI는 independent phrase/phase와 uncertainty flag를 atomic하게 먼저
저장한 뒤에만 Gemini phrase/phase를 공개한다. Blind 단계의 browser payload에는
rollout outcome뿐 아니라 episode identity, waypoint rank/step, progress와 원본
frame filename도 포함하지 않는다. `annotation_view`와 human reference
`media_layout`은 별도 provenance로 기록해, P0의 LEFT annotation과 공통 3-view
human reference가 혼동되지 않게 한다.

### Gate 6 — Coverage와 Stage 4

아래는 human-reviewed core 실행 계약이다. 현재 materialized 결과는 같은
mechanical 경로에 `user_authorized_assumed_review` 기반 automatic provisional
phase를 넣은 진단 실행이며 semantic 완료로 승격하지 않는다.

1. Reviewed annotation superset을 coverage `0.3/0.4/0.5`로 filter한다.
2. Coverage마다 raw cluster ID를 유지한 채
   `(task_description, reviewed_phase)` phase group과 assignment를 별도로 만든다.
3. Gate 0에서 선택한 SAE의 sparse top-k shard만 재사용한다. 기존 automatic
   raw-cluster score matrix는 reviewed phase-group row와 assignment가 다르므로
   재사용하지 않는다.
4. Partition×coverage마다 score matrix를 `W=5`, `event-step-scale=5`로 새로
   계산한다.
5. Ranking은 이미 coverage-filtered phase group에 대해 top-5,
   `min-coverage=0.0`으로 생성한다. Merged phase-group coverage에 다시
   `0.3/0.4/0.5`를 적용하지 않는다.
6. SAE를 재학습하거나 이 통제 실험 결과를 보고 checkpoint를 다시 선택하지 않는다.

### Gate 7 — Final audit와 report

1. Source, waypoint, media, feature, partition, annotation, review와 score hash를
   한 manifest에 연결한다.
2. 각 비교의 실제 n과 제외 사유를 기록한다.
3. Cluster-weighted와 event-weighted 수치를 분리한다.
4. Task family와 instruction cell별 수치를 함께 제시한다.
5. 관측된 차이, human-reviewed semantic 결과와 downstream feature ranking을
   인과 주장과 구분한다.

## 7. 비교 지표

### 7.1 공통 기계적 지표

- Events와 anchor-source count
- Raw/selected cluster count
- Cluster size median/max와 singleton event fraction
- Instruction별 episode coverage
- Exact join 누락·중복 수
- 공통 sample에서 ARI/NMI

### 7.2 Q1 — Annotation view

P0의 두 annotation view는 동일 cluster, exemplar sample과 frame index를 사용한다.

- Human phase exact match의 paired transition table
- Phrase–phase consistency
- LEFT와 3-view annotation agreement
- Mixed/visually-insufficient cluster 수
- Task-family별 numerator/denominator

단순 machine-valid JSON 비율은 annotation 품질 지표로 사용하지 않는다.

### 7.3 Q2 — Clustering view

- P0/P1의 913 common events 전체에 대한 ARI/NMI
- Raw/selected cluster와 singleton 차이
- Human-reviewed mixed-cluster rate
- Reviewed phase coverage와 instruction별 분포
- Stage 4 top-5 overlap/rank correlation

Cluster 수가 많거나 phase 종류가 많다는 사실만으로 더 좋은 clustering이라고
판정하지 않는다.

### 7.4 Q3 — Gripper anchor

- R-PG의 `position`, `gripper_close`, `both` source count
- Gripper-only events의 recurring-cluster 진입률
- Human-reviewed grasp/grasp-handle 및 interaction phase의 cluster/event coverage
- P1/P2의 공통 position anchors에 대한 partition 변화
- 추가 events가 만든 mixed-cluster 또는 singleton 증가

Closing peak는 semantic grasp 정답이 아니다. Human review 전에는 “grasp를
회수했다”고 쓰지 않는다.

### 7.5 Q4 — Relative/absolute AWE

- R-PG/A-PG event 수와 common anchor 수
- 공통 anchors의 ARI/NMI
- frame별로만 존재하는 anchor의 task/instruction 분포
- Human-reviewed mixed rate와 phase coverage
- Stage 4 top-5 overlap/rank correlation

Relative와 absolute의 event set이 다르므로 전체 partition ARI 하나만으로
좌표계 효과를 요약하지 않는다.

## 8. Artifact와 provenance 계약

계획과 실제 materialized 경로를 합친 root:

```text
logs/groot_n15/experiments/anchor_view_controlled_ablation_v1/
├── experiment_manifest.json
├── inputs/
│   └── source_manifest.json
├── waypoints/
│   ├── r_pos/
│   ├── r_pos_gripper/
│   └── a_pos_gripper/
├── media/
│   ├── r_pos/{left,right,wrist}/
│   ├── r_pos_gripper/{left,right,wrist}/
│   └── a_pos_gripper/{left,right,wrist}/
├── features/
│   ├── r_pos/
│   ├── r_pos_gripper/
│   └── a_pos_gripper/
├── partitions/
│   ├── p0_r_pos_left/
│   ├── p1_r_pos_3view/
│   ├── p2_r_pos_gripper_3view/
│   └── p3_a_pos_gripper_3view/
├── annotation_media/
│   ├── p0_left/
│   ├── p0_multiview/
│   ├── p1_multiview/
│   ├── p2_multiview/
│   └── p3_multiview/
├── annotations/
│   └── <e0...e4 condition_id>/
│       ├── attempt*.jsonl
│       ├── frozen_annotations.jsonl
│       └── provisional_finalized_annotations.jsonl
├── reviews/
├── stage4/
│   └── <condition_id>/<cov0p3|cov0p4|cov0p5>/
│       ├── filtered_finalized_annotations.jsonl
│       ├── phase_groups/
│       └── <checkpoint_id>/
│           ├── event_feature_scores.pt
│           └── rankings/
└── audits/
    └── final_audit.json
```

각 directory는 source path/hash, code commit, config/profile hash, expected/actual
count와 생성 시간을 기록한다. P0 partition을 복사해 별도 identity로 만들지
않고 두 annotation condition이 같은 P0 hash를 참조한다. 기존 historical raw
response와 cluster assignment는 historical artifact로 보존하며 새 결과로
덮어쓰지 않는다. 현재 automatic 실행은 condition별
`frozen_annotations.jsonl`을 고정 입력으로 사용하고, coverage마다
`filtered_finalized_annotations.jsonl`을 만든다. 여기서 `frozen`과
`finalized`는 pipeline bytes의 고정을 뜻하며 human approval을 뜻하지 않는다.

## 9. 중단 조건

다음은 사전 계획의 중단 조건이다. 현재 3-checkpoint sensitivity deviation은
manifest에서 명시적으로 승인됐으므로 첫 조건의 무단 누락과 구분한다.

다음 중 하나라도 발생하면 downstream 실행을 중단한다.

- Stage 4 `selected_topk_run_dir`가 비어 있거나 세 후보 밖의 값을 가리킴
- Annotation attempt merge에 cluster ID 중복·누락 또는 prompt/config hash 혼합
- R-P/R-PG/A-PG count가 913/1,227/1,278과 불일치
- AWE geometric threshold 위반
- sample ID 중복 또는 event–media–feature exact join 누락
- 3-view panel의 sample/frame synchronization 불일치
- P0 paired condition의 partition, exemplar sample ID 또는 frame-order 불일치
- task-family resolver가 instruction을 유일하게 분류하지 못함
- prompt text/version/hash 또는 generation config가 condition 중간에 변경
- annotation output에 허용되지 않은 phase, API error 또는 unresolved parse error
- human review가 끝나기 전에 automatic phase를 canonical semantic 결과로 사용

## 10. 완료 조건과 주장 범위

Core 통제 실험 완료 조건:

1. 선택한 Stage 4 top-k와 annotation merge contract가 profile/manifest에 고정
2. P0–P3 exact join audit 통과
3. 다섯 semantic condition의 `coverage≥0.3` frozen annotation 완료
4. 모든 selected cluster human review 완료
5. Coverage `0.3/0.4/0.5`별 phase group·score·ranking materialization
6. Pairwise Q1–Q4 표와 condition별 provenance manifest 작성
7. 관련 test와 `git diff --check` 통과

현재 automatic 실행은 2·3·5의 mechanical artifact와 automatic Q1–Q4 audit을
materialize했지만 4의 human review를 충족하지 않았다. 1도 계획의 단일
checkpoint 선택 대신 승인된 세 checkpoint deviation을 사용했다. 따라서 결과
상태는 `gate6_complete_provisional_automatic`이며 Core 통제 실험의 semantic
완료 조건을 만족했다고 쓰지 않는다.

이 실험이 직접 제공하는 것은 현재 150 episodes와 5 instruction/scene cells 안의
**diagnostic evidence**다. Human-reviewed annotation 비교는 visual labeling과
cluster semantic 품질에 관한 근거지만, policy 성능이나 인과 효과를 뜻하지 않는다.
다른 scene, instruction paraphrase와 held-out rollout 일반성은 별도 실험 없이는
주장하지 않는다.
