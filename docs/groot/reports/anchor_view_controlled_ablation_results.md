# GR00T Anchor/view 통제 실험 — Automatic 결과 snapshot

[GR00T 현황](../README.md) ·
[실행 규약](../protocols/anchor_view_controlled_ablation.md) ·
[Current phase-feature analysis](phase_feature_analysis.md)

> 이 문서의 90-row annotation, 45/45 Stage 4 결과와
> `gate6_complete_provisional_automatic` 상태는 V11 artifact snapshot이다.
> 이후 replacement 결과는 이 문서의 phase group·feature ranking에 반영되지
> 않았다. 전체 실행 이력은
> [Annotation execution lineage](annotation_lineage.md#5-execution-lineage)에 있다.

- 갱신일: 2026-07-27
- Artifact 생성 시각: `2026-07-24T05:34:05Z`
- 실행 root:
  `logs/groot_n15/experiments/anchor_view_controlled_ablation_v1/`
- 결과 상태: `gate6_complete_provisional_automatic`
- Semantic 상태: **automatic provisional; human review 0/90**

계획 계약은
[`configs/groot/anchor_view_controlled_ablation_v1.json`](../../../configs/groot/anchor_view_controlled_ablation_v1.json)에
보존한다. 실제 실행 범위와 deviation의 source of truth는 Git에서 제외된
local artifact
`logs/groot_n15/experiments/anchor_view_controlled_ablation_v1/experiment_manifest.json`과
`logs/groot_n15/experiments/anchor_view_controlled_ablation_v1/audits/final_audit.json`이다.
계획 config의 `selected_topk_run_dir`는 여전히 `null`이며 단일 checkpoint 선택
계약을 담고 있다. 실제 실행에서는 사용자 승인에 따라 보존된 세 checkpoint를
sensitivity 축으로 모두 사용했다.

## 0. 결과 수치와 평가 범위

| 항목 | 실제 범위 |
| --- | ---: |
| Source episodes | 150 |
| Instruction/scene cells | 5 |
| Semantic conditions | 5 (`E0`–`E4`) |
| Coverage thresholds | 3 (`0.3`, `0.4`, `0.5`) |
| SAE checkpoints | 3 |
| Phase-group bundles | 15/15 |
| Score artifacts | 45/45 |
| Ranking artifacts | 45/45 |
| Run-level candidate rows | 900 |
| Bundle 안의 phase-group rows | 129 |
| `event_aligned` checkpoint별 phase rows | 387 |
| `window_mean` checkpoint별 phase rows | 387 |
| Automatic annotation rows | 90 |
| Human-reviewed annotation rows | 0/90 |

`129`는 condition×coverage bundle 안의 행을 합한 수다. 같은 semantic phase가
여러 coverage bundle에 반복될 수 있으므로 129개의 서로 다른 semantic phase라는
뜻이 아니다. 각 phase ranking row에는 Top-20이 저장되고 결과 UI는 첫 Top-5를
표시한다.

### 0.1 Coverage별 raw cluster와 phase group

| Condition | Coverage | 조건을 만족한 raw cluster | Phase group |
| --- | ---: | ---: | ---: |
| E0 · rel position · LEFT cluster · LEFT label | 0.3 / 0.4 / 0.5 | 17 / 12 / 9 | 11 / 10 / 8 |
| E1 · rel position · LEFT cluster · 3-view label | 0.3 / 0.4 / 0.5 | 17 / 12 / 9 | 9 / 9 / 7 |
| E2 · rel position · 3-view cluster · 3-view label | 0.3 / 0.4 / 0.5 | 18 / 12 / 7 | 11 / 9 / 5 |
| E3 · rel position + gripper · 3-view cluster/label | 0.3 / 0.4 / 0.5 | 18 / 12 / 11 | 9 / 8 / 8 |
| E4 · abs position + gripper · 3-view cluster/label | 0.3 / 0.4 / 0.5 | 20 / 14 / 10 | 9 / 9 / 7 |

Coverage를 바꿔도 raw partition과 automatic annotation은 다시 만들지 않는다.
대신 threshold를 만족하는 raw cluster를 exact filter한 뒤
`(task_description, phase)`를 coverage마다 다시 병합하고, 각 SAE checkpoint의
score와 ranking을 다시 계산한다.

### 0.2 Q1–Q4 통제 비교

아래 수치는 `audits/final_audit.json`의 read-only mechanical audit 결과다.
ARI/NMI는 서로 다른 instruction의 cluster ID를 직접 합치지 않고, instruction별
지표를 계산한 뒤 5개 cell을 동일 가중한 `within_task_macro`다.

| 질문 | 비교와 실제 분모 | 결과 |
| --- | --- | --- |
| Q1 · annotation view | E0↔E1의 paired 17 clusters, 419 events | Phase exact match `10/17` (58.8%, cluster 동일 가중), `216/419` (51.6%, cluster event 수 가중); phrase exact match `6/17` (35.3%) |
| Q2 · clustering view | P0 LEFT↔P1 3-view의 공통 position anchor 913개 | Macro ARI `0.7067`, NMI `0.8810` |
| Q3 · gripper-close anchor | P1↔P2의 공통 position anchor 913개; P2 gripper-only event 314개 | 공통 anchor macro ARI `0.7207`, NMI `0.8996`; gripper-only 중 `coverage≥0.3` cluster 진입 `60/314` (19.1%) |
| Q4 · position frame | P2 relative↔P3 absolute의 공통 anchor 1,096개; relative-only 131개, absolute-only 182개 | 공통 anchor macro ARI `0.8941`, NMI `0.9625` |

Q1은 `gemini_v11_provisional_automatic_not_human_reviewed` 결과이며 human review는
`0/90`이다. 따라서 view 변경에 따른 **automatic label sensitivity**만 나타낸다.
Q2–Q4의 ARI/NMI는 공통 anchor에서 partition이 얼마나 유지되는지를 나타내며
semantic cluster 품질이 아니다. Q3의 `60/314`도 gripper-only anchor가 recurring
cluster에 들어간 비율일 뿐 grasp precision이 아니다. 어느 view·anchor·position
frame이 실제 action phase를 더 정확히 나타낸다는 결론은
**confounded — 판정 보류**다.

## 1. Confound audit

| Gate | 판정 | 근거 |
| --- | --- | --- |
| Length | **FAIL** | 같은 150-episode source에는 알려진 success/timeout 길이 차이가 있고, 이 실행은 fixed-time·dwell-matched behavioral evaluation이 아니다. |
| Task identity | PASS | Phase ranking과 join은 instruction-local이며 `(task_description, phase_group_id)`를 사용한다. Cross-instruction 일반성은 주장하지 않는다. |
| Instruction balance | N/A | Instruction paraphrase나 instruction-balanced detector 성능을 측정하지 않았다. |
| In-sample rescue | N/A | Detector, steering 또는 held-out policy evaluation을 수행하지 않았다. |
| Rollout pooling | PASS | Event를 cluster–episode 안에서 먼저 평균하고 episode를 동일 가중한다. |
| Phase/dwell | **FAIL** | C0 descriptor에 progress가 있고 condition-scoped annotation human review가 0/90이다. |
| Observation ≠ causation | PASS | ARI/NMI와 feature ranking을 geometry·후보 진단으로만 기록하며 policy 효과로 해석하지 않는다. |
| Scene-local ≠ general | **FAIL** | 결과 범위가 5 instruction/scene cells에 한정되고 held-out scene·camera·instruction 검증이 없다. |

Claim strength: **diagnostic evidence**.

따라서 현재 automatic phase의 semantic correctness, clustering/annotation 품질,
feature detector 성능과 policy 효과는 **confounded — 판정 보류**다. `45/45`,
`3/3 checkpoint Top-5 준비됨`은 artifact와 join 완전성을 뜻할 뿐 품질 합격을
뜻하지 않는다.

## 2. Artifact 무결성 계약

Coverage `c`마다 loader가 다음 exact invariant를 확인한다.

```text
S(c) = {raw cluster ID | episode_coverage >= c}
     = filtered_finalized_annotations의 cluster ID
     = phase group source_cluster_ids의 union

S(0.5) ⊆ S(0.4) ⊆ S(0.3)
```

추가 fail-fast 검증:

- Raw cluster, filtered annotation과 phase-group source에 누락·중복이 없다.
- Source cluster의 task·coverage와 annotation의 task·phase·source phrase가
  phase group에 exact match한다.
- Summary와 각 phase-group row의
  `actual_human_review_completed`가 일치한다.
- `event_aligned`와 `window_mean`의 key 집합은
  `(task_description, phase_group_id)` 기준으로 phase group과 exact match한다.
- Ranking의 phase·phrase는 해당 canonical phase-group row와 일치한다.
- 각 checkpoint의 `ranking_config.json`에 기록된 `scores_pt`와
  `topk_run_dir`가 현재 run provenance와 일치한다.
- Phase row의 첫 Top-5는 고유 feature ID와 finite score를 가진다.
- 15개 condition×coverage bundle 모두에서 세 checkpoint의 score artifact
  SHA-256은 서로 다르다. Feature ID는 checkpoint-local이다.

## 3. 실제 실행 deviation

Manifest에 기록된 승인된 deviation은 다음과 같다.

1. Core protocol의 한 checkpoint 선택 대신 세 checkpoint를 sensitivity 축으로
   모두 실행했다.
2. Human review를 초기 automatic 결과 이후로 미뤘으며 모든 phase group에
   `actual_human_review_completed=false`를 유지했다.
3. Annotation pilot은 생략했다.
4. Gemini retry의 transport timeout은 달랐지만 model, prompt, media,
   temperature와 JSON schema는 고정했다.

`frozen_annotations.jsonl` 또는 `filtered_finalized_annotations.jsonl`의
`frozen`·`finalized`는 pipeline 입력 bytes가 고정됐다는 뜻이다. Human approval을
뜻하지 않는다.

## 4. 결과 UI와 검증

통합 결과 UI 실행 명령은
[GR00T read-only commands](../README.md#7-read-only-commands)를 따른다.

2026-07-27 현재 구현 검증:

```bash
PYTHONPATH=. conda run --no-capture-output -n event-sae-dev \
  python -m pytest -q \
  tests/test_cluster_review_app.py \
  tests/test_experiment_results_ui.py
```

현재 두 UI test file은 `44 passed`이며 전체 suite 개수가 아닌 revision
snapshot이다. 실제 artifact loader도 45 run, 15 phase set, 129 bundle
phase rows를 strict validation으로 통과했다. 데스크톱과 320px viewport에서
5-frame 재생, 이전/다음 rollout, source raw-cluster 이동, phase cell에서 정확한
ranking 행 이동, 키보드 탭과 45-cell matrix를 확인했다. 이 browser smoke는
로컬 최종 확인이며 별도 test file로 저장하지 않았다.

동일 서버의 `Oracle phase 실험` 탭은 별도 `/api/oracle` payload와 media
endpoint를 사용한다. `Oracle ↔ E3 정렬` 탭은 별도
`/api/directional-alignment` payload에서 primary 10k/W5의 Fine·Coarse4
Top-5를 비교한다. 이 비교는 source별 `matrix_raw` 순위를 유지하고 source
score를 합치지 않는다. 방향 표기 `↑`·`↓`·`ᴾ`는 aggregate 대표 template이며
rollout consistency나 causal effect가 아니다.

최종 alignment 비교 수치는 다음과 같다.

| Level | Fine | Coarse4 |
| --- | --- | --- |
| Instruction | 14 rows · ID 27 · same direction 9/27 | 13 rows · ID 28 · same direction 10/28 |
| Task family | 3 rows · ID 7 · same direction 4/7 | 4 rows · ID 10 · same direction 5/10 |
| Task agnostic | grasp 1 row · 3/3 | grasp 1 row · 3/3 |

이 표는 최신 directional artifact의 descriptive view다. V11의 45-run
historical snapshot을 대체하거나 automatic annotation의 semantic correctness를
입증하지 않는다.

## 5. 남은 semantic gate

다음 단계는 새 feature 후보를 더 고르는 일이 아니라 90개 condition-scoped
annotation row의 blind human review다.

1. Automatic phrase/phase를 보기 전에 independent visual assessment를 기록한다.
2. Mixed·visually insufficient cluster를 별도 상태로 남긴다.
3. Human-reviewed phase를 사용해 coverage별 phase group과 Stage 4 score/ranking을
   새 run ID에 다시 만든다.
4. Automatic 결과와 reviewed 결과를 같은 canonical 표에 섞지 않는다.
