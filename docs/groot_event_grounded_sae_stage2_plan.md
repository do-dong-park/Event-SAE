# GR00T Event-Grounded SAE Stage 2 — 실행 계획

- 작성일: 2026-07-21
- 갱신일: 2026-07-22
- 대상: GR00T N1.5, RoboCasa PQ3
- 단계: Kinematic keyframe extraction
- 기본 방법: AWE dynamic-programming waypoint selection
- 상태: **입력 adapter와 video timeline 구현 완료, 원격 export·AWE 실행 전**

Stage 2의 목적은 closed-loop rollout의 end-effector trajectory를 소수의
kinematic waypoint로 압축하는 것이다. 이 waypoint는 Stage 3의 event window
중심점으로 사용하며 SAE checkpoint와 독립적이다. 따라서 Stage 1의
1.2k/4k/10k/20k SAE가 모두 동일한 waypoint set을 공유한다.

## 0. 계획 요약

Stage 2는 다음 순서로 진행한다.

1. 원격 PQ3 PKL에서 trajectory field만 추출한다.
2. 기존 AWE CLI가 읽는 `trajectory_records.jsonl`로 변환한다.
3. `pos_only, η=0.05`를 baseline으로 150개 episode 전체에서 waypoint를 추출한다.
4. `η∈{0.02, 0.05, 0.10}` sweep으로 압축률과 trajectory 오차의 민감도를 확인한다.
5. 입력 무결성, reconstruction error, waypoint density와 event proximity를 audit한다.
6. baseline `waypoint_summary.json`을 Stage 3 handoff로 고정한다.

예상 산출물:

```text
logs/groot_n15/pq3_stage2_keyframes/
├── trajectory_records.jsonl
├── trajectory_manifest.json
├── trajectory_audit.json
├── dp_pos_only_err0p02/waypoint_summary.json
├── dp_pos_only_err0p05/waypoint_summary.json
├── dp_pos_only_err0p10/waypoint_summary.json
└── waypoint_audit.json
```

현재 repo에는 exporter, JSONL audit, GR00T video timeline과 합성 테스트까지
준비되어 있다. 아직 완료하지 않은 것은 원격 150개 PKL의 실제 export, AWE
dependency 설치·고정, threshold sweep과 결과 audit이다.

## 1. 목표와 범위

### 1.1 핵심 질문

> End-effector trajectory의 형상을 허용 오차 안에서 보존하면서, Stage 3의
> event anchor로 사용할 소수의 waypoint를 안정적으로 선택할 수 있는가?

Stage 2는 feature를 선택하거나 해석하지 않는다. Kinematic waypoint는
SAE activation이 아닌 rollout trajectory만 사용한다.

### 1.2 포함 범위

- PQ3 150개 rollout의 trajectory inventory
- GR00T PKL → AWE 입력 schema 변환
- AWE `pos_only` baseline 실행
- error threshold sensitivity sweep
- trajectory reconstruction 및 waypoint 분포 audit
- Stage 3용 canonical `waypoint_summary.json` 지정

### 1.3 제외 범위

- SAE feature activation과 waypoint의 결합
- keyframe 주변 image bundle 생성
- VLM annotation과 event clustering
- feature ranking, steering, ablation
- “waypoint가 의미 있는 semantic event다”라는 인과적 주장

Image bundle 생성부터는 Stage 3 범위다.

## 2. 확인된 시작점

### 2.1 Source

Stage 1과 동일한 원격 source를 사용한다.

```text
/home/kimseungjun/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/
  phase_event_pq3/raw_rollouts/
```

| 항목 | 값 |
| --- | --- |
| Rollout PKL | 150 |
| Instruction/scene cell | 5 |
| Task family | 2 |
| Policy records | 12,041 |
| Exact-stem MP4 | 150 / 150 |
| MP4 총 frame 수 | 30,127 |
| PKL↔MP4 누락/extra | 0 / 0 |
| SAE 의존성 | 없음 |

### 2.2 GR00T PKL에서 확인된 trajectory field

원격 PQ3 rollout을 read-only로 검사한 결과 각 PKL은 `states`와 `actions`를
policy record 단위로 저장한다.

| PQ3 변환 대상 | GR00T PKL source |
| --- | --- |
| `eef_pos` | `states[*]["observation.state.eef_pos_rel"]` |
| `eef_quat` | `states[*]["observation.state.eef_quat_rel"]` |
| `gripper_qpos` | `states[*]["observation.state.gripper_qpos"]` |
| gripper command 후보 | `actions[*]["action.gripper_close"]` |
| success | `episode_success` |
| task/cell provenance | `robocasa_task`, `cell_id`, instruction field |
| diagnostic event | `event_steps`, `grasp_steps`, `drop_steps` |

`eef_pos_rel`은 robot-relative frame의 3D position이다. 모든 episode에서 같은
frame을 사용한다면 AWE의 trajectory shape 비교에는 사용할 수 있다. Quaternion
순서와 gripper command의 실제 실행 step 대응은 `geometric_gripper` mode를
사용하기 전에 별도 검증해야 한다.

### 2.3 현재 repo의 구현 상태

- [`scripts/groot/export_pq3_trajectories.py`](../scripts/groot/export_pq3_trajectories.py)
  - trusted PKL을 공통 `trajectory_records.jsonl`과 manifest로 변환
  - inventory, shape, finite, step continuity와 exact video path audit
- [`event_sae/keyframes/extract.py`](../event_sae/keyframes/extract.py)
  - 기존 episode grouping, filtering, AWE 호출을 수정 없이 재사용
- [`scripts/extract_keyframes.py`](../scripts/extract_keyframes.py)
  - 기존 `trajectory_records.jsonl` 입력과 `waypoint_summary.json` 출력 재사용
  - GR00T output path 추론 지원
- [`event_sae/events/video_timeline.py`](../event_sae/events/video_timeline.py)
  - policy record와 rendered video frame 사이의 정수 시간축 변환
- [`scripts/extract_keyframe_media.py`](../scripts/extract_keyframe_media.py)
  - manifest-relative exact MP4 lookup과 missing-video 정책 지원

남은 gap:

1. `event-sae-dev` 환경에 `waypoint_extraction` module이 아직 설치되어 있지 않다.
2. 원격 150개 PKL에 exporter를 실행하고 작은 JSON 산출물을 회수해야 한다.

## 3. 입력 계약

### 3.1 `trajectory_records.jsonl`

한 줄은 한 episode의 한 policy record다. 최소 schema는 다음과 같다.

| Field | Type | 의미 |
| --- | --- | --- |
| `episode_num` | int | 150개 파일 전체에서 유일한 episode 번호 |
| `task_id` | int | PKL과 파일명에 기록된 RoboCasa source task index |
| `task_episode_idx` | int | task/cell 내부 episode 순번 |
| `step_in_episode` | int | 0부터 시작하는 policy record index |
| `eef_pos` | float[3] | robot-relative end-effector position |
| `done` | bool | 마지막 record에서 rollout success |
| `task_description` | str | canonical task description |
| `prompt_task_description` | str | 실제 policy instruction |

이번 GR00T export에서 `eef_quat: float[4]`와 `gripper_qpos: float[2]`도
required field로 저장한다. 따라서 한 record의 end-effector pose는
`eef_pos[3] + eef_quat[4]`인 7D다. `actions[*]["action.gripper_close"]`는
scalar가 아니라 `[16,1]` planning chunk이므로 임의로 scalar화하지 않는다.
`event_steps`, `grasp_steps`, `drop_steps`는 중복을 피하기 위해 episode별
manifest에 보존한다.

### 3.2 변환 invariant

각 source PKL `e`에 대해

```text
len(states_e) = trajectory record 수 T_e
Σ_e T_e      = 12,041
episode 수    = 150
```

을 만족해야 한다. Episode 내부에서는

```text
step_in_episode = 0, 1, ..., T_e - 1
eef_pos.shape   = [T_e, 3]
```

이어야 한다. 모든 numeric field는 finite여야 하며 `episode_num`과
`(source_file, cell_id, task_episode_idx)` mapping은 manifest에 보존한다.

### 3.3 Export 방식

Raw PKL과 hidden activation은 원격에 유지한다. Stage 1과 같은 방식으로
standalone trajectory exporter만 원격에 전달하고, 작은 JSONL과 manifest만
로컬로 가져온다.

구현된 CLI:

```text
scripts/groot/export_pq3_trajectories.py
├── export: trusted PKL → trajectory_records.jsonl + manifest
└── audit:  inventory/schema/finite/step continuity와 video inventory 재검사
```

PKL loading은 신뢰한 source에서만 `--trust-pkl`로 허용한다.

## 4. AWE 방법과 실험 조건

### 4.1 Baseline

Baseline은 기존 OpenVLA/OpenPI runbook과 같은 설정이다.

| 설정 | 값 |
| --- | --- |
| Waypoint mode | `pos_only` |
| Error threshold `η` | 0.05 |
| Episode filter | all |
| Compute | CPU |
| SAE checkpoint | 사용하지 않음 |

`pos_only`는 end-effector position `p_t∈R^3`만 사용한다. 선택된 waypoint
집합을 `W={w_0,...,w_m}`이라 하면 인접 waypoint 사이를 선형 보간해 원래
trajectory를 근사한다.

```text
p_hat_t = linear_interpolate(p_wj, p_w(j+1), t)
e_t     = ||p_t - p_hat_t||_2
```

`η`는 trajectory approximation에 허용하는 위치 오차 budget이다.
`eef_pos_rel` 단위가 meter라면 `η=0.05`는 5 cm에 해당한다. 실제 AWE
implementation의 error aggregation 방식과 단위는 smoke run에서 독립
reconstruction audit와 함께 확인한다.

### 4.2 Threshold sweep

| Run | `η` | 목적 |
| --- | ---: | --- |
| tight | 0.02 | 더 세밀한 trajectory 보존 |
| baseline | 0.05 | 기존 repo/paper 기본값 |
| coarse | 0.10 | 더 강한 waypoint 압축 |

예상되는 단조 관계는

```text
η 증가 → waypoint 수 감소 또는 동일
```

이다. 이 관계가 다수 episode에서 역전되면 adapter, step ordering 또는
AWE 호출을 점검한다.

### 4.3 `geometric_gripper`의 위치

`geometric_gripper`는 이번 baseline 완료 조건에 포함하지 않는다. 다음이
확인된 뒤 secondary ablation으로만 실행한다.

1. `eef_quat_rel`의 quaternion ordering과 normalization
2. policy record와 실제 실행 gripper command의 정렬
3. `action.gripper_close`에서 scalar command를 선택하는 규칙
4. AWE가 요구하는 `robosuite` quaternion convention과의 호환성

Gripper transition은 먼저 baseline waypoint의 diagnostic recall을 계산하는 데
사용하며, AWE 입력에는 강제로 섞지 않는다.

## 5. 구현 및 실행 절차

### 5.1 Dependency 준비

AWE fork는 commit `7197bb86a20784666dabed90e6eabcf8bb1e9912`,
`robosuite==1.4.0`으로 고정한다. `event-sae-dev`의 NumPy 1.26 계약을 보존하기
위해 OpenCV도 `4.11.0.86`으로 고정한다. 설치 후 다음 smoke test를 통과한다.

```bash
conda run -n event-sae-dev python -c   "from waypoint_extraction import dp_waypoint_selection; print('AWE import OK')"
```

2026-07-22 기준 AWE import, synthetic DP 호출, `pip check`와 repo test 22개를
통과했다. `external/awe`는 gitignored editable clone이며 재현 가능한 pin은
`environment-sae-dev.yml`에 기록한다. Repo test는 upstream AWE test가 자동
수집되지 않도록 `python -m pytest -q tests`로 실행한다.

### 5.2 Trajectory export

원격 raw rollout 옆에서 standalone exporter만 실행한다.

```bash
python scripts/groot/export_pq3_trajectories.py export \
  --input-dir /home/kimseungjun/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/phase_event_pq3/raw_rollouts \
  --output-dir pq3_stage2_keyframes \
  --trust-pkl
```

로컬로 `trajectory_records.jsonl`과 `trajectory_manifest.json`만 회수한 뒤
다시 audit한다. Audit는 exact-stem 영상 inventory도 함께 확인한다.

```bash
conda run -n event-sae-dev python scripts/groot/export_pq3_trajectories.py audit \
  --trajectory-records-path logs/groot_n15/pq3_stage2_keyframes/trajectory_records.jsonl \
  --manifest logs/groot_n15/pq3_stage2_keyframes/trajectory_manifest.json \
  --video-root /path/to/phase_event_pq3/raw_rollouts
```

현재 원격 inventory는 150개 PKL과 exact-stem MP4 150개가 모두 대응한다.
최종 audit에는 `--require-complete-videos`를 붙여 이 조건을 강제한다.

### 5.3 AWE baseline

```bash
conda run -n event-sae-dev python scripts/extract_keyframes.py   --trajectory-records-path logs/groot_n15/pq3_stage2_keyframes/trajectory_records.jsonl   --output-dir logs/groot_n15/pq3_stage2_keyframes/dp_pos_only_err0p05   --waypoint-mode pos_only   --err-threshold 0.05   --success-filter all
```

같은 명령을 `η=0.02`와 `η=0.10`에 대해 반복한다.

### 5.4 Video timeline handoff

GR00T 영상은 policy record와 1:1이 아니다. Episode의 record 수를 `R`,
`A=n_action_steps`, `S=steps_per_render`라 하면 다음 계약을 쓴다.

```text
expected_frames = ceil(R * A / S)
frame_start(r)  = ceil(r * A / S)
frame_stop(r)   = ceil((r + 1) * A / S) - 1
```

현재 PQ3의 `A=5`, `S=2`에서는 144 records가 360 frames, 35 records가
88 frames에 대응한다. 원격 150개 전체를 검사한 결과 CSV의 12,041 records와
MP4의 30,127 frames가 episode별로 이 공식을 모두 만족했다. Keyframe 주변
offset은 먼저 record 공간에서 적용한
뒤 각 record의 대표 frame으로 변환한다. 기본 대표 frame은 interval의 첫
frame이다. 이 식은 기존 GR00T annotation 코드의 계약을 그대로 따른다. 실제
overlay를 만들 때는 gripper transition을 기준으로 ±1 frame 오프셋을 수동
확인하고, 차이가 있으면 manifest의 calibration 결과로 기록한다.

Stage 3 media packaging 준비 명령은 다음과 같다. 최종 실행에서는
`--require-complete-videos`를 사용한다. 부분 inventory smoke test에서만 이
옵션을 생략하며, 그 경우 없는 episode는 명시적으로 skip된다.

```bash
python scripts/extract_keyframe_media.py \
  --waypoint-summary-path logs/groot_n15/pq3_stage2_keyframes/dp_pos_only_err0p05/waypoint_summary.json \
  --trajectory-manifest-path logs/groot_n15/pq3_stage2_keyframes/trajectory_manifest.json \
  --video-root /path/to/phase_event_pq3/raw_rollouts \
  --frame-anchor first \
  --require-complete-videos
```

### 5.5 Waypoint audit

구현된 audit는 source JSONL과 세 `waypoint_summary.json`을 함께 읽는다.

```bash
conda run -n event-sae-dev python scripts/groot/audit_pq3_keyframes.py \
  --trajectory-records-path logs/groot_n15/pq3_stage2_keyframes/trajectory_records.jsonl \
  --trajectory-manifest logs/groot_n15/pq3_stage2_keyframes/trajectory_manifest.json \
  --waypoint-root logs/groot_n15/pq3_stage2_keyframes \
  --event-tolerance 2 \
  --output logs/groot_n15/pq3_stage2_keyframes/waypoint_audit.json
```

`--event-tolerance 2`는 reference event와 waypoint가 ±2 policy records 안에
있으면 proximity match로 세는 diagnostic 설정이다.

## 6. 검증 metric

Episode `e`의 step 수를 `T_e`, waypoint 수를 `W_e`라 한다.

### 6.1 압축과 분포

```text
waypoint density_e  = W_e / T_e
compression factor_e = T_e / W_e
```

- Waypoint density가 낮을수록 더 강한 압축이다.
- Compression factor가 1이면 사실상 모든 step을 선택한 것이다.
- 평균만 보지 않고 cell, task, success/failure별 median과 분포를 보고한다.

고정된 “좋은 waypoint 수”를 미리 정하지 않는다. Episode 길이와 trajectory
복잡도가 다르므로 압축률은 reconstruction error와 함께 해석한다.

### 6.2 Trajectory reconstruction

```text
RMSE_e      = sqrt((1/T_e) Σ_t ||p_t - p_hat_t||_2^2)
MaxError_e  = max_t ||p_t - p_hat_t||_2
```

Baseline `η=0.05`에서 `MaxError`가 threshold 계약을 만족하는지 확인한다.
AWE 내부 error 정의와 독립 계산식이 다르면 두 정의를 모두 기록하고 차이를
설명한다.

### 6.3 Boundary와 numerical integrity

Episode마다 다음을 검사한다.

- waypoint index가 `[0,T_e)` 범위에 있음
- waypoint가 strict ascending이며 중복이 없음
- 시작·종료점 포함 여부
- `eef_pos`와 waypoint position이 finite
- summary의 `num_steps`, `num_waypoints`와 실제 배열 길이가 일치
- 150개 episode가 누락 없이 출력됨

### 6.4 Event proximity diagnostic

PKL의 `event_steps`, `grasp_steps`와 gripper transition을 reference `G_e`로 두고

```text
match(g) = 1[min_w |g-w| ≤ δ]
event recall = Σ_g match(g) / |G_e|
```

를 계산한다. 기본 `δ=2` policy records다.

이 수치는 Stage 2 통과 기준이 아니라 sanity check다. 기존 event label은
환경 heuristic에서 생성됐을 수 있으므로 독립적인 semantic ground truth로
취급하지 않는다.

### 6.5 Threshold stability

각 episode에서 `η=0.02,0.05,0.10`의 waypoint 수가 대체로 non-increasing인지
확인한다. Baseline waypoint가 tight/coarse run에서 얼마나 유지되는지도
nearest-step matching으로 보고한다.

## 7. 실험 매트릭스와 완료 조건

### 7.1 필수 실험

| ID | 설정 | 역할 |
| --- | --- | --- |
| S2-A | adapter/export audit | 입력 계약 확정 |
| S2-B | `pos_only, η=0.02` | tight sensitivity |
| S2-C | `pos_only, η=0.05` | canonical baseline |
| S2-D | `pos_only, η=0.10` | coarse sensitivity |
| S2-E | trajectory/event audit | 품질 및 sanity check |

`geometric_gripper`는 필수 실험 이후의 optional ablation이다.

### 7.2 구현 준비 검증

2026-07-22 기준 합성 검증 결과는 다음과 같다.

- GR00T PKL exporter와 JSONL/manifest audit 구현
- `144→360`, `35→88` timeline 계산 검증
- record offset을 frame index로 변환하는 media 통합 검증
- exact-stem 영상이 없을 때 다른 실험 영상으로 fallback하지 않음
- 신규 합성 테스트 13개 통과

### 7.3 Repo-scope 완료 조건

Stage 2는 다음 조건을 모두 만족하면 완료한다.

- [ ] AWE dependency revision이 고정되고 import smoke test를 통과한다.
- [ ] 150 episodes, 12,041 policy records가 JSONL로 누락 없이 변환된다.
- [ ] 모든 required field의 shape, finite, step continuity audit를 통과한다.
- [ ] 세 threshold에서 전체 episode의 `waypoint_summary.json`이 생성된다.
- [ ] Waypoint index ordering, range와 summary consistency를 통과한다.
- [ ] Position reconstruction RMSE/MaxError와 waypoint density가 기록된다.
- [ ] Threshold 증가에 따른 waypoint count의 단조성을 검사한다.
- [ ] 5 cells에서 각 2 episodes, 총 10 episodes를 trajectory overlay로 수동 확인한다.
- [ ] `η=0.05` baseline을 canonical Stage 3 handoff로 지정한다.
- [ ] 실행 명령, code revision, source manifest와 audit 결과를 최종 보고서에 기록한다.

### 7.4 Stage 3 handoff

Canonical output:

```text
logs/groot_n15/pq3_stage2_keyframes/dp_pos_only_err0p05/waypoint_summary.json
```

Stage 3는 이 waypoint를 중심으로 frame window를 만들고 vision/state descriptor를
구성한다. Stage 1 SAE와 Stage 2 waypoint는 이 시점까지 독립적으로 유지하며,
두 축의 결합은 event-feature scoring 단계에서 수행한다.
