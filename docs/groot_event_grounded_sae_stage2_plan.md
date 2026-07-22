# GR00T Event-Grounded SAE Stage 2 — 실행 보고서

- 작성일: 2026-07-21
- 갱신일: 2026-07-22
- 대상: GR00T N1.5, RoboCasa PQ3
- 단계: Kinematic keyframe extraction
- 기본 방법: AWE geometric objective + 연결 보장 exact dynamic programming
- 상태: **완료 — 150 episodes export, threshold sweep, audit 및 수동 trajectory 검증 통과**

Stage 2의 목적은 closed-loop rollout의 end-effector trajectory를 소수의
kinematic waypoint로 압축하는 것이다. 이 waypoint는 Stage 3의 event window
중심점으로 사용하며 SAE checkpoint와 독립적이다. 따라서 Stage 1의
1.2k/4k/10k/20k SAE가 모두 동일한 waypoint set을 공유한다.

## 0. 실행 결과 요약

Stage 2는 다음 순서로 실행했다.

1. 원격 PQ3 PKL에서 trajectory field만 추출한다.
2. 기존 AWE CLI가 읽는 `trajectory_records.jsonl`로 변환한다.
3. `pos_only, η=0.05`를 baseline으로 150개 episode 전체에서 waypoint를 추출한다.
4. `η∈{0.02, 0.05, 0.10}` sweep으로 압축률과 trajectory 오차의 민감도를 확인한다.
5. 입력 무결성, reconstruction error, waypoint density와 event proximity를 audit한다.
6. baseline `waypoint_summary.json`을 Stage 3 handoff로 고정한다.

산출물:

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

원격 150개 PKL export, AWE dependency 고정, 세 threshold 전체 실행과
독립 audit를 완료했다. 최종 Stage 3 handoff는 `η=0.05` 결과로 고정한다.

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

실행 시점의 dependency와 source inventory는 모두 고정했다. AWE는 commit
`7197bb86a20784666dabed90e6eabcf8bb1e9912`, exporter는 Event-SAE commit
`8ed41c568ce682a8ae035253d8be465660a7f082`를 사용했다.

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

입력 field, position-only 목적식과 `η=0.05`는 기존 OpenVLA/OpenPI runbook과
같다. Selector는 8.3절의 검증 결과에 따라 연결을 보장하는 exact DP를 사용한다.

| 설정 | 값 |
| --- | --- |
| Waypoint mode | `pos_only` |
| DP implementation | `exact_pos_only` |
| Error threshold `η` | 0.05 |
| Episode filter | all |
| Compute | CPU |
| SAE checkpoint | 사용하지 않음 |

`pos_only`는 end-effector position `p_t∈R^3`만 사용한다. AWE fork의
`pos_only_geometric_waypoint_trajectory`는 각 record를 시간축 보간점과
비교하지 않고, 해당 waypoint 구간의 3D line segment까지 최단거리로 비교한다.

```text
e_t^AWE = dist(p_t, segment(p_wj, p_w(j+1)))
E^AWE   = max_t e_t^AWE
segment accept iff E^AWE < η
```

`η`는 이 최대 geometric deviation의 budget이다. `eef_pos_rel` 단위가
meter라면 `η=0.05`는 5 cm에 해당한다. AWE가 반환하지 않는 첫 record `0`은
error 계산에서 implicit anchor로 prepend된다.

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

2026-07-22 기준 AWE import, synthetic DP 호출, `pip check`와 repo test 26개를
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
conda run -n event-sae-dev python scripts/extract_keyframes.py \
  --trajectory-records-path logs/groot_n15/pq3_stage2_keyframes/trajectory_records.jsonl \
  --output-dir logs/groot_n15/pq3_stage2_keyframes/dp_pos_only_err0p05 \
  --waypoint-mode pos_only \
  --dp-implementation exact_pos_only \
  --err-threshold 0.05 \
  --success-filter all
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

AWE threshold 계약은 `awe_geometric_max_error = E^AWE`로 직접 재계산한다.
이와 별개로 시간 진행까지 보존하는지 보기 위해 record index 비율로 선형
보간한 독립 metric도 계산한다.

```text
alpha_t     = (t - w_j) / (w_(j+1) - w_j)
p_hat_t     = (1 - alpha_t) p_wj + alpha_t p_w(j+1)
RMSE_e      = sqrt((1/T_e) Σ_t ||p_t - p_hat_t||_2^2)
MaxError_e  = max_t ||p_t - p_hat_t||_2
```

따라서 Stage 2의 threshold 통과 여부는 `awe_geometric_max_error < η`로
판정하며, RMSE와 MaxError는 더 엄격한 독립 trajectory-shape 진단값으로
함께 보고한다.

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
- Stage 2 포함 repo test 26개 통과 (`exact_pos_only` 회귀 테스트 4개 포함)

### 7.3 Repo-scope 완료 조건

Stage 2는 다음 조건을 모두 만족하면 완료한다.

- [x] AWE dependency revision이 고정되고 import smoke test를 통과한다.
- [x] 150 episodes, 12,041 policy records가 JSONL로 누락 없이 변환된다.
- [x] 모든 required field의 shape, finite, step continuity audit를 통과한다.
- [x] 세 threshold에서 전체 episode의 `waypoint_summary.json`이 생성된다.
- [x] Waypoint index ordering, range와 summary consistency를 통과한다.
- [x] Position reconstruction RMSE/MaxError와 waypoint density가 기록된다.
- [x] Threshold 증가에 따른 waypoint count의 단조성을 검사한다.
- [x] 5 cells에서 각 2 episodes, 총 10 episodes를 trajectory overlay로 수동 확인한다.
- [x] `η=0.05` baseline을 canonical Stage 3 handoff로 지정한다.
- [x] 실행 명령, code revision, source manifest와 audit 결과를 최종 보고서에 기록한다.

### 7.4 Stage 3 handoff

Canonical output:

```text
logs/groot_n15/pq3_stage2_keyframes/dp_pos_only_err0p05/waypoint_summary.json
```

Stage 3는 이 waypoint를 중심으로 frame window를 만들고 vision/state descriptor를
구성한다. Stage 1 SAE와 Stage 2 waypoint는 이 시점까지 독립적으로 유지하며,
두 축의 결합은 event-feature scoring 단계에서 수행한다.

## 8. 실험 결과 및 결론

### 8.1 입력 및 실행 무결성

- PKL 150개를 episode 150개, policy record 12,041개로 누락 없이 변환했다.
- exact-stem MP4도 150/150개 대응하며, 전체 30,127 frames가 episode별 timeline 식을 만족했다.
- 세 threshold 모두 150개 episode를 처리했고 누락, 중복, 범위 밖 waypoint는 없었다.
- 모든 episode에서 `awe_geometric_max_error < η`를 만족했다.
- `η`가 증가할 때 episode별 waypoint 수가 증가한 경우는 0/150이었다.
- 5개 cell에서 2개씩 선택한 episode 0, 1, 30, 31, 60, 61, 90, 91, 120, 121의 3D trajectory overlay를 수동 확인했다. 시작점, 종료점과 궤적 굴곡이 waypoint segment에 정상 연결됐다.

입력 산출물 SHA-256은 다음과 같다.

```text
trajectory_records.jsonl  7f66ba6953dda8957f3b23cbfa06ad7f90f7058a320849caf4c93592e7445729
trajectory_manifest.json   81df6c80a109ba458b1e49c39444db1f42a5e1f4f9324bcd41fbb94b40d1c89c
trajectory_audit.json      ff87cee5b82b8e745b0f1931b1d4c82f099c643cd3f7cabd75bc99213cf3be14
```

### 8.2 Threshold sweep

| η | 전체 waypoint | 평균 waypoint/episode | 평균 density | 중앙 compression factor | 평균 geometric max error | 전체 geometric max error | 평균 RMSE | event recall | threshold 위반 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.02 | 1,673 | 11.15 | 15.89% | 6.50× | 0.01926 | 0.01999 | 0.03521 | 63.84% | 0 |
| 0.05 | 913 | 6.09 | 9.50% | 11.00× | 0.04788 | 0.04999 | 0.06831 | 41.88% | 0 |
| 0.10 | 603 | 4.02 | 6.59% | 14.71× | 0.09360 | 0.09996 | 0.11660 | 48.97% | 0 |

`η`는 source `eef_pos_rel`과 같은 위치 단위의 최대 허용 기하 편차다.
`Density=W/T`는 episode별 waypoint 비율의 평균이고, compression factor는
`T/W`의 episode별 중앙값이다. Geometric max error는 원 궤적의 각 점에서
인접 waypoint를 잇는 3D 선분까지의 최단거리 중 최댓값이다. RMSE는 별도의
시간 인덱스 선형 보간 오차이므로 geometric threshold보다 클 수 있으며, 이는
threshold 실패를 뜻하지 않는다.

Event recall은 PKL의 heuristic event가 waypoint ±2 policy records 안에 있는
비율이다. Waypoint 집합은 threshold 사이에서 서로 nested일 필요가 없으므로
`η=0.10`의 recall이 `η=0.05`보다 높게 나온 것은 모순이 아니다. 이 값은
semantic ground truth가 아니며 canonical threshold 선택의 단독 기준으로 쓰지 않는다.

### 8.3 AWE upstream DP 검증과 교정

초기 실행은 AWE commit의 `dp_waypoint_selection`을 그대로 사용했다. 그러나
세 threshold 모두 150/150 episode에서 반환 경로의 geometric threshold를
재계산하면 위반했다. 원인은 부분 문제에서 검사한 구간 시작점이 최종 반환
waypoint path에 보존되지 않아, 검사한 segment와 실제 연결 segment가 달라지는
구현 문제였다.

따라서 AWE의 목적식과 strict condition `E < η`는 유지하되, trajectory index를
node로 하고 조건을 만족하는 contiguous segment만 edge로 허용하는 shortest-path
DP인 `exact_pos_only`를 추가했다. Upstream `awe` 옵션은 기존 OpenVLA/OpenPI
호환성을 위해 남겼으며, PQ3 최종 결과는 모두 `exact_pos_only`로 재생성했다.
합성 회귀 테스트는 직선, 급격한 굴곡, wrapper routing과 잘못된 threshold를 포함한다.

### 8.4 최종 판단

`η=0.02`는 event proximity와 형상 보존이 가장 좋지만 평균 11.15개 waypoint를
요구한다. `η=0.10`은 4.02개까지 줄지만 시간축 RMSE가 크게 증가한다.
`η=0.05`는 geometric threshold를 전 episode에서 만족하면서 평균 6.09개,
평균 density 9.50%, 중앙 압축 배수 11.0×를 제공한다. 기존 repo 설정과도
일치하므로 Stage 3의 canonical handoff로 채택한다.

이 결론은 kinematic compression이 유효하다는 뜻이다. Waypoint가 grasp, contact,
placement 같은 semantic event를 나타낸다는 결론은 아직 내리지 않는다. 그 검증은
원격 MP4의 frame window를 결합하는 Stage 3에서 수행한다.

최종 산출물 SHA-256:

```text
η=0.02 waypoint_summary.json  f421f12e06b2f04c9325774fd8388d3b1bb7706b416faeb38de68da6dcaf7a3f
η=0.05 waypoint_summary.json  736f50da3c71ca8020a61cd08b4dd99cff5a2dbca9cf130a4a19b0d798b81f0e
η=0.10 waypoint_summary.json  fa21c2d60c40fa04ca776014fff22c6a96ea8423b73d42eb95a670a6a1f5e63e
waypoint_audit.json           6b7c6f8d228875e54a4d23e65ae95a1f3a25f4d3b2f05419e1478f6be1fad0df
```
