# GR00T Event-Grounded SAE Stage 2 — 최종 보고서

- 작성일: 2026-07-22
- 대상: GR00T N1.5, RoboCasa PQ3
- 단계: Kinematic waypoint extraction
- 상태: **완료**

Stage 2에서는 150개 closed-loop rollout의 end-effector trajectory를 소수의
waypoint로 압축했다. 세 error threshold를 비교한 결과 `η=0.05`가 모든
episode에서 기하 오차 조건을 만족하면서 평균 6.09개 waypoint와 중앙값 기준
11.0배 압축을 제공했다. 이 결과를 Stage 3의 canonical event anchor로 사용한다.

## 0. Stage 2 결과 요약

| 항목 | 결과 |
| --- | --- |
| 입력 | 150 episodes, 12,041 policy records |
| 영상 대응 | exact-stem MP4 150/150, 총 30,127 frames |
| 방법 | `pos_only`, 연결 보장 exact dynamic programming |
| 비교 조건 | `η∈{0.02, 0.05, 0.10}` |
| 선택 조건 | `η=0.05` |
| 선택 결과 | 총 913개, 평균 6.09개/episode |
| 압축 | 평균 density 9.50%, 중앙 compression factor 11.0× |
| 기하 오차 | 전체 최댓값 0.04999, threshold 위반 0/150 |
| 단조성 | threshold 증가 시 waypoint 수 증가 0/150 |
| 검증 | repo test 26개 및 10-episode trajectory overlay 통과 |

Stage 2의 결론은 “trajectory를 허용된 기하 오차 안에서 압축할 수 있다”이다.
선택된 waypoint가 grasp나 placement 같은 semantic event라는 결론은 아직
내리지 않는다. 그 검증은 영상과 SAE activation을 결합하는 Stage 3의 범위다.

## 1. 전체 목표 중 어떤 단계인가?

```text
Stage 1  GR00T activation으로 SAE 학습
Stage 2  End-effector trajectory에서 kinematic waypoint 추출  ← 현재 단계
Stage 3  Waypoint 주변 영상·상태를 event descriptor로 구성
Stage 4  Event와 SAE feature의 대응 관계 산출
Stage 5  Feature 검증 및 steering/ablation
```

Stage 1의 SAE는 representation 축을 제공하고, Stage 2의 waypoint는 시간축
anchor를 제공한다. Stage 2는 activation이나 SAE checkpoint를 사용하지 않으므로
1.2k/4k/10k/20k checkpoint가 동일한 waypoint set을 공유한다.

핵심 질문은 다음과 같다.

> End-effector trajectory의 형상을 허용 오차 안에서 보존하면서, 후속 event
> 분석에 사용할 소수의 시점을 안정적으로 선택할 수 있는가?

포함 범위는 trajectory export, waypoint 추출, threshold sensitivity와
reconstruction audit이다. Image bundle, VLM annotation, feature ranking과
steering은 포함하지 않는다.

## 2. 데이터셋 구성

### 2.1 Source inventory

원격 source는 Stage 1과 같은 PQ3 rollout이다.

```text
/home/kimseungjun/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/
  phase_event_pq3/raw_rollouts/
```

| Cell | Task index | Task family | Episodes | Records | Success |
| --- | ---: | --- | ---: | ---: | ---: |
| `pq3_drawer_left` | 8 | OpenDrawer | 30 | 2,567 | 17 |
| `pq3_drawer_right` | 7 | OpenDrawer | 30 | 2,954 | 16 |
| `pq3_ppcc_bread` | 5 | PickPlaceCounterToCabinet | 30 | 2,017 | 23 |
| `pq3_ppcc_beer` | 15 | PickPlaceCounterToCabinet | 30 | 2,786 | 19 |
| `pq3_ppcc_pizza_cutter` | 16 | PickPlaceCounterToCabinet | 30 | 1,717 | 26 |
| **전체** | — | 2 families | **150** | **12,041** | **101** |

Episode 길이는 최소 25, 중앙값 53.5, 평균 80.27, 최대 144 policy records다.
각 PKL에는 같은 stem의 MP4가 존재하며 누락과 extra는 모두 0이다.

### 2.2 Trajectory 계약

| Export field | PKL source | 용도 |
| --- | --- | --- |
| `eef_pos` | `observation.state.eef_pos_rel` | Stage 2 입력인 3D 위치 |
| `eef_quat` | `observation.state.eef_quat_rel` | 후속 pose ablation용 quaternion |
| `gripper_qpos` | `observation.state.gripper_qpos` | 후속 상태 해석용 |
| `done` | `episode_success` | rollout 성공 여부 |

각 JSONL record에는 episode provenance를 위한 `episode_num`, `task_id`,
`task_episode_idx`, `cell_id`, instruction과 `step_in_episode`도 저장한다.
`event_steps`, `grasp_steps`, `drop_steps`는 episode 단위 manifest에 보존한다.
`eef_pos[3] + eef_quat[4]`가 원래 문의한 end-effector 7D pose이며, Stage 2의
baseline은 이 중 position 3D만 사용하고 quaternion 4D도 손실 없이 보존한다.

변환 후 다음 조건을 강제했다.

```text
episode 수                = 150
Σ episode length         = 12,041
step_in_episode          = 0, 1, ..., T_e - 1
eef_pos.shape            = [T_e, 3]
PKL stem ↔ MP4 stem      = exact match
모든 필수 numeric field  = finite
```

PKL은 임의 코드를 실행할 수 있으므로 신뢰한 source에 대해서만
`--trust-pkl`을 사용한다.

## 3. Waypoint 추출 조건

### 3.1 Position-only objective

Episode `e`의 record `t`에서 end-effector 위치를 `p_t∈R³`라 한다. 시작점
`w_j`와 종료점 `w_(j+1)` 사이의 오차는 각 원 trajectory point에서 두
waypoint를 잇는 3D 선분까지의 최단거리다.

```text
e_t = dist(p_t, segment(p_wj, p_w(j+1)))
E(segment) = max_t e_t
segment 허용 조건: E(segment) < η
```

Index 0은 implicit start anchor이고 마지막 record는 항상 반환 waypoint에
포함된다. `η`의 단위는 source `eef_pos_rel`과 같다. 표준 RoboCasa 좌표가
meter라면 `η=0.05`는 5 cm의 최대 기하 편차에 해당한다.

### 3.2 연결 보장 exact DP

AWE commit `7197bb86a20784666dabed90e6eabcf8bb1e9912`의
`dp_waypoint_selection`을 먼저 실행했으나, 반환된 전체 경로를 독립적으로
재계산하면 세 threshold 모두 150/150 episode에서 조건을 위반했다.

원인은 부분 문제에서 검사한 segment 시작점이 최종 waypoint path에 보존되지
않아, 검사한 segment와 실제로 연결되는 segment가 달라지는 구현 문제였다.
따라서 AWE의 거리 목적식과 strict condition `E<η`는 유지하면서, trajectory
index를 node로 하고 조건을 만족하는 contiguous segment만 edge로 허용하는
shortest-path DP인 `exact_pos_only`를 구현했다.

Upstream `awe` 호출은 OpenVLA/OpenPI 호환성을 위해 옵션으로 남겼다. PQ3 최종
결과는 모두 `exact_pos_only`로 다시 생성했다.

### 3.3 실험 조건

| 조건 | Tight | Baseline | Coarse |
| --- | ---: | ---: | ---: |
| Error threshold `η` | 0.02 | **0.05** | 0.10 |
| Waypoint mode | `pos_only` | `pos_only` | `pos_only` |
| DP implementation | `exact_pos_only` | `exact_pos_only` | `exact_pos_only` |
| Episode filter | all | all | all |
| Compute | CPU | CPU | CPU |

`geometric_gripper`는 필수 조건에서 제외했다. `eef_quat_rel`의 quaternion
ordering과 planning chunk의 gripper command를 policy record에 대응시키는 규칙이
확정되지 않았기 때문이다. Position-only 결과에 임의의 scalar gripper 값을
섞는 것보다 별도 ablation으로 남기는 편이 안전하다.

## 4. 코드 상 실행 절차

### 4.1 관련 코드

- [`scripts/groot/export_pq3_trajectories.py`](../scripts/groot/export_pq3_trajectories.py): trusted PKL을 JSONL과 manifest로 변환한다.
- [`event_sae/keyframes/extract.py`](../event_sae/keyframes/extract.py): upstream AWE wrapper와 `exact_pos_only` DP를 제공한다.
- [`scripts/extract_keyframes.py`](../scripts/extract_keyframes.py): threshold별 `waypoint_summary.json`을 생성한다.
- [`scripts/groot/audit_pq3_keyframes.py`](../scripts/groot/audit_pq3_keyframes.py): 오차, 압축률과 event proximity를 독립 계산한다.
- [`event_sae/events/video_timeline.py`](../event_sae/events/video_timeline.py): policy record와 video frame의 시간축을 변환한다.

### 4.2 환경 준비

재현 환경은 [`environment-sae-dev.yml`](../environment-sae-dev.yml)에 고정했다.
AWE revision은 `7197bb86a20784666dabed90e6eabcf8bb1e9912`, NumPy는 1.26.4,
OpenCV는 4.11.0.86, robosuite는 1.4.0이다.
Exact DP와 독립 audit 구현은 Event-SAE commit
`629bcd6b28f360d158f7b578fc422a9b2518700c`에 기록돼 있다.

```bash
conda env update -n event-sae-dev -f environment-sae-dev.yml
conda run -n event-sae-dev python -c \
  "from waypoint_extraction import dp_waypoint_selection; print('AWE import OK')"
conda run -n event-sae-dev python -m pip check
conda run -n event-sae-dev python -m pytest -q tests
```

최종 실행에서는 dependency 오류가 없었고 repo test 26개가 통과했다.
`external/awe`는 gitignored editable clone이며 revision은 환경 파일에 기록돼 있다.

### 4.3 원격 trajectory export

Raw PKL과 MP4는 원격에 유지하고 trajectory JSON만 회수한다.

```bash
python scripts/groot/export_pq3_trajectories.py export \
  --input-dir /home/kimseungjun/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/phase_event_pq3/raw_rollouts \
  --output-dir pq3_stage2_keyframes \
  --trust-pkl
```

실제 exporter 실행에는 Event-SAE commit
`8ed41c568ce682a8ae035253d8be465660a7f082`를 사용했다. 회수한 입력의
SHA-256은 다음과 같다.

```text
trajectory_records.jsonl  7f66ba6953dda8957f3b23cbfa06ad7f90f7058a320849caf4c93592e7445729
trajectory_manifest.json   81df6c80a109ba458b1e49c39444db1f42a5e1f4f9324bcd41fbb94b40d1c89c
trajectory_audit.json      ff87cee5b82b8e745b0f1931b1d4c82f099c643cd3f7cabd75bc99213cf3be14
```

### 4.4 Threshold sweep

Baseline 명령은 다음과 같다.

```bash
conda run -n event-sae-dev python scripts/extract_keyframes.py \
  --trajectory-records-path logs/groot_n15/pq3_stage2_keyframes/trajectory_records.jsonl \
  --output-dir logs/groot_n15/pq3_stage2_keyframes/dp_pos_only_err0p05 \
  --waypoint-mode pos_only \
  --dp-implementation exact_pos_only \
  --err-threshold 0.05 \
  --success-filter all
```

같은 명령에서 output directory와 `--err-threshold`를 각각 `0.02`, `0.10`으로
바꿔 tight와 coarse 결과를 생성한다.

### 4.5 최종 audit

```bash
conda run -n event-sae-dev python scripts/groot/audit_pq3_keyframes.py \
  --trajectory-records-path logs/groot_n15/pq3_stage2_keyframes/trajectory_records.jsonl \
  --trajectory-manifest logs/groot_n15/pq3_stage2_keyframes/trajectory_manifest.json \
  --waypoint-root logs/groot_n15/pq3_stage2_keyframes \
  --event-tolerance 2 \
  --output logs/groot_n15/pq3_stage2_keyframes/waypoint_audit.json
```

`logs/`는 gitignored이다. 코드와 실행 계약은 Git에 남고, 큰 입력과 실행
산출물은 로컬 또는 원격 저장소에 유지한다.

## 5. Metric 정의와 계산

Episode의 record 수를 `T`, 반환 waypoint 수를 `W`라 한다. `W`에는 마지막
record가 포함되지만 implicit start anchor인 index 0은 AWE가 반환하지 않으면
포함되지 않는다.

### 5.1 Geometric max error

```text
GeometricMaxError = max_t dist(p_t, segment(p_wj, p_w(j+1)))
```

이 값이 `η`보다 작은지가 추출 알고리즘의 직접적인 통과 조건이다. 최종
audit의 `awe_geometric_max_error`가 이 값을 의미한다. 결과표의 평균은
150개 episode별 최댓값의 평균이고, 전체 값은 그중 다시 취한 최댓값이다.
`위반`은 이 값이 `η` 이상인 episode 수다.

### 5.2 Time-indexed reconstruction

기하 오차는 이동 속도를 무시하므로, 시간 진행을 함께 보는 독립 지표를 추가했다.

```text
α_t       = (t - w_j) / (w_(j+1) - w_j)
p_hat_t   = (1 - α_t)p_wj + α_t p_w(j+1)
RMSE      = sqrt((1/T) Σ_t ||p_t - p_hat_t||²)
MaxError  = max_t ||p_t - p_hat_t||
```

RMSE는 root mean squared error다. 이 값은 시간축 선형 보간을 사용하므로
geometric threshold보다 클 수 있으며, 그 자체가 threshold 실패는 아니다.

### 5.3 압축률

```text
WaypointDensity   = W / T
CompressionFactor = T / W
```

Density가 낮고 compression factor가 높을수록 더 강한 압축이다. Episode 길이가
서로 다르므로 평균 waypoint 수만 보지 않고 density와 함께 해석한다.
결과표의 평균과 중앙값은 episode별 metric을 동일 가중치로 집계한 값이다.

### 5.4 Event proximity와 단조성

Manifest의 heuristic event step `g`와 가장 가까운 waypoint의 거리가 2 policy
records 이내이면 match로 센다.

```text
EventRecall = matched reference events / all reference events
```

이 값은 semantic ground truth가 아니라 sanity check다. 또한 threshold별
waypoint 집합은 서로 nested일 필요가 없으므로 recall은 단조일 필요가 없다.
반면 `η`가 커질수록 episode별 waypoint 수는 감소하거나 같아야 한다.

## 6. 실험 결과 및 해석

### 6.1 Threshold sweep

| `η` | 전체 waypoint | 평균 waypoint | 평균 density | 중앙 compression | 평균 geometric max error | 전체 geometric max error | 평균 RMSE | Event recall | 위반 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.02 | 1,673 | 11.15 | 15.89% | 6.50× | 0.01926 | 0.01999 | 0.03521 | 63.84% | 0 |
| **0.05** | **913** | **6.09** | **9.50%** | **11.00×** | **0.04788** | **0.04999** | **0.06831** | 41.88% | **0** |
| 0.10 | 603 | 4.02 | 6.59% | 14.71× | 0.09360 | 0.09996 | 0.11660 | 48.97% | 0 |

`η=0.02`는 형상 보존과 event proximity가 가장 좋지만 평균 11.15개 waypoint를
요구한다. `η=0.10`은 평균 4.02개로 가장 강하게 압축하지만 시간축 RMSE가
가장 크다. `η=0.05`는 모든 episode에서 기하 조건을 만족하면서 waypoint를
평균 6.09개로 줄여 fidelity와 압축 사이의 중간점을 제공한다.

Event recall이 `η=0.10`에서 `η=0.05`보다 높은 것은 오류가 아니다. 서로 다른
최소 경로가 선택되면서 일부 coarse waypoint가 heuristic event 근처에 우연히
위치한 결과다. 이 수치를 threshold 선택의 단독 기준으로 사용하지 않았다.

### 6.2 Upstream AWE와 exact DP 비교

| `η` | Upstream 위반 episode | Upstream 전체 geometric max error | Exact DP 위반 episode |
| ---: | ---: | ---: | ---: |
| 0.02 | 150/150 | 0.11609 | 0/150 |
| 0.05 | 150/150 | 0.16608 | 0/150 |
| 0.10 | 150/150 | 0.23254 | 0/150 |

이 비교는 threshold를 느슨하게 바꿀 문제가 아니라 반환 경로의 segment 연결을
고쳐야 한다는 것을 보여준다. Exact DP 결과는 세 threshold 모두 조건을
만족했고, threshold 증가 시 waypoint 수가 증가한 episode도 0/150이었다.

### 6.3 Cell별 baseline 안정성

| Cell | 평균 waypoint | 평균 density | 평균 RMSE |
| --- | ---: | ---: | ---: |
| `pq3_drawer_left` | 6.03 | 8.35% | 0.07198 |
| `pq3_drawer_right` | 5.97 | 6.40% | 0.07846 |
| `pq3_ppcc_bread` | 6.00 | 11.91% | 0.05928 |
| `pq3_ppcc_beer` | 6.43 | 8.18% | 0.06999 |
| `pq3_ppcc_pizza_cutter` | 6.00 | 12.65% | 0.06182 |

Cell별 평균 waypoint는 5.97–6.43으로 유사하다. Density 차이는 주로 episode
길이 차이에서 온다. 특정 task에만 waypoint가 과도하게 집중된 증거는 없다.

5개 cell에서 2개씩, 총 10개 episode의 3D trajectory overlay도 수동 확인했다.
시작점과 종료점이 연결됐고 큰 굴곡에는 waypoint가 배치됐으며 범위 밖 index나
비연속 segment는 발견되지 않았다.

## 7. 결론과 Stage 3 handoff

Stage 2는 repo 범위에서 완료됐다. `η=0.05`, `pos_only`, `exact_pos_only`를
canonical 조건으로 채택한다.

```text
logs/groot_n15/pq3_stage2_keyframes/
├── trajectory_records.jsonl
├── trajectory_manifest.json
├── trajectory_audit.json
├── dp_pos_only_err0p02/waypoint_summary.json
├── dp_pos_only_err0p05/waypoint_summary.json  ← canonical
├── dp_pos_only_err0p10/waypoint_summary.json
└── waypoint_audit.json
```

GR00T 영상은 policy record와 1:1이 아니다. Record 수를 `R`, 실제 실행 action
수를 `A`, render 간격을 `S`라 하면 다음 mapping을 사용한다.

```text
expected_frames = ceil(R × A / S)
frame_start(r)  = ceil(r × A / S)
frame_stop(r)   = ceil((r + 1) × A / S) - 1
```

PQ3에서는 `A=5`, `S=2`이며 150개 MP4 모두 이 식을 만족했다. Stage 3는
canonical waypoint를 이 mapping으로 video frame에 연결한 뒤 주변 window를
추출하고, vision/state descriptor와 Stage 1 SAE activation을 결합한다.

현재 결과가 보장하는 것은 kinematic fidelity다. Semantic event 의미, SAE
feature 대응과 behavior 영향은 Stage 3 이후의 독립 검증 대상으로 남긴다.
