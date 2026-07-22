# GR00T Event-Grounded SAE Stage 3 — 실행 계획

- 작성일: 2026-07-22
- 대상: GR00T N1.5, RoboCasa PQ3
- 상태: **원격 미디어 패키저 구현·로컬 테스트 완료, 원격 pilot 대기**
- 로컬 환경: `event-sae-dev`
- 원격 미디어 환경: `event-sae-media`
- temporal_vla 구현: `feat/groot-event-sae-media`, `86a81ae`

Stage 3의 목표는 Stage 2에서 찾은 end-effector waypoint를 영상과 robot state가 결합된 event sample로 바꾸고, 같은 task 안에서 반복되는 event를 묶어 사람이 읽을 수 있는 phrase와 phase를 부여하는 것이다. 이 단계에서는 SAE activation이나 checkpoint를 사용하지 않는다. SAE feature와 event를 연결하는 작업은 Stage 3 완료 뒤 ranking bridge에서 수행한다.

## 0. 결정 요약

1. Stage 2의 `pos_only`, `exact_pos_only`, `η=0.05` waypoint 913개를 고정 입력으로 사용한다.
2. 원본 PKL과 MP4는 원격에 둔다. 원격은 MP4 decode와 JPEG frame bundle 생성만 수행한다.
3. 원격 코드는 temporal_vla의 `scripts/event_sae/`, 산출물은 `outputs/event_sae/`에 둔다. 원격 서버에서 코드를 직접 수정하지 않는다.
4. 원격 `event-sae-media`는 Python, imageio, FFmpeg backend, Pillow만 포함하는 최소 환경이다. PyTorch, SigLIP, clustering, Gemini는 설치하지 않는다.
5. 선택된 JPEG와 provenance JSON만 로컬로 회수한다. 이후 embedding, clustering, annotation, audit은 모두 기존 `event-sae-dev`에서 수행한다.
6. 한 waypoint당 record offset `[-4,-2,0,2,4]`의 5개 frame을 사용한다. `first/center/last` frame anchor는 task별 success/failure pilot으로 결정한다.
7. baseline descriptor는 `[mean SigLIP vision, eef_pos, progress]`, 가중치는 `1.0/0.5/0.4`다.
8. success/failure와 heuristic event label은 clustering 입력이나 threshold 선택에 사용하지 않고, 설정을 고정한 뒤 confound audit에만 사용한다.
9. VLM label은 설명용 metadata다. Cluster membership이나 SAE feature score를 label 문구로 정하지 않는다.

## 1. 목표와 주장 범위

핵심 질문은 다음과 같다.

> Stage 2 waypoint 주변의 시각 변화와 end-effector state를 사용하면, 같은 task의 여러 episode에서 반복되는 동작 event를 안정적으로 묶을 수 있는가?

Stage 3가 검증할 것은 다음 네 가지다.

- policy record와 MP4 frame의 시간축 연결이 정확한가?
- approach/withdraw처럼 방향이 반대인 event가 무분별하게 섞이지 않는가?
- cluster가 progress, episode boundary, success/failure만 대리하지 않는가?
- drawer와 pick-place 모두에 일관된 phrase와 phase를 붙일 수 있는가?

Stage 3 결과는 PQ3의 고정 scene/camera 안에서 반복되는 event structure에 대한 기술이다. Task description과 cell이 1:1이므로 task 의미와 scene 효과를 분리했다는 주장은 하지 않는다. Cluster와 VLM label을 인과적 robot skill로도 간주하지 않는다.

## 2. 고정 입력과 규모

### 2.1 Stage 2 handoff

```text
logs/groot_n15/pq3_stage2_keyframes/
├── trajectory_records.jsonl
├── trajectory_manifest.json
└── dp_pos_only_err0p05/waypoint_summary.json
```

```text
trajectory_records.jsonl  7f66ba6953dda8957f3b23cbfa06ad7f90f7058a320849caf4c93592e7445729
trajectory_manifest.json   81df6c80a109ba458b1e49c39444db1f42a5e1f4f9324bcd41fbb94b40d1c89c
waypoint_summary.json      736f50da3c71ca8020a61cd08b4dd99cff5a2dbca9cf130a4a19b0d798b81f0e
```

`trajectory_manifest.json`은 PKL과 MP4의 상대경로, task/cell, success, `n_action_steps`, `steps_per_render`를 제공한다. `waypoint_summary.json`은 episode별 waypoint record index를 제공한다. 원격 패키징에는 이 두 파일만 필요하고, `trajectory_records.jsonl`은 로컬 descriptor 생성에서 사용한다.

### 2.2 Inventory

| 항목 | 값 |
| --- | ---: |
| Episodes | 150 |
| Policy records | 12,041 |
| Exact-stem MP4 | 150 / 150 |
| Video frames | 30,127 |
| Canonical waypoints / samples | 913 |
| JPEG frames | 4,565 |
| Task/cell | 5 |
| Success / failure episodes | 101 / 49 |

150개 원본 MP4의 실제 합계는 106,318,617 bytes, 즉 101.393 MiB다. 전체 영상을 옮길 수도 있는 크기지만, 데이터 소스가 바뀌어도 같은 절차를 재사용할 수 있도록 raw video 인접 위치에서 frame을 뽑는 구조를 채택한다.

| Task | Episodes | Success | Failure | Waypoints |
| --- | ---: | ---: | ---: | ---: |
| Open the left drawer. | 30 | 17 | 13 | 181 |
| Open the right drawer. | 30 | 16 | 14 | 179 |
| Pick beer and place in cabinet. | 30 | 19 | 11 | 193 |
| Pick bread and place in cabinet. | 30 | 23 | 7 | 180 |
| Pick pizza cutter and place in cabinet. | 30 | 26 | 4 | 180 |

## 3. 실행 구조와 환경

### 3.1 Local/remote 경계

```text
Stage 2 JSON ──push──> remote temporal_vla
                         │
raw MP4 ──decode─────────┤  event-sae-media
                         │
                         └── JPEG + samples.jsonl + hash audit
                                      │
                                   pull
                                      ▼
local Event-SAE / event-sae-dev
  └── SigLIP → descriptor → task-local clustering → VLM annotation → audit
```

| 위치 | 책임 |
| --- | --- |
| Remote `event-sae-media` | MP4 lookup/decode, record-to-frame 변환, JPEG 저장, hash audit |
| Local `event-sae-dev` | bundle import, SigLIP, descriptor, clustering, VLM, 최종 audit |
| 원격에 남김 | raw PKL/MP4와 임시 bundle |
| 로컬로 회수 | JPEG, `samples.jsonl`, manifest, packaging audit |

Stage 1·2와 Stage 3의 연구 코드는 모두 로컬 `event-sae-dev`를 canonical project environment로 사용한다. 원격 환경은 연구 환경의 복제본이 아니라 비디오 codec 차이를 격리하는 작은 media adapter다.

### 3.2 코드와 산출물 위치

```text
# local authoring worktree
/home/dongkyu/pkt_ws/temporal_vla/scripts/event_sae/

# remote checked-out code
/home/kimseungjun/workspace/temporal_vla/scripts/event_sae/

# remote generated artifacts
/home/kimseungjun/workspace/temporal_vla/outputs/event_sae/groot_n15/
├── pq3_stage3_inputs/
└── pq3_stage3_media/
```

코드는 로컬에서 test와 commit을 통과한 뒤 Git으로 원격에 동기화한다. 모든 원격 명령과 작은 결과 회수는 temporal_vla의 `scripts/utils/remote_compute.sh`만 사용한다. `scripts/event_sae/` 아래에는 생성 결과를 쓰지 않는다.

### 3.3 원격 최소 환경

`scripts/event_sae/environment-media.yml`의 direct dependency는 다음뿐이다.

```text
Python 3.10
NumPy 1.26
imageio 2.37.3
Pillow 11.3.0
imageio-ffmpeg 0.6.0
```

이 환경을 별도로 두는 이유는 원격의 기존 temporal_vla 환경을 건드리지 않고 MP4 decode 결과를 재현하기 위해서다. GPU, CUDA, 모델 weight, API credential은 필요하지 않다.

## 4. Media bundle 계약

### 4.1 Record-to-frame 변환

Episode의 policy record 수를 `N`, 한 record가 예측하는 action 수를 `A`, 한 번의 render마다 소비하는 action 수를 `R`이라 하면 예상 video frame 수는 다음과 같다.

```math
F = \left\lceil \frac{N A}{R} \right\rceil
```

Record `i`에 대응하는 frame 구간은 다음과 같다.

```math
s_i = \left\lceil \frac{iA}{R} \right\rceil,
\qquad
e_i = \left\lceil \frac{(i+1)A}{R} \right\rceil - 1
```

`first`, `center`, `last` anchor는 각각 `s_i`, `floor((s_i+e_i)/2)`, `e_i`를 선택한다. PQ3에서는 `A=5`, `R=2`다. 패키저는 모든 episode에서 실제 MP4 frame 수가 `F`와 정확히 같은지 확인하고 하나라도 다르면 중단한다.

Waypoint record `w`의 요청 window는 다음과 같다.

```math
W_w = \{w-4,\;w-2,\;w,\;w+2,\;w+4\}
```

Episode 경계를 넘는 경우 5개 서로 다른 record를 유지하도록 window 전체를 이동한다. 이때 실제 waypoint가 bundle 안에 없을 수 있으므로 `anchor_frame_position=null`을 허용하고 `record_window_shift`와 `boundary_shift_category`를 반드시 기록한다. 이를 숨기거나 중앙 frame으로 오인하지 않는다.

### 4.2 저장 형식과 provenance

- image: RGB JPEG, quality 95
- sample metadata: newline-delimited JSON인 `samples.jsonl`
- bundle metadata: `media_manifest.json`
- integrity report: `packaging_audit.json`
- format id: `event_sae_stage3_media_v1`

각 frame에는 상대경로, SHA-256, byte 수, width/height, record index, video frame index를 기록한다. 각 sample에는 task/cell, episode, success, waypoint, 실제 record/frame delta, boundary category, 원본 video 상대경로와 SHA-256을 기록한다. Success는 provenance일 뿐 descriptor에는 포함하지 않는다.

`SHA-256`은 파일 내용으로부터 계산하는 256-bit hash다. 전송 전후 hash가 같으면 내용이 바뀌지 않았음을 검사할 수 있다. `samples.jsonl`은 한 줄이 한 sample인 JSON 형식이라 전체 파일을 메모리에 올리지 않고 순차 처리할 수 있다.

### 4.3 현재 구현 검증

temporal_vla의 `stage3_media.py`는 `package`와 `audit` 명령을 제공한다. 로컬 `event-sae-dev`에서 다음 검증을 통과했다.

```text
pytest tests/test_event_sae_stage3_media.py  → 6 passed
python -m py_compile scripts/event_sae/stage3_media.py → passed
stage3_media.py --help → passed
```

테스트 범위는 timeline 수식, episode 시작/끝 window shift, 너무 짧은 episode 거부, 상대경로 탈출 거부, 합성 MP4의 package→JPEG→SHA-256 audit 전체 경로다.

## 5. 로컬 event 표현과 clustering

### 5.1 Baseline descriptor

5개 frame의 SigLIP embedding을 `v_j`, waypoint의 3D end-effector position을 `p`, episode progress를 `t=w/(N-1)`라 한다. Baseline descriptor는 다음과 같다.

```math
\bar v = \operatorname{L2}\!\left(\frac{1}{5}\sum_{j=1}^{5}v_j\right)
```

```math
x = \operatorname{L2}\left[
1.0\bar v,\;0.5\operatorname{zscore}(p),\;0.4\operatorname{zscore}(t)
\right]
```

`L2`는 vector 길이가 1이 되도록 정규화하는 연산이고, `zscore`는 전체 sample의 평균을 빼고 표준편차로 나누는 표준화다. SigLIP은 image-text representation model이며 여기서는 frozen vision encoder로만 사용한다.

현재 GR00T trajectory에는 검증된 scalar gripper state가 없으므로 baseline에 gripper와 quaternion을 임의로 넣지 않는다. Mean pooling은 frame 순서를 직접 표현하지 못한다. 반대 방향 event 혼합이 반복될 때만 center embedding과 `last-first` 차이를 결합한 temporal descriptor를 후속 ablation으로 검토한다.

### 5.2 Task-local clustering

Clustering은 cosine distance를 사용하는 agglomerative clustering으로 수행한다. Cosine distance는 두 vector 방향의 차이다.

```math
d_{cos}(x,y) = 1 - \frac{x^\top y}{\|x\|_2\|y\|_2}
```

Agglomerative clustering은 각 sample을 별도 cluster로 시작해, 거리가 가까운 cluster를 threshold까지 반복해서 합치는 계층적 방법이다. 서로 다른 task/cell은 합치지 않는다.

```text
distance_threshold ∈ {0.12, 0.15, 0.18, 0.21, 0.24}
vision/state/progress = 1.0/0.5/0.4
```

다음 두 ablation으로 state/progress 지배 여부를 확인한다.

| ID | Vision | State | Progress | 확인 질문 |
| --- | ---: | ---: | ---: | --- |
| C0 | 1.0 | 0.5 | 0.4 | reproduction baseline이 안정적인가? |
| C1 | 1.0 | 0.5 | 0.0 | progress가 cluster를 지배하는가? |
| C2 | 1.0 | 0.0 | 0.0 | visual information만으로 구조가 유지되는가? |

Canonical 설정은 결측·중복이 없고, 거의 모두 singleton이거나 task 전체를 하나로 합치지 않으며, 인접 threshold에서 구조가 비교적 안정적이고, exemplar가 같은 target과 동작 방향을 보이는 설정으로 고른다. 조건이 비슷할 때만 기존 default에 가까운 `0.18`을 선택한다. Success/failure 분리가 좋아지는 threshold를 고르지 않는다.

### 5.3 Annotation

Cluster representative frame을 VLM에 제공해 짧은 phrase와 다음 보조 phase를 생성한다.

```text
pre_grasp, immobilization, contact, detach, post_grasp, transition
```

Frame은 연속 frame이 아니라 시간순으로 sampling한 frame임을 prompt에 명시한다. Phrase가 주 설명이고 phase는 논문 호환용 tag다. Drawer가 대부분 `transition`으로 몰리면 taxonomy limitation으로 기록하고 phrase 중심으로 보고한다.

## 6. 실행 절차와 게이트

### Gate 0 — 코드와 환경

- [x] 원격 media adapter와 environment YAML 작성
- [x] 로컬 단위·통합 test 6개 통과
- [x] temporal_vla branch commit/push (`86a81ae`)
- [ ] 원격 temporal_vla HEAD 검증
- [ ] 원격 `event-sae-media` 생성 및 import/decode smoke test
- [ ] Stage 2 manifest/waypoint hash를 원격 입력에서 재검증

### Gate 1 — 10-episode frame-anchor pilot

각 task에서 success/failure 한 episode씩 선택해 `first`, `center`, `last` 세 bundle을 만든다. 다음을 비교해 anchor 하나를 고정한다.

```text
episode_num = 0 2 30 31 60 61 90 91 120 128
pilot samples = 58
```

- `actual_frames == expected_frames`
- 알려진 event/grasp 시점과 영상의 contact/motion 정렬
- waypoint 전후 변화가 5-frame bundle에 포함되는지
- start/end shifted sample이 자연스러운지
- JPEG를 로컬로 회수한 뒤 hash와 image decode가 통과하는지

Pilot bundle의 sample당 byte를 `b_pilot`이라 하면 전체 JPEG 예상량은 `913 × b_pilot`으로 외삽한다. 예상량이 로컬 여유 공간에 비해 비정상적으로 크면 full packaging 전에 JPEG quality 또는 보존 범위를 다시 정한다.

### Gate 2 — Full media와 embedding

- [ ] sample 913개와 JPEG 4,565개, skip 0개
- [ ] sample id와 frame path가 모두 unique
- [ ] Stage 2 task/episode/waypoint와 exact join
- [ ] SigLIP model revision과 processor config 고정
- [ ] embedding 913개가 같은 dimension이며 finite
- [ ] success가 metadata에만 존재하고 descriptor에는 미포함

### Gate 3 — Canonical clustering

- [ ] 모든 sample이 cluster 하나에 정확히 배정됨
- [ ] threshold sweep과 C0/C1/C2 ablation 완료
- [ ] task별 cluster size, episode coverage, singleton 비율 기록
- [ ] within-cluster distance와 threshold 간 구조 안정성 기록
- [ ] progress, boundary, success association을 설정 고정 뒤 사후 보고
- [ ] representative contact sheet를 사람이 검토

`episode coverage`는 cluster에 포함된 서로 다른 episode 수의 비율이다. `singleton`은 sample 하나뿐인 cluster다. Success-associated cluster나 boundary-dominated cluster의 존재 자체는 실패가 아니지만, 이를 보편적 semantic phase로 해석하지 않고 flag해야 한다.

### Gate 4 — VLM annotation

- [ ] drawer와 pick-place를 모두 포함한 10~15 cluster prompt pilot
- [ ] model id, prompt version, temperature, raw response 기록
- [ ] valid 결과를 보존하는 resume와 실패 row만 재시도하는 경로 검증
- [ ] duplicate/missing cluster, API error, parse error 0개
- [ ] canonical cluster의 exemplar와 phrase/phase 수동 검토

## 7. 실행 명령

### 7.1 원격 환경 생성

```bash
conda env create -f scripts/event_sae/environment-media.yml
conda run -n event-sae-media python scripts/event_sae/stage3_media.py --help
```

### 7.2 Pilot packaging

```bash
conda run -n event-sae-media python scripts/event_sae/stage3_media.py package \
  --waypoint-summary outputs/event_sae/groot_n15/pq3_stage3_inputs/waypoint_summary.json \
  --trajectory-manifest outputs/event_sae/groot_n15/pq3_stage3_inputs/trajectory_manifest.json \
  --video-root /home/kimseungjun/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/phase_event_pq3/raw_rollouts \
  --output-dir outputs/event_sae/groot_n15/pq3_stage3_media/pilot_center \
  --frame-anchor center \
  --episode-num 0 2 30 31 60 61 90 91 120 128 \
  --expected-samples 58
```

`first`와 `last`도 서로 다른 output directory에 동일하게 실행한다. 이미 내용이 있는 output directory에는 쓰지 않으므로 재실행 시 새 경로를 사용한다.

### 7.3 Full packaging과 audit

```bash
conda run -n event-sae-media python scripts/event_sae/stage3_media.py package \
  --waypoint-summary outputs/event_sae/groot_n15/pq3_stage3_inputs/waypoint_summary.json \
  --trajectory-manifest outputs/event_sae/groot_n15/pq3_stage3_inputs/trajectory_manifest.json \
  --video-root /home/kimseungjun/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/phase_event_pq3/raw_rollouts \
  --output-dir outputs/event_sae/groot_n15/pq3_stage3_media/full_<anchor> \
  --frame-anchor <selected-anchor> \
  --expected-samples 913

conda run -n event-sae-media python scripts/event_sae/stage3_media.py audit \
  --bundle-dir outputs/event_sae/groot_n15/pq3_stage3_media/full_<anchor> \
  --expected-samples 913
```

## 8. 산출물

원격 portable media bundle은 다음 구조다.

```text
outputs/event_sae/groot_n15/pq3_stage3_media/<run_id>/
├── frames/<sample_id>