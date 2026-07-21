# GR00T Event-Grounded SAE 재현 계획 — 1/6: L15 SAE 학습

**문서 상태:** **Stage 1 완료** — full-data audit를 통과한 L15 1.2k early-stop checkpoint 선택
**작성일:** 2026-07-21
**현재 과정:** 전체 6단계 중 **1단계**
**원 논문 파이프라인 대응:** Phase 1의 step (a) activation collection과
step (b) SAE training 완료

## Index

- [1. 전체 목표와 현재 위치](#1-전체-목표와-현재-위치)
- [2. Stage 0 전제조건: 이미 확보된 데이터](#2-stage-0-전제조건-이미-확보된-데이터)
- [3. Stage 1의 목표](#3-stage-1의-목표)
  - [L15 선택의 의미](#l15-선택의-의미)
- [4. 고정할 SAE dataset identity](#4-고정할-sae-dataset-identity)
- [5. 세부 실행 순서](#5-세부-실행-순서)
  - [1.1 Source audit와 계약 동결](#11-source-audit와-계약-동결)
  - [1.2 PQ3 token-wise loader 개정](#12-pq3-token-wise-loader-개정)
  - [1.3 기존 smoke의 지위](#13-기존-smoke의-지위)
  - [1.4 L15 calibration pilot](#14-l15-calibration-pilot)
  - [1.5 Checkpoint 품질 audit](#15-checkpoint-품질-audit)
  - [1.6 L15 production 학습](#16-l15-production-학습)
- [6. 산출물 계약](#6-산출물-계약)
- [7. 완료 gate](#7-완료-gate)
- [8. 다음 단계로의 handoff](#8-다음-단계로의-handoff)

## 1. 전체 목표와 현재 위치

최종 목표는 Event-Grounded Sparse Autoencoder 방법을 GR00T N1.5와
RoboCasa rollout에 적용해 재현하는 것이다. 여기서 재현은 원 논문의 특정
수치를 그대로 복제한다는 뜻이 아니라, 다음 방법론적 연결을 GR00T에서
끝까지 구성하고 검증한다는 뜻이다.

```text
closed-loop activation
        -> sparse SAE feature
        -> SAE와 독립적으로 만든 event cluster
        -> event-cluster × SAE-feature temporal score
        -> 선택 feature의 closed-loop causal intervention
```

전체 작업을 실행 단위 기준으로 다음 6단계로 나눈다.

|     전체 순서 | 과정                                         | 원 파이프라인 대응 | 핵심 산출물                                |
| ------------: | -------------------------------------------- | ------------------ | ------------------------------------------ |
| **1/6** | **GR00T DiT L15 SAE 학습과 품질 검증** | Phase 1, (a)–(b)  | 검증된 L15 SAE checkpoint                  |
|           2/6 | Kinematic keyframe 추출                      | Phase 2, (c)       | episode별 AWE waypoint                     |
|           3/6 | Keyframe media와 event descriptor 생성       | Phase 3, (d)–(e)  | frame bundle, vision/state descriptor      |
|           4/6 | Event clustering과 VLM annotation            | Phase 3, (f)–(g)  | task-local labeled event cluster           |
|           5/6 | Event-feature scoring과 feature ranking      | bridge, (h)–(j)   | cluster × SAE feature score, 후보 feature |
|           6/6 | Closed-loop intervention                     | Phase 4, (k)       | feature별 SR 변화와 인과 검증              |

이 문서는 **1/6만** 다룬다. Keyframe, SigLIP, VLM annotation, event
clustering, feature ranking, intervention은 현재 단계의 범위가 아니다.

## 2. Stage 0 전제조건: 이미 확보된 데이터

Activation collection은 새로 수행하지 않는다. SAE 학습의 **유일한 원천은
PQ3 full-token rollout**로 고정한다. Raw activation은 승준 서버의 HDD에
read-only로 두고, 로컬 workspace/NVMe로 복제하지 않는다.

```text
/home/kimseungjun/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/
  phase_event_pq3/raw_rollouts/
```

2026-07-21 원격 원본에서 확인한 입력 계약은 다음과 같다.

| 항목                    | 확인값                                                            |
| ----------------------- | ----------------------------------------------------------------- |
| rollout PKL             | 150 episodes = 5 instruction cells × 30                           |
| record별 DiT activation | `[L=7, K=4, T=49, D=1536]`, fp16                                  |
| capture layer ID        | `[0, 2, 4, 8, 10, 12, 15]`                                        |
| denoise step            | `0, 1, 2, 3`                                                       |
| source model token      | 49개 = state 1 + future 32 + action 16, 평균 없이 전부 보존        |
| feature kind            | `groot_n15_dit_block_residual_full_tokens_denoise`                |
| feature axes            | `layer, denoise_step, model_token, feature_dim`                   |
| capture token mode      | `all_token_full`                                                   |
| 부가 정렬 정보          | `states`, `feature_phases`, `action_vectors`, task/scene/success metadata |

`all_token_full`은 **원본 capture 계약**이다. SAE가 49개 token을 전부 학습한다는
뜻이 아니다. OpenPI π0.5 action-expert 경로에 대응하도록 Stage 1 SAE는 이
tensor에서 마지막 `model_action_horizon=16` action token만 선택해 사용한다.
state 1개와 future 32개는 source와 provenance에는 보존하지만 SAE 학습 row에는
넣지 않는다.

고정된 5개 cell은 다음과 같다.

| RoboCasa task               | cell                    | PKL |
| --------------------------- | ----------------------- | --: |
| `OpenDrawer`                | `pq3_drawer_left`       |  30 |
| `OpenDrawer`                | `pq3_drawer_right`      |  30 |
| `PickPlaceCounterToCabinet` | `pq3_ppcc_bread`        |  30 |
| `PickPlaceCounterToCabinet` | `pq3_ppcc_beer`         |  30 |
| `PickPlaceCounterToCabinet` | `pq3_ppcc_pizza_cutter` |  30 |

`phase_event_6p`의 `[L=7,K=4,D=1536]` activation은 마지막 16개 action
token을 이미 평균한 PQ2 자료다. Token-wise SAE의 원천으로 사용하거나 PQ3와
혼합하지 않는다. `steer_eval_pq2`에는 평가 TSV/JSON만 남아 있고 activation은
없으므로 역시 입력 경로가 아니다.

PKL은 Python code execution이 가능한 형식이므로, 이 프로젝트에서 생성하고
위 경로로 provenance가 고정된 archive에만 `--trust-pkl`을 사용한다. Source
PKL은 수정하거나 재저장하지 않는다.

위 절대 경로는 원격 서버에서만 유효하다. 원격에는 의존성이 없는 standalone
exporter 한 파일만 Git bundle로 전달했다. Exporter가 PKL을 하나씩 audit하고
L15 action-token fp16 shard만 만들었으며, raw PKL은 전송하거나 수정하지 않았다.
검증된 shard 150개(2.3 GiB)만 로컬로 회수해 단일 activation cache로 병합했고,
SAE 학습과 checkpoint audit는 로컬 GPU 5에서 수행했다.

## 3. Stage 1의 목표

GR00T N1.5 DiT **physical layer 15 residual stream**을 재구성하면서도 sparse한
BatchTopK SAE를 학습하고, 이후 event-feature scoring과 closed-loop hook에서
재사용할 수 있는 checkpoint를 만든다.

완료 상태는 단순히 `ae.pt`가 생성된 상태가 아니다. 다음 세 조건을 모두
만족해야 한다.

1. SAE 입력 activation 계약이 manifest에 기록돼 있다.
2. 학습이 끝나고 checkpoint를 다시 load할 수 있다.
3. reconstruction과 sparsity 기본 진단에서 numerical failure나 명백한
   collapse가 없다.

검증 범위는 이 repository의 OpenVLA/OpenPI SAE 학습 pipeline과 같은 수준으로
제한한다. Stage 1은 학습 가능한 activation과 후속 단계에서 load 가능한
checkpoint를 확인하는 단계다. Temporal metadata join, reconstruction hook,
held-out 일반화와 feature 의미 검증은 각각 실제로 사용하는 후속 단계에서
다룬다.

### L15 선택의 의미

L15는 16-block DiT에서 마지막으로 capture한 physical block이다. 최종 action
prediction에 가까운 motor-proximal residual stream을 우선 분석한다는
architectural prior와 연구자 선택에 따라 primary layer로 사전 고정한다.

이 선택은 “L15가 다른 layer보다 우수하다는 실험 결과”를 뜻하지 않는다.
현재 L15의 Event-Grounded SAE 성능 비교는 아직 수행되지 않았다. 본 재현의
primary checkpoint를 L15로 고정하고, layer ablation이 필요할 때만 별도
계획에서 L8 등과 비교한다.

코드는 L15가 tensor의 마지막 slot이라고 추정하면 안 된다. 반드시
`capture_layers.index(15)`로 physical layer ID를 array position으로 변환한다.

## 4. 고정할 SAE dataset identity

이번 checkpoint의 identity는 다음 조합 전체다.

```text
model             = GR00T N1.5
benchmark         = RoboCasa
dataset scope     = phase_event_pq3 / 5 instruction cells / 150 rollouts
pathway           = DiT block residual, action-token slice from full-token capture
physical layer    = 15
denoise policy    = all: K=4 states를 각각 독립 row로 사용
token policy      = action-only: 마지막 A=16 token을 각각 독립 row로 사용
activation dim    = 1536
dictionary size   = 1536 (1× expansion)
BatchTopK k       = 64
```

현재 source는 RoboCasa 전체 task 분포가 아니라 위 5개 instruction cell이다.
따라서 산출물을 “RoboCasa-general SAE”라고 부르지 않는다.

선택한 L15의 원본은 `[N,K=4,T=49,D=1536]`이다. Source metadata로
`T = state 1 + future 32 + action 16` 계약을 확인한 뒤 마지막 16개 action
token만 잘라 `[N,K=4,A=16,D=1536]`으로 만든다. K축과 A축은 평균하지 않고
`[N×4×16,1536]`으로 펼치며, 각 denoise-action-token residual vector가 SAE의
한 학습 row다. Manifest에는 source file 순서와 row flatten 순서를 기록한다.

49개 전체를 공유 SAE 하나에 넣는 실험은 Stage 1 primary 계약이 아니다.
필요하면 별도 checkpoint와 명시적인 ablation으로만 수행한다.

## 5. 세부 실행 순서

### 1.1 Source audit와 계약 동결

전체 PKL을 읽어 다음을 검증한다.

- file, record, activation-row 수와 5개 cell별 file inventory
- `feature_kind`, `feature_axes`, `capture_layers`
- `capture_token_mode == all_token_full`
- 모든 `hidden_states[t]`의 `[7, 4, 49, 1536]` shape
- `model_action_horizon == 16`과 token layout `1 + 32 + 16 == 49`
- SAE 입력 slice가 absolute model-token index `33..48`인지 여부
- 선택한 L15 action-token activation의 NaN/Inf 부재
- source activation dtype의 일관성
- 선택 layer가 physical L15인지 여부
- source file 목록과 file별 record/row 범위

전체 audit와 bounded export는 `scripts/groot/export_pq3_activation_shards.py`를
standalone으로 원격에서 실행했다. 원격 접속과 전송에는 temporal_vla의
`scripts/utils/remote_compute.sh`만 사용했다.

```bash
# 원격: PKL 하나씩 검증하고 L15 action-token shard 생성
python export_pq3_activation_shards.py export \
  --input-dir /home/kimseungjun/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/phase_event_pq3/raw_rollouts \
  --output-dir pq3_l15_action_shards \
  --trust-pkl --layer 15

# 로컬: 전송된 150개 shard를 검증하며 단일 cache로 병합
conda run -n event-sae-dev python scripts/groot/export_pq3_activation_shards.py merge \
  --shard-dir /home/dongkyu/pkt_ws/temporal_vla/outputs/event_sae_stage1/pq3_l15_action_shards \
  --output-cache logs/groot_n15/pq3_l15_activation_cache/l15_action_tokens.pt
```

이 audit은 의도적으로 SAE 입력 activation 계약만 검사한다. `states`,
`feature_phases`, `action_vectors`의 temporal join은 Stage 2와 Stage 5에서 해당
metadata를 실제로 사용할 때 검증하며, Stage 1 완료 조건에는 포함하지 않는다.

### 1.2 PQ3 token-wise loader 구현

기존 `scripts/groot/train_sae_robocasa.py`는 아래 PQ2 pooled 계약을 사용했다.

```text
feature_axes = [layer, denoise_step, feature_dim]
hidden shape = [L, K, D]
rows         = [record × K, D]
```

현재 loader는 다음 PQ3 계약을 구현하고 합성 test로 고정한다.

- exact `feature_kind`, `feature_axes`, `capture_token_mode`,
  `capture_layers=[0,2,4,8,10,12,15]` 검증
- physical layer ID를 `capture_layers.index(layer_id)`로 선택
- `model_action_horizon=16`을 metadata에서 검증하고 `T=49`의 마지막 16개만 선택
- 선택 layer의 `[K,A=16,D]`를 pooling 없이 `[K×A,D]` row로 변환
- `source_files`, file별 record/row 범위와 flatten `row_order`를 manifest에 기록
- 기본 전체 실행에서 5개 cell별 30개와 전체 150개 file inventory를 엄격히 검증
- `--max-files` 또는 `--allow-partial-inventory`를 준 경우에만 부분집합 허용
- PQ2 pooled activation, 서로 다른 token count/dtype, state/future token 혼입 거부
- 전체 record 수와 `records×4×16` 학습 row 수를 audit manifest에 기록
- resident activation은 source fp16으로 유지하고 현재 학습 batch만 float32로 변환
- 선택 activation materialization 예상 peak가 `--max-ram-gib`를 넘으면 학습 전 실패

합성 contract/CLI/cache round-trip test 9개와 실제 원격 150개 PKL 전체 audit가
통과했다. 실제 inventory는 12,041 records, 770,624 action-token rows,
`[770624,1536]` fp16, 선택 activation 2,367,356,928 bytes다. Reconstruction
hook에서 full residual을 교체하는 동작은 Stage 6 intervention 구현과 함께
검증한다.

### 1.3 기존 smoke의 지위

7개 capture layer 모두에서 100-step plumbing smoke가 완료돼 있다. L15 smoke
checkpoint는 다음 위치에 있다.

```text
logs/groot_n15/ppcs_apple_all_layers_smoke/layer_15/trainer_0/ae.pt
```

이 checkpoint는 **PQ2 pooled loader 기준으로** scheduler, training loop와
저장 경로가 동작했다는 제한된 증거다. PQ3 `[L,K,T,D]` loader 검증이나
token-wise SAE 품질의 증거가 아니며, production SAE나 feature 해석에
사용하지 않는다.

실제 PQ3 cache로 수행한 별도 smoke는
`logs/groot_n15/pq3_l15_smoke_100/`에 저장했다. 100-step 학습, final
`ae.pt/config.json` 생성, 새 process reload와 8,192-row audit가 모두
통과했다. Training FVE는 step 90에서 0.647이었다. 이 run은 upstream
`threshold_start_step=1000` 이전이라 inference threshold가 `-1`인 plumbing
smoke이며, sparsity 품질 checkpoint로는 사용하지 않는다.

### 1.4 L15 calibration pilot

BatchTopK의 inference threshold는 기본값 `-1`이고 upstream trainer에서
`step > 1000`일 때부터 갱신된다. 정확히 1,000 steps로는 inference sparsity를
검증할 수 없으므로 pilot을 1,200 steps로 실행해 199회의 threshold update를
포함했다.

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n event-sae-dev python \
  scripts/groot/train_sae_robocasa.py \
  --activation-cache logs/groot_n15/pq3_l15_activation_cache/l15_action_tokens.pt \
  --save-dir logs/groot_n15/pq3_l15_pilot_1200 \
  --layer 15 --dict-size 1536 --sae-k 64 --lr 1e-4 \
  --steps 1200 --batch-size 4096 --warmup-steps 100 \
  --save-every 600 --log-steps 100 --device cuda:0
```

저장 threshold는 19.956이었고 full-data audit에서 FVE 0.987958, cosine
0.997290, L0 63.979, dead 252/1536(16.4%)이었다. 이 단계는 동일 source
분포의 calibration이며 held-out 성능을 뜻하지 않는다.

### 1.5 Checkpoint 품질 audit

각 checkpoint 후보를 새 process에서 reload하고 동일한 source의 결정론적
row 순열에 대해 다음 값을 계산한다.

- reconstruction MSE
- fraction of variance explained
- input/reconstruction cosine similarity
- 평균 L0
- feature별 firing frequency
- dead feature 수와 비율
- feature activation magnitude
- encode/decode NaN/Inf

`scripts/groot/audit_sae_checkpoint.py`는 별도 평가 공식을 구현하지 않는
얇은 GR00T adapter다. PQ3 입력은 `load_layer_activations()`, checkpoint
reload는 기존 `load_batch_topk_sae()`, L2/L0/FVE/cosine/alive 평가는
`dictionary_learning.evaluation.evaluate()`를 그대로 재사용한다. GR00T
전용 추가 로직은 source/checkpoint identity 검증, deterministic row sample,
feature별 firing 통계와 JSON/NPZ 저장뿐이다.

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n event-sae-dev python \
  scripts/groot/audit_sae_checkpoint.py \
  --activation-cache logs/groot_n15/pq3_l15_activation_cache/l15_action_tokens.pt \
  --sae-checkpoint logs/groot_n15/pq3_l15_stage1_selected/trainer_0/ae.pt \
  --batch-size 512 --max-audit-rows 0 --device cuda:0
```

기본 출력은 checkpoint run root의 `sae_quality.json`과
`sae_quality_by_feature.npz`다. 8,192-row 표본은 희소 firing feature를 많이
놓쳐 pilot dead 수를 1,392개로 과대평가했다. 따라서 최종 checkpoint 선택에는
전체 complete batch인 770,560 rows를 사용했다. Stage 1의 audit은 동일 source
분포이므로 이를 “held-out 성능”이나 layer 일반화 증거라고 부르지 않는다.
별도의 reconstruction 수치 threshold는 사전 등록하지 않았지만, full-data에서
dictionary 대부분이 전혀 사용되지 않는 후보는 utilization collapse로
거부했다.

### 1.6 L15 production 학습

Pilot gate 통과 뒤 다음 최초 production 후보를 GPU 5에서 실행했다.

```text
dict_size    = 1536
k            = 64
lr           = 1e-4
steps        = 10,000
batch_size   = 4,096
warmup_steps = 1,000
seed         = 0
```

실행 자체와 checkpoint 저장은 완료됐지만 최종 후보는 full-data utilization
gate에서 거부했다.

| 후보 | FVE | cosine | L0 | dead / 1536 | 결정 |
| --- | ---: | ---: | ---: | ---: | --- |
| 1.2k calibration, full rows | 0.987958 | 0.997290 | 63.979 | 252 (16.4%) | **선택** |
| 10k final, full rows | 0.993197 | 0.998570 | 63.987 | 1303 (84.8%) | 거부 |

10k 후보는 reconstruction은 더 좋지만 threshold를 쓰지 않는 training-style
batch-top-k 진단에서도 100k rows 중 141개 feature만 사용했다. 따라서 문제는
inference threshold calibration이 아니라 장기 최적화 중 dictionary 사용이
소수 feature로 집중된 현상이다. Stage 1 범위를 hyperparameter sweep으로
확장하지 않고 full-data audit를 통과한 1.2k early-stop checkpoint를 handoff
산출물로 선택했다. 10k run은
`logs/groot_n15/pq3_l15_production_10000/`에 rejection diagnostic으로 보존한다.

## 6. 산출물 계약

선택된 canonical handoff는 다음 위치에 고정했다.

```text
logs/groot_n15/pq3_l15_stage1_selected/
├── groot_source_manifest.json
├── sae_quality.json
├── sae_quality_by_feature.npz
├── selection.json
└── trainer_0/
    ├── ae.pt
    └── config.json
```

`selection.json`은 1.2k early-stop 선택 근거와 10k 후보의 rejection 지표를
함께 기록한다.

`groot_source_manifest.json`에는 최소한 다음 provenance를 기록한다.

- source root와 source file inventory
- feature kind/axes와 capture layer IDs
- selected physical layer 15
- source capture token mode와 source token 수
- SAE token scope, action horizon과 absolute slice
- denoise policy와 denoise-step 수
- record와 activation-row 수
- activation dimension과 source/resident dtype
- materialization 예상 memory

Dictionary size, k, learning rate, step 수 등 trainer 설정은
`trainer_0/config.json`을 기준으로 한다. Publication용 source hash나 두
repository의 revision bundle은 Stage 1 학습 검증의 필수 산출물로 두지 않는다.

로컬 `pq3_l15_activation_cache/l15_action_tokens.pt`와 전송 shard는 학습
재현을 위한 derived intermediary다. Stage 2/5 handoff의 기준 산출물은 위
selected checkpoint/config/source manifest/quality/selection 파일이며 raw PKL은
계속 원격 source-of-truth로 둔다.

## 7. 완료 gate

다음을 모두 만족해야 Stage 1을 완료로 표시하고 Stage 2 keyframe 추출로
넘어간다.

- [x] PQ3 5 cell × 30 = 150 PKL source contract가 혼입 없이 검증됨
- [x] `feature_kind`, four-axis `feature_axes`, `all_token_full`이 검증됨
- [x] physical L15가 metadata를 통해 선택됨
- [x] `model_action_horizon=16`과 token layout `1+32+16=49`가 검증됨
- [x] 모든 `[record,4,16]` action-token row가 pooling 없이 누락 없이 구성됨
- [x] state/future 33개 token이 SAE 학습 row에서 배제됨
- [x] PQ2 `phase_event_6p` activation이 입력에서 배제됨
- [x] 선택 activation과 train/encode/decode에서 NaN/Inf가 없음
- [x] PQ3 전용 100-step smoke에서 `ae.pt`와 `config.json`이 생성되고 reload됨
- [x] threshold update를 포함한 1.2k pilot checkpoint가 reload되고 full-data audit됨
- [x] reconstruction MSE, FVE, cosine, L0, firing/dead-feature 진단이 저장됨
- [x] selected checkpoint에 numerical failure나 명백한 utilization collapse가 없음
- [x] selected checkpoint와 config/source manifest/quality/selection이 함께 저장됨

다음 항목은 Stage 1 완료 gate가 아니다.

- `states`, phase, action metadata의 temporal join 검증
- full residual reconstruction에서 앞 33개 token의 bitwise 보존 검증
- held-out 성능, layer ablation과 feature의 event 의미
- checkpoint SHA와 publication용 revision bundle

Stage 1에서는 SAE feature가 특정 event를 의미한다고 주장하지 않는다. 그
연결은 Stage 5의 temporal scoring에서 처음 만들어지고, 인과적 의미는 Stage
6 closed-loop intervention 이후에만 평가한다.

## 8. 다음 단계로의 handoff

Stage 1이 완료되면 Stage 2는 같은 rollout PKL의
`states[*]["observation.state.eef_pos_rel"]` trajectory에서 AWE waypoint를
추출한다. Keyframe은 SAE activation을 보고 고르지 않는다.

Stage 5에서는 Stage 1의 L15 checkpoint로 원본 activation을 다시 encode한 뒤,
episode/inference/denoise/action-token offset을 통해 event waypoint 주변
activation과 결합한다. 이때 activation row와 trajectory metadata의 정렬을
Stage 5 입력 audit로 별도 검증한다.
