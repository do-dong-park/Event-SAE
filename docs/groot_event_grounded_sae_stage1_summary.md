# GR00T Event-Grounded SAE — Stage 1 결과 요약

**상태:** 완료

**선택 checkpoint:** `logs/groot_n15/pq3_l15_stage1_selected/trainer_0/ae.pt`

**상세 실행 기록:** [groot_event_grounded_sae_stage1.md](groot_event_grounded_sae_stage1.md)

## 결론

GR00T N1.5의 RoboCasa PQ3 activation으로 physical layer 15 BatchTopK SAE를
학습했다. Stage 1 범위에서 reconstruction, 목표 sparsity, dictionary 사용률,
수치 안정성과 checkpoint reload가 모두 정상인 **1.2k short-schedule run**을
handoff checkpoint로 선택했다.

10k run은 reconstruction은 더 좋아졌지만 전체 데이터에서 1,536개 feature 중
233개만 발화해 해석용 dictionary의 utilization gate를 통과하지 못했다. 두
후보는 총 step뿐 아니라 warmup/decay schedule도 다른 독립 run이므로, 이 결과만
가지고 “오래 학습한 것” 하나를 원인으로 단정하지 않는다.

## 무엇을 학습했는가

| 항목 | 값 |
| --- | --- |
| source | PQ3 전체 150 PKL = 5 instruction cells × 30 |
| records | 12,041 |
| SAE layer | physical L15 |
| token scope | 마지막 action token 16개만 사용, state/future 33개 제외 |
| denoise scope | 4개 step을 각각 독립 row로 사용 |
| unique activation rows | `12,041 × 4 × 16 = 770,624` |
| activation shape/dtype | `[770624, 1536]`, fp16 resident |
| dictionary | BatchTopK, `dict_size=1536`, `k=64` |

다른 physical layer `0,2,4,8,10,12`, PQ2 pooled activation, state/future token은
이 checkpoint의 학습 입력이 아니다.

## 770k rows와 1.2k/10k steps의 관계

`770,624`는 unique row 수이고, 1.2k/10k는 optimizer update 수다. Batch
size가 4,096이므로 한 update에서 4,096개 row의 reconstruction loss를 계산한다.

| run | optimizer steps | row presentations | dataset 사용량 | warmup | decay start |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.2k short schedule | 1,200 | 4,915,200 | 약 6.38회 | 100 | 960 |
| 10k long schedule | 10,000 | 40,960,000 | 약 53.15회 | 1,000 | 8,000 |

Loader는 shuffle마다 188개 complete batch, 즉 770,048 rows를 사용하고 나머지
576개를 무작위로 제외한다. 매 epoch 다시 shuffle하므로 두 run 모두 770,624개
unique row를 실질적으로 반복 학습한다. 770k optimizer steps가 필요한 것은
batch size가 1일 때뿐이다.

## 선택 checkpoint 결과

평가는 train과 같은 source 분포의 770,560 complete-batch rows에서 수행했다.

| metric | 값 | 의미 |
| --- | ---: | --- |
| reconstruction MSE | 10.2842 | activation 원소별 제곱오차 평균. Raw scale에 의존한다. |
| L2 loss | 123.1901 | row별 reconstruction error L2 norm 평균. |
| FVE | 0.987958 | source activation variance의 약 98.8%를 재구성했다. |
| cosine similarity | 0.997290 | 입력과 reconstruction의 방향이 매우 유사하다. |
| L2 ratio | 0.997136 | reconstruction norm이 입력 norm과 거의 같다. |
| relative reconstruction bias | 0.999883 | 1에 가까워 체계적인 scale bias가 작다. |
| L0 | 63.979 | row당 활성 feature 수가 목표 `k=64`와 일치한다. |
| L1 loss | 20,603.80 | feature activation magnitude 기준값이며 sparsity 개수가 아니다. |
| alive / dead | 1,284 / 252 | 전체 feature의 83.6%가 한 번 이상 발화했다. |
| inference threshold | 19.956 | batch-independent inference에서 평균 L0를 조정하는 cutoff다. |
| finite checks | 모두 통과 | input/encoding/reconstruction에 NaN/Inf가 없다. |

MSE와 L1/L2 값은 activation scale에 의존하므로 단독 절대 기준으로 해석하지
않는다. Reconstruction 판정은 FVE, cosine, norm ratio를 함께 보고, SAE 판정은
L0와 alive/dead까지 함께 본다.

## 후보 비교와 선택 이유

| 후보 | MSE | FVE | cosine | L0 | alive / dead | 결정 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1.2k short schedule | 10.2842 | 0.987958 | 0.997290 | 63.979 | 1284 / 252 | **선택** |
| 10k long schedule | 5.8102 | 0.993197 | 0.998570 | 63.987 | 233 / 1303 | 거부 |

10k checkpoint는 낮은 MSE만 보면 우수하지만 feature의 84.8%가 full-data
inference에서 한 번도 발화하지 않았다. Threshold를 끈 training-style
batch-top-k에서도 100k rows 중 141개 feature만 사용했으므로 threshold
calibration만의 문제는 아니다.

선택된 1.2k checkpoint에 대한 Stage 1 판정은 다음과 같다.

- reconstruction: 통과
- target sparsity `k=64`: 통과
- dictionary utilization: 명백한 collapse 없음
- numerical stability와 checkpoint reload: 통과
- held-out 일반화와 feature 의미: 아직 평가하지 않음

따라서 **현재 Stage 1 범위에서는 잘 학습된 checkpoint**다. 이 결론은 feature가
특정 event를 의미하거나 다른 RoboCasa task에 일반화한다는 주장은 아니다.

## Inference threshold

BatchTopK 학습은 batch 전체에서 평균 `k`개 feature를 선택한다. 이를 그대로
추론에 사용하면 한 sample의 결과가 batch 구성에 의존하므로, 추론에서는 학습
중 추적한 scalar threshold보다 큰 feature만 남긴다.

Upstream 기본 threshold는 `-1`이고 `step > 1000`부터 갱신된다. 따라서
100-step 또는 정확히 1,000-step checkpoint는 sparsity 평가에 사용할 수 없다.
선택 run은 1,200 steps로 199회의 threshold update를 포함한다.

## 산출물과 검증 범위

Canonical handoff:

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

- 원격 PQ3 150 PKL contract와 모든 선택 activation의 finite 여부를 검증했다.
- GR00T loader/export/cache/audit 회귀 테스트 9개가 통과했다.
- Raw PKL은 원격 source-of-truth에 유지하고 L15 action-token derived cache만
  로컬 학습에 사용했다.
- Temporal metadata join, held-out 평가, layer ablation, event semantics와
  intervention은 Stage 1 범위가 아니다.

Stage 2 keyframe 추출은 SAE와 독립적으로 진행한다. Selected checkpoint는
동결해 두고 Stage 5 event-feature scoring에서 사용한다.
