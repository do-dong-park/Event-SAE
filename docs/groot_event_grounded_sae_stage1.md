# GR00T Event-Grounded SAE Stage 1 — 최종 보고서

- 작성일: 2026-07-21
- 코드 revision: `878e77054f792f0706d69ea58478c2ba015b5e71`
- 대상: GR00T N1.5, RoboCasa PQ3, DiT physical layer 15
- 상태: **repo-scope offline Stage 1 완료**
- Provisional handoff: `logs/groot_n15/pq3_l15_stage1_selected/trainer_0/ae.pt`

이 문서는 Stage 1의 목적, 데이터 계약, 학습 조건, 재현 절차와 실험 결과를
한곳에 기록한다. Stage 1은 SAE의 offline 품질까지만 확정하며, policy 행동 보존과
feature의 event 의미·인과성은 후속 검증 대상으로 남긴다.

## 0. Stage 1 결과 요약

12,041개 policy records에서 GR00T L15의 4개 denoise step과 16개 action token을
각각 독립 sample로 사용해 770,624개 activation row를 구성했다. 같은 데이터로
BatchTopK SAE를 1.2k, 4k, 10k, 20k optimizer step 동안 각각 독립 학습했다.

| Run | Dataset passes | MSE ↓ | FVE ↑ | 평균 L0 | Alive feature |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.2k | 6.38 | 10.2842 | 0.987958 | 63.979 | 1,284 / 1,536 (83.59%) |
| 4k | 21.26 | 7.3981 | 0.991338 | 63.908 | 262 / 1,536 (17.06%) |
| 10k | 53.15 | 5.8102 | 0.993197 | 63.987 | 233 / 1,536 (15.17%) |
| 20k | 106.30 | 5.3015 | 0.993793 | 63.997 | 192 / 1,536 (12.50%) |

`MSE`는 reconstruction 오차, `FVE`는 입력 분산 중 reconstruction이 설명한 비율,
`L0`는 activation row 하나에서 0이 아닌 SAE code 수다. `Alive feature`는
전체 audit에서 한 번이라도 발화한 dictionary feature 수다. 정확한 계산식과
해석은 5절에 정의한다.

학습량이 늘수록 reconstruction은 일관되게 개선되었지만, 사용되는 dictionary
feature 수는 감소했다. 현재는 후속 event-feature 탐색에 넓은 후보군을 제공하는
1.2k를 **provisional handoff**로 유지하고, 4k/10k/20k를 reconstruction이 더 좋은
비교 checkpoint로 보존한다. 어느 checkpoint가 policy 행동을 가장 잘 보존하는지는
reconstruction-only closed-loop 평가 전에는 결정할 수 없다.

## 1. 전체 목표 중 어떤 단계인가?

전체 연구는 다음 질문을 순서대로 다룬다.

| 단계 | 핵심 질문 | 산출물 |
| ---: | --- | --- |
| **1/6** | **VLA residual을 sparse하게 재구성할 수 있는가?** | **SAE checkpoint와 품질 audit** |
| 2/6 | 행동 궤적의 중요한 시점은 어디인가? | AWE waypoint |
| 3/6 | 각 시점을 어떻게 표현할 것인가? | vision/state descriptor |
| 4/6 | 반복되는 event 유형은 무엇인가? | annotated event cluster |
| 5/6 | 어떤 SAE feature가 event와 정렬되는가? | feature ranking |
| 6/6 | 선택 feature가 행동에 영향을 주는가? | closed-loop ΔSR |

Stage 1의 판정 축은 둘로 나뉜다.

1. **Offline fidelity:** reconstruction, sparsity, feature usage와 numerical integrity
2. **Behavioral fidelity:** 원 activation 대신 `Dec(Enc(x))`를 주입해도 policy의
   success rate가 유지되는지

현재 완료 범위는 첫 번째 축이다. 따라서 “SAE가 source activation을 잘 근사한다”는
결론은 가능하지만, “원 policy 행동을 보존한다”거나 “feature가 특정 event를
나타낸다”는 결론은 아직 내리지 않는다.

## 2. 데이터셋 구성

### 2.1 Source inventory

원격 source-of-truth:

```text
/home/kimseungjun/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/
  phase_event_pq3/raw_rollouts/
```

| RoboCasa task | Instruction cell | PKL |
| --- | --- | ---: |
| `OpenDrawer` | `pq3_drawer_left` | 30 |
| `OpenDrawer` | `pq3_drawer_right` | 30 |
| `PickPlaceCounterToCabinet` | `pq3_ppcc_beer` | 30 |
| `PickPlaceCounterToCabinet` | `pq3_ppcc_bread` | 30 |
| `PickPlaceCounterToCabinet` | `pq3_ppcc_pizza_cutter` | 30 |
| **합계** | **5 cells** | **150** |

150개 PKL에는 12,041개 activation record가 있다. 이는 2개 task family와
5개 instruction/scene cell에 대한 데이터이며 RoboCasa 전체 분포를 대표하지 않는다.

### 2.2 Activation 계약

각 record의 activation tensor는 다음 계약을 갖는다.

| 속성 | 값 |
| --- | --- |
| Record shape | `[L=7, K=4, T=49, D=1536]`, fp16 |
| Physical layers | `[0, 2, 4, 8, 10, 12, 15]` |
| Token layout | state 1 + future 32 + action 16 |
| 선택 범위 | physical L15, action token `[33,49)` |
| SAE 입력 cache | `[770624,1536]`, fp16 |

`L`은 저장된 layer 수, `K`는 denoise step 수, `T`는 model token 수,
`D`는 residual dimension이다. `[33,49)`는 token index 33 이상 49 미만,
즉 마지막 16개 action token을 뜻한다.

### 2.3 SAE row 구성

한 record에서 선택되는 row는 다음과 같다.

```text
[L=7, K=4, T=49, D=1536]
→ physical L15 선택
→ action token [33,49) 선택
→ [K=4, A=16, D=1536]
→ pooling 없이 K와 A를 sample 축으로 flatten
→ [64, D=1536]
```

여기서 `A`는 선택한 action token 수다. 총 unique row 수 `N`은

```text
N = records × denoise steps × action tokens
  = 12,041 × 4 × 16
  = 770,624
```

이다. 각 row는 하나의 `(policy record, denoise step, action-token offset)`에
대응한다. State/future token, 다른 physical layer와 PQ2 pooled activation은
포함하지 않는다.

이 구성은 action expert residual을 denoise forward와 token offset별로 분석한다는
점에서 Event-Grounded SAE의 per-token 관점과 정렬된다. 다만 `K=4`와 `A=16`은
GR00T PQ3의 고유 계약이며 모든 VLA에 공통인 논문 상수는 아니다.

Raw PKL은 원격에 유지하고 standalone exporter로 만든 activation shard만 로컬로
전송한다. 로컬 학습 cache는 다음 경로에 있다.

```text
logs/groot_n15/pq3_l15_activation_cache/l15_action_tokens.pt
```

## 3. SAE 학습 조건

### 3.1 Model과 objective

입력 residual `x ∈ R^1536`에 대해 BatchTopK SAE는

```text
u     = W_enc x + b_enc
z     = BatchTopK_k(u)
x_hat = W_dec z + b_dec
L     = MSE(x, x_hat) + λ_aux L_aux
```

로 계산한다. `z`는 sparse code, `x_hat`은 reconstruction이다. BatchTopK는
학습 batch 전체에서 평균적으로 row당 `k`개 feature를 남긴다. `L_aux`는 주
reconstruction에 선택되지 않은 feature의 학습을 보조하는 auxiliary loss다.

모든 run의 공통 조건은 다음과 같다.

| 항목 | 값 |
| --- | --- |
| Activation dimension | 1,536 |
| Dictionary size | 1,536 |
| Expansion ratio | 1.0 |
| Active budget `k` | 64 |
| Learning rate | `1e-4` |
| Batch size | 4,096 rows |
| Optimizer seed | 0 |
| Training dtype | float32 |
| Activation normalization | 사용 |

Dictionary size는 SAE feature 수다. Expansion ratio는
`dictionary size / activation dimension`이며, `k=64`는 row당 평균 active
feature 예산이다. 따라서 목표 sparsity 비율은

```text
k / dictionary size = 64 / 1,536 = 0.04167 = 4.17%
```

이다.

### 3.2 네 학습 schedule

| Run | Steps | Warmup | Decay start | Row presentations | Dataset passes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1.2k | 1,200 | 100 | 960 | 4,915,200 | 6.38 |
| 4k | 4,000 | 1,000 | 3,200 | 16,384,000 | 21.26 |
| 10k | 10,000 | 1,000 | 8,000 | 40,960,000 | 53.15 |
| 20k | 20,000 | 1,000 | 16,000 | 81,920,000 | 106.30 |

`Row presentations`는 optimizer가 본 row의 누적 횟수이며 unique row 수와 다르다.

```text
row presentations = batch size × optimizer steps
dataset passes      = row presentations / 770,624
```

예를 들어 20k run은 `4,096 × 20,000 = 81,920,000`회 row를 제시했고,
`81,920,000 / 770,624 = 106.30` dataset passes에 해당한다. Loader는 pass마다
shuffle하므로 모든 run이 770,624개 unique row를 반복 사용한다.

네 run은 서로 독립적이다. 특히 1.2k는 warmup도 짧으므로 네 행의 차이를
optimizer step 하나의 인과 효과로 해석할 수는 없다.

### 3.3 논문 조건과의 관계

Event-Grounded SAE는 OpenVLA에 4k step, PaliGemma/action-expert stream에
10k step을 사용하지만 batch size는 40,000이다. 따라서 optimizer step 수보다
`batch × steps`로 계산한 row presentations가 학습량 비교에 더 적절하다.
해당 논문은 suite당 약 10 tasks × task당 50 rollouts를 사용하지만 unique
activation row 수는 공개하지 않아 우리 데이터와 정확한 선형 환산은 불가능하다.

[Swann et al. (2026)](https://arxiv.org/html/2603.19183v1)은 ER1 TopK+AuxK SAE를
batch 4,096으로 100 epochs 학습한다. 우리의 20k는 약 106.30 passes라 횟수만
보면 가깝지만, SAE 구조·normalization·token pooling과 데이터가 다르므로
동등 조건으로 간주하지 않는다. 특히 해당 논문의 본문 결과는 주로 timestep별
mean-pooled activation을 사용하고, 우리는 action token을 pooling하지 않는다.

현재 데이터는 2개 task family, 5 cells, 150 rollouts로 비교적 좁다. 이 때문에
필요 학습량을 task 수에 단순 비례시키는 대신 1.2k–20k를 실제 학습하고 동일한
full-data audit로 비교했다.

## 4. 코드 상 실행 절차

### 4.1 관련 코드

- 환경: [`environment-sae-dev.yml`](../environment-sae-dev.yml)
- 원격 export/로컬 merge:
  [`export_pq3_activation_shards.py`](../scripts/groot/export_pq3_activation_shards.py)
- SAE 학습:
  [`train_sae_robocasa.py`](../scripts/groot/train_sae_robocasa.py)
- Checkpoint audit:
  [`audit_sae_checkpoint.py`](../scripts/groot/audit_sae_checkpoint.py)
- 테스트:
  [`test_groot_train_sae_robocasa.py`](../tests/test_groot_train_sae_robocasa.py)

### 4.2 환경과 테스트

```bash
conda env create -f environment-sae-dev.yml

conda run -n event-sae-dev python -m pytest -q   tests/test_groot_train_sae_robocasa.py
```

기록된 repo-scope 결과는 `9 passed`다.

### 4.3 원격 activation export

신뢰한 원격 PKL source에서 L15 action-token shard를 만든다.

```bash
python export_pq3_activation_shards.py export   --input-dir /home/kimseungjun/datasets/temporal_vla_outputs/eval/robocasa/groot_n15/phase_event_pq3/raw_rollouts   --output-dir pq3_l15_action_shards   --trust-pkl --layer 15
```

정상 완료 조건은 `groot_source_manifest.json`, 150개 shard,
`num_activation_rows=770624`다. `--trust-pkl`은 provenance가 고정된 이
source에만 사용한다.

### 4.4 로컬 cache merge

```bash
conda run -n event-sae-dev python   scripts/groot/export_pq3_activation_shards.py merge   --shard-dir LOCAL_SHARD_DIR   --output-cache logs/groot_n15/pq3_l15_activation_cache/l15_action_tokens.pt
```

Merge 단계는 manifest, shard 수, tensor shape, row 수와 finite 여부를 검사한다.

### 4.5 SAE 학습

아래 명령에서 `RUN_DIR`, `STEPS`, `WARMUP`, `DECAY`를 3.2절의 값으로
치환한다. `CUDA_VISIBLE_DEVICES=5`일 때 process 내부의 `cuda:0`은 physical
GPU 5를 가리킨다.

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n event-sae-dev python   scripts/groot/train_sae_robocasa.py   --activation-cache logs/groot_n15/pq3_l15_activation_cache/l15_action_tokens.pt   --save-dir RUN_DIR   --layer 15 --dict-size 1536 --sae-k 64 --lr 1e-4   --steps STEPS --batch-size 4096 --warmup-steps WARMUP   --save-every STEPS --log-steps 100 --device cuda:0   --run-tag RUN_TAG
```

실험에서 사용한 run directory는 다음과 같다.

| Run | `RUN_DIR` |
| --- | --- |
| 1.2k selected | `logs/groot_n15/pq3_l15_stage1_selected` |
| 4k | `logs/groot_n15/pq3_l15_candidate_4000` |
| 10k | `logs/groot_n15/pq3_l15_production_10000` |
| 20k | `logs/groot_n15/pq3_l15_candidate_20000` |

현재 `logs`에는 각 run의 최종 `ae.pt`, `config.json`, source manifest와
full-audit 결과만 보존한다. Smoke run과 중간 checkpoint는 제거했다.

### 4.6 Full-data audit

```bash
CUDA_VISIBLE_DEVICES=5 conda run -n event-sae-dev python   scripts/groot/audit_sae_checkpoint.py   --activation-cache logs/groot_n15/pq3_l15_activation_cache/l15_action_tokens.pt   --sae-checkpoint RUN_DIR/trainer_0/ae.pt   --batch-size 512 --max-audit-rows 0 --device cuda:0
```

전체 770,624 rows 중 complete batch만 평가하므로 실제 audit row 수는

```text
floor(770,624 / 512) × 512 = 1,505 × 512 = 770,560
```

이다. 제외되는 64 rows는 전체의 약 0.0083%다. 결과는 run root의
`sae_quality.json`과 `sae_quality_by_feature.npz`에 저장된다.

## 5. Metric 정의와 계산

### 5.1 Reconstruction과 sparsity

입력 matrix를 `X ∈ R^(N×D)`, reconstruction을 `X_hat`, sparse code를
`Z ∈ R^(N×M)`이라 하자. 여기서는 `N=770,560`, `D=M=1,536`이다.

| Metric | 계산 | 의미 |
| --- | --- | --- |
| MSE | `(1/ND) Σ_i Σ_d (X_id-X_hat_id)^2` | 원소당 제곱 reconstruction 오차; 낮을수록 좋음 |
| FVE | `1-Var(X-X_hat)/Var(X)` | 입력 분산 중 reconstruction이 설명한 비율; 1에 가까울수록 좋음 |
| Cosine | `(1/N) Σ_i cos(X_i,X_hat_i)` | row별 방향 보존; 1에 가까울수록 좋음 |
| Average L0 | `(1/N) Σ_i ||Z_i||_0` | row당 0이 아닌 feature 수; 목표 `k=64`와 비교 |
| Alive fraction | `|{j: ∃i, Z_ij≠0}|/M` | source audit에서 한 번 이상 사용된 feature 비율 |
| Dead count | `M-alive count` | audit에서 한 번도 사용되지 않은 feature 수 |

FVE는 scale-normalized reconstruction 지표지만 MSE는 activation scale에
의존하므로 서로 다른 model·layer·normalization 사이에서 직접 비교하지 않는다.
Average L0가 64에 가깝다는 것은 sparse budget을 지켰다는 뜻이지 feature의
의미가 좋다는 뜻은 아니다.

### 5.2 Inference threshold

BatchTopK는 학습 중 batch 전체 순위로 feature를 선택한다. 저장된 inference
threshold `τ`는 batch가 하나이거나 크기가 달라도 독립적으로 encode하기 위한
cutoff다. 개념적으로 inference에서는 pre-activation이 `τ`를 넘는 항목을
선택한다.

Threshold의 절댓값은 latent activation scale과 함께 변하므로 “낮을수록 좋다”는
품질 metric이 아니다. 이 보고서에서는 threshold가 유한한지, checkpoint에
저장되었는지와 실제 inference L0가 `k` 근처인지 함께 확인한다.

### 5.3 Alive, general, important의 구분

세 용어는 서로 다른 질문에 답한다.

- **Alive:** 현재 source에서 한 번이라도 발화했는가?
- **General:** 여러 episode·task에서 동일한 의미의 event에 일관되게 반응하는가?
- **Behaviorally important:** ablation·steering·reconstruction hook이 실제 행동을
  바꾸는가?

[Swann et al. (2026)](https://arxiv.org/html/2603.19183v1)은 general feature를
episode coverage, onset count, activation magnitude와 run length로 분류했다.
보고된 general feature는 LIBERO PG5에서 `32/2,044`, OpenVLA Goal L8에서
`8/1,775`로 소수였다. 그러나 이는 “dead feature가 많을수록 좋다”는 뜻이 아니다.
해당 수치에서 분모는 분석 가능한 feature이고, 그중 task/scene을 넘어서는
general feature가 소수라는 뜻이다.

따라서 우리 20k의 alive 12.5%를 이 논문만으로 정당화할 수도, 실패라고 단정할
수도 없다. Stage 1의 alive는 dictionary 사용 현황이고, feature generality와
인과적 중요성은 Stage 5/6에서 별도로 측정한다.

### 5.4 Behavioral fidelity

동일한 closed-loop protocol에서

```text
Raw SR    = raw policy 성공 episode 수 / 전체 episode 수
Hooked SR = reconstruction hook policy 성공 episode 수 / 전체 episode 수
ΔSR       = Hooked SR - Raw SR
```

를 계산한다. `Hooked SR`은 원 residual `x`를 `Dec(Enc(x))`로 교체했을 때의
success rate다. Offline FVE가 높아도 작은 reconstruction 오차가 action에
누적될 수 있으므로, policy 보존의 최종 판정은 이 metric으로 해야 한다.

## 6. 실험 결과 및 해석

### 6.1 Full-data audit

| Run | MSE ↓ | FVE ↑ | Cosine ↑ | L0 | Alive / dead | Alive % | Threshold |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1.2k | 10.2842 | 0.987958 | 0.997290 | 63.979 | 1,284 / 252 | 83.59% | 19.9559 |
| 4k | 7.3981 | 0.991338 | 0.998131 | 63.908 | 262 / 1,274 | 17.06% | 16.3785 |
| 10k | 5.8102 | 0.993197 | 0.998570 | 63.987 | 233 / 1,303 | 15.17% | 13.0924 |
| 20k | 5.3015 | 0.993793 | 0.998713 | 63.997 | 192 / 1,344 | 12.50% | 11.8220 |

모든 행은 같은 770,560 rows에서 계산했다. `Alive / dead`는 1,536개
dictionary feature 중 전체 audit에서 한 번 이상 발화한 수와 한 번도 발화하지
않은 수다. `Threshold`는 5.2절의 batch-independent inference cutoff다.

### 6.2 결과 해석

**Reconstruction.** 1.2k에서 20k로 갈수록 MSE는 `10.2842 → 5.3015`로
48.45% 감소하고 FVE는 `0.987958 → 0.993793`으로 증가했다. 이 데이터에서
reconstruction만 비교하면 20k가 가장 좋다. 다만 10k에서 20k로 늘렸을 때의
FVE 증가는 약 0.000596으로, 후반부 개선 폭은 작아진다.

**Sparsity.** 네 run 모두 평균 L0가 63.91–64.00으로 목표 `k=64`와 일치한다.
즉 reconstruction 개선이 row당 더 많은 feature를 켜서 얻어진 것은 아니다.
각 row는 평균적으로 dictionary의 약 4.17%만 사용한다.

**Dictionary usage.** Alive feature는 1.2k의 1,284개에서 4k의 262개로 크게
감소한 뒤, 10k 233개, 20k 192개로 계속 줄었다. 이는 긴 schedule이 source
분포를 더 적은 feature로 압축했다는 관측이다. 그것이 유용한 specialization인지,
feature collapse인지, 혹은 BatchTopK 학습 dynamics인지는 alive 수만으로
구분할 수 없다.

**Schedule confound.** 1.2k는 warmup 100, 나머지는 warmup 1,000인 독립
run이다. 따라서 1.2k와 4k 사이의 큰 alive 차이를 optimizer step만의 효과로
주장하지 않는다. 엄밀한 학습량 ablation에는 동일 warmup/decay schedule과
여러 random seed가 필요하다.

### 6.3 Checkpoint 선택

| 용도 | Checkpoint | 판단 |
| --- | --- | --- |
| 후속 feature discovery handoff | 1.2k | 넓은 alive coverage를 우선한 provisional 선택 |
| 짧은 논문 step 비교 | 4k | reconstruction은 개선됐지만 alive가 급감한 경계점 |
| Event-Grounded PG/AE step 비교 | 10k | 높은 reconstruction의 offline-valid alternate |
| 장기/약 100-pass 비교 | 20k | 최고 reconstruction의 offline-valid alternate |

현재 1.2k 선택은 “가장 잘 학습된 SAE”라는 최종 판정이 아니다. Stage 2–5에서
많은 candidate feature를 살펴보기 위한 실용적 handoff다. 반대로 20k도 alive가
적다는 이유만으로 실패가 아니다. 네 checkpoint 모두 input/code/reconstruction
finite 검사, save/reload와 full-data audit를 통과했다.

최종 선택에는 최소한 다음 증거가 더 필요하다.

1. 동일 rollout set의 Raw SR과 1.2k/4k/10k/20k Hooked SR
2. episode/task별 feature coverage와 event-aligned onset 분석
3. 주요 feature의 여러 seed 재현성
4. 선택 feature의 ablation 또는 steering 결과

## 7. 결론

| 검증 범위 | 상태 | 근거 또는 남은 조건 |
| --- | --- | --- |
| Data contract | 완료 | 150 PKL, 12,041 records, 770,624 rows audit |
| Training pipeline | 완료 | 1.2k/4k/10k/20k train, save/reload |
| Offline reconstruction | 완료 | 동일 770,560-row MSE/FVE/Cosine |
| Offline sparsity/usage | 완료 | L0, alive/dead, threshold와 finite 검사 |
| Repo-scope Stage 1 | **완료** | provisional handoff와 비교 checkpoint 보존 |
| Behavioral fidelity | 미완료 | Raw SR 대비 reconstruction-only Hooked SR 필요 |
| Event semantics/causality | 범위 밖 | Stage 5/6에서 검증 |

최종 산출물 경로:

```text
logs/groot_n15/
├── pq3_l15_activation_cache/l15_action_tokens.pt
├── pq3_l15_stage1_selected/trainer_0/ae.pt
├── pq3_l15_candidate_4000/trainer_0/ae.pt
├── pq3_l15_production_10000/trainer_0/ae.pt
└── pq3_l15_candidate_20000/trainer_0/ae.pt
```

각 run root에는 `groot_source_manifest.json`, `sae_quality.json`,
`sae_quality_by_feature.npz`와 `trainer_0/config.json`이 함께 있다.

Stage 1은 현재 repo에서 재현 가능한 offline 단계로 완료했다. 네 run 모두
sparse reconstruction에는 성공했고, 학습량 증가에 따라 reconstruction과
dictionary usage 사이의 trade-off가 관찰되었다. 현재 handoff는 1.2k지만
paper-aligned final checkpoint는 아직 정하지 않는다. 그 결정은 동일한
closed-loop Hooked SR과 후속 feature generality·causality 검증 이후에 내린다.
