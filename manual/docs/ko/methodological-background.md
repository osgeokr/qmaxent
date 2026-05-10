# 방법론 해설

본 챕터는 *어떻게*의 뒤에 있는 *왜* — QMaxent의 기본값을 정당화하는
학술적 추론을 정리합니다. 모델 선택을 이해하거나 인용하고 싶은 사용자를
위한 것이며, 처음 사용자에게는 사용 가이드 챕터들이 우선입니다.

## Maxent와 최대 엔트로피 원리

Maxent는 출현 위치에서 관측된 경험적 모멘트와 일치하는 **최대 엔트로피
분포**를 적합합니다 ([Phillips, Anderson & Schapire 2006](references.md)).
직관적으로: 출현 표본과 동일한 환경 변수 평균값을 만드는 연구 영역
상의 모든 분포 가운데, 최대 엔트로피 분포는 *가장 정보가 적은* 것 —
데이터가 실제로 말하는 것 너머에는 가정을 추가하지 않는 분포입니다.

이 추정량은 출현-배경 대조에 적합된 정규화 비균질 포아송 점-과정 모델과
수학적으로 동등하며 ([Fithian & Hastie 2013](references.md)), 이 동등성이
elapid의 구현이 최적화를 scikit-learn의 로지스틱-회귀 루틴에 위임할 수
있게 해줍니다 ([Anderson 2023](references.md);
[Pedregosa et al. 2011](references.md)).

## 피처 선택을 위한 maxnet 자동 규칙

Maxent 피처 클래스(Linear, Quadratic, Product, Hinge, Threshold)는
**유연성**과 **과적합 위험** 사이의 절충입니다.
[Phillips & Dudík 2008](references.md)의 수백 종에 대한 벤치마크는
표본 크기 기반 규칙을 확립했고, 이후의 벤치마크에서도 이 규칙은
유지되었습니다:

| 표본 크기 | 허용 피처 |
|---|---|
| ≤ 10 | L |
| ≤ 30 | L, Q |
| ≤ 80 | L, Q, H |
| > 80 | L, Q, P, H, T |

QMaxent의 **Auto** 모드는 이 규칙을 직접 적용합니다. 이 규칙의 근거가
되는 벤치마크는 Maxent 문헌에서 가장 광범위하게 복제된 튜닝
권고이며, ENMeval 식
([Muscarella et al. 2014](references.md)) hyperparameter 검색 없이 이
규칙을 벗어나는 것은 발표에서 정당화하기 어렵습니다.

## 공간 교차검증이 중요한 이유

Random K-Fold CV는 무작위 폴드가 학습 점에 지리적으로 *인접한* 점을
포함할 가능성이 높기 때문에 공간 데이터에서 과도하게 낙관적인 AUC를
산출합니다. 공간 자기상관이 holdout 점을 사소하게 예측 가능하게
만듭니다.

[Roberts et al. 2017](references.md)은 많은 SDM 벤치마크에서 이를
경험적으로 입증: random-K-Fold AUC는 새로운 지역에서 실제로 보이는
성능을 0.05–0.15만큼 체계적으로 과대 추정합니다. 그들은 모든 공간
모델에서 공간 블록 CV를 기본값으로 권고합니다.

따라서 QMaxent는 **Geographic K-Fold (k = 5, 고정 시드)** —
Roberts et al. 권장 설계 — 를 기본값으로 합니다. *Random* K-Fold는
오래된 연구의 정확한 재현을 허용하기 위해서만 제공됩니다.

## 폴드별 vs 통합 AUC

폴드 전체에 대한 두 요약 통계가 흔히 보고됩니다:

- **폴드별 AUC 후 평균** — 각 폴드 내에서 AUC를 계산, 폴드 간 평균,
  평균 ± SD 보고.
- **통합 AUC** — 폴드 전체의 holdout 예측을 연결, 단일 AUC 계산.

QMaxent는 **폴드별 평균 ± SD**를 헤드라인 숫자로 보고합니다 —
폴드 간 분산을 담고 있고, 그 자체가 정보적 진단이기 때문입니다 —
높은 분산은 모델의 지리적 전이성이 불안정함을 신호합니다. 통합 AUC
대안은 부 메트릭으로 XLSX의 Table 3에 포함됩니다.

## Jackknife 변수 중요도

고전적 Maxent jackknife
([Phillips, Anderson & Schapire 2006](references.md))는 변수마다 두
개의 추가 모델을 적합: 그 변수만 가진 모델(*with-only*)과 그 변수가
제거된 모델(*without*). 전체에서 *without*으로의 검증 AUC 하락은 다른
변수로부터 회복하기 어려운 고유 기여를 가진 변수를 식별합니다.

QMaxent는 변수당 네 개의 AUC(only-train, only-test, without-train,
without-test)와 검증측 하락 열을 보고하며, 가장 고유하게 정보적인
변수가 맨 위에 오도록 하락 내림차순으로 정렬합니다. 이는 Phillips et
al. 2006 이후 발표된 Maxent 문헌에서 사용된 형식입니다.

## 임계값 방법 (MTP / T10 / MaxSSS)

이진 적합/부적합 지도가 필요할 때 (예: 보호구역 설계 응용) cloglog
출력의 임계값을 선택해야 합니다. SDM 문헌은 세 가지 옵션으로
표준화됩니다 ([Liu, Newell & White 2013](references.md)):

- **MTP — Minimum Training Presence** — 학습 출현 지점 중 최저
  cloglog 값. 가장 관대; 민감도 우선.
- **T10 — 10번째 백분위수 학습 출현** — 학습 출현의 하위 10%를
  이상치로 제거. 고전적 Java-MaxEnt 기본값.
- **MaxSSS — Max Sensitivity + Specificity** — 학습 ROC에서 민감도 +
  특이도 합을 최대화하는 임계값. 발표된 비교 대부분에서 가장 좋은
  보정.

QMaxent는 XLSX 내보내기의 Table 3에 세 가지 임계값을 모두 보고합니다.
Java-MaxEnt 연구를 재현(T10)하거나 관대(MTP) 또는 엄격한 분류를
선호할 특정 이유가 없는 한 **MaxSSS**를 선택하세요.

## 출력 변환 비교

**Spatial Projection**에서 세 가지 출력 변환을 사용할 수 있습니다:

| 변환 | 범위 | 해석 |
|---|---|---|
| **cloglog** | 0–1 | 종의 일반적 표본을 가정한 출현 확률, [Phillips et al. 2017](references.md) 권장 기본값 |
| **logistic** | 0–1 | 더 오래된 [Phillips & Dudík 2008](references.md) 매개변수화; 일부 baseline에서 여전히 사용 |
| **raw** | 무한 | 정규화되지 않은 지수 출력; 고급 후처리 |

QMaxent는 **cloglog**를 기본값으로 합니다 —
[Phillips et al. 2017](references.md)이 명시적으로 현대 기본값으로
권고하는 형태이고, "출현 확률" 해석이 모델링 비전문 이해당사자에게
가장 직접적으로 전달 가능하기 때문입니다.

## 범주형 변수 처리

Java MaxEnt는 래스터의 속성 테이블을 통해 범주형 래스터를 인코딩합니다;
elapid는 내부적으로 **one-hot 확장**으로 인코딩합니다
([Anderson 2023](references.md)). 두 접근은 학습에는 동등하지만,
래스터가 학습 중 보지 못한 클래스 코드를 포함할 때 투영 시점에
갈라집니다:

- **Java MaxEnt**는 보지 못한 코드를 무작위 클래스에 침묵으로 매핑하여
  영향받는 셀에서 임의의 적합도 값을 만듭니다.
- **QMaxent**는 unified preflight 대화상자를 통해 보지 못한 코드를
  감지하고 외삽 대신 **NoData로 자동 마스킹**합니다.

이는 권장 관행입니다
([Elith, Kearney & Phillips 2010](references.md)) — 학습 영역을 벗어난
외삽은 근본적으로 정의되지 않으며 침묵으로 채워서는 안 되고
표시되어야 합니다.

## 표본 편향 보정

출현이 공간적으로 군집화될 때 (예: 도로 편향, 시민과학 군집),
모델은 종의 서식지 선호와 함께 *편향 표면*을 적합합니다. 두 보완적
보정이 존재합니다:

- **표본 가중치 다운웨이팅**
  ([Phillips et al. 2009](references.md)) — 군집화된 출현에 분수
  가중치를 적용. QMaxent는 이를 *Down-weight spatially clustered
  points* 옵션으로 구현합니다.
- **외부 KDE 편향 래스터** — 명시적 공변량으로 사용되는, 조사 노력을
  나타내는 래스터. QMaxent v0.1.x에서 구현되지 않음; 가장 가까운
  유사물은 다운웨이트 옵션이며 많은 발표 연구에서 비슷하게 작동합니다.

더 포괄적인 처리를 위해 [Boria et al. 2014](references.md)는 모델링
전 보완 단계로 출현 레이어의 **공간적 솎아내기**를 권고합니다 — *NNJoin*
플러그인이나 standalone *spThin* 도구를 통해 QGIS에서 구현 가능.

## 왜 이런 선택인가

QMaxent에 모인 기본값은 (a) 벤치마크 문헌이 많은 종과 연구 영역에서
잘 작동한다고 보여주는 것과 (b) [Roberts et al. 2017](references.md),
[Araújo et al. 2019](references.md), [Elith, Kearney & Phillips
2010](references.md)에 기록된 침묵의 실패 모드를 최소화하는 것의
합집합입니다. 이로부터의 분기는 특정 연구에는 올바른 움직임일 수도
있습니다; [Pitta nympha 예제](examples/pitta-nympha.md)는
[Lee et al. 2025](references.md)의 LQH/RM=4 고정 hyperparameter 설정에
대해 그것이 어떻게 보이는지 보여줍니다.
