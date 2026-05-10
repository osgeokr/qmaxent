# Bradypus variegatus

세발가락나무늘보 *Bradypus variegatus* 는 표준 Maxent 테스트 데이터셋입니다 —
[Phillips, Anderson & Schapire (2006)](../references.md) 와 함께
원래 발표되었고 이후 거의 모든 Maxent 논문에서 재사용되었습니다. 본 장에서는 이를
**QMaxent의 모든 기능 입문 투어** 로 사용합니다 — 데이터 로딩, 파라미터 선택, 공간
교차검증, Jackknife 중요도, 투영, 조사 계획. 본 장이 끝나면 사용자는 **완성된**
Bradypus 서식적합도 모델을 만들고 그것을 학술적으로 방어할 수 있게 됩니다.

## 데이터셋

Phillips et al. (2006) 데이터셋은 다음을 포함합니다:

| 레이어 | 유형 | 설명 |
|---|---|---|
| `bradypus.shp` | 벡터 점 | 남미와 중미에 걸친 116개 출현 기록 |
| `bio1, bio5, bio6, bio7, bio8` | 연속 래스터 | 기온 변수 (WorldClim) |
| `bio12, bio16, bio17` | 연속 래스터 | 강수 변수 (WorldClim) |
| `biome` | 범주형 래스터 | 생물군계 유형 (Olson et al. 2001) |

모든 래스터가 동일한 그리드를 공유: EPSG:4326, 0.5° × 0.5° 셀, 아메리카 전역 커버.
전체 데이터셋 크기 100 MB 미만.

**플러그인 → QMaxent → Download Example Dataset → Bradypus variegatus** 로 다운로드.
레이어가 자동으로 QGIS 프로젝트에 추가됩니다:

![중남미 지역의 bio17 위에 표시된 Bradypus 출현 지점](../images/maps/example-bradypus-loaded.png)

## Analysis 도크에 데이터 불러오기

**플러그인 → QMaxent → QMaxent Analysis** 를 엽니다. **① Data** 탭에서 **Presence
Points Layer** 드롭다운에서 `bradypus` 선택 — QMaxent가 즉시 `116 presence points
loaded` 를 보고합니다.

**Add from project** 클릭으로 로드된 모든 래스터 레이어를 한 번에 추가, `biome` 을
`[categorical]` 로 표시. **Check Raster Consistency** 클릭으로 그리드 확인:

![Bradypus 출현 레이어, 7개 환경 래스터, biome이 categorical, Check Raster Consistency 통과](../images/ui/dock-1-data-with-bradypus.png)

상태 줄은
`✓ All 9 rasters share grid (CRS: EPSG:4326, resolution: 0.5 × 0.5)` —
번들 데이터셋에서 정확히 기대되는 것입니다.

## 모델 설정

**② Parameters** 로 전환. 이 투어에서는 모든 기본값을 그대로 사용:

- **피처 유형**: Auto (116 출현 지점에서 maxnet 규칙이 LQPHT 모두 선택)
- **정규화 배수**: 1.0
- **공간 평가**: Geographic K-Fold, 5 folds, fixed seed = 0
- **Jackknife 변수 중요도**: 활성화
- **출력 파일**: `qmaxent_output/model.pkl` 와 `qmaxent_output/results.xlsx`

고정 무작위 시드는 **이 튜토리얼을 다시 실행하는 누구나 비트 단위로 동일한 결과**
를 얻는다는 의미입니다 — 재현성에 중요합니다.

## 학습과 교차검증 실행

**▶ Run Maxent** 클릭. 학습 탭이 인계받고 약 30초 후 완료됩니다:

![100% 완료된 학습 탭과 전체 로그](../images/ui/dock-3-training-completed.png)

로그를 위에서 아래로 읽으면 전체 이야기가 드러납니다:

```text
→ 10,000 background points sampled
Extracting raster covariates for presence points…
Extracting raster covariates for background points…
→ Presence: 116, Background: 9,997
→ Feature types: ['linear', 'quadratic', 'product', 'hinge', 'threshold']
Training MaxentModel…
→ Model training complete
→ Model saved: …/model.pkl
Computing ROC curve…
→ Training AUC = 0.9562
Running cross-validation…
  Fold 1: 22 test presences, AUC = 0.7453
  Fold 2: 21 test presences, AUC = 0.7839
  Fold 3: 39 test presences, AUC = 0.8097
  Fold 4: 26 test presences, AUC = 0.8614
  Fold 5: 8 test presences, AUC = 0.5903
→ CV AUC = 0.7581 ± 0.0920  (n=5 fold(s))
```

**Train AUC = 0.956** 인 반면 **CV AUC = 0.758 ± 0.092**. 그 격차는 공간적으로
구분되는 검증 세트를 보류한 비용 — 부풀려진 학습 AUC만 보는 것보다 훨씬 정직한
실세계 예측 성능 지표입니다.

Fold 5는 가장 낮은 AUC(0.59)와 가장 작은 검증 세트(8 출현 지점)을 가집니다 — 공간
CV에서는 폴드가 의도적으로 면적이 고르지 않아, 한 폴드가 작고 비전형적인 영역에
떨어질 수 있습니다. 통합 CV AUC는 이 변동을 평균화합니다.

## 변수 행동 확인

### 반응곡선

**④ Results → Response Curves** 에서 `bio1`(연평균 기온) 선택:

![학습 범위가 음영 표시된 bio1 반응곡선](../images/ui/dock-4-response-curve-bio1.png)

반응은 비단조적 — 약 150과 290 부근에 봉우리(WorldClim 관행에 따라 0.1 °C 단위)와
240 부근의 함몰을 가집니다. 모델이 이 단절을 포착하기 위해 hinge 피처를 사용했습니다.
음영 "Training range" 영역은 데이터셋에 실제로 존재하는 값을 커버합니다 — −50(극저온)
근처와 320 너머의 예측은 순수 외삽으로 간주해야 합니다.

드롭다운에서 다른 변수를 시도해 보세요 — Maxent가 각각에 어떤 피처를 모집했는지
볼 수 있습니다. 부드러운 U자 또는 봉우리 모양 곡선은 quadratic 항을 시사하고, 날카로운
각진 단절은 hinge 또는 threshold 피처에서 옵니다.

### Jackknife 중요도와 ROC

**Jackknife Importance** 하위 탭은 ROC와 변수별 막대를 한 그림에 결합:

![9개 Bradypus 변수의 ROC 패널과 Jackknife 패널](../images/ui/dock-4-jackknife-bars.png)

ROC 읽기:

- **Training ROC** (실선, AUC 0.956): 표본 내 적합
- **Mean CV ROC** (점선, AUC 0.758): 5개 공간 폴드 평균
- **Per-fold ROC** (희미함): 분산은 모델의 공간 부분 표본 안정성

Jackknife 읽기:

어두운 막대(이 변수만 가진 모델)와 밝은 막대(이 변수 없는 모델)가 각 변수의 *고유*
기여를 알려줍니다. Bradypus의 경우:

- **`biome`** (범주형) 가 가장 강한 단변량 신호(AUC ≈ 0.78)을 가지며, 제거 시 모델이
    의미있게 손실 — 생물군계 경계가 나무늘보 분포와 밀접히 매핑됨.
- **`bio7`** (기온 연주기 범위)가 두 번째.
- **`bio1`** 은 단독으로 정보를 주지만 여러 변수와 중복(제거 시 작은 하락).
- **`bio5`** 는 가장 낮은 단독 신호(~0.54) — 무작위에 가까움.

이는 정확히 원 Phillips et al. (2006) 논문이 jackknife를 사용해 biome과 계절성이
공동으로 Bradypus에 가장 많은 정보를 가진다고 주장한 방식입니다.

## 공간 투영

같은 결과 탭의 **Spatial Projection** 으로 전환. **cloglog** 출력과 **Auto-load
result as QGIS layer** 켠 상태에서 **▶ Run Spatial Projection** 클릭:

![실행 후 Projection 하위 탭 — 출력 GeoTIFF 경로 표시](../images/ui/dock-4-projection-done.png)

지도가 흰색-초록으로 자동 스타일링되어 QGIS에 나타납니다:

![중남미 전역의 Bradypus 서식적합도 지도](../images/maps/quickstart-final-suitability.png)

고적합도 핵심은 브라질 남동부 대서양림과 아마존 분지를 커버 — 둘 다 잘 알려진
나무늘보 거점 — 그리고 중미에 걸친 부차적 패치. 모델이 안데스(추움, 고도)와 매우
건조한 브라질 북동부(카아칭가)의 부적합성을 정확히 식별합니다.

## 출력 저장

두 파일이 자동으로 생성되었습니다:

- `qmaxent_output/model.pkl` — 직렬화된 학습 모델. 나중에 Data 탭의 **Load
    existing model (.pkl)…** 버튼에서 다시 불러오거나 협업자와 공유. 보안 고려는
    [모델 저장 및 재사용](../saving-models.md) 참고.
- `qmaxent_output/results.xlsx` — 실험 설정·변수 목록·교차검증·Jackknife·반응곡선
    breakpoint를 담은 다중 시트 Excel 보충. 시트별 설명은 [결과
    내보내기](../exporting-results.md) 참고.

투영 전에 **Save analysis charts as PNG** 를 켰다면, 반응곡선·ROC·Jackknife 패널의
발표 가능한 300-dpi PNG도 GeoTIFF 옆에 있습니다.

## 선택: 우선조사 후보지

학습된 모델로 후속 조사를 계획하는 것이 자연스러운 다음 단계입니다.
**⑤ Priority Sites for Survey** 로 전환, **Discovery** 모드 선택, 자동 설정된 최소
적합도 0.9 유지, **▶ Extract Priority Sites** 클릭:

![추출 후 Discovery 모드 Priority Sites 탭](../images/ui/dock-5-priority-sites-extracted.png)

20개 후보 위치(빨간 점)가 적합도 지도 위에 나타나며, Nominatim 역지오코딩으로
주소가 채워집니다:

![Bradypus 적합도 지도 위에 빨간 점으로 표시된 우선조사 후보지](../images/maps/priority-sites-on-canvas.png)

각 후보는 알려진 출현 지점으로부터 최소 1 km, 다른 모든 후보로부터 최소 500 m
떨어져 있어 단일 현장 여행이 여러 개를 합리적으로 커버할 수 있습니다.

## 다음 단계

- **지저분한 래스터로 같은 워크플로**: [Ariolimax 예제](ariolimax.md) 는 좌표계나
    해상도가 공유되지 않는 래스터에서 시작 — Check + Harmonize 도구를 사용합니다.
- **발표된 연구와 같은 워크플로 비교**: [Pitta nympha 예제](pitta-nympha.md) 는
    발표된 Java MaxEnt 분석을 QMaxent에서 재현하고 두 파이프라인이 일치/차이를
    보이는 지점을 논의합니다.
- **더 깊은 이론**: [방법론 해설](../methodological-background.md) 이 본 투어에서
    수용한 각 기본값이 왜 올바른 선택인지 설명합니다.
