# ⑤ 우선조사 후보지

다섯 번째 탭은 서식적합도 래스터를 **현장 조사용 후보지 목록** 으로 변환합니다.
새 개체군 발견과 모델 검증 — 두 가지 명확히 다른 목적이 있고, QMaxent는 별도의
표본 추출 전략으로 둘 다 지원합니다.

## 조사 목적

하나 선택:

| 모드 | 목표 | 참고문헌 |
|---|---|---|
| **Discovery** | 미조사 영역에서 새 개체군 발견 | Williams et al. 2009 |
| **Model validation** | 적합도 그래디언트가 출현/부재를 예측하는지 검증 | Rhoden, Peterman & Taylor 2017 |

두 모드는 모델에 매우 다른 질문을 던지므로, 패널 나머지가 선택된 목적에 따라 두
가지 다른 구성으로 전환됩니다.

## Discovery 모드

![Discovery 모드의 Priority Sites 탭](images/ui/dock-5-priority-sites-discovery.png)

학습된 모델이 **높은 적합도** 를 예측한 위치에서 후보지를 생성해, 현장 팀이 효율적
탐색을 할 수 있도록 안내합니다.

### Discovery 설정

- **Minimum suitability**: 셀이 표본에서 제외되는 임계값. 기본 0.9000은 *raster
    maximum × 0.9* 로 자동 설정 — 가장 유망한 영역만 포착. 더 넓은 후보 풀을
    원하면 0.7 등으로 낮추세요.
- **Sampling order**:
    - **Random**: 고적합도 영역 안에서 균일 무작위 표본. 후보 풀의 공간 커버리지를
        원할 때.
    - **Top-N (highest first)**: 가장 적합도 높은 *N*개 셀 선택. 정말 최고 사이트만
        방문해야 할 때.

### 표본 추출 전략

목적과 무관하게 후보 집합을 형성하는 세 개의 수치 컨트롤:

- **Number of priority sites**: 생성할 후보 수(기본 20).
- **Min. distance from existing presences (m)**: 어떤 출현 지점으로부터의 이 반경
    이내 표본을 금지. 기본 1,000 m. 출현 지점이 GPS 부정확하거나 알려진 곳에서
    멀리 떨어진 진정한 새 개체군을 찾고 싶으면 증가.
- **Min. distance between sites (m)**: 이미 선택된 후보로부터의 이 반경 이내 표본을
    금지. 기본 500 m. 후보를 고적합도 영역에 분산.

### 역지오코딩

**Add administrative address (province/city/district)** 체크 — QMaxent가
[OpenStreetMap Nominatim](https://nominatim.openstreetmap.org/) 에 질의해 각
후보지에 사람이 읽을 수 있는 주소를 추가합니다. API 키 불필요 — Nominatim
[사용 정책](https://operations.osmfoundation.org/policies/nominatim/) 준수. 약 20개
사이트의 지오코딩에 10–20초 소요.

## Model validation 모드

검증 모드는 새 현장 조사로 적합도 그래디언트를 확인 또는 반증하기 위한 것입니다.
Rhoden, Peterman & Taylor (2017)에 따라, **적합도 사분위 4개에 걸쳐 표본을
층화** 하여 각 대역이 조사 설계에 대표적으로 포함되도록 합니다.

### 임계값 방법

가장 낮은 사분위의 하한이 어떻게든 설정되어야 합니다. QMaxent는 표준 방법들을 제공:

| 방법 | 정의 | 참고문헌 |
|---|---|---|
| **MTP** | *Minimum Training Presence* — 어떤 출현 지점에서의 최저 적합도 | Pearson et al. 2007 |
| **T10** | *10% Training Presence* — 일부 이상치 출현에 강건 | — |
| **MaxSSS** | *Maximum Sum of Sensitivity and Specificity* — ROC의 최적 임계값 | Liu, White & Newell 2013 |
| **Custom** | 사용자가 직접 값 선택 | — |

`MaxSSS` 가 검증 목적에 권장됩니다 — 모델의 분별력을 최대화하는 임계값이기 때문.
`MTP` 는 더 관대하며 가장 낮은 역사적 점유 사이트도 생물학적 의미를 가질 때 유용합니다.

## 출력

- **Output vector layer**: GeoPackage(`.gpkg`)가 다른 QMaxent 출력 옆에 작성됩니다.
    각 후보가 점이며 속성 `suitability`, `quartile`(검증 모드), `min_distance_m`,
    그리고 지오코딩이 활성화된 경우 Nominatim 주소 필드를 가집니다.
- **Auto-add to QGIS project** *(기본 켜짐)*: 새 레이어가 기본 빨간 점 심볼과 함께
    로드되어 적합도 지도 위에 후보를 즉시 볼 수 있습니다.

Bradypus 모델에 Discovery 실행 후:

![추출 후 Priority Sites 탭 — 20개 추출, 19/20 지오코딩됨](images/ui/dock-5-priority-sites-extracted.png)

상태바는 "20 priority sites extracted (MinSuit = 0.9000); 19/20 geocoded" 를
보고합니다. 한 사이트는 외해 위에 떨어져 지오코딩 실패 — Nominatim에 그곳의
주소 기록이 없기 때문입니다.

후보가 적합도 지도 위에 표시됩니다:

![Bradypus 적합도 지도 위에 빨간 점으로 표시된 우선조사 후보지](images/maps/priority-sites-on-canvas.png)

## 조사 설계 팁

- **Discovery와 Validation 둘 다** 현장 예산이 있으면 함께 실행. Discovery는
    확인할 새 개체군을 주고, Validation은 모델 정확도 추정을 줍니다.
- **자동 설정 Minimum suitability를 재정의** — 연구 시스템의 "높음"이 모델과
    생물학적으로 다를 때. Bradypus의 cloglog 0.9는 우림 핵심을 포착, 고산종은 더
    낮게 조정.
- **간격 제약을 활용** — 조사 여정이 실제로 사이트에 도달할 수 있도록. 사이트 간
    기본 500 m는 도로 접근 종에 적당, 사이트당 다일 원정이 필요한 서식지는 증가.
