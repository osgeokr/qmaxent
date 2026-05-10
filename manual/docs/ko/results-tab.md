# ④ 결과 탭

성공적인 실행 후 결과 탭의 세 하위 탭이 활성화됩니다:

| 하위 탭 | 표시 내용 |
|---|---|
| **Response Curves** | 각 변수가 데이터 범위에서 예측 적합도를 어떻게 형성하는지 |
| **Jackknife Importance** | 어떤 변수가 가장 중요한지, ROC와 함께 |
| **Spatial Projection** | 예측을 GeoTIFF로 작성하고 스타일링된 QGIS 레이어로 불러오기 |

## Response Curves

상단 드롭다운에서 변수를 하나 선택하세요. QMaxent가 변수의 실제 학습 범위(음영
영역)에 걸쳐 예측 cloglog 적합도를 그리며, 평균을 수직 기준선으로 표시합니다:

![학습 범위에 걸친 bio1의 예측 cloglog 적합도 반응곡선](images/ui/dock-4-response-curve-bio1.png)

반응곡선 읽는 법:

- **Y축** — cloglog 적합도 확률(0–1)
- **X축** — 변수의 원래 단위 값. 음영 "Training range" 영역이 모델이 실제로 본
    데이터 범위 — 이 영역 밖으로 곡선이 연장되면 외삽이며 신중하게 해석해야 합니다
    ([방법론 해설](methodological-background.md) 논의 참고)
- **날카로운 봉우리나 계단** 은 보통 hinge 또는 threshold 피처 — 부드러운 곡선은
    linear와 quadratic 피처에서 옴
- **변수별 해석**: 곡선은 다른 모든 변수를 평균으로 고정한 채 한 변수에 대한 부분
    반응을 보여줌 — 생태학적 해석에 사용하고, 예측에는 사용하지 마세요.

드롭다운에서 자유롭게 변수를 전환하세요 — 모델을 재실행하지 않고 그래프가 다시
그려집니다.

## Jackknife 중요도

Jackknife 하위 탭은 플러그인에서 가장 정보 밀도가 높은 화면입니다 — ROC 분석
(학습과 폴드별 CV)을 변수 중요도 막대와 함께 한 사이드-바이-사이드 그림에 결합:

![9개 Bradypus 변수의 ROC 곡선과 Jackknife 변수 중요도](images/ui/dock-4-jackknife-bars.png)

### ROC 패널 (좌측)

- **Training ROC** (실선): 표본 내 적합. 항상 낙관적.
- **Mean CV ROC** (점선): 모든 CV 폴드의 평균. 이것이 모델의 핵심 성능 추정.
- **Per-fold ROC** (희미한 선): 각 공간 CV 폴드의 ROC. 분산은 모델이 공간 부분
    표본에 걸쳐 얼마나 안정적인지 시각화.
- **Random** (대각선): 무작위 기준선.

Bradypus 예제는 train AUC 0.956과 CV AUC 0.758을 보여줍니다 — 모델이 진짜 신호를
학습했고 심각한 과적합이 아님을 시사하는 전형적이고 건강한 격차입니다.

### Jackknife 패널 (우측)

각 변수에 두 개의 막대:

- **With only variable** (어두움): *이 변수만* 가진 모델의 예측 성능. 높음 = 변수가
    강한 단변량 신호 보유.
- **Without variable** (밝음): *이 변수 없이* 모델의 성능. (전체 모델 AUC 대비)
    낮음 = 변수가 다른 변수와 중복되지 않는 고유 정보 보유.

**어두움이 높고 밝음에 눈에 띄는 하락을 만드는** 변수는 명백히 중요합니다. 어두움이
높지만 밝음이 거의 변하지 않는 변수는 다른 변수와 중복(상관 가능성 높음). 각 막대의
검증/학습 분할은 같은 화면에 표시 — 값 라벨은 CV 폴드의 보류 검증 AUC.

## Spatial Projection

이 하위 탭은 학습된 모델을 전체 래스터 스택에 적용해 연속 서식적합도 표면을
생성합니다.

![실행 후 Spatial Projection 하위 탭 — 출력 GeoTIFF 경로 표시](images/ui/dock-4-projection-done.png)

### 출력 변환

| 변환 | 범위 | 해석 |
|---|---|---|
| **cloglog** *(기본)* | 0–1 | 평균 출현율에서의 출현 확률 (Phillips et al. 2017) |
| **logistic** | 0–1 | 이전 logistic 변환 — 하위 호환성용 |
| **raw** | 무제한 | 상대 출현율 — 연구 영역 전체 합이 1 |

`cloglog` 가 거의 모든 연구의 권장 기본값입니다 — 적절한 확률적 해석을 가지며,
상대 출현율과 선형으로 스케일하고, 현대 Maxent 문헌이 보고하는 형식입니다.

### Auto-load result as QGIS layer

기본값으로 켜져 있으며, 결과 GeoTIFF가 작성되는 즉시 QGIS 프로젝트에 추가되어
자동 적용된 흰색-초록 연속 색상으로 스타일됩니다. Bradypus의 결과:

![중남미 전역의 Bradypus 서식적합도 지도, 흰색=낮음, 초록=높음](images/maps/quickstart-final-suitability.png)

이후 레이어의 **Symbology** 패널에서 색상을 재정의할 수 있습니다 — QMaxent의 자동
스타일링은 합리적인 기본값일 뿐입니다.

### Save analysis charts as PNG

체크하면 QMaxent가 Response Curves, ROC, Jackknife 그래프의 고해상도 PNG 사본도
GeoTIFF 옆에 작성합니다. 이 그림들은 발표 가능한 수준이며 본 탭에 보이는 것과
일치합니다 — 원고 그림 직접 붙여넣기용 크기(300 dpi)로 저장됩니다.

## 메모리와 성능

공간 투영은 각 래스터 셀을 셀 단위로 읽어 모델을 적용합니다. 매우 큰 래스터(30 m
해상도의 대륙 규모)는 투영 단계가 실행 시간을 지배합니다. 두 가지 팁:

- **연구 범위로 미리 타일링** — QGIS의 `Clip raster by extent` 또는 `Warp` 알고리즘
    사용
- **해상도 거칠게** — 30 m 기후 래스터는 거의 의미 없음. 250 m–1 km가 보통 SDM에
    충분하고 훨씬 빠름.
