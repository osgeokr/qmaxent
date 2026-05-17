# 빠른 시작

이 5분 워크스루는 갓 설치한 플러그인에서 출발해 *Bradypus variegatus*(세발가락나무늘보)의
완성된 서식적합도 지도까지 가는 과정을 다룹니다 — 원 Maxent 논문이 사용한 종입니다.
본 장의 모든 스크린샷은 사용자가 직접 재현할 수 있는 실제 수치를 보여줍니다.

!!! tip "사전 준비"
    [설치](installation.md) 와 [의존성 관리](dependencies.md) 를 먼저 완료하세요 —
    **QMaxent environment ready** 배너가 초록색이어야 진행할 수 있습니다.

## 1단계 · 예제 데이터셋 다운로드

**플러그인 → QMaxent → Download Example Dataset** 을 열고, **Bradypus variegatus**
선택, 저장 경로 기본값 유지, **Download** 클릭.

![Bradypus 선택된 Download Example Dataset 다이얼로그](images/ui/dialog-download-example-dataset.png)

몇 초 뒤, 출현 지점 레이어와 9개 환경 변수 래스터가 QGIS 프로젝트에 추가됩니다:

![중남미 지역의 환경 래스터 위에 표시된 Bradypus 출현 지점](images/maps/example-bradypus-loaded.png)

## 2단계 · Analysis 도크 열고 데이터 불러오기

**플러그인 → QMaxent → QMaxent Analysis** 를 엽니다. 도크에는 다섯 개의 번호 매겨진
탭이 있고 순서대로 진행합니다. **① Data** 부터 시작 — **Presence Points Layer**
드롭다운에서 `bradypus` 를 선택하고, **Add from project** 를 클릭해 로드된 모든 래스터
레이어를 한 번에 추가합니다. `biome` 은 `[categorical]` 로 표시해 QMaxent가 연속 변수가
아닌 이산 요인으로 처리하게 합니다.

**Check Raster Consistency** 를 클릭. 예제 데이터는 이미 정렬되어 있어 초록 ✓ 가
표시됩니다:

![Bradypus 출현 레이어(116점)와 7개 환경 래스터, biome이 categorical로 표시되고, Check Raster Consistency 결과 모든 래스터가 그리드 일치](images/ui/dock-1-data-with-bradypus.png)

도크 하단 상태바에 `presence=116 background=10,104` 이 표시됩니다.

## 3단계 · 기본 파라미터 사용

**② Parameters** 로 이동합니다. 기본값은 거의 모든 첫 실행에 발표 가능한 수준이
되도록 의도적으로 설정되어 있습니다: **Auto** 피처 선택(maxnet 규칙),
**정규화 배수 1.0**, **Geographic K-Fold(k=5)** 공간 교차검증, **Jackknife 변수 중요도**
활성화.

![기본 설정의 Parameters 탭](images/ui/dock-2-parameters-defaults.png)

Output Files 섹션에는 학습된 모델용 `.pkl` 과 발표용 보충 표용 `.xlsx` 경로가 지정되어
있습니다. 나중에 변경할 수 있으니 지금은 기본값을 그대로 둡니다.

## 4단계 · Maxent 실행

도크 하단의 초록색 **▶ Run Maxent** 버튼을 클릭합니다. **③ Training** 탭으로 전환되며
진행 상황이 실시간으로 보고됩니다.

약 30초 후 실행이 완료됩니다:

![100% 완료된 Training 탭과 전체 로그: 116 presence, 10,104 background, train AUC=0.9569, CV AUC=0.7436 ± 0.0750, 변수별 jackknife](images/ui/dock-3-training-completed.png)

로그를 위에서 아래로 읽으면 모델의 전체 이야기를 알 수 있습니다:

- 배경 표본 추출과 공변량 추출 완료
- 자동 규칙이 선택한 피처 유형: linear, quadratic, product, hinge, threshold
- **Training AUC = 0.9569**(표본 내 적합)
- **CV AUC = 0.7436 ± 0.0750**(5개 공간 폴드의 보류 검증 성능)
- 9개 변수 각각의 Jackknife 결과

도크 하단 상태바에 핵심 수치가 표시됩니다: `train AUC=0.9569 · CV AUC=0.7436`.

## 5단계 · 결과 확인

**④ Results** 탭이 활성화됩니다. 세 개의 하위 탭을 둘러보며 모델이 학습한 내용을
이해해 보세요.

### Response Curves(반응곡선)

드롭다운에서 변수를 하나 선택 — 여기서는 `bio1`(연평균 기온):

![학습 범위에 걸친 bio1 cloglog 적합도 반응곡선](images/ui/dock-4-response-curve-bio1.png)

음영 영역은 실제 데이터 범위를 나타냅니다. 이 범위 밖의 예측은 외삽이며 신중하게
해석해야 합니다.

### Jackknife 중요도와 ROC

Jackknife 하위 탭은 두 가지 진단을 한 화면에 결합합니다 — 좌측의 ROC 곡선(학습 +
폴드별 CV)과 우측의 Jackknife 막대(각 변수만 / 각 변수 제외 시 모델 AUC):

![9개 Bradypus 변수의 ROC 곡선과 Jackknife 변수 중요도](images/ui/dock-4-jackknife-bars.png)

평균 CV ROC(점선)는 무작위 대각선을 한참 위에 있고, Jackknife 막대는 `bio7`,
`biome`, `bio12` 가 이 종에 가장 많은 정보를 담고 있음을 보여줍니다.

### Spatial Projection(공간 투영)

Spatial Projection 하위 탭은 학습된 모델을 전체 환경 변수 래스터에 적용해 서식적합도
지도를 생성합니다. **cloglog** 가 권장 출력 변환입니다(Phillips et al. 2017).
**Auto-load result as QGIS layer** 체크를 유지한 채 **▶ Run Spatial Projection** 클릭:

![실행 후 Spatial Projection 하위 탭 — 출력 GeoTIFF 경로 표시](images/ui/dock-4-projection-done.png)

결과 래스터가 흰색-초록 연속 색상으로 자동 스타일링되어 QGIS에 즉시 추가됩니다:

![중남미 전역의 Bradypus 서식적합도 지도, 흰색=낮음, 초록=높음](images/maps/quickstart-final-suitability.png)

## 6단계 · 우선조사 후보지 생성 *(선택)*

**⑤ Priority Sites for Survey** 로 이동합니다. **Discovery** 모드(새 개체군 발견)를
선택하고 `Minimum suitability` 임계값을 높게(예: 0.9), **Number of priority sites** 를
20으로, 역지오코딩을 활성화한 뒤 **▶ Extract Priority Sites** 클릭:

![추출 후 Discovery 모드 Priority Sites 탭 — 20개 추출, 19/20 지오코딩됨](images/ui/dock-5-priority-sites-extracted.png)

후보 위치가 적합도 지도 위에 빨간 점으로 표시됩니다:

![Bradypus 적합도 지도 위에 빨간 점으로 표시된 우선조사 후보지](images/maps/priority-sites-on-canvas.png)

출력 GeoPackage의 속성 테이블에는 각 후보지의 좌표, 적합도 점수, (Nominatim이
해결할 수 있던 경우) 행정 주소가 포함되어 있어 현장 조사에 즉시 사용할 수 있습니다.

## 다음 단계

축하합니다 — 완전한 Maxent 워크플로를 실행했습니다. 이제부터는:

- **학습된 모델을 나중에 재사용** → [모델 저장 및 재사용](saving-models.md)
- **결과를 논문으로 정리** → [결과 내보내기](exporting-results.md)
- **기본값의 근거를 이해** → [방법론 해설](methodological-background.md)
- **더 깊이 있는 사례 분석** → [실전 예제](examples/index.md) 가 Bradypus, Ariolimax,
    Pitta nympha를 차례대로 다룹니다.
