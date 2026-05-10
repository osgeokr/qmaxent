# 결과 내보내기

QMaxent 학습 실행은 매번 `.pkl` 모델 옆에 **다중 시트 XLSX 워크북** 을
작성합니다. 워크북은 의도적으로 학술 논문의 **보충 표 블록** 으로 읽히도록
포맷됐습니다 — Times New Roman, 표 번호 헤딩, 각 측정 지표가 어떻게
계산됐는지 설명하는 행 푸터. 시트를 논문 부록에 직접 붙여넣을 수 있습니다.

## 출력 폴더 레이아웃

학습 + 공간 투영 + `Save analysis charts as PNG` 체크박스가 켜진 실행
직후 출력 폴더는 다음과 같습니다:

![QMaxent 출력 폴더 — model.pkl, prediction.tif, results.xlsx, priority_sites.gpkg, 그리고 PNG로 저장된 세 분석 차트](images/results/output-folder.png)

| 파일 | 생성자 | 용도 |
|---|---|---|
| `model.pkl` | 학습 | *Load existing model (.pkl)…* 로 나중에 재로드 |
| `prediction.tif` | 공간 투영 | 서식 적합도 래스터 |
| `prediction.tif.aux` | 공간 투영 | QGIS 보조 메타데이터 |
| `results.xlsx` | 학습 | 다중 시트 보충 표 |
| `priority_sites.gpkg` | Priority Sites 탭 | 현장 가능 후보 지점 |
| `prediction_roc.png` | 공간 투영 | ROC 곡선 (300 dpi) |
| `prediction_jackknife.png` | 공간 투영 | Jackknife 막대 (300 dpi) |
| `prediction_response_curves.png` | 공간 투영 | 모든 반응 곡선 (300 dpi) |

## 시트별 설명

XLSX는 어떤 기능을 실행했는지에 따라 4–5개 시트를 가집니다. 아래 스크린샷은
Pitta nympha 실행에서 캡처 — *형식* 은 모든 데이터셋에 동일하며, 값만 변합니다.

### Table 1 — 실험 설정

전체 실행 구성을 한 페이지로 기록. Reviewer가 시드, 변수 수, 정규화, CV
방식, 학습/CV AUC를 한눈에 검증할 수 있습니다.

![Table 1 — 실험 설정, 학습 데이터, 실행 단위 측정 지표](images/results/xlsx-1-experimental-setup.png)

이 시트는 **Methods 섹션 재현성** 을 위한 단일 최고의 산출물입니다.
"Table S1" 으로 인용하면 독자가 분석을 비트 단위로 재실행할 수 있습니다.

### Table 2 — 예측 변수

모든 환경 변수, 그 유형 (`continuous` / `categorical`), 학습 데이터 범위
나열. 연속 변수는 `[min, max]`, 범주형은 학습 시 만난 이산적 수준 집합.

![Table 2 — 예측 변수, 유형, 학습 데이터 범위](images/results/xlsx-2-predictor-variables.png)

이 시트로 **외삽 범위** 를 문서화하세요 — 이 범위 밖의 환경 값에 적용된
어떤 예측이든 정의상 외삽이며 결과 논의에서 표시되어야 합니다.

### Table 3 — 교차검증 결과

Fold별 보류 AUC와 fold 사이의 평균 ± 표준편차 — 핵심 성능 추정.

![Table 3 — 교차검증 결과 (Random K-Fold)](images/results/xlsx-3-cv-results.png)

degenerate 검증 셋을 가진 fold (예: 공간 분할 후 모두 양성) 는 `NaN` 으로
보고되며 평균에서 제외됩니다. 이는 fold가 유효한 AUC를 만들 수 없는 드문
경우를 원본 Maxent 문헌이 처리하는 방식과 일치합니다.

### Table 4 — Jackknife 변수 중요도

각 변수에 대해 4개의 AUC (only-train, only-test, without-train,
without-test), 그리고 기여 크기를 스캔하기 쉽게 만드는 "Train AUC drop"
및 "Test AUC drop" 열. 변수는 *Test AUC drop 내림차순* 으로 정렬 —
가장 고유하게 정보 있는 변수가 맨 위에.

![Table 4 — Jackknife 변수 중요도](images/results/xlsx-4-jackknife.png)

푸터 노트가 drop 열을 설명:
*Drop = full-model AUC − without-variable AUC; 큰 drop은 변수의 고유한
기여가 다른 변수에서 회복하기 어려움을 가리킵니다.*

### 선택 — Table 5 — 우선조사 후보지

**Priority Sites for Survey** 탭을 실행했다면 같은 워크북에 다섯 번째
시트가 추가되며 후보 위치당 한 행을 가집니다. 컬럼은 `lat`, `lon`,
`suitability`, 그리고 — 역지오코딩이 활성화됐을 때 — `country`,
`province`, `city_county`, `district`, 사람이 읽을 수 있는 `display_name`.
형식은 QMaxent가 GeoPackage 안에서 사용하는 것과 동일:

![Priority Sites 속성 테이블 — 한국어 역지오코딩된 행정 주소를 가진 20개 후보](images/results/attribute-table-priority-sites.png)

## 내보내기 커스터마이즈

**② Parameters** 탭의 기본 파일 경로는 `<home>/qmaxent_output/results.xlsx`
입니다. 쓰기 가능한 어떤 곳으로든 변경 가능 — 상대 경로는 현재 QGIS 프로젝트
폴더에 대해 해석됩니다.

**XLSX 작성을 멈추려면** (드물지만 — 보통 파일 무시가 비활성화보다 빠름)
경로 필드를 비우면 됩니다. QMaxent는 내보내기를 건너뛰고 학습 로그에 알림을
보냅니다.

PNG 분석 차트는 **④ Results → Spatial Projection** 하위 탭의 **Save
analysis charts as PNG** 체크박스로 제어됩니다. 체크된 상태에서 투영 실행
시점에 GeoTIFF 옆에 세 개의 추가 PNG가 작성됩니다 — **단일 컬럼 원고
그림에 직접 붙여넣을 수 있는 크기** (300 dpi, 약 1.6 MB 각).

## 워크북 인용

XLSX는 발표된 보충 표가 인용되는 방식으로 인용되도록 설계됐습니다:

> Yu, B.-H. (2026). QMaxent results workbook for *<species>*
> [Supplementary table]. https://github.com/osgeokr/qmaxent

모델이 재현 가능하다면 (고정 시드 + 동일 입력 래스터) `model.pkl` 을 XLSX와
함께 배포해 reviewer가 모델을 다시 로드해 예측을 독립적으로 검사하게 할 수
있습니다.
