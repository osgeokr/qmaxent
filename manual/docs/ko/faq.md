# 자주 묻는 질문 및 문제 해결

흔한 질문, 오류 메시지, 우회 방법. 사용자의 문제가 보이지 않으면
[GitHub에 이슈를 등록](https://github.com/osgeokr/qmaxent/issues)해 주세요.

## 설치 문제

### "QMaxent environment not ready"가 설치 후에도 빨강

**Plugins → QMaxent → Manage Dependencies** 대화상자를 확인하세요.
필수 패키지(`elapid`, `numpy`, `scikit-learn`, `pandas`, `openpyxl`)
중 하나라도 누락이라고 보고되면 **Install missing**을 클릭합니다.

대화상자 자체가 오류를 일으키면 QGIS Python에 `pip`이 누락되었을 수
있습니다. Windows에서는 **OSGeo4W Shell → `python -m ensurepip --upgrade`**
실행; macOS/Linux에서는 `pip`이 활성화된 OSGeo4W 또는 Conda 채널을 통해
QGIS를 설치하세요.

### Windows에서 "ensurepip ... returned non-zero exit status 1"

QGIS 3.44.15 이상의 Windows판에서 v0.1.x의 의존성 설치가 가상환경 안의
`ensurepip` 단계에서 멈추던 문제입니다. v0.2.0은 Windows에서 가상환경을 만들지 않고
QGIS 자체 pip로 QMaxent 패키지 폴더(`%USERPROFILE%\.qgis_qmaxent\venv_py3.12\Lib\site-packages`
처럼 QGIS의 Python 버전을 따름)에 설치하므로 이 오류가 생기지 않습니다. 플러그인을
업데이트한 뒤 의존성을 다시 설치하세요. 환경변수 `QMAXENT_FORCE_VENV=1` 이 설정되어
있다면 해제하세요([의존성 관리](dependencies.md) 참고).

### "Could not find a version that satisfies the requirement elapid"

QGIS Python이 3.9보다 오래된 경우입니다. QMaxent는 elapid가 설정한
하한선인 **Python ≥ 3.9**가 필요합니다. QGIS를 **LTR ≥ 3.34** 릴리스로
업그레이드하세요; 번들 Python은 3.9+입니다.

### 플러그인 관리자가 QMaxent를 보여주지만 "Install"이 회색

이미 *user* 플러그인 디렉터리에 QMaxent가 설치되어 있고 플러그인
관리자가 *system* 패키지 목록을 표시하고 있을 가능성이 높습니다.
**Plugins → Manage and Install Plugins → Installed**를 확인 — 오래된
사본을 먼저 제거한 다음 최신을 다시 설치합니다.

## 모델링 오류

### "ValueError: All presence points fell on NoData cells"

출현 레이어의 CRS나 범위가 래스터 스택과 겹치지 않습니다.
**① Data**에서 확인:

1. QGIS의 출현 점을 클릭하여 좌표 읽기.
2. 대략 동일한 영역의 래스터 셀을 클릭하여 래스터에 그곳 값이
   있는지 확인.
3. 일치하지 않으면 레이어가 잘못 투영된 것 — **Vector → Save Layer
   As…**로 래스터의 CRS로 출현 레이어를 다시 내보냅니다.

일부 출현점만 빠진 경우에는 학습 로그의
`⚠ k of n presence points fall outside the rasters or on NoData cells and were
excluded.` 줄이 제외된 개수를 알려 줍니다
([③ 학습 탭](training-tab.md) 참고).

### "Convergence not reached after 500 iterations"

모델이 반복 한도 내에서 가능도 최대를 찾지 못했습니다. 흔한 세 가지
원인:

- **선택한 피처 클래스에 비해 출현이 너무 적음**. **② Parameters**에서
  **자동 (출현점 수에 따른 maxnet 기본 규칙)** 으로 전환하면 출현점 수에
  맞춰 피처 집합이 줄어듭니다.
- **고도로 공선적인 변수**. 결과 탭의 **Jackknife Importance**로
  거의 영에 가까운 기여자를 식별하여 제거합니다.
- **매우 희귀한 클래스를 가진 범주형 변수**. 모델링 전에 희귀
  클래스를 `Other` 범주로 다시 묶으세요.

### "Grid mismatch — CRS, extent, resolution differ across rasters"

이는 **Check Raster Consistency** 실패 모드입니다. **Harmonize to
Folder…**를 클릭하여 QMaxent가 모든 래스터를 가장 고해상도로
재투영하게 하세요. [Ariolimax 예제](examples/ariolimax.md)가 이
워크플로를 단계별로 안내합니다.

### "래스터 N개의 좌표계(CRS)에 문제가 있습니다"

**래스터 일관성 점검** 이 다른 좌표계로 변환할 수 없는 래스터를 찾았다는
뜻입니다. 좌표계 정보가 없거나, `.prj` 에 `"NAD83(NSRS2007) / Michigan South"`
같은 이름만 있고 datum이 없는 로컬(engineering) 좌표계인 경우입니다. 안내창에
표시된 파일을 레이어 패널에서 우클릭하고 `레이어 좌표계 > 레이어 좌표계
설정...` 에서 올바른 좌표계(안내창이 제안한 EPSG 코드가 있으면 그것)를 지정한 뒤
점검을 다시 실행하세요. 지정한 좌표계는 점검과 Harmonize에 함께 반영됩니다
([① 데이터 탭](data-tab.md) 참고).

### "다중 밴드 래스터는 쓸 수 없다"는 안내

QMaxent는 환경변수 하나에 밴드 하나를 씁니다. `프로세싱 > GDAL > 래스터 변환 >
밴드 재배열` 로 밴드마다 파일 하나씩 저장한 뒤 단일 밴드 파일을 추가하세요.

### 같은 설정인데 실행할 때마다 결과가 조금씩 다릅니다

**② Parameters** 의 **랜덤 시드 고정** 이 켜져 있는지 확인하세요. v0.2.0부터
시드가 배경점 추출에도 적용됩니다(v0.1.x에서는 실행마다 배경점이 달랐습니다).
배경점은 래스터 목록의 첫 래스터에서 뽑으므로, 래스터 순서가 바뀌어도 결과가
달라질 수 있습니다.

## 성능

### 학습이 매우 느림 (출현 100개에 10분 이상)

흔한 두 가지 원인:

- **매우 큰 래스터 스택** (예: 대륙 범위에 30 m 해상도). 배경
  지점 추출이 병목입니다. 래스터 해상도를 종의 행동권 크기에
  적합한 값으로 줄이거나, 먼저 래스터를 클리핑하여 분석을
  관련 ROI로 제한하세요.
- **너무 많은 배경 지점**. 기본 10,000은 대부분의 연구에 적절합니다;
  50,000 이상의 값은 거의 도움이 되지 않고 비례적으로 실행을
  느리게 합니다.

### 공간 투영은 빠르지만 RAM을 많이 사용

투영은 셀 단위로 전체 래스터 스택을 메모리에 로드합니다. 변수 10개의
대륙 1-km 스택의 경우 ~2 GB 피크를 예상하세요. RAM이 빡빡하면
투영 전에 래스터를 다운샘플링 — 학습 해상도에서의 투영은 방법론적으로
방어 가능합니다 ([Elith, Kearney & Phillips 2010](references.md)).

## 출력 및 투영

### "왜 일부 셀이 NoData인가요? 래스터가 그 영역을 커버하는데"

unified preflight 대화상자는 두 셀 범주를 NoData로 자동 마스킹합니다:

1. 학습 중 나타나지 않은 범주형 클래스 코드.
2. 어떤 연속 변수가 학습 범위 밖에 있는 셀 — *관련 안전 마스크가
   활성화된 경우에만*.

이는 의도적입니다 ([Elith, Kearney & Phillips 2010](references.md)) —
학습 영역을 벗어난 외삽은 정의되지 않으며 침묵으로 채워서는 안 되고
표시되어야 합니다.

### 학습한 곳과 다른 지역에 투영할 수 있나요?

가능 — 그러나 preflight 대화상자가 새 지역 범위가 학습 범위를 초과하는
모든 연속 변수를 표시하고 보지 못한 범주형 코드를 자동 마스킹합니다.
v0.2.0부터는 학습 자료에 없던 범주가 남아 있어도 예측이 멈추지 않으며, 그 셀은
범주 기여가 없는 것으로 계산됩니다.
**Yes**를 클릭하기 전에 대화상자를 읽으세요; 보고된 외삽은 논문에서
논의할 발견으로 취급합니다.

### 출력 cloglog 값이 일관되게 낮게 보입니다

대체로 정상 — cloglog 값
([Phillips et al. 2017](references.md))은 "이 위치에서 종의 일반적인
표본이 주어졌을 때의 출현 확률"로 보정되며, 이는 어떤 종의 분포역
대부분에서 진정으로 낮습니다 (<0.5). 봉우리(0.7+)가 중요한 부분이며;
절대 척도는 종 간이나 연구 간에 직접 비교할 수 없습니다.

### 조사지점을 다시 선정했더니 `survey_sites_1.gpkg` 가 생겼습니다

같은 이름의 GeoPackage가 이미 있으면 QMaxent는 프로젝트에서 그 레이어를 빼고 새
결과로 교체합니다. 다른 프로그램(다른 QGIS 창, 파일 탐색기 미리보기 등)이 파일을
열고 있어 지울 수 없을 때만 `survey_sites_1.gpkg` 처럼 다음 빈 이름으로 저장합니다.

## 좋은 질문을 하는 방법

GitHub에 이슈를 등록할 때 다음을 포함해 주세요:

1. **QMaxent 버전** (Plugins → Manage and Install Plugins → Installed,
   또는 `metadata.txt`).
2. **QGIS 버전** (Help → About).
3. **정확한 오류 메시지** (Training 탭 로그 또는 QGIS Python 콘솔에서
   복사).
4. **최소 재현기** — 이상적으로는 문제를 유발하는 작은 데이터셋, 또는
   구성을 볼 수 있도록 **① Data**와 **② Parameters** 탭의 스크린샷.

가장 빨리 해결되는 이슈는 유지 관리자가 자기 컴퓨터에서 실패를
재현할 수 있는 모든 것이 함께 제출되는 이슈입니다.
