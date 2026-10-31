# 의존성 관리

QMaxent의 모델링 엔진 — [elapid](https://github.com/earth-chris/elapid) Python
라이브러리와 동반 패키지([rasterio](https://rasterio.readthedocs.io/),
[geopandas](https://geopandas.org/), [scikit-learn](https://scikit-learn.org/),
[matplotlib](https://matplotlib.org/)) — 는 QGIS에 기본 포함되지 않습니다.
플러그인을 처음 사용할 때, **QMaxent 전용 폴더**에 이 의존성들을 설치하여
시스템 Python이나 QGIS 자체에 영향을 주지 않습니다(Linux와 macOS는 격리 가상환경,
Windows는 패키지 폴더. 아래 [설치 위치와 방식](#install-location) 참고).

이는 약 5분 정도의 일회성 설정입니다.

## 단계별 따라하기

### 1단계. QMaxent 의존성 패널 열기

플러그인을 설치하면 이 패널이 저절로 열립니다. 메뉴에서 언제든 다시 열 수도 있습니다.

![플러그인 메뉴에서 QMaxent와 QMaxent 의존성이 강조된 화면](images/steps/ko/i04_menu_qmaxent_deps.png){ .qm-shot width="562" }

1. **플러그인** 메뉴를 엽니다.
2. **QMaxent**에 마우스를 올립니다.
3. **QMaxent 의존성**을 누릅니다.

### 2단계. 의존성 설치

![설치 전 QMaxent 의존성 패널](images/steps/ko/d01_deps_not_installed.png){ .qm-shot width="380" }

1. **환경 상태**에 *Dependencies not installed*가 표시됩니다.
2. **의존성 설치 / 업데이트**를 누릅니다.

패키지는 [PyPI](https://pypi.org)에서 받아 패널 아래쪽에 표시된 경로
`~/.qgis_qmaxent`(Windows는 `%USERPROFILE%\.qgis_qmaxent`)에 설치됩니다. 내려받는 양은
300~500 MB 정도(설치 후 약 590 MB)이고, 일반 인터넷 환경에서 3~8분 걸립니다. 기다리는
동안 버튼 아래에 진행 막대와 설치 중인 패키지 이름이 표시됩니다.

![패키지를 설치하고 있는 QMaxent 의존성 패널](images/steps/ko/d02_deps_installing.png){ .qm-shot width="380" }

### 3단계. 준비 완료 확인

![설치가 끝나 초록색 준비 완료 상태가 표시된 QMaxent 의존성 패널](images/steps/ko/d03_deps_ready.png){ .qm-shot width="380" }

1. **환경 상태**가 초록색 *✓ QMaxent environment ready*로 바뀝니다. 패널을 닫고
   **QMaxent 분석**을 열면 됩니다.

## 환경 상태 메시지

| 메시지 | 의미 |
|---|---|
| ✗ **Dependencies not installed** | 처음 한 번 설치가 필요합니다 |
| ✓ **QMaxent environment ready** | 모두 설치되어 모델링을 시작할 수 있습니다 |
| ✗ **Dependencies need updating** | 플러그인 업그레이드로 패키지 목록이나 설치 방식이 바뀌었습니다. **의존성 설치 / 업데이트**를 다시 누르세요 |

## 설치 위치와 방식 {#install-location}

설치 폴더 이름은 QGIS에 들어 있는 Python 버전을 따릅니다(예: Python 3.12이면
`venv_py3.12`).

| 운영체제 | 방식 | 설치 위치 (예) |
|---|---|---|
| Linux, macOS | 가상환경(`venv`)을 만들고 그 안의 pip로 설치 | `~/.qgis_qmaxent/venv_py3.12/` (가상환경 폴더) |
| Windows (v0.2.0) | QGIS 자체 Python의 pip로 패키지 폴더에 설치 (`pip install --target`), 별도 인터프리터 없음 | `%USERPROFILE%\.qgis_qmaxent\venv_py3.12\Lib\site-packages` |

Windows 방식이 바뀐 이유: QGIS 3.44.15 이상의 Windows판에서는 가상환경 안에서
`ensurepip` 가 실패해("ensurepip ... returned non-zero exit status 1") 패키지를
하나도 내려받기 전에 설치가 멈췄습니다. QMaxent는 설치한 패키지를 QGIS 안에서
불러오기만 하므로 Windows에서는 가상환경 인터프리터가 필요 없습니다. v0.2.0은 QGIS
자체 pip로 QMaxent 패키지 폴더에 설치해 이 문제를 피합니다. Linux와 macOS는 기존
가상환경 방식을 그대로 씁니다.

두 환경변수로 동작을 바꿀 수 있습니다. QGIS를 시작하기 전에 설정하세요.

| 환경변수 | 효과 |
|---|---|
| `QMAXENT_CACHE_DIR` | 기본 `~/.qgis_qmaxent` 대신 다른 위치에 설치 (예: 홈 폴더에 쓸 수 없거나 공간이 부족할 때) |
| `QMAXENT_FORCE_VENV=1` | Windows에서도 이전(v0.1.x) 가상환경 방식을 사용 |

## 설치되는 패키지

QMaxent는 다음 패키지와 그 의존성을 설치합니다:

- **[elapid](https://github.com/earth-chris/elapid)** — Maxent 엔진
    (maxnet 알고리즘, 공간 교차검증, 투영)
- **[rasterio](https://rasterio.readthedocs.io/)** — 래스터 입출력
- **[geopandas](https://geopandas.org/)** — 출현 지점 벡터 입출력
- **[scikit-learn](https://scikit-learn.org/)** — 교차검증, ROC, AUC
- **[scipy](https://scipy.org/)** + **[numpy](https://numpy.org/)** — 수치 연산 핵심
- **[matplotlib](https://matplotlib.org/)** — 반응곡선, ROC, Jackknife 그래프

QMaxent가 테스트한 버전으로 핀이 고정되어 있으며, 각 릴리스 노트에 포함된 버전이
기록되어 있습니다.

## Remove Environment

**Remove Environment** 는 QMaxent 가상환경 전체를 디스크에서 삭제합니다. 다음과
같은 경우에 사용합니다:

- QMaxent를 더 이상 사용하지 않아 디스크 공간 확보
- 무언가 망가졌을 때 깨끗한 재설치 강제
- 다른 패키지 버전을 핀하는 다른 QMaxent 버전으로 전환

제거 후 상태 배너는 *Dependencies not installed* 로 돌아가고, **Install / Update
Dependencies** 가 다시 활성화됩니다.

## 문제 해결

??? warning "네트워크 또는 SSL 오류로 설치 실패"
    대부분 회사 프록시나 방화벽이 [PyPI](https://pypi.org) 를 차단하기 때문입니다.
    관리자에게 `pypi.org` 와 `files.pythonhosted.org` 허용을 요청하세요. 설치 도구는
    QGIS의 프록시 설정을 따르므로, **설정 → 옵션 → 네트워크** 를 구성하면 보통 해결됩니다.

??? warning "Windows에서 'ensurepip ... returned non-zero exit status 1' 오류"
    v0.1.x를 QGIS 3.44.15 이상의 Windows판에서 쓸 때 생기던 오류입니다. 플러그인을
    v0.2.0 이상으로 업데이트한 뒤 **Install / Update Dependencies** 를 다시
    클릭하세요. v0.2.0은 가상환경을 만들지 않고 QGIS 자체 pip로 설치하므로 이
    단계를 거치지 않습니다([설치 위치와 방식](#install-location) 참고).
    `QMAXENT_FORCE_VENV=1` 을 설정해 두었다면 해제하세요.

??? warning "설치 중 디스크 공간 부족"
    설치 중에는 임시로 최종 크기의 약 두 배가 필요합니다. 홈 디렉터리가 있는 드라이브에
    최소 1 GB를 확보하고 다시 시도하세요.

??? warning "성공적인 설치 후에도 "No module named …" 오류"
    QGIS를 한 번 재시작하세요. Python의 import 캐시는 새 환경을 인식하기 위해 깨끗한
    프로세스가 필요합니다.

??? warning "플러그인은 시작되지만 Run Maxent 버튼이 비활성화"
    **Dependencies** 다이얼로그를 열어 상태가 초록인지 확인하세요. 초록이면 QGIS를
    재시작, 그렇지 않으면 **Install / Update Dependencies** 를 다시 클릭하세요.

## 가상환경 보안 안내

QMaxent의 환경은 Linux와 macOS에서는 표준 Python `venv` 모듈로, Windows에서는
QGIS 자체 pip의 패키지 폴더 설치로 만들어지며, 어느 경우든 PyPI에서만 설치됩니다.
플러그인은 런타임에 원격 소스의 임의 코드를 실행하지 않습니다. 저장된 모델 파일
(`.pkl`)은 별도의 고려사항이며, 신뢰할 수 없는 소스의 pickle을 불러올 때의 보안
주의는 [모델 저장 및 재사용](saving-models.md)을 참고하세요.
