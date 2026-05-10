# 의존성 관리

QMaxent의 모델링 엔진 — [elapid](https://github.com/earth-chris/elapid) Python
라이브러리와 동반 패키지([rasterio](https://rasterio.readthedocs.io/),
[geopandas](https://geopandas.org/), [scikit-learn](https://scikit-learn.org/),
[matplotlib](https://matplotlib.org/)) — 는 QGIS에 기본 포함되지 않습니다.
플러그인을 처음 사용할 때, **플러그인 전용 격리 가상환경**에 이 의존성들을 설치하여
시스템 Python이나 QGIS 자체에 영향을 주지 않습니다.

이는 약 5분 정도의 일회성 설정입니다.

## Dependencies 다이얼로그 열기

**플러그인 → QMaxent → QMaxent Dependencies** 를 선택하세요.

![설치 전 QMaxent Dependencies 다이얼로그 — Status: Dependencies not installed](images/ui/dialog-dependencies.png)

다이얼로그는 QMaxent 가상환경의 현재 상태, 플러그인 버전, 설치될 항목 요약을 보여줍니다.

## Environment Status 패널

상단 상태 배너가 다음 단계를 정확히 알려줍니다:

| 배너 | 의미 |
|---|---|
| ✗ **Dependencies not installed** | 최초 설치 필요(위 화면) |
| ✓ **QMaxent environment ready** | 모든 의존성이 설치되어 모델링 준비 완료 |
| ⚠ **Update available** | 의존성 중 일부 업데이트 권장 |

## 의존성 설치 / 업데이트

**Install / Update Dependencies** 를 클릭하세요. 플러그인이 [PyPI](https://pypi.org)에서
패키지들을 다운로드해 다이얼로그 하단에 표시된 경로에 설치합니다(기본:
macOS/Linux는 `~/.qgis_qmaxent`, Windows는 `%USERPROFILE%\.qgis_qmaxent`).
총 다운로드는 플랫폼에 따라 약 **300–500 MB** 이며, 일반 광대역에서 3–8분 소요됩니다.

설치가 끝나면 상태 배너가 초록으로 바뀝니다:

![설치 후 QMaxent Dependencies 다이얼로그 — Status: ✓ QMaxent environment ready](images/ui/dialog-dependencies-installed.png)

이제 다이얼로그를 닫고 **QMaxent Analysis** 를 열면 모델링 워크플로가 활성화됩니다.

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

QMaxent의 환경은 표준 Python `venv` 모듈로 생성되며 PyPI에서만 설치됩니다.
플러그인은 런타임에 원격 소스의 임의 코드를 실행하지 않습니다. 저장된 모델 파일
(`.pkl`)은 별도의 고려사항이며, 신뢰할 수 없는 소스의 pickle을 불러올 때의 보안
주의는 [모델 저장 및 재사용](saving-models.md)을 참고하세요.
