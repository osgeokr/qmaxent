# 설치

QMaxent는 일반적인 QGIS 플러그인으로 설치됩니다. 플러그인 자체가 설치된 뒤에는,
일회성 의존성 설정으로 Maxent에 필요한 Python 라이브러리들이 격리된 가상환경에
다운로드됩니다 — 시스템 Python이나 QGIS 자체에는 영향을 주지 않습니다.

## 시스템 요구사항

- **QGIS** 3.44 LTR 이상
- **운영체제**: Windows 10/11, macOS 12+, 또는 최신 Linux
- **디스크 공간**: Python 가상환경용 약 500 MB
- **인터넷 접속**은 의존성 다운로드를 위한 최초 1회만 필요합니다. 일상적인 모델
    학습과 투영은 완전히 오프라인에서 동작합니다.

## QGIS 플러그인 저장소에서 설치

QMaxent는 [공식 QGIS 플러그인 저장소](https://plugins.qgis.org/plugins/qmaxent/)에
공개되어 있습니다.

1. QGIS에서 **플러그인 → 플러그인 관리 및 설치…** 를 엽니다.
2. **All** 탭의 검색창에 `qmaxent` 를 입력하면 결과 목록에 플러그인이 나타납니다.
3. **Install Plugin** 을 클릭합니다.

설치가 완료되면 **플러그인 → QMaxent** 메뉴에 세 가지 진입점이 추가됩니다:

![플러그인 → QMaxent 메뉴에 표시된 세 가지 진입점](images/ui/menu-plugins-qmaxent.png)

| 메뉴 항목 | 설명 |
|---|---|
| **QMaxent Analysis** | 메인 분석 도크 — 플러그인의 핵심 |
| **QMaxent Dependencies** | 의존성 관리 다이얼로그 (다음 장) |
| **Download Example Dataset** | 튜토리얼용 SDM 예제 데이터셋 (*예제 데이터* 장 참고) |

## 설치 확인

**플러그인 → QMaxent → QMaxent Dependencies** 를 선택하세요. 다이얼로그가 정상적으로
열린다면 플러그인은 올바르게 설치된 것입니다. 처음에는 **Environment Status** 배너가
*Dependencies not installed* 로 표시되는데, 이는 정상이며 다음 장에서 설치합니다.

## 새 버전으로 업데이트

새 릴리스가 배포되면 QGIS 플러그인 매니저가 QMaxent를 업데이트 가능 상태로 표시합니다.
**Installed** 탭의 **Upgrade Plugin** 을 클릭하세요. 패치 릴리스(0.1.x → 0.1.y)
사이에는 보통 의존성 재설치가 필요하지 않습니다 — 변경이 있으면 **Dependencies**
다이얼로그가 알려줍니다.

## 문제 해결

??? warning "검색 결과에 QMaxent가 보이지 않습니다"
    QGIS 버전이 3.44 이상인지 확인하세요(`도움말 → QGIS 정보`). 더 낮은 버전에서는
    호환성 문제로 필터링됩니다. **Settings** 의 **Show experimental plugins** 옵션도
    원하는 대로 설정되어 있는지 확인하세요 — QMaxent의 안정 버전은 이 옵션이 필요하지
    않습니다.

??? warning "플러그인 설치는 됐는데 메뉴 항목이 보이지 않습니다"
    QGIS를 재시작하세요. 일부 QGIS 버전은 새 플러그인의 최상위 메뉴 항목을 등록하기
    위해 재시작이 필요합니다.

??? warning "회사 프록시나 제한된 네트워크 환경"
    플러그인 설치 자체는 QGIS의 네트워크 스택을 사용하므로 QGIS의 프록시 설정을 따릅니다.
    의존성 설치(다음 장)는 [PyPI](https://pypi.org)에서 다운로드합니다 — 회사가 PyPI를
    차단하는 경우, 네트워크 관리자에게 `pypi.org`와 `files.pythonhosted.org` 허용을
    요청하세요.
