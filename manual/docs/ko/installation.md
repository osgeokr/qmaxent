# 설치

QMaxent는 일반적인 QGIS 플러그인으로 설치됩니다. 플러그인 자체가 설치된 뒤에는,
일회성 의존성 설정으로 Maxent에 필요한 Python 라이브러리들이 QMaxent 전용 폴더에
다운로드됩니다(Linux와 macOS는 가상환경, Windows는 패키지 폴더). 시스템 Python이나
QGIS 자체에는 영향을 주지 않습니다.

## 시스템 요구사항

- **QGIS** 3.44 LTR 이상
- **운영체제**: Windows 10/11, macOS 12+, 또는 최신 Linux
- **디스크 공간**: 의존성 패키지용 약 600 MB
- **인터넷 접속**은 의존성 다운로드를 위한 최초 1회만 필요합니다. 일상적인 모델
    학습과 투영은 완전히 오프라인에서 동작합니다.

## QGIS 플러그인 저장소에서 설치

QMaxent는 [공식 QGIS 플러그인 저장소](https://plugins.qgis.org/plugins/qmaxent/)에
공개되어 있습니다.

### 1단계. 플러그인 관리자 열기

![플러그인 메뉴에서 플러그인 관리 및 설치가 강조된 화면](images/steps/ko/i01_menu_manage_plugins.png){ .qm-shot width="365" }

1. **플러그인** 메뉴를 엽니다.
2. **플러그인 관리 및 설치...**를 누릅니다.

### 2단계. QMaxent를 찾아 설치

![QMaxent를 찾은 플러그인 관리자와 플러그인 설치 버튼](images/steps/ko/i02_plugin_manager_search.png){ .qm-shot width="887" }

1. 왼쪽에서 **모두**가 선택되어 있는지 확인합니다.
2. 검색창에 `QMaxent`를 입력합니다.
3. 결과 목록에서 **QMaxent**를 누르면 오른쪽에 설명이 나타납니다.
4. **플러그인 설치**를 누릅니다. 설치가 끝나면 **닫기**를 누릅니다.

### 3단계. QMaxent 메뉴 확인

![QMaxent 하위 메뉴가 열린 플러그인 메뉴](images/steps/ko/i04_menu_qmaxent_deps.png){ .qm-shot width="562" }

1. **플러그인** 메뉴를 다시 엽니다.
2. **QMaxent**에 마우스를 올리면 세 항목이 보입니다.
3. **QMaxent 의존성**은 다음 장에서 설명하는 의존성 설치 화면을 엽니다.

| 메뉴 항목 | 하는 일 |
|---|---|
| **QMaxent 분석** | 모든 모델링을 하는 분석 도크를 엽니다 |
| **QMaxent 의존성** | 의존성 설치 화면을 엽니다(다음 장) |
| **예제 데이터셋 다운로드...** | 이 매뉴얼에서 쓰는 예제 데이터를 받습니다([예제 데이터](example-datasets.md) 참고) |

## 설치 확인

설치가 끝나면 QGIS 오른쪽에 두 패널이 열립니다. 위쪽이 **QMaxent 의존성**, 아래쪽이
**QMaxent 분석** 도크입니다.

![설치 직후 QMaxent 의존성 패널이 강조된 QGIS 주 화면](images/steps/ko/d00_main_first_start.png){ .qm-shot width="1440" }

1. 처음에는 **QMaxent 의존성** 패널에 *Dependencies not installed*가 표시됩니다.
   정상이며, 다음 장에서 설치합니다.

패널이 보이지 않으면 **플러그인 → QMaxent → QMaxent 의존성**을 누르세요. 화면이 열리면
플러그인은 올바르게 설치된 것입니다.

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

??? warning "Windows에서 'ensurepip ... returned non-zero exit status 1' 오류"
    QGIS 3.44.15 이상의 Windows판에서 v0.1.x의 의존성 설치가 이 오류로 멈추던
    문제는 v0.2.0에서 해결되었습니다. 플러그인을 v0.2.0 이상으로 업데이트한 뒤
    **QMaxent Dependencies** 에서 다시 설치하세요. 자세한 설치 방식은
    [의존성 관리](dependencies.md)를 참고하세요.

??? warning "회사 프록시나 제한된 네트워크 환경"
    플러그인 설치 자체는 QGIS의 네트워크 스택을 사용하므로 QGIS의 프록시 설정을 따릅니다.
    의존성 설치(다음 장)는 [PyPI](https://pypi.org)에서 다운로드합니다 — 회사가 PyPI를
    차단하는 경우, 네트워크 관리자에게 `pypi.org`와 `files.pythonhosted.org` 허용을
    요청하세요.
