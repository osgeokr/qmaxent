# 예제 데이터

QMaxent는 두 가지 표준 SDM 예제 데이터셋을 한 번의 클릭으로 다운로드할 수 있는
기능을 **플러그인 → QMaxent → 예제 데이터셋 다운로드...** 메뉴에서 제공합니다.
지난 20년간 Maxent 문헌에서 사용된 동일한 데이터셋이라, 결과를 발표 벤치마크와
직접 비교할 수 있습니다.

## 단계별 따라하기

### 1단계. 다운로드 창 열기

![플러그인 메뉴에서 QMaxent와 예제 데이터셋 다운로드가 강조된 화면](images/steps/ko/e00_menu_example.png){ .qm-shot width="562" }

1. **플러그인** 메뉴를 엽니다.
2. **QMaxent**에 마우스를 올립니다.
3. **예제 데이터셋 다운로드...**를 누릅니다.

### 2단계. 데이터셋과 저장 폴더 고르기

![Bradypus가 선택된 예제 데이터셋 다운로드 창](images/steps/ko/e01_example_dialog_bradypus.png){ .qm-shot width="540" }

1. 데이터셋을 고릅니다. 처음이라면 **Bradypus variegatus**가 좋습니다.
2. **저장 위치**를 확인합니다. 기본값은 홈 폴더 아래 `qmaxent_examples` 폴더이며,
   **...**를 눌러 바꿀 수 있습니다.
3. **다운로드**를 누릅니다.

**Ariolimax**를 고르면 **Variant**(변형) 선택이 함께 나타납니다.

![Ariolimax와 Variant 선택이 강조된 예제 데이터셋 다운로드 창](images/steps/ko/e01_example_dialog_ariolimax.png){ .qm-shot width="540" }

1. **Ariolimax**를 선택합니다.
2. **Variant**: **Pre-harmonized (default)**는 공개된 래스터를 그대로, 모두 같은
   격자로 불러옵니다. **Mismatch demo**는 좌표계, 범위, 해상도를 일부러 바꾼 사본을
   불러오므로 래스터 일관성 점검을 연습할 수 있습니다
   ([래스터가 맞지 않을 때](data-tab.md#fix-mismatch) 참고).
3. **저장 위치**는 Bradypus와 같습니다.
4. **다운로드**를 누릅니다.

### 3단계. 다운로드 기다리기

![창 안의 다운로드 진행 막대](images/steps/ko/e02_example_downloading_bradypus.png){ .qm-shot width="540" }

1. 진행 막대와 그 아래 줄에 지금 받고 있는 파일이 표시됩니다. 멈추려면 **취소**를
   누릅니다.

다운로드가 끝나면 파일이 저장된 위치를 알려 주는 메시지가 나타납니다.

![예제 데이터셋 준비가 끝났다는 메시지](images/steps/ko/e03_example_done_msg_bradypus.png){ .qm-shot width="444" }

1. **OK**를 누릅니다.

### 4단계. 레이어 확인

![Bradypus 레이어가 올라온 QGIS 주 화면](images/steps/ko/e04_main_bradypus_loaded.png){ .qm-shot width="1440" }

1. 출현 레이어와 환경변수 래스터가 **레이어** 패널에 추가되고 지도에 그려집니다.
   바로 [빠른 시작](quick-start.md)으로 넘어가면 됩니다.

## Bradypus variegatus 데이터셋

*세발가락나무늘보*. Phillips, Anderson & Schapire (2006)와 함께 발표된 표준 Maxent
테스트 데이터셋이며, 이후 거의 모든 Maxent 논문에서 재사용된 데이터입니다.

| 레이어 | 설명 |
|---|---|
| `bradypus.shp` | 남미·중미에 분포한 116개의 출현 지점 |
| `bio1, bio5, bio6, bio7, bio8, bio12, bio16, bio17` | WorldClim 생기후 변수(연속) |
| `biome` | 생물군계 분류(범주형) |

다운로드 직후 레이어들이 QGIS 캔버스에 즉시 표시됩니다:

![중남미 지역에 표시된 Bradypus 출현 지점과 bio17 래스터](images/maps/example-bradypus-loaded.png)

데이터셋은 100 MB 미만으로 작고, Maxent 학습이 보통 30초 이내에 끝나므로 첫 실행과
본 매뉴얼 학습용으로 이상적입니다.

## Ariolimax 데이터셋

*태평양바나나민달팽이*. [elapid](https://github.com/earth-chris/elapid)의 기본 테스트
데이터셋입니다.

!!! note "Pre-harmonized와 Mismatch demo 중 무엇을 고를까요?"

    공개된 Ariolimax 래스터는 이미 좌표계, 범위, 해상도가 모두 같아서
    **래스터 일관성 점검**을 바로 통과합니다. 격자가 맞지 않는 상황을 직접 보고 고쳐
    보고 싶다면 **Variant**에서 **Mismatch demo**를 고르세요.
    [Ariolimax 실전 예제](examples/ariolimax.md)와 같은 상황이 만들어집니다. 변형된
    사본은 `mismatch_demo` 하위 폴더에 저장되며 원본 파일은 바뀌지 않습니다.

## 저장 위치 및 프로젝트 구조

어느 데이터셋을 선택하든 종 이름의 하위 폴더가 만들어집니다:

```text
qmaxent_examples/
├── bradypus/
│   ├── bradypus.shp     (그리고 .dbf, .shx, .prj, .cpg)
│   ├── bio1.tif         …
│   └── biome.tif
└── ariolimax/
    ├── ariolimax.shp
    └── …
```

같은 저장 위치로 다이얼로그를 다시 실행하면 이전 다운로드 파일을 덮어쓰므로,
튜토리얼을 깨끗하게 다시 시작하고 싶을 때 "기본값 리셋" 버튼처럼 안전하게 사용할 수
있습니다.

## 레이어 수동 불러오기

플러그인이 자동으로 레이어를 추가하지만, 직접 다시 불러와야 할 경우 QGIS의
**Browser** 패널에서 `.shp` 와 `.tif` 파일을 **Layers** 패널로 드래그하면 됩니다 —
다른 데이터셋과 동일한 방식입니다.

## 데이터 출처

두 데이터셋 모두 다운로드 시점에 원본 아카이브 URL에서 가져옵니다 — QMaxent는
이를 플러그인 자체에 포함하지 않습니다. 이 방식은 플러그인 패키지를 작게 유지하고
항상 업스트림 표준 버전을 받을 수 있게 합니다. 오프라인 환경이 필요하다면, 인터넷이
연결된 컴퓨터에서 다운로드한 뒤 `qmaxent_examples` 폴더를 오프라인 워크스테이션으로
복사하세요.

두 데이터셋의 사본은 QMaxent 저장소의 [`example_data`](https://github.com/osgeokr/qmaxent/tree/main/example_data) 폴더에도
있습니다(약 15 MB). 원본 파일을 그대로 복사했으며, 출처와 인용 정보를 담은
`README.txt`가 함께 들어 있습니다. 파일을 받아 레이어를 QGIS 프로젝트에 추가하면
됩니다.
