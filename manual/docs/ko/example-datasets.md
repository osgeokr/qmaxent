# 예제 데이터

QMaxent는 두 가지 표준 SDM 예제 데이터셋을 한 번의 클릭으로 다운로드할 수 있는
기능을 **플러그인 → QMaxent → Download Example Dataset** 메뉴에서 제공합니다.
지난 20년간 Maxent 문헌에서 사용된 동일한 데이터셋이라, 결과를 발표 벤치마크와
직접 비교할 수 있습니다.

## 다이얼로그 열기

**플러그인 → QMaxent → Download Example Dataset** 을 선택하세요.

![Download Example Dataset 다이얼로그 — Bradypus 선택 상태](images/ui/dialog-download-example-dataset.png)

두 데이터셋 중 하나를 라디오 버튼으로 선택하고, 저장 위치(기본은 홈 디렉터리 아래
`qmaxent_examples` 폴더)를 정한 뒤 **Download** 를 클릭합니다. 플러그인이 데이터를
가져와 압축을 풀고, **현재 QGIS 프로젝트에 자동으로 레이어를 추가**해 즉시 모델링을
시작할 수 있게 합니다.

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

*태평양바나나민달팽이*. [elapid](https://github.com/earth-chris/elapid) 의 기본
테스트 데이터셋으로, 다른 형태의 도전 과제를 보여줍니다 — 환경 변수 래스터들이
**공통 좌표계, 범위, 해상도를 공유하지 않습니다**. 이 데이터셋은 의도적으로 QMaxent의
**Check Raster Consistency** 와 **Harmonize Rasters** 도구를 보여주기 위해 사용됩니다 —
[Ariolimax 실전 예제](examples/ariolimax.md)를 참고하세요.

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
