# ① 데이터 탭

데이터 탭은 **어떤 종**을 모델링하고 **어떤 환경 변수**를 사용할지 QMaxent에
알려주는 곳입니다. 세 가지 컨트롤이 핵심입니다 — 출현 지점 선택기, **Check
Raster Consistency** 동작이 포함된 환경 변수 래스터 목록, 배경 지점 표본 수.

![초기 빈 상태의 Data 탭](images/ui/dock-1-data-empty.png)

## 출현 지점 레이어

첫 번째 드롭다운은 현재 QGIS 프로젝트의 모든 벡터 점 레이어를 보여줍니다. 종 출현
기록이 담긴 레이어를 선택하세요. 선택 즉시 QMaxent가 형상을 읽어 점 개수를
드롭다운 아래에 표시합니다(예: `116 presence points loaded`).

이 레이어 준비 시 팁:

- **형상**은 `Point` 또는 `MultiPoint` 여야 합니다.
- **좌표계**는 QGIS가 지원하는 어떤 CRS도 가능 — QMaxent가 내부적으로 래스터의
    좌표계로 재투영합니다.
- **중복 점 제거**는 사용자가 직접 — 동일 좌표에 여러 기록이 있다면 미리 정리하세요
    (`벡터 → 지오프로세싱 도구 → 중복 형상 제거`).
- **공간 필터링**: 부분 영역만 모델링하려면 미리 해당 영역으로 잘라낸 레이어를
    추가하세요.

## 환경 변수 래스터 목록

아래 큰 목록 패널이 모델이 사용할 모든 래스터를 담습니다. 버튼으로 관리:

- **Add from project** – 현재 프로젝트의 모든 래스터 레이어를 한 번에 추가.
    bioclim 변수가 가득한 폴더를 드래그한 직후 가장 빠른 방법.
- **Remove selected** – 선택된 항목 제거.
- **▲ / ▼** – 래스터 순서 변경. 적합한 모델에는 영향이 없지만, 결과 XLSX의 컬럼
    순서에 영향을 주므로 의미있게 정렬해 두면 좋습니다.

각 래스터 우측에 `[continuous]` / `[categorical]` 토글이 있습니다.
**범주형 래스터는 반드시 categorical로 표시**하세요 — QMaxent는 이를 one-hot
인코딩하며, 범주형을 연속으로 처리하면 무의미한 결과를 만듭니다. 토지피복 분류
인덱스, 토양 유형, 생물군계 ID 같은 변수가 여기 해당됩니다.

Bradypus 예제 데이터를 불러오고 `biome` 을 categorical로 표시한 뒤:

![Bradypus 출현 레이어(116점)와 7개 환경 래스터, biome이 categorical, Check Raster Consistency가 모든 래스터의 그리드 일치 확인](images/ui/dock-1-data-with-bradypus.png)

## Check Raster Consistency

QMaxent의 *조용한 오류 차단기* 입니다. 모든 래스터가 **동일한 좌표계, 범위, 해상도**
를 공유하는지 검증합니다. Maxent는 정합되지 않은 래스터에서도 오류를 던지지 않습니다 —
오프셋 셀에서 공변량을 표본화해 단지 잘못된 예측을 만들 뿐입니다. 래스터 목록을 변경할
때마다 이 검사를 실행하시길 강력히 권장합니다.

세 가지 가능한 결과:

| 상태 | 의미 |
|---|---|
| ✓ All rasters share grid (CRS, resolution) | 진행 가능 |
| ⚠ CRS mismatch 또는 extent mismatch | **Harmonize Rasters** 사용 (다음 절) |
| ⚠ Resolution mismatch | **Harmonize Rasters** 로 공통 그리드로 리샘플 |

검사가 통과하면 결과·공유 좌표계·해상도가 버튼 아래 상태 줄에 기록됩니다 —
프로젝트 저장·재로드 후에도 유지되는 정합성 체크입니다.

검사가 실패했을 때의 흐름은 [Ariolimax 실전 예제](examples/ariolimax.md) 참고.

## Harmonize Rasters

**Check Raster Consistency** 가 불일치를 발견하면 같은 영역에서 열리는 Harmonize
Rasters 다이얼로그가 모든 래스터를 사용자가 선택한 공통 그리드로 재투영·자르기·
리샘플링합니다. 내부적으로
[`gdalwarp`](https://gdal.org/programs/gdalwarp.html)를 사용 — QGIS의 **Warp
(Reproject)** 알고리즘이 감싸는 동일한 도구입니다. 입력 파일은 변경되지 않으며,
조화화된 사본이 프로젝트 옆에 기록됩니다.

## 배경 지점

Maxent는 실제 부재 지점을 필요로 하지 않습니다 — 대신 **배경 지점** 으로 *가용*
환경 공간을 표본화합니다. 기본값 10,000은 Phillips et al. (2017)의 권장값이며 거의
모든 연구에서 잘 작동합니다.

기본값 변경이 필요한 경우:

- **작은 연구 영역** (단일 유역, 섬): 10,000개의 고유 셀이 없을 수 있음 —
    5,000 또는 `n_cells × 0.5` 같은 작은 값 사용.
- **대륙 또는 전 지구 모델링**: 10,000으로 유지해도 무방 — 기본 maxnet 알고리즘이
    잘 확장됩니다.
- **출현 지점에 강한 공간 표본 편향**: [파라미터 탭](parameters-tab.md)의
    *Down-weight spatially clustered points* 옵션과 함께 사용.

상태바는 표본 후 실제 개수를 보고합니다(예: `background=10,104`); 요청보다 약간 작은
숫자는 NaN 셀 제외에 의한 것입니다.

## 기존 모델 (.pkl) 불러오기

우측 상단의 **Load existing model (.pkl)…** 버튼은 저장된 모델을 열고 각 모델 변수를
현재 QGIS 래스터에 매핑하는 절차를 안내합니다. 다른 영역으로 재투영, 두 연구 비교,
협업자와의 모델 공유에 유용합니다. 전체 절차와 [pickle 파일 보안 안내](saving-models.md)
는 [모델 저장 및 재사용](saving-models.md) 참고.

## 도크가 기억하는 것

레이어 선택과 categorical/continuous 토글은 QGIS 프로젝트(`.qgs` / `.qgz`)와 함께
저장됩니다. 프로젝트를 다시 열면 이 탭의 모든
## 도크가 기억하는 것

레이어 선택과 categorical/continuous 토글은 QGIS 프로젝트(`.qgs` / `.qgz`)와
함께 저장됩니다. 프로젝트를 다시 열면 이 탭의 모든 설정이 복원되어 구성을 다시
만들지 않고도 모델링을 이어갈 수 있습니다.
