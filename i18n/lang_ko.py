"""Korean (한국어) translation strings for QMaxent.

Keys   = English source strings (as used in the code).
Values = Korean translations shown to the user.

To add or update a translation, edit this file and reload QGIS.

Translation policy:
  * Tab labels, group titles, section headers, and ordinary UI prose
    are translated to natural Korean.
  * Standard Maxent / SDM technical terms (Maxent, Jackknife, AUC,
    cloglog, hinge knots, beta multiplier, K-Fold, LOO, Checkerboard,
    Geographic K-Fold, Buffered LOO, etc.) are kept in English. This
    matches Korean SDM literature where these terms are used verbatim,
    and avoids visual asymmetry across the UI.
"""

STRINGS: dict[str, str] = {

    # ── Plugin / dock titles ────────────────────────────────────────────────
    "QMaxent — Dependencies":               "QMaxent — 의존성",

    # ── Tab labels ──────────────────────────────────────────────────────────
    "① Data":                               "① 데이터",
    "② Parameters":                         "② 매개변수",
    "③ Training":                           "③ 훈련",
    "④ Results":                            "④ 결과",

    # ── Result sub-tabs ─────────────────────────────────────────────────────
    "Response Curves":                      "반응 곡선",
    "Jackknife Importance":                 "Jackknife 중요도",
    "Spatial Projection":                   "공간 예측",

    # ── Data tab ────────────────────────────────────────────────────────────
    "Presence Points Layer":                "출현 포인트 레이어",
    "Select a layer to see point count.":   "레이어를 선택하면 포인트 수가 표시됩니다.",
    "Environmental Rasters":                "환경변수 래스터 레이어",
    "Add from project":                     "프로젝트에서 추가",
    "Remove selected":                      "선택 제거",
    "Background Points":                    "배경점",
    "Sample count:":                        "샘플 수:",
    "No raster layers in project.":         "프로젝트에 래스터 레이어가 없습니다.",
    "No new raster layers to add.":         "새로 추가할 래스터 레이어가 없습니다.",
    "{n} presence points loaded":           "{n}개 포인트 로드됨",
    "Cannot read layer info.":              "레이어 정보를 읽을 수 없습니다.",

    # ── Parameters tab ──────────────────────────────────────────────────────
    "Feature Types":                        "피처 타입",
    "Auto (sample-size based, Phillips et al. 2017)":  "자동 (샘플 수 기반, Phillips et al. 2017)",
    "Manual selection":                     "수동 선택",
    "Regularization":                       "정규화",
    "Regularization multiplier:":           "정규화 배율:",
    "Advanced":                             "고급 설정",
    "Hinge knots:":                         "Hinge knot 수:",
    "Threshold knots:":                     "Threshold knot 수:",
    "Add presences to background":          "출현점을 배경에 추가",
    "Add presences to the background sample so that the fitted "
    "density is consistent over the full study area "
    "(addsamplestobackground; Phillips et al. 2006, 2017).":
        "출현점을 배경 표본에도 포함하여 학습된 밀도가 전체 연구지역에 "
        "일관되도록 합니다 (addsamplestobackground; Phillips et al. 2006, 2017).",

    # Spatial evaluation group
    "Spatial evaluation":                   "공간 평가",
    "Method:":                              "방법:",
    "None (no cross-validation; training AUC only)":  "없음 (교차 검증 안 함, 훈련 AUC만 계산)",
    "Geographic K-Fold (Anderson 2023)":
        "Geographic K-Fold (Anderson 2023)",
    "Fix random seed:":                      "랜덤 시드 고정:",
    "On (default): use the seed value below for every "
    "stochastic operation (CV fold partitions, hold-out "
    "splits, priority-site shuffling, background draws). "
    "Same seed → identical results across runs.\n\n"
    "Off: a fresh random seed is drawn from the OS each "
    "run; results will vary slightly between runs. Useful "
    "when checking robustness to fold assignment without "
    "having to manually try multiple values.":
        "켬 (기본값): 아래 시드 값으로 모든 확률적 연산을 수행 (CV fold "
        "분할, hold-out 분할, 우선 조사대상지 셔플, background 추출). "
        "같은 시드 → 매 실행마다 동일한 결과.\n\n"
        "끔: 매 실행마다 OS에서 새로운 랜덤 시드를 가져와 결과가 약간씩 "
        "달라집니다. 여러 시드를 수동으로 시도하지 않고도 fold 분할에 "
        "대한 결과 robustness를 확인할 때 유용합니다.",
    "The seed value. The Overview sheet of results.xlsx "
    "records this value (or 'random (not fixed)' when the "
    "checkbox is off) so the run is fully reproducible.":
        "시드 값. results.xlsx의 Overview 시트에 이 값 (또는 체크박스가 "
        "꺼져 있으면 'random (not fixed)') 이 기록되어 실행을 완전히 "
        "재현할 수 있습니다.",
    "Random K-Fold (Phillips 2006)":
        "Random K-Fold (Phillips 2006)",
    "Geographic K-Fold (default) provides spatially-independent "
    "folds and is the recommended choice for unbiased "
    "generalization assessment when presence points show any "
    "spatial clustering (Roberts 2017).\n\n"
    "Random K-Fold is included for direct comparability with "
    "maxent.jar / ENMeval / dismo workflows; note it tends to "
    "inflate AUC estimates relative to spatial methods when "
    "presences are spatially autocorrelated.":
        "Geographic K-Fold (기본값)은 공간적으로 독립된 fold를 제공하며, "
        "출현점에 공간 군집이 존재할 때 일반화 성능을 편향 없이 "
        "평가할 수 있는 권장 방법입니다 (Roberts 2017).\n\n"
        "Random K-Fold는 maxent.jar / ENMeval / dismo 워크플로우와의 "
        "직접 비교를 위해 포함되었습니다. 출현점이 공간적으로 "
        "자기상관을 가질 때 공간 분할 방법보다 AUC 추정치가 "
        "부풀려지는 경향이 있습니다.",
    "Checkerboard (single spatial split; Muscarella 2014)":
        "Checkerboard (single spatial split; Muscarella 2014)",
    "Buffered LOO (Pearson 2007; Ploton 2020)":
        "Buffered LOO (Pearson 2007; Ploton 2020)",
    "Folds:":                               "Folds:",
    "Grid size (m):":                       "그리드 크기 (m):",
    "Buffer (m):":                          "버퍼 (m):",
    "Buffer distance in the CRS units of the presence layer "
    "(metres for projected CRSs; degrees for EPSG:4326). "
    "Choose a value appropriate to the species' dispersal range "
    "(Roberts et al. 2017; Ploton et al. 2020).":
        "출현 레이어 CRS 단위로 해석되는 버퍼 거리입니다 "
        "(투영 CRS는 m, EPSG:4326은 degree). 종의 분산 거리에 "
        "따라 적절히 설정하세요 (Roberts et al. 2017; Ploton et al. 2020).",
    "Each variable contributes 2*nknots-2 hinge features "
    "(maxnet default: 50).":
        "변수당 2*nknots-2 개의 hinge feature 가 생성됩니다 "
        "(maxnet 디폴트: 50).",
    "Each variable contributes 2*nknots-2 threshold features "
    "(maxnet default: 50).":
        "변수당 2*nknots-2 개의 threshold feature 가 생성됩니다 "
        "(maxnet 디폴트: 50).",

    "Jackknife variable importance":        "Jackknife 변수 중요도 분석",

    "Output Files":                         "출력 파일",
    "Model (.pkl):":                        "모델 파일 (.pkl):",
    "Results XLSX:":                        "결과 XLSX:",
    "Excel files (*.xlsx)":                 "Excel 파일 (*.xlsx)",
    "Enter path or click [...]":            "경로를 입력하거나 [...] 클릭",

    # ── Training tab ────────────────────────────────────────────────────────
    "Waiting...":                           "대기 중...",
    "Clear log":                            "로그 지우기",

    # ── Results tab — response curves ───────────────────────────────────────
    # Chart labels (axis titles, legend entries, plot titles) are kept
    # English in matplotlib by design — see the comment in
    # main_dock._show_response_curve. Only UI prose around the chart is
    # translated here.
    "Variable:":                            "변수:",
    "Response curve error: {e}":            "반응 곡선 오류: {e}",

    # ── Results tab — jackknife / variable importance ───────────────────────
    # Chart labels here are also kept English by design.
    "No jackknife or CV results.\nEnable them in Parameters and re-run.":
        "Jackknife 또는 CV 결과가 없습니다.\n매개변수 탭에서 활성화하고 다시 실행하세요.",
    "Chart error: {e}":                     "차트 오류: {e}",

    # ── Results tab — projection ─────────────────────────────────────────────
    "Applies the trained model to environmental rasters to produce a habitat suitability map.\n"
    "Uses the layers set in the ① Data tab.":
        "학습된 모델을 환경변수 래스터에 적용하여 서식지 적합도 지도를 생성합니다.\n"
        "래스터는 '① 데이터' 탭에서 설정한 레이어를 사용합니다.",
    "Output transform:":                    "출력 변환:",
    "Output raster:":                       "출력 래스터:",
    "Output path (.tif)":                   "저장 경로 (.tif)",
    "Auto-load result as QGIS layer":       "결과를 QGIS 레이어로 자동 로드",
    "▶  Run Spatial Projection":            "▶  공간 예측 실행",
    "Projecting...":                        "예측 진행 중...",
    "Projection complete.":                 "예측 완료.",
    "No model. Run Maxent first.":          "모델이 없습니다. 먼저 Maxent를 실행하세요.",
    "Enter output raster path.":            "출력 래스터 경로를 입력하세요.",
    "Set raster layers in ① Data tab.":     "① 데이터 탭에서 래스터 레이어를 설정하세요.",
    "Running projection...":                "공간 예측 중...",
    "Done: {path}":                         "완료: {path}",
    "Layer loaded: {name}":                 "레이어 로드: {name}",
    "Layer load error: {e}":                "레이어 로드 오류: {e}",

    # ── Bottom bar ──────────────────────────────────────────────────────────
    "Model not yet trained.":               "모델이 학습되지 않았습니다.",
    "▶  Run Maxent":                        "▶  Maxent 실행",
    "■  Stop":                              "■  중지",
    "Stopping...":                          "취소 중...",
    "Select a presence point layer.":       "출현 포인트 레이어를 선택하세요.",
    "Add environmental raster layers.":     "환경변수 래스터 레이어를 추가하세요.",
    "Data error: {e}":                      "데이터 오류: {e}",
    "All analysis complete.":               "모든 분석이 완료됐습니다.",
    "✓ All analysis complete.":             "✓ 모든 분석이 완료됐습니다.",

    # ── Setup dock ──────────────────────────────────────────────────────────
    "Environment Status":                   "환경 상태",
    "About":                                "안내",

    # About — three-block layout (plugin info, citation, dependencies)
    "<b>QMaxent</b> 0.1.0<br>"
    "Author: Byeong-Hyeok Yu &lt;bhyu@knps.or.kr&gt;<br>"
    "License: MIT — Copyright © 2026 Byeong-Hyeok Yu<br>"
    "Repository: "
    "<a href=\"https://github.com/osgeokr/qmaxent\">"
    "github.com/osgeokr/qmaxent</a>":
        "<b>QMaxent</b> 0.1.0<br>"
        "개발자: Byeong-Hyeok Yu &lt;bhyu@knps.or.kr&gt;<br>"
        "라이선스: MIT — Copyright © 2026 Byeong-Hyeok Yu<br>"
        "저장소: "
        "<a href=\"https://github.com/osgeokr/qmaxent\">"
        "github.com/osgeokr/qmaxent</a>",

    "<b>Dependencies:</b><br>"
    "QMaxent installs its dependencies into an isolated virtual "
    "environment so they do not affect QGIS.<br><br>"
    "One-time setup installs:<br>"
    "&nbsp;&nbsp;• elapid (Maxent engine)<br>"
    "&nbsp;&nbsp;• rasterio, geopandas (spatial I/O)<br>"
    "&nbsp;&nbsp;• scikit-learn, scipy, numpy<br>"
    "&nbsp;&nbsp;• matplotlib (result plots)<br><br>"
    "Approximate size: 300–500 MB":
        "<b>의존성:</b><br>"
        "QMaxent는 의존성 패키지를 격리된 가상 환경에 설치하여 "
        "QGIS 환경에 영향을 주지 않습니다.<br><br>"
        "최초 설치 항목:<br>"
        "&nbsp;&nbsp;• elapid (Maxent 엔진)<br>"
        "&nbsp;&nbsp;• rasterio, geopandas (공간 데이터 입출력)<br>"
        "&nbsp;&nbsp;• scikit-learn, scipy, numpy<br>"
        "&nbsp;&nbsp;• matplotlib (결과 시각화)<br><br>"
        "예상 용량: 300~500 MB",
    "Install location: ":                   "설치 위치: ",
    "Install / Update Dependencies":        "의존성 설치 / 업데이트",
    "Cancel":                               "취소",
    "Remove Environment":                   "환경 삭제",
    "Starting installation...":             "설치 시작 중...",
    "Cancelling...":                        "취소 중...",
    "Checking...":                          "확인 중...",
    "Remove the QMaxent virtual environment?\n"
    "You will need to reinstall dependencies before using QMaxent again.":
        "QMaxent 가상 환경을 삭제하시겠습니까?\n"
        "삭제 후 QMaxent를 사용하려면 의존성을 다시 설치해야 합니다.",

    # ── Categorical variable handling (Data tab) ───────────────────────────
    "[continuous]":                         "[연속형]",
    "[categorical]":                        "[범주형]",
    "Treated as a continuous numeric variable "
    "(e.g. temperature, precipitation, elevation).":
        "연속형 수치 변수로 처리됩니다 "
        "(예: 기온, 강수량, 고도).",
    "Treated as categorical: one-hot encoded inside the model "
    "(e.g. land cover, biome, soil type).":
        "범주형으로 처리됩니다: 모델 내부에서 one-hot 인코딩됩니다 "
        "(예: 토지피복, 생물군계, 토양유형).",

    # ── distance_weights (sample bias correction) ──────────────────────────
    "Down-weight spatially clustered points":
        "공간 밀집 출현점 가중치 감소",
    "Reduce the influence of spatially clustered presences to "
    "correct for sample selection bias (Phillips et al. 2009; "
    "elapid distance_weights). Recommended when occurrence data "
    "come from opportunistic sources (e.g. GBIF, citizen "
    "science). Not recommended for systematic surveys or when "
    "clustering reflects genuine habitat preference.":
        "공간상 밀집된 출현점의 영향을 줄여 "
        "표본 선택 편향을 보정합니다 (Phillips et al. 2009; "
        "elapid distance_weights). GBIF 등 기회주의적 출처의 "
        "데이터에 권장되며, 체계적 조사 데이터나 군집이 "
        "실제 서식지 선호를 반영하는 경우에는 권장되지 않습니다.",

    # ── Raster consistency check & harmonization (explicit workflow) ──────
    "Check Raster Consistency":
        "래스터 일관성 점검",
    "Inspect every raster's CRS, extent, and resolution. "
    "If any of them disagree, a 'Harmonize to Folder…' button "
    "will appear so you can write aligned copies to a folder "
    "of your choosing (Hijmans 2024; SDMSelect Prepare_r_multi).":
        "모든 래스터의 CRS, 범위, 해상도를 점검합니다. 불일치가 "
        "발견되면 'Harmonize to Folder…' 버튼이 나타나며, 사용자가 "
        "지정한 폴더에 정합된 사본을 저장할 수 있습니다 "
        "(Hijmans 2024; SDMSelect Prepare_r_multi).",
    "Harmonize to Folder...":
        "폴더에 정합...",
    "Harmonize Rasters":
        "래스터 정합",
    "Harmonizing rasters...":
        "래스터 정합 중...",
    "Select harmonized output folder":
        "정합 결과 폴더 선택",
    "Could not create output folder:\n{e}":
        "출력 폴더 생성 실패:\n{e}",
    "Harmonization failed:\n{msg}":
        "정합 실패:\n{msg}",
    "Harmonized {n} raster(s) to:\n{path}":
        "{n}개의 래스터를 다음 위치에 정합:\n{path}",
    "Replace the unaligned originals in this project with "
    "the harmonized versions?":
        "프로젝트의 정합되지 않은 원본을 정합된 버전으로 교체하시겠습니까?",
    "Replace":
        "교체",
    "Add as additional layers":
        "추가 레이어로 추가",
    "Status: harmonized files written to {path}; "
    "project not changed.":
        "상태: 정합된 파일이 {path}에 작성됨, 프로젝트는 변경되지 않음.",
    "Status: not checked yet.":
        "상태: 아직 점검되지 않음.",
    "Status: ⚠ Could not read rasters ({err}).":
        "상태: ⚠ 래스터를 읽을 수 없음 ({err}).",
    "Status: ✓ All {n} rasters share grid (CRS: {crs}, "
    "resolution: {res}).":
        "상태: ✓ {n}개의 래스터가 모두 동일 격자를 공유 "
        "(CRS: {crs}, 해상도: {res}).",
    "Status: ⚠ Grid mismatch — {dims} differ across "
    "rasters. Click \"Harmonize to Folder…\" to align.":
        "상태: ⚠ 격자 불일치 — 래스터 간 {dims} 차이. "
        "'Harmonize to Folder…' 클릭으로 정합하세요.",
    "CRS":          "CRS",
    "extent":       "범위",
    "resolution":   "해상도",
    "Environmental rasters do not share a common grid "
    "({mismatches} differ). Run \"Check Raster Consistency\" "
    "and \"Harmonize to Folder…\" in the ① Data tab to "
    "align them before training.":
        "환경 래스터가 공통 격자를 공유하지 않습니다 "
        "({mismatches} 불일치). 학습 전 ① Data 탭에서 "
        "'Check Raster Consistency'와 'Harmonize to Folder…'를 "
        "실행하여 정합하세요.",

    # ── Sample-size validation errors ──────────────────────────────────────
    "No valid presence points after removing rows with NoData "
    "in the environmental rasters. Check that your presence "
    "layer overlaps the raster extents and that the rasters "
    "have valid values at the presence locations.":
        "환경 래스터의 NoData 행을 제거한 후 유효한 출현점이 "
        "없습니다. 출현 레이어가 래스터 범위와 겹치는지, "
        "그리고 출현 위치에 유효한 래스터 값이 있는지 "
        "확인해 주세요.",
    "Too few presence points after removing NoData "
    "(n={n} < 5). Maxent is not reliable below 5 records; "
    "please collect more presences or expand the study area.":
        "NoData 제거 후 출현점이 부족합니다 (n={n} < 5). "
        "Maxent는 5개 미만의 기록에서는 신뢰하기 어렵습니다. "
        "더 많은 출현점을 수집하거나 연구 영역을 확장해 주세요.",
    "Too few background points after removing NoData "
    "(n={n} < 100). Increase the background sample count or "
    "check the raster coverage of the study area.":
        "NoData 제거 후 배경점이 부족합니다 (n={n} < 100). "
        "배경 샘플 수를 늘리거나 연구 영역의 래스터 커버리지를 "
        "확인해 주세요.",

    # ── Pickle security note (model load dialog) ──────────────────────────
    "⚠ Note: .pkl is a Python pickle file. Only load models from "
    "sources you trust (typically models you produced with this "
    "plugin); a malicious .pkl can execute arbitrary code on load.":
        "⚠ 참고: .pkl은 Python pickle 파일입니다. 신뢰할 수 있는 "
        "출처의 모델만 불러오세요 (일반적으로 본 플러그인으로 "
        "만든 모델). 악의적인 .pkl은 불러오는 시점에 임의 "
        "코드를 실행할 수 있습니다.",
    "Reduce the influence of spatially clustered presences to "
    "correct for sample selection bias (Phillips et al. 2009; "
    "elapid distance_weights). Recommended when occurrence data "
    "come from opportunistic sources (e.g. GBIF, citizen "
    "science). Not recommended for systematic surveys or when "
    "clustering reflects genuine habitat preference.":
        "공간상 밀집된 출현점의 영향을 줄여 "
        "표본 선택 편향을 보정합니다 (Phillips et al. 2009; "
        "elapid distance_weights). GBIF 등 기회주의적 출처의 "
        "데이터에 권장되며, 체계적 조사 데이터나 군집이 "
        "실제 서식지 선호를 반영하는 경우에는 권장되지 않습니다.",

    # ── Regularization tooltip ─────────────────────────────────────────────
    "Regularization multiplier (Phillips et al. 2017). "
    "Higher values produce smoother, more conservative response "
    "curves; lower values fit training data more tightly. "
    "Default 1.0 follows the maxent.jar / maxnet standard.":
        "정규화 배율 (Phillips et al. 2017). "
        "값이 클수록 더 매끄럽고 보수적인 반응 곡선이 생성되며, "
        "값이 작을수록 학습 데이터에 더 가깝게 적합됩니다. "
        "기본값 1.0은 maxent.jar / maxnet 표준을 따릅니다.",

    # ── Grid size tooltip ──────────────────────────────────────────────────
    "Checkerboard cell size in the CRS units of the presence "
    "layer (metres for projected CRSs; degrees for EPSG:4326). "
    "Choose a value matching the environmental autocorrelation "
    "length of the study area (Muscarella et al. 2014, ENMeval).":
        "체커보드 격자 크기 — 출현점 레이어 CRS 단위로 입력 "
        "(투영 좌표계는 미터; EPSG:4326은 도). "
        "연구 영역의 환경 자기상관 길이에 맞는 값을 선택하세요 "
        "(Muscarella et al. 2014, ENMeval).",

    # ── Example data dialog ────────────────────────────────────────────────
    "Download Example Dataset...":          "예제 데이터셋 다운로드...",
    "Download Example Dataset":             "예제 데이터셋 다운로드",
    "Downloads a small canonical SDM example dataset directly "
    "from its archival URL. After download, layers will be "
    "added to the current QGIS project automatically.":
        "표준 SDM 예제 데이터셋을 보관 URL에서 직접 다운로드합니다. "
        "다운로드 후 현재 QGIS 프로젝트에 레이어가 자동으로 추가됩니다.",
    "Dataset":                              "데이터셋",
    # Registry labels (passed through tr() in the dialog)
    "Bradypus variegatus (Phillips et al. 2006 standard)":
        "Bradypus variegatus (Phillips et al. 2006 표준)",
    "Brown-throated three-toed sloth occurrence + 8 bioclimatic "
    "rasters and 1 categorical biome raster covering South "
    "America. The canonical Maxent benchmark dataset, mirrored "
    "from the CRAN dismo R package.":
        "갈색목세발가락나무늘보 출현 + 남아메리카 영역의 8개 "
        "생물기후 래스터 및 1개 범주형 biome 래스터. CRAN dismo "
        "R 패키지에서 미러링한 Maxent 표준 검증 데이터.",
    "Ariolimax (banana slug, elapid default)":
        "Ariolimax (바나나민달팽이, elapid 기본 데이터)",
    "Banana slug occurrence + 6 cloud cover, leaf area index, "
    "and surface temperature rasters covering California. "
    "elapid's built-in example dataset.":
        "바나나민달팽이 출현 + 캘리포니아 영역의 운량·엽면적지수·"
        "지표온도 래스터 6개. elapid의 내장 예제 데이터.",
    "Save to:":                             "저장 위치:",
    "Select destination directory":         "저장 디렉토리 선택",
    "Please choose a destination directory.":
        "저장 디렉토리를 선택해 주세요.",
    "Please select a dataset.":
        "데이터셋을 선택해 주세요.",
    "Could not create destination directory:\n{e}":
        "저장 디렉토리를 만들 수 없습니다:\n{e}",
    "Starting download...":                 "다운로드 시작 중...",
    "Cancelling...":                        "취소 중...",
    "Download failed.":                     "다운로드 실패.",
    "Download failed:\n{msg}":              "다운로드 실패:\n{msg}",
    "Files were downloaded but could not be added "
    "to the project automatically:\n{e}":
        "다운로드는 완료되었지만 프로젝트에 자동으로 "
        "추가할 수 없습니다:\n{e}",
    "Example dataset '{ds}' is ready in:\n{path}\n\n"
    "Open the QMaxent Analysis dock to start training.":
        "예제 데이터셋 '{ds}' 준비 완료:\n{path}\n\n"
        "QMaxent 분석 도크를 열어 학습을 시작하세요.",
    "Example dataset '{ds}' is ready in:\n{path}\n\n"
    "Note: rasters were left in their original .grd "
    "format because rasterio is not available yet. "
    "Install plugin dependencies (Plugins → QMaxent → "
    "QMaxent Dependencies) and re-download to get "
    "GeoTIFF copies. The .grd files work fine for now.":
        "예제 데이터셋 '{ds}' 준비 완료:\n{path}\n\n"
        "참고: rasterio 가 설치되지 않아 래스터를 원본 .grd "
        "형식 그대로 유지했습니다. GeoTIFF 사본을 받으려면 "
        "플러그인 의존성을 먼저 설치한 뒤 (Plugins → QMaxent "
        "→ QMaxent Dependencies) 다시 다운로드 하세요. 지금 "
        "상태에서도 .grd 파일은 정상적으로 사용 가능합니다.",
    "Cancel":                               "취소",
    "Download":                             "다운로드",

    # ── Menu actions ────────────────────────────────────────────────────────
    "QMaxent Analysis":                     "QMaxent 분석",
    "QMaxent — Analysis":                   "QMaxent — 분석",
    "QMaxent Dependencies":                 "QMaxent 의존성",

    # ── File dialogs ────────────────────────────────────────────────────────
    "Select save location":                 "저장 위치 선택",
    "Pickle files (*.pkl)":                 "Pickle 파일 (*.pkl)",
    "GeoTIFF (*.tif)":                      "GeoTIFF (*.tif)",
    "Load QMaxent model":                   "QMaxent 모델 불러오기",

    # ── Model load flow ─────────────────────────────────────────────────────
    "Already trained a model?":             "이미 학습한 모델이 있습니까?",
    "Load existing model (.pkl)...":        "기존 모델 불러오기 (.pkl)...",
    "Load a previously saved QMaxent model and project it to "
    "rasters in the current QGIS project. You will be asked to "
    "match the model's variables to your raster layers.":
        "이전에 저장한 QMaxent 모델을 불러와 현재 QGIS 프로젝트의 "
        "래스터에 투영합니다. 모델의 변수와 래스터 레이어를 매칭하는 "
        "단계가 진행됩니다.",
    "Map model variables to rasters":       "모델 변수를 래스터에 매칭",
    "This model was trained on {n} variables in a specific order. "
    "Map each model variable to a QGIS raster layer below. "
    "Order matters: predictions are computed by raster position, "
    "not by name.":
        "이 모델은 {n}개의 변수를 특정 순서로 학습했습니다. "
        "아래에서 각 모델 변수에 QGIS 래스터 레이어를 매칭하세요. "
        "예측은 이름이 아닌 래스터의 *순서*로 계산되므로 순서가 중요합니다.",
    "⚠ No raster layers in the QGIS project. Add the required "
    "rasters to the project first, then load the model again.":
        "⚠ QGIS 프로젝트에 래스터 레이어가 없습니다. 필요한 래스터를 "
        "먼저 프로젝트에 추가한 뒤 모델을 다시 불러오세요.",
    "Model variable:":                      "모델 변수:",
    "QGIS raster layer:":                   "QGIS 래스터 레이어:",
    "— Select layer —":                     "— 레이어 선택 —",
    "Load model":                           "모델 불러오기",
    "⚠ The same raster is assigned to two or more variables. "
    "Each model variable needs its own raster.":
        "⚠ 같은 래스터가 둘 이상의 변수에 매칭되어 있습니다. "
        "각 모델 변수에는 서로 다른 래스터가 필요합니다.",
    "✓ All {n} variables matched.":         "✓ 모든 {n}개 변수가 매칭됐습니다.",
    "Auto-matched {n}/{total} by name. Pick layers for the rest.":
        "이름으로 {n}/{total} 개를 자동 매칭했습니다. 나머지는 직접 선택하세요.",
    "Loaded file has no QMaxent metadata "
    "(was it saved by this plugin?).":
        "선택한 파일에 QMaxent 메타데이터가 없습니다 "
        "(이 플러그인이 저장한 .pkl인지 확인하세요).",
    "Model load cancelled.":                "모델 불러오기를 취소했습니다.",
    "✓ Model loaded: {name}  ({n} variables)":
        "✓ 모델 로드 완료: {name}  (변수 {n}개)",

    # ── Status summary ──────────────────────────────────────────────────────
    "presence={n}":                         "출현={n}",
    "background={n}":                       "배경={n}",
    "train AUC={v:.4f}":                    "훈련 AUC={v:.4f}",
    "CV AUC={v:.4f}":                       "CV AUC={v:.4f}",

    # ── Priority Sites for Survey ──────────────────────────────────────────
    "Priority Sites":                       "우선순위 지점",

    "Threshold method:":                    "임계값 방법:",
    "10th percentile training presence (T10; Pearson 2007)":
        "10퍼센타일 훈련 출현 (T10; Pearson 2007)",
    "Minimum training presence (MTP; Pearson 2007)":
        "최소 훈련 출현 (MTP; Pearson 2007)",
    "Maximum sum of sensitivity + specificity (MaxSSS; Liu 2013)":
        "민감도+특이도 최대합 (MaxSSS; Liu 2013)",
    "Custom value...":                      "사용자 지정...",
    "Computed threshold value:":            "계산된 임계값:",

    "Number of priority sites:":            "우선순위 지점 수:",
    "Min. distance from existing presences (m):":
        "기존 출현지로부터 최소 거리 (m):",
    "Min. distance between sites (m):":     "지점 간 최소 거리 (m):",

    "Add administrative address (province/city/district)":
        "행정주소 자동 추가 (도/시·군·구/읍·면·동)",

    "Output vector layer:":                 "출력 벡터 layer:",
    "Output path (.gpkg)":                  "출력 경로 (.gpkg)",
    "GeoPackage (*.gpkg)":                  "GeoPackage (*.gpkg)",

    "▶  Extract Priority Sites":            "▶  우선순위 지점 추출",
    "Starting...":                          "시작 중...",
    "Failed.":                              "실패.",
    "No sites extracted.":                  "추출된 지점 없음.",

    "Train a model first (or load an existing .pkl).":
        "먼저 모델을 학습하거나 기존 .pkl 을 로드해 주세요.",
    "Run a spatial projection first — the priority sampling needs a "
    "prediction raster.\n\nTip: ④ Results → Spatial Projection → "
    "▶ Run Spatial Projection.":
        "먼저 공간 예측을 수행해 주세요 — 우선순위 추출에는 예측 래스터가 "
        "필요합니다.\n\n안내: ④ Results → Spatial Projection → "
        "▶ Run Spatial Projection.",
    "No presence layer selected (① Data tab).":
        "출현 layer 가 선택되지 않았습니다 (① Data 탭).",
    "Could not read presence layer:\n{e}":
        "출현 layer 를 읽을 수 없습니다:\n{e}",
    "Presence layer has no point features.":
        "출현 layer 에 포인트 피처가 없습니다.",
    "Reverse geocoding {n} sites at 1 request/second will take about "
    "{mins}m {rem}s. The progress bar will report each step.":
        "{n} 지점을 1초당 1건 속도로 역지오코딩하는 데 약 {mins}분 "
        "{rem}초가 소요됩니다. 진행률 바에서 단계별로 확인할 수 있습니다.",
    "Priority site extraction failed:\n{msg}":
        "우선순위 지점 추출 실패:\n{msg}",
    "Could not write output:\n{e}":
        "출력 파일을 쓸 수 없습니다:\n{e}",
    "Could not auto-load priority sites layer: {e}":
        "우선순위 지점 layer 를 자동 로드할 수 없습니다: {e}",
    "✓ {n} priority sites extracted ({m} = {t:.4f}); "
    "{g}/{n} geocoded. Output: {p}":
        "✓ 우선순위 지점 {n}개 추출 완료 ({m} = {t:.4f}); "
        "{g}/{n} 지오코딩 완료. 출력: {p}",
    "⚠ {dropped} of {total} priority sites could not be "
    "written to the GeoPackage and were dropped.":
        "⚠ 우선순위 지점 {total}개 중 {dropped}개를 GeoPackage 에 "
        "쓰지 못해 제외되었습니다.",

    # ── v0.1.0 newly added strings ──────────────────────────────────────────
    # Tab labels that gained the ⑤ prefix or were renamed.
    "⑤ Priority Sites for Survey":          "⑤ 우선 조사대상지",

    # Priority Sites tab — group titles after numeric prefixes were dropped.
    "Sampling strategy":                     "샘플링 전략",
    "Reverse geocoding":                     "역지오코딩",
    "Output":                                "출력",

    "Uses OpenStreetMap Nominatim. No API key required. "
    "Rate limit 1 req/sec is applied automatically. "
    "Results carry © OpenStreetMap contributors attribution.":
        "OpenStreetMap Nominatim 사용. API 키 불필요. "
        "1 req/sec 속도 제한이 자동 적용됩니다. "
        "결과는 © OpenStreetMap 기여자 표시를 포함합니다.",

    # Priority Sites — auto-add label simplified (labels removed).
    "Auto-add to QGIS project":              "QGIS 프로젝트에 자동 추가",

    # Priority Sites — main-thread geocoding progress messages.
    "Reverse geocoding 0/{n}…":              "역지오코딩 0/{n}…",
    "Reverse geocoding {i}/{n}…":            "역지오코딩 {i}/{n}…",
    "Priority site extraction failed:\n{e}\n\n"
    "Full traceback in: View → Panels → Log "
    "Messages → QMaxent.":
        "우선 조사대상지 추출 실패:\n{e}\n\n"
        "자세한 traceback 확인: View → Panels → Log "
        "Messages → QMaxent.",

    # Spatial Projection — chart-export checkbox + tooltip.
    "Save analysis charts as PNG "
    "(Response Curves, ROC, Jackknife, Permutation)":
        "분석 차트를 PNG로 저장 "
        "(Response Curves, ROC, Jackknife, Permutation)",
    "When the projection finishes, four sets of PNG files are "
    "written next to the GeoTIFF:\n"
    "  • <name>_response_curves.png — one image with all response curves\n"
    "  • <name>_roc.png — ROC panel (training + CV folds + mean)\n"
    "  • <name>_jackknife.png — variable-importance bars (jackknife)\n"
    "  • <name>_permutation.png — variable-importance bars (permutation)\n"
    "Uncheck to skip this step entirely.":
        "투영이 완료되면 GeoTIFF 옆에 4개 PNG 파일이 생성됩니다:\n"
        "  • <name>_response_curves.png — 모든 변수의 응답 곡선\n"
        "  • <name>_roc.png — ROC 패널 (학습 + CV folds + 평균)\n"
        "  • <name>_jackknife.png — 변수 중요도 막대 (jackknife)\n"
        "  • <name>_permutation.png — 변수 중요도 막대 (permutation)\n"
        "체크 해제 시 이 단계 전체를 건너뜁니다.",

    # Spatial Projection — pre-flight messages added in v0.1.0.
    "Preflight: sampling rasters...":        "사전 점검: 래스터 샘플링 중...",
    "Projection cancelled.":                 "투영이 취소되었습니다.",
    "Projection blocked: unknown categorical codes":
        "투영 중단: 알 수 없는 categorical 코드",
    "The new rasters contain categorical codes that "
    "the model has never seen during training, so "
    "the projection cannot run — elapid's encoder "
    "would reject them mid-projection.\n\n"
    "{details}\n\n"
    "To proceed: either clip the projection extent "
    "to remove those areas, or retrain the model on "
    "a region that includes those categories.":
        "새 래스터에 모델이 학습 시 본 적 없는 categorical 코드가 "
        "포함되어 있어 투영을 진행할 수 없습니다 — elapid 인코더가 "
        "투영 중 해당 코드를 거부합니다.\n\n"
        "{details}\n\n"
        "해결 방법: 투영 영역을 잘라 해당 지역을 제외하거나, "
        "그 카테고리를 포함하는 영역에서 모델을 다시 학습하세요.",
    "Environmental extrapolation":           "환경 외삽 (extrapolation)",
    "Some new rasters extend beyond the model's "
    "training ranges. Predictions outside the "
    "training envelope are extrapolations and may "
    "be unreliable (Elith et al. 2010).\n\n"
    "{details}\n\n"
    "Proceed with the projection anyway?":
        "일부 래스터가 모델의 학습 범위를 벗어납니다. "
        "학습 영역 밖의 예측은 외삽(extrapolation)이며 "
        "신뢰도가 낮을 수 있습니다 (Elith et al. 2010).\n\n"
        "{details}\n\n"
        "그래도 투영을 진행하시겠습니까?",

    # Harmonize / consistency check status messages.
    "Status: add at least one raster, then click Check Raster Consistency.":
        "상태: 래스터를 1개 이상 추가한 후 Check Raster Consistency 를 클릭하세요.",
    "Status: harmonization cancelled by user.":
        "상태: 사용자가 래스터 조화 작업을 취소했습니다.",

    # Output folder shortcut.
    "Open folder in file manager":           "파일 관리자에서 폴더 열기",

    # ── Discovery vs Validation mode (UI redesign) ──────────────────────────
    "Survey purpose":                        "조사 목적",
    "Discovery mode — find new populations in unsurveyed areas":
        "Discovery 모드 — 미조사 지역에서 새 개체군 발견",
    "Validation mode — test the suitability gradient":
        "Validation 모드 — 적합도 gradient 검증",
    "Sample sites within a high-suitability band, focused on "
    "the most likely habitat for the species.":
        "고적합도 구간 내에서 지점을 샘플링하여 가장 유력한 "
        "서식지에 조사 노력을 집중합니다.",
    "Sample equal numbers from four suitability quartiles "
    "(Rhoden 2017). Will include lower-suitability sites by "
    "design — useful for evaluating the model gradient.":
        "적합도 4분위에서 동일 개수씩 샘플링합니다 "
        "(Rhoden 2017). 의도적으로 낮은 적합도 지점도 포함되며, "
        "모델 gradient 평가에 유용합니다.",
    "Discovery settings":                    "Discovery 설정",
    "Validation settings":                   "Validation 설정",
    "Minimum suitability:":                  "최소 적합도:",
    "Cells with suitability ≥ this value form the candidate "
    "pool. Default is auto-filled to (raster max × 0.9) when "
    "the prediction raster is selected.":
        "적합도가 이 값 이상인 픽셀이 후보 풀이 됩니다. "
        "예측 래스터가 선택되면 (래스터 최댓값 × 0.9) 값으로 "
        "자동 채워집니다.",
    "Sampling order:":                       "샘플링 순서:",
    "Random":                                "무작위",
    "Top-N (highest first)":                 "Top-N (적합도 높은 순)",
    "Threshold value is 0 or unset. Enter a value > 0 "
    "or pick a different threshold method.":
        "임계값이 0이거나 설정되지 않았습니다. 0보다 큰 값을 입력하거나 "
        "다른 임계값 방법을 선택하세요.",

    # ── v0.1.7 additions ────────────────────────────────────────────────────
    # Setup dock — About block (now with Homepage and Manual rows).
    "<b>QMaxent</b> {version}<br>"
    "Author: Byeong-Hyeok Yu &lt;bhyu@knps.or.kr&gt;<br>"
    "License: MIT — Copyright © 2026 Byeong-Hyeok Yu<br>"
    "Homepage: "
    "<a href=\"https://osgeokr.github.io/qmaxent/\">"
    "osgeokr.github.io/qmaxent</a><br>"
    "Manual: "
    "<a href=\"https://osgeokr.github.io/qmaxent/manual/\">"
    "osgeokr.github.io/qmaxent/manual</a><br>"
    "Repository: "
    "<a href=\"https://github.com/osgeokr/qmaxent\">"
    "github.com/osgeokr/qmaxent</a>":
        "<b>QMaxent</b> {version}<br>"
        "저자: Byeong-Hyeok Yu &lt;bhyu@knps.or.kr&gt;<br>"
        "라이선스: MIT — Copyright © 2026 Byeong-Hyeok Yu<br>"
        "홈페이지: "
        "<a href=\"https://osgeokr.github.io/qmaxent/\">"
        "osgeokr.github.io/qmaxent</a><br>"
        "매뉴얼: "
        "<a href=\"https://osgeokr.github.io/qmaxent/manual/\">"
        "osgeokr.github.io/qmaxent/manual</a><br>"
        "저장소: "
        "<a href=\"https://github.com/osgeokr/qmaxent\">"
        "github.com/osgeokr/qmaxent</a>",

    # Setup dock — Dependencies block size update.
    "<b>Dependencies:</b><br>"
    "QMaxent installs its dependencies into an isolated virtual "
    "environment so they do not affect QGIS.<br><br>"
    "One-time setup installs:<br>"
    "&nbsp;&nbsp;• elapid (Maxent engine)<br>"
    "&nbsp;&nbsp;• rasterio, geopandas (spatial I/O)<br>"
    "&nbsp;&nbsp;• scikit-learn, scipy, numpy<br>"
    "&nbsp;&nbsp;• matplotlib (result plots)<br><br>"
    "Approximate size: ~590 MB":
        "<b>의존성:</b><br>"
        "QMaxent는 QGIS에 영향을 주지 않도록 의존성을 격리된 가상 환경에 "
        "설치합니다.<br><br>"
        "최초 설치 항목:<br>"
        "&nbsp;&nbsp;• elapid (Maxent 엔진)<br>"
        "&nbsp;&nbsp;• rasterio, geopandas (공간 I/O)<br>"
        "&nbsp;&nbsp;• scikit-learn, scipy, numpy<br>"
        "&nbsp;&nbsp;• matplotlib (결과 플롯)<br><br>"
        "디스크 용량: 약 590 MB",

    # Data tab — Export for external Maxent radio labels (renamed).
    "Samples + Raster (samples CSV + .asc layers)":
        "Samples + Raster (샘플 CSV + .asc 레이어)",
    "SWD (CSV pair, extracted values)":
        "SWD (CSV 쌍, 추출된 값)",
    "A short samples CSV (Species, Longitude, Latitude only) "
    "plus one ESRI ASCII Grid (.asc) per environmental raster "
    "in a layers/ folder. maxent.jar fits the model AND "
    "produces a projection raster in a single run. Larger "
    "files but ready to use without additional arguments.":
        "샘플 CSV (Species, Longitude, Latitude만)와 환경 래스터마다 "
        "하나씩 .asc (ESRI ASCII Grid) 파일을 layers/ 폴더에 저장합니다. "
        "maxent.jar가 한 번의 실행으로 모델 적합과 예측 래스터 생성을 "
        "모두 수행합니다. 파일 크기는 더 크지만 추가 인자 없이 바로 "
        "사용할 수 있습니다.",
    "Two CSVs (presence.csv + background.csv) with environmental "
    "values pre-extracted at each point. maxent.jar will use "
    "these exact points and values — no further sampling. "
    "Smaller files, but produces no projection raster unless "
    "you also pass projectionlayers= to maxent.jar.":
        "두 개의 CSV 파일 (presence.csv + background.csv)에 각 지점의 "
        "환경 값이 미리 추출되어 있습니다. maxent.jar는 이 정확한 지점과 "
        "값을 그대로 사용하며 추가 샘플링을 하지 않습니다. 파일 크기는 "
        "작지만 projectionlayers= 인자를 별도로 전달하지 않으면 예측 "
        "래스터가 생성되지 않습니다.",
    "Exporting samples + raster...":
        "Samples + Raster 내보내는 중...",
    "Samples + Raster export complete.\n\n"
    "Samples:   {s} ({np} rows)\n"
    "Layers:    {l} ({nr} ASCII grids)\n"
    "README:    {r}\n\n"
    "The README contains the maxent.jar command "
    "line to run on this dataset.":
        "Samples + Raster 내보내기 완료.\n\n"
        "샘플:    {s} ({np}개 행)\n"
        "레이어:  {l} ({nr}개 ASCII grid)\n"
        "README:  {r}\n\n"
        "이 데이터셋에 대한 maxent.jar 실행 명령은 "
        "README 파일에 있습니다.",

    # Results sub-tabs — explanation strips (Response Curves, Jackknife,
    # Permutation Importance). All three render as plain prose in
    # v0.1.7 — the grey background on the Permutation strip was
    # removed for visual consistency.
    "Marginal effect: how the predicted suitability changes "
    "across the range of one variable while the other "
    "variables are held at their training-set means. The "
    "Y-axis reads as relative habitat suitability (cloglog: "
    "0 to 1, saturates near 1 in the most suitable region). "
    "The shaded green band marks the variable's training-data "
    "range — curves outside it are extrapolation and should "
    "be interpreted with caution; the dotted line marks the "
    "training-data mean. Useful for inspecting whether the "
    "model has learned ecologically plausible shapes (e.g. "
    "unimodal temperature optima) and for spotting variables "
    "whose curves are nearly flat — candidates for removal "
    "in a parsimonious model.":
        "주변 효과 (marginal effect): 다른 변수를 학습 데이터의 평균값으로 "
        "고정한 상태에서, 하나의 변수가 변할 때 예측 적합도가 어떻게 "
        "변하는지 보여줍니다. Y축은 상대적 서식지 적합도 (cloglog: 0~1, "
        "가장 적합한 영역에서 1에 가까이 포화)입니다. 녹색 음영은 해당 "
        "변수의 학습 데이터 범위로, 음영 밖의 곡선은 외삽 영역이므로 "
        "해석에 주의해야 합니다. 점선은 학습 데이터의 평균값입니다. "
        "모델이 생태학적으로 타당한 형태 (예: 단봉형 온도 최적값)를 "
        "학습했는지 확인하고, 곡선이 거의 평평한 변수 — 단순화된 "
        "모델에서 제거 후보 — 를 발견하는 데 유용합니다.",
    "Retraining test: trains the model with each variable "
    "alone (per-variable AUC) and with each variable removed "
    "(per-variable drop in AUC), then averages across CV "
    "folds. Robust to correlated variables — unlike "
    "permutation importance — because each retrained model "
    "sees a different variable set. Slower (requires N×2 "
    "retrains) but the standard Maxent variable-importance "
    "diagnostic since Phillips et al. (2006).":
        "재학습 테스트: 각 변수 단독 (변수별 AUC), 각 변수 제외 "
        "(변수별 AUC 감소량)으로 모델을 학습한 뒤 CV fold에 걸쳐 "
        "평균을 냅니다. 매 재학습마다 변수 집합이 달라지므로 — "
        "permutation importance와 달리 — 상관관계가 있는 변수들에도 "
        "강건합니다. 학습 횟수가 많아 (N×2회 재학습) 느리지만 Phillips "
        "et al. (2006) 이래 Maxent의 표준 변수 중요도 진단입니다.",
    # The Permutation Importance description was already translated in
    # earlier versions; no change needed here.

    # Run Maxent finish — training log auto-save status line.
    "Training log saved: {path}":
        "훈련 로그 저장됨: {path}",
    "(Could not save training log: {e})":
        "(훈련 로그 저장 실패: {e})",

    # Save log as... tooltip (auto-save location wording change).
    "Save the entire training log to a text file at a "
    "location of your choice. This is independent of the "
    "auto-save next to model.pkl.":
        "전체 훈련 로그를 원하는 위치에 텍스트 파일로 저장합니다. "
        "model.pkl 옆에 자동 저장되는 파일과는 별개입니다.",
}
