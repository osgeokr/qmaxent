# 참고문헌

QMaxent의 기본값, 평가 절차, 우선조사 후보지 워크플로는 종 분포 모델링(SDM)
문헌에 근거합니다. 본 장은 플러그인이 인용하는 문헌을 주제별로 묶고, 각
참고문헌이 QMaxent 안에서 어떻게 활용되는지 한 줄 주석을 덧붙였습니다.
플러그인 소스 코드의 관련 모듈(`workers/maxent_worker.py`,
`bridge/elapid_bridge.py`, `bridge/priority_sites.py`,
`core/venv_manager.py`)에 인라인 인용이 포함되어 있습니다.

## Maxent — 핵심 방법론

Phillips, S. J., Anderson, R. P., & Schapire, R. E. (2006). Maximum entropy
modeling of species geographic distributions. *Ecological Modelling*, 190(3–4),
231–259. <https://doi.org/10.1016/j.ecolmodel.2005.03.026>
:   Maxent 원논문. Linear·Quadratic·Hinge·Product·Threshold 피처(LQPHT)와
    정규화 프레임워크, 그리고 출력값을 상대적 출현 확률로 해석하는 근거를
    정의합니다.

Phillips, S. J., & Dudík, M. (2008). Modeling of species distributions with
Maxent: new extensions and a comprehensive evaluation. *Ecography*, 31(2),
161–175. <https://doi.org/10.1111/j.0906-7590.2008.5203.x>
:   QMaxent의 "Regularization" 파라미터가 노출하는 정규화 배수 방식을
    도입했고, 다른 SDM 방법과 Maxent를 비교 평가합니다.

Phillips, S. J., Anderson, R. P., Dudík, M., Schapire, R. E., & Blair, M. E.
(2017). Opening the black box: an open-source release of Maxent.
*Ecography*, 40(7), 887–893. <https://doi.org/10.1111/ecog.03049>
:   현재 표준 보고 형식으로 자리잡은 cloglog 출력 변환을 정의한 릴리스.
    QMaxent는 Spatial Projection 하위 탭에서 cloglog를 기본값으로 사용합니다.

Elith, J., Phillips, S. J., Hastie, T., Dudík, M., Chee, Y. E., & Yates, C. J.
(2011). A statistical explanation of MaxEnt for ecologists. *Diversity and
Distributions*, 17(1), 43–57.
<https://doi.org/10.1111/j.1472-4642.2010.00725.x>
:   Maxent를 페널티가 부과된 로그선형(Poisson) 모델로 재유도하여, elapid
    (그리고 QMaxent)가 내부에서 사용하는 maxnet 구현의 이론적 근거를
    제공합니다.

Merow, C., Smith, M. J., & Silander, J. A. (2013). A practical guide to
MaxEnt for modeling species' distributions: what it does, and why inputs
and settings matter. *Ecography*, 36(10), 1058–1069.
<https://doi.org/10.1111/j.1600-0587.2013.07872.x>
:   표준 실무 가이드. QMaxent가 권장하는 배경 표본 추출, 피처 복잡도,
    정규화 설정은 본 논문을 그대로 따릅니다.

Fithian, W., & Hastie, T. (2013). Finite-sample equivalence in statistical
models for presence-only data. *Annals of Applied Statistics*, 7(4),
1917–1939. <https://doi.org/10.1214/13-AOAS667>
:   Maxent와 충분한 배경 표본 위에서의 가중 Poisson 회귀가 등가임을
    증명 — maxnet 알고리즘의 근거입니다.

## 표본 선택 편향과 배경 지점

Phillips, S. J., Dudík, M., Elith, J., Graham, C. H., Lehmann, A., Leathwick,
J., & Ferrier, S. (2009). Sample selection bias and presence-only
distribution models: implications for background and pseudo-absence data.
*Ecological Applications*, 19(1), 181–197.
<https://doi.org/10.1890/07-2153.1>
:   QMaxent의 "Down-weight spatially clustered points" 옵션과, 출현 지점과
    동일 영역에서 배경 지점을 추출하는 권장 관행의 근거.

## 공간 교차검증

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J.,
Guillera-Arroita, G., et al. (2017). Cross-validation strategies for data
with temporal, spatial, hierarchical, or phylogenetic structure.
*Ecography*, 40(8), 913–929. <https://doi.org/10.1111/ecog.02881>
:   QMaxent의 Geographic K-Fold와 Buffered LOO 방식의 개념적 토대.
    데이터에 공간 자기상관이 있을 때 무작위 K-fold가 성능을 과대평가함을
    보입니다.

Muscarella, R., Galante, P. J., Soley-Guardia, M., Boria, R. A., Kass, J. M.,
Uriarte, M., & Anderson, R. P. (2014). ENMeval: an R package for conducting
spatially independent evaluations and estimating optimal model complexity
for Maxent ecological niche models. *Methods in Ecology and Evolution*,
5(11), 1198–1205. <https://doi.org/10.1111/2041-210X.12261>
:   QMaxent의 Checkerboard 분할 방식의 출처.

Valavi, R., Elith, J., Lahoz-Monfort, J. J., & Guillera-Arroita, G. (2019).
blockCV: an R package for generating spatially or environmentally separated
folds for k-fold cross-validation of species distribution models. *Methods
in Ecology and Evolution*, 10(2), 225–232.
<https://doi.org/10.1111/2041-210X.13107>
:   동반 소프트웨어 참고문헌. QMaxent의 공간 블록 분할은 Python에서
    동일한 논리를 따릅니다.

Ploton, P., Mortier, F., Réjou-Méchain, M., Barbier, N., Picard, N.,
Rossi, V., et al. (2020). Spatial validation reveals poor predictive
performance of large-scale ecological mapping models. *Nature
Communications*, 11, 4540. <https://doi.org/10.1038/s41467-020-18321-y>
:   무작위 CV와 공간 CV 성능 추정치 사이의 격차에 대한 실증 근거.
    QMaxent가 Buffered LOO 폴드에서 통합 AUC를 기본값으로 사용하는 근거.

## 모델 복잡도, 정규화, 평가

Radosavljevic, A., & Anderson, R. P. (2014). Making better Maxent models of
species distributions: complexity, overfitting and evaluation. *Journal of
Biogeography*, 41(4), 629–643. <https://doi.org/10.1111/jbi.12227>
:   표본 크기 기반 피처 자동 선택 규칙과 정규화 배수 기본값 1.0의 근거.

Lobo, J. M., Jiménez-Valverde, A., & Real, R. (2008). AUC: a misleading
measure of the performance of predictive distribution models. *Global
Ecology and Biogeography*, 17(2), 145–151.
<https://doi.org/10.1111/j.1466-8238.2007.00358.x>
:   주의 환기를 위한 인용. QMaxent는 AUC 단독 의존을 막기 위해 반응곡선과
    Jackknife 변수 중요도를 함께 보고합니다.

## 임계값 방법과 소표본 평가

Liu, C., White, M., & Newell, G. (2013). Selecting thresholds for the
prediction of species occurrence with presence-only data. *Journal of
Biogeography*, 40(4), 778–789. <https://doi.org/10.1111/jbi.12058>
:   QMaxent Priority Sites 탭의 MaxSSS(Maximum Sum of Sensitivity and
    Specificity) 임계값과 MTP·T10 비교 논의의 출처.

Pearson, R. G., Raxworthy, C. J., Nakamura, M., & Peterson, A. T. (2007).
Predicting species distributions from small numbers of occurrence records:
a test case using cryptic geckos in Madagascar. *Journal of Biogeography*,
34(1), 102–117. <https://doi.org/10.1111/j.1365-2699.2006.01594.x>
:   QMaxent가 소표본(≤ 25 출현 지점) 데이터셋에 구현한 Buffered
    Leave-One-Out(LOO) 워크플로의 정의.

## SDM 출력 기반 조사 계획

Williams, J. N., Seo, C., Thorne, J., Nelson, J. K., Erwin, S., O'Brien, J. M.,
& Schwartz, M. W. (2009). Using species distribution models to predict new
occurrences for rare plants. *Diversity and Distributions*, 15(4),
565–576. <https://doi.org/10.1111/j.1472-4642.2009.00567.x>
:   QMaxent Priority Sites 탭의 Discovery 모드(모델이 예측한 고적합도
    지역으로 신규 현장 조사를 안내) 개념적 근거.

Rhoden, C. M., Peterman, W. E., & Taylor, C. A. (2017). Maxent-directed
field surveys identify new populations of narrowly endemic habitat
specialists. *PeerJ*, 5, e3632. <https://doi.org/10.7717/peerj.3632>
:   QMaxent Validation 모드(예측 검증을 위해 적합도 사분위에 걸친 층화
    표본 추출)의 출처.

## 소프트웨어 의존성

Anderson, C. B. (2023). elapid: species distribution modeling tools for
Python. *Journal of Open Source Software*, 8(84), 4930.
<https://doi.org/10.21105/joss.04930>
:   QMaxent가 통합한 Python 라이브러리. maxnet 엔진, 공간 CV 분할기,
    투영 유틸리티를 제공합니다.

Friedman, J., Hastie, T., & Tibshirani, R. (2010). Regularization paths for
generalized linear models via coordinate descent. *Journal of Statistical
Software*, 33(1), 1–22. <https://doi.org/10.18637/jss.v033.i01>
:   maxnet R 패키지(그리고 elapid의 Python 포팅)가 기반으로 하는 glmnet
    알고리즘.

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel,
O., et al. (2011). Scikit-learn: machine learning in Python. *Journal of
Machine Learning Research*, 12, 2825–2830.
<https://www.jmlr.org/papers/v12/pedregosa11a.html>
:   QMaxent의 평가 파이프라인이 사용하는 교차검증 프리미티브, ROC 계산,
    측정 지표 유틸리티 제공.

Harris, C. R., Millman, K. J., van der Walt, S. J., Gommers, R., Virtanen,
P., Cournapeau, D., et al. (2020). Array programming with NumPy. *Nature*,
585(7825), 357–362. <https://doi.org/10.1038/s41586-020-2649-2>
:   위 모든 라이브러리의 수치 연산 토대.

## 플러그인 아키텍처

Wu, Q. (2026). GeoAI: A Python package for integrating artificial
intelligence with geospatial data analysis and visualization. *Journal of
Open Source Software*, 11(118), 9605.
<https://doi.org/10.21105/joss.09605>
:   QMaxent의 의존성 설치 워크플로(`core/venv_manager.py`)를 차용한 QGIS
    플러그인. 특히 Windows QGIS에서 `sys.executable`이 `qgis-bin.exe`를
    가리키는 문제를 우회하기 위한 `_get_qgis_python()` 패턴과, `pip install`
    실행 중 서브프로세스 파이프 처리 방식을 그대로 따랐습니다.

## 관련 SDM 플랫폼

Kass, J. M., Vilela, B., Aiello-Lammens, M. E., Muscarella, R., Merow, C., &
Anderson, R. P. (2018). Wallace: a flexible platform for reproducible
modeling of species niches and distributions built for community
expansion. *Methods in Ecology and Evolution*, 9(4), 1151–1156.
<https://doi.org/10.1111/2041-210X.12945>
:   Introduction에서 관련 플랫폼으로 인용. Wallace는 Shiny 기반 SDM GUI이며,
    QMaxent는 같은 영역을 다루되 독립 웹 앱이 아닌 QGIS 내부에서 동작합니다.

## 사례 연구 참고문헌

Lee, S., Cho, M., Yu, B.-H., Lee, S., Lee, S., Wolfe, J. D., & Oh, H.-S.
(2025). Breeding habitat prediction and nest-site characteristics of the
fairy pitta (*Pitta nympha*) in Geoje-si, South Korea: insights from a
species distribution model.
:   *Pitta nympha* 실전 예제에서 재현하는 발표 연구. 원래 고전 Java MaxEnt로
    분석되었으며, 예제 챕터에서 QMaxent로 동일 분석을 수행하고 결과의
    일치·차이를 논의합니다.
