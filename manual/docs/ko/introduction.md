# 소개

QMaxent는 [maxnet](https://cran.r-project.org/package=maxnet) /
[elapid](https://github.com/earth-chris/elapid) Maxent 알고리즘을 QGIS의
**도크 기반 GUI**로 감싸서, 이미 QGIS 환경에 익숙한 생태학자, 보전 관련
실무자, 학생들이 별도의 R/Python 스크립트나 Java MaxEnt 명령행 없이
종 분포 모델(SDM)을 구축할 수 있도록 합니다.

## QMaxent란

**Plugins → QMaxent → QMaxent Analysis** 메뉴 하나로 다섯 개 탭의
도크가 열리고, 사용자는 다음 워크플로를 차례로 진행합니다:

1. **① Data** — 출현 점 레이어와 환경 래스터 등록
2. **② Parameters** — 피처 클래스, 정규화, 공간 CV 방식 선택
3. **③ Training** — 진행 막대를 보면서 모델 학습
4. **④ Results** — 반응 곡선, Jackknife 중요도, 공간 투영 확인
5. **⑤ Priority Sites for Survey** — 적합도 지도에서 현장 조사 후보지 추출

각 탭은 독립적인 챕터에서 자세히 다룹니다.

## 누구를 위한 도구인가

| 사용자 | QMaxent의 가치 |
|---|---|
| **현장 생태학자** | 코드 작성 없이 적합도 지도 구축; 즉시 활용 가능한 우선조사 후보지 |
| **대학원생** | 발표 논문 재현; 다중 시트 XLSX 보충표로 학위논문 부록 작성 |
| **보전 단체** | QGIS 안에서 멸종위기종의 잠재 서식지 신속 평가 |
| **방법론 연구자** | Java MaxEnt와의 정량적 비교; ENMeval/Wallace 결과 검증 |

## Java MaxEnt 및 R 패키지와의 차이

| 항목 | Java MaxEnt | maxnet/dismo (R) | **QMaxent** |
|---|---|---|---|
| GUI | 자체 Swing GUI | 없음 (스크립트) | QGIS 도크 |
| 백엔드 | Java MaxEnt | maxnet (glmnet) | elapid (Python) |
| 공간 CV | 75/25 분할 | 사용자 구현 | Geographic K-Fold (기본값) |
| 자동 hyperparameter rule | 없음 | 사용자 정의 | maxnet 자동 규칙 |
| 범주형 외삽 처리 | 무작위 클래스 | 명시적 처리 필요 | 자동 NoData 마스킹 |
| 출력 보충표 | TXT/HTML | 사용자 구현 | 다중 시트 XLSX |
| 우선조사 후보지 추출 | 없음 | 없음 | 내장 |

## 방법론 계보

QMaxent의 기본값은 다음 핵심 SDM 문헌의 권고를 따릅니다:

- 알고리즘 자체: [Phillips, Anderson & Schapire 2006](references.md);
  [Phillips et al. 2017](references.md);
  [Fithian & Hastie 2013](references.md)
- 피처 선택 규칙: [Phillips & Dudík 2008](references.md);
  [Radosavljevic & Anderson 2014](references.md)
- 공간 CV 기본값: [Roberts et al. 2017](references.md)
- 출력 cloglog 변환: [Phillips et al. 2017](references.md)
- 우선조사 후보지 설계:
  [Williams et al. 2009](references.md);
  [Rhoden, Peterman & Taylor 2017](references.md)
- 보고 표준: [Araújo et al. 2019](references.md);
  [Elith et al. 2011](references.md);
  [Merow, Smith & Silander 2013](references.md)

전체 인용 목록은 [참고문헌](references.md)에 정리되어 있습니다.
방법론적 의사결정의 *왜*에 대한 해설은
[방법론 해설](methodological-background.md)을 참고하세요.

## 다음 단계

설치를 진행하려면 [설치](installation.md) 챕터로 이동하세요. 30 초 이내에
첫 실행 가능한 모델을 만들고 싶다면 [빠른 시작](quick-start.md)부터
시작해도 좋습니다.
