# 실전 예제

QMaxent의 서로 다른 측면과 서로 다른 실제 워크플로를 강조하는 세 가지
종단간 사례 연구.

## 예제 개요

| 예제 | 종 | 주안점 | 배우는 내용 |
|---|---|---|---|
| [Bradypus variegatus](bradypus.md) | 세발가락나무늘보 | 번들 데이터셋; 9 변수, 116 출현 | QMaxent 전체 기능 투어 — 원 Maxent 논문과 동일한 워크플로 |
| [Ariolimax](ariolimax.md) | 태평양 바나나 슬러그 | 정렬되지 않은 래스터 | **Check Raster Consistency** + **Harmonize to Folder…** preflight |
| [Pitta nympha](pitta-nympha.md) | 팔색조 | [Lee et al. 2025](../references.md) 재현 | 발표된 Java-MaxEnt 연구를 QMaxent에서 재현하고 결과 비교하는 방법 |

세 예제는 **순서대로 완료**하도록 설계되었습니다: Bradypus는
워크플로를 소개하고, Ariolimax는 래스터 조화를 추가하며, Pitta
nympha는 학술 비교가 포함된 발표 품질의 재현을 추가합니다.

## 따라하는 방법

모든 예제는 [설치](../installation.md)를 완료했고 **QMaxent
environment ready** 배너가 녹색이라고 가정합니다.

세 데이터셋 중 둘은 플러그인에 번들로 제공되어 — **Plugins → QMaxent
→ Download Example Dataset → \<species name\>**으로 로드합니다:

- **Bradypus variegatus** — 9 변수, 116 출현, 플러그인에 번들. Phillips
  et al. 2006 데이터셋과 비트 단위 동일.
- **Ariolimax** — 6 변수, 3,732 출현, elapid 라이브러리 기본 태평양
  연안 슬러그 데이터셋. **공식 다운로드는 사전 정렬됨** — Ariolimax
  실전 예제를 정확히 따르려면 예제 헤더에 설명된 의도적으로 비동기화된
  버전을 사용해야 합니다.
- **Pitta nympha** — 10 변수, 한국 거제시 47개 둥지. Lee et al. 2025
  보충 아카이브에서 가져옴; 라이선스 고려로 플러그인에 번들되지 않음.

## 재현성

세 예제는 모두 **고정된 무작위 시드**로 실행되므로 스크린샷의 숫자가
사용자 컴퓨터에서 보일 것과 정확히 일치합니다. AUC에서 0.001을 초과하는
차이는 어떤 설정이 예제 지침과 다르다는 신호입니다; **② Parameters**
탭 구성을 한 줄씩 확인하세요.
