# GODAD 지오코딩 편향: 제작자 입장과 관련 문헌

조사일 2026-11-01

## 1. GODAD 제작자의 검증 (Bomprezzi et al., "Wedded to Prosperity?", CEPR DP 18878, 부록 B)
**지오코딩 방식**
- CRS 사업 제목과 설명에서 자연어처리(NER)로 지명을 추출한다.
- 그 지명을 Google Maps 지오코더(GoogleV3)로 좌표화한다.

**정확도**
- 사람이 코딩한 1,000건과 비교한 지명 추출 정확도는 89%다. 1,000건 중 145건에 오류가 있었다.
- 무작위 10,000건 점검에서 위치가 빠지거나 틀린 사업은 5%였다.
- 주된 오류 원인: 같은 지명이 여러 나라에 있는 경우(예: Córdoba), 문장 구조가 엉성한 경우, 영어가 아닌 텍스트.

**대표성** (B.5: 지오코딩된 사업과 안 된 사업의 비교)
- 지오코딩 확률은 다음에 따라 달라진다.
  - 최근 연도일수록 높다.
  - 분야: 인프라·보건·농업은 높고, 금융·식량 지원은 낮다.
  - 공여국과 재원 유형(무상 > 지분투자 등).
  - 설명 텍스트가 충실할수록 높다.
- 사업 규모(약정액)와는 무관하다.
- 제작자 결론: "결과는 모든 원조가 아니라 **지오코딩된 사업 집합**에 대한 것으로 해석해야 한다."

**우리 질문과 직결되는 제작자의 서술**
- 지오코더는 "거의 항상 **행정구역(주·시·군)이나 도시**를 찾아낸다."
- "아주 작은 지역(동네, 작은 마을)을 모든 나라에서 일관되게 정확히 잡아주는 지오코딩 도구는 없다."
- "대부분의 설명은 **큰 도시나 지방자치단체**를 언급해 군 단위로 지오코딩이 가능하다."
- → **작은 마을만 언급한 사업은 빠지거나 더 거친 단위(주)로 잡힐 가능성이 크다는 뜻이다.** 외진 곳 사업일수록 위치가 덜 잡힐 수 있다는 우리의 우려는 제작자의 서술과 일치한다. 즉 근거 없는 가정이 아니다.
- 다만 제작자는 이 편향이 **접근성이나 외딴 정도와 얼마나 관련되는지는 직접 검정하지 않았다.**

**제작자의 대응 방식**
- 군(ADM2) 고정효과와 주(ADM1)×연도 고정효과를 넣는다.
- 공여국별로 따로 회귀한다.
- 외부 자료와 비교한다: 우간다 AIMS(224건) 대비 GODAD는 4,929건이다. 비교한 AIMS 11개국 모두에서 GODAD의 포괄 범위가 더 넓다.

## 2. 관련 문헌
- **Runfola, Marty, Goodman, LeFew, BenYishay (2017), geoSIMEX**: 위치 부정확성을 무시하면 중국 원조가 식생에 부정적 영향을 준 것으로 나왔다. 부정확성을 반영하자 효과가 사라졌다. → **공간 부정확성은 결론을 바꿀 수 있다.**
- **Marty et al. (2018)**: 텍스트에서 추출한 위치에 남는 공간 불확실성은 잠재적 편향의 원천이다.
- **BenYishay et al. (AidData WP 64)**: 거친 위치 정보의 측정오차는 "무해하지도, 무작위적이지도 않을 것"이라고 지적한다.
- **Opršal (2025, Development Policy Review)**: 원조 위치 정보 공개는 "파편적이고 때로 오해를 부르는" 상태다.
- **AidData 연구들**: 분야 구성에 따라 정밀도가 달라진다(교통 사업이 더 정밀하게 잡힘).

## 3. 우리 연구에 주는 시사점
1. **편향의 방향**: 제작자 서술상 작은 마을 사업이 덜 잡힌다. 따라서 편향이 있다면 접근성 효과를 **부풀리는 방향**이다. 우리의 점검 방향은 맞다.
2. **편향의 크기**: 알려져 있지 않다. 그래서 범위로 보고하는 게 맞다.
   - 현실적 시나리오: 주 단위 사업을 주 안에 균등 분할하면 +120%, p = 0.006.
   - 극단 시나리오: 전부 가장 외진 군에 배정하면 +61%, p = 0.12.
3. **우리가 더할 수 있는 점검** (제작자가 하지 않은 것)
   - ① 정밀도와 접근성의 관계: 이미 했다. 주 단위 순위 상관 +0.30.
   - ② **설명이 충실한 사업만**으로 다시 추정한다. 오류가 텍스트 부실에서 오므로, 충실한 사업에서도 효과가 유지되면 편향 설명이 약해진다.
   - ③ **인구 비례 분할 시나리오**: geoSIMEX의 발상처럼 주 단위 사업을 인구 비율로 나눈다. 극단 가정보다 현실적인 중간값이다.
   - ④ 공여국·분야별 일관성: 이미 했다. 유럽·미국과 중국에서 같은 방향이다.
4. **서술 방식**: "제작자가 인정한 한계(작은 지역의 누락)가 우리 추정을 부풀릴 수 있으며, 우리는 이를 범위와 여러 점검으로 보고한다."

## 출처
- [GODAD 홈](https://godad.uni-goettingen.de/home/)
- [GODAD 코드북](https://godad.uni-goettingen.de/uploads/GODAD_codebook.pdf)
- [Bomprezzi et al. (2025) Wedded to Prosperity? (GODAD 검증 부록 포함)](https://godad.uni-goettingen.de/uploads/Spouses.pdf)
- [AidData geoSIMEX](https://www.aiddata.org/publications/geosimex-a-generalized-approach-to-modeling-spatial-imprecision)
- [Marty et al., Chinese aid in Burundi and Rwanda under spatial imprecision](https://www.aiddata.org/publications/assessing-the-causal-impact-of-chinese-aid-on-vegetative-land-cover-in-burundi-and-rwanda-under-conditions-of-spatial-imprecision)
- [AidData WP 64 Connective Financing](https://docs.aiddata.org/ad4/pdfs/WPS64_Connective_Financing_Chinese_Infrastructure_Projects_and_the_Diffusion_of_Economic_Activity_in_Developing_Countries.pdf)
- [AidData Geocoding Methodology 2017](https://docs.aiddata.org/ad4/files/geocoding-methodology-updated-2017-06.pdf)
- [Opršal (2025) Publish where you fund! (IDEAS)](https://ideas.repec.org/a/bla/devpol/v43y2025i6ne70043.html)
