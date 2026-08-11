# 03-Appendix. DP2 백데이터 — RAG 검색 전략 실측 평가

> 본 문서는 `03_DP2_Permission_Aware_Dataset_Strategy.md`의 §7 Trade-off 평가에서 `[Expected]`로 표기했던
> 설계 예상 KPI를, SWE-bench PoC 페이지(`/swebench`, `src/frontend/swebench.html`)에 이미 인덱싱된 실제 데이터에 대해
> **직접 측정한 결과**로 대체/보강하기 위한 발표용 어펜딕스입니다.

---

## 0. 측정 방법

| 항목 | 내용 |
|---|---|
| 데이터 소스 | SWE-bench Lite/Full 이슈 **866건** (전량), `swebench_issues` 테이블 |
| RAG 인덱싱 방식 | Legacy(AST 청킹) / BasicRAG / RaptorRAG / ROIRAG — 4종 |
| 검색 전략 | Flat(통합 DB 무조건 검색) / PostFilter(Option B) / Routed(Option A) — 3종 |
| 조합 수 | 4 × 3 = 12 |
| Top-K / Prefetch-K | 3 / 10 (`backend/rag/swebench_rag_engines.py`와 동일) |
| 측정 로직 | `backend/routers/swebench.py`의 `/api/swebench/evaluate` 및 `swebench_rag_engines.retrieve()`와 동일한
컬렉션 선택·필터·라우팅 로직을 재현하여 직접 측정 (임베딩은 이슈당 1회만 계산 후 재사용) |
| DP2 대응 관계 | SWE-bench의 **repo + version 경계**를 DP2의 "프로젝트/부서 권한 + 요청 Source Version 범위"의
대리 지표(proxy)로 사용. Flat = 권한/버전 통제가 전혀 없는 baseline(DP2 후보 아님, 비교용) |
| 원자료 | `src/data/dp2_backdata.json` |
| 생성일 | 2026-08-11 |

> 주의: 보안성·근거추적성 지표는 SWE-bench의 repo/version 경계를 권한·버전 스코프의 대리 지표로 사용한 PoC 실측치입니다.
> 실제 조직의 RBAC/부서·프로젝트 권한 체계 적용 시 재측정이 필요합니다.

---

## 1. Appendix A. RAGAS 스타일 검색 품질 평가

SWE-bench 이슈 866건 전체에 대해 12개 조합의 Top-3 검색 결과를 다음 정의로 채점했습니다.

- **Hit@3**: 정답 파일이 Top-3 청크에 하나라도 포함되면 1, 아니면 0
- **Context Precision@3**: Top-3 중 정답 파일을 포함한 청크 비율
- **Context Recall@3**: 전체 정답 파일 중 Top-3에서 회수된 비율
- **MRR**: 정답이 처음 등장한 순위의 역수 평균

### 1.1 차트

![Hit@3 / Context Precision@3](assets/dp2_appendix/appendix_a_hit_precision.png)

![Context Recall@3 / MRR](assets/dp2_appendix/appendix_a_recall_mrr.png)

### 1.2 조합별 실측 결과 (n=866)

| RAG 엔진 | 전략 | Hit@3 | Precision@3 | Recall@3 | MRR | 평균 지연(ms) |
|---|---|---:|---:|---:|---:|---:|
| Legacy | Flat | 68.9% | 37.3% | 62.6% | 0.603 | 58.3 |
| Legacy | PostFilter | 78.3% | 53.1% | 72.5% | 0.724 | 59.3 |
| Legacy | Routed | 83.6% | 43.3% | 77.6% | 0.754 | 18.9 |
| BasicRAG | Flat | 60.5% | 36.6% | 55.7% | 0.535 | 52.3 |
| BasicRAG | PostFilter | 68.1% | 49.7% | 63.8% | 0.627 | 55.2 |
| BasicRAG | Routed | 71.3% | 44.5% | 67.0% | 0.646 | 19.3 |
| RaptorRAG | Flat | 65.5% | 44.2% | 59.1% | 0.579 | 123.1 |
| RaptorRAG | PostFilter | 77.7% | 59.4% | 71.1% | 0.720 | 129.2 |
| RaptorRAG | Routed | 81.8% | 55.0% | 75.2% | 0.744 | 26.4 |
| ROI-RAG | Flat | 64.0% | 36.1% | 60.9% | 0.557 | 54.1 |
| ROI-RAG | PostFilter | 73.8% | 51.4% | 71.8% | 0.679 | 58.6 |
| ROI-RAG | Routed | 80.0% | 44.8% | 78.0% | 0.713 | 18.4 |

### 1.3 전략별 평균 (4개 RAG 엔진 평균)

| 전략 | Hit@3 | Precision@3 | Recall@3 | MRR |
|---|---:|---:|---:|---:|
| Flat | 64.7% | 38.6% | 59.6% | 0.568 |
| PostFilter | 74.5% | 53.4% | 69.8% | 0.688 |
| Routed | 79.2% | 46.9% | 74.4% | 0.714 |

**읽는 법**: 권한/버전 스코프를 좁히는 두 전략(PostFilter, Routed) 모두 Flat 대비 Hit@3·Recall·MRR이 뚜렷하게 향상됩니다.
Routed가 Hit@3·Recall·MRR에서 최고치를 기록하지만, Precision@3는 PostFilter가 근소하게 앞섭니다 — Routed는 파티션이
작을수록(이슈당 코드 조각 수가 적을수록) Top-3 채움에 관련 없는 항목이 섞일 수 있기 때문입니다. 즉 **검색 정확도만 보면
Option A/B 중 어느 한쪽이 압도적이지 않으며**, DP2 §7.2가 "검색 정확도 외 QA 축으로 결정해야 한다"고 판단한 근거와 부합합니다.

---

## 2. Appendix B. RAG 검색 전략 선택 지표 — 보안성 / 응답성 / 확장성·운용성 / 근거추적성

DP2 §7.0에서 정의한 4개 QA 축을 동일 실측 데이터로 계산했습니다. 모든 값은 4개 RAG 엔진의 평균입니다.

### 2.1 지표 정의

| QA 축 | 지표명 | 정의 | 방향 |
|---|---|---|---|
| ① 보안성 | Unauthorized Scope Exposure Rate | 반환된 Top-3 청크 중 질의의 repo/version과 불일치하는 청크 비율 | 낮을수록 좋음 |
| ② 응답성 | 평균/P95 검색 지연시간 | 임베딩 계산을 제외한, 전략 자체의 검색 처리 시간 | 낮을수록 좋음 |
| ③ 확장성/운용성 | Index Operation Count | 4개 RAG 엔진 전체가 운영해야 하는 ChromaDB 컬렉션(인덱스) 총 개수 | 상황에 따라 다름 (§2.3 참고) |
| ③ 보조 | Post-filter Drop Rate | PostFilter가 Prefetch한 결과 중 스코프 불일치로 폐기하는 비율 | 낮을수록 좋음 |
| ④ 근거추적성 | Verifiable Citation Coverage | 반환된 청크 중 올바른 스코프로 검증 가능한(=repo/version이 일치하는) 근거 비율 | 높을수록 좋음 |
| ④ 보조 | Scope Declared Rate | 검색 시점에 권한/버전 스코프 조건을 명시적으로 선언했는지 여부 | 높을수록 좋음 |

### 2.2 차트

![① 보안성 / ② 응답성](assets/dp2_appendix/appendix_b_security_latency.png)

![③ 확장성·운용성 / ④ 근거추적성](assets/dp2_appendix/appendix_b_indexops_citation.png)

### 2.3 전략별 실측 결과

| 전략 | DP2 대응 | 노출률(①) | 평균 지연(②) | P95 지연(②) | Index 운영 수(③) | Post-filter Drop(③) | 인용 검증률(④) | 스코프 사전선언(④) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Flat | 해당 없음 (baseline) | **62.2%** | 72.0ms | 104.3ms | 4 | — | 37.8% | 0% |
| PostFilter | Option B | 0.25% | 75.6ms | 117.8ms | 4 | **72.1%** | 99.75% | 100% |
| Routed | **Option A (선택안)** | **0.21%** | **20.8ms** | 79.6ms | **276** | — | **99.80%** | 100% |

※ Index Operation Count: Flat/PostFilter는 엔진당 통합 컬렉션 1개(4개 엔진 × 1 = 4), Routed는 엔진당 실제 생성된
repo×version 파티션 69개(4개 엔진 × 69 = 276).

### 2.4 해석 — DP2 Decision과의 정합성

1. **보안성 · 응답성**: Routed(Option A)가 노출률 0.21%, 평균 지연 20.8ms로 두 축 모두 실측 최우수입니다. DP2 §8.1이
   Option A를 우선 선택한 근거("보안성과 버전 정합성이 가장 명확하다")를 실측치로 뒷받침합니다. 특히 응답 속도는
   Routed가 Flat/PostFilter 대비 약 3.5배 빠른데, 이는 파티션 컬렉션의 검색 대상이 작아 ANN 탐색 비용이 줄기 때문입니다.

2. **확장성/운용성**: 그러나 Routed는 파티션 컬렉션을 276개(엔진당 69개, repo×version 조합 수에 비례) 운영해야 하며
   Flat/PostFilter(4개)보다 운영 부담이 큽니다. 이는 DP2 §7.2 "Source Scope가 늘어날수록 운영 복잡도가 증가한다"는
   판단과 정확히 일치하며, DP2 §9.1의 Hybrid Source Scope Strategy(공통/저위험 데이터는 통합 Index, 민감 데이터만
   분리)가 필요한 근거를 실측으로 보여줍니다.

3. **근거추적성**: PostFilter도 사후 필터링만으로 노출률 0.25%, 인용 검증률 99.75%까지 낮출 수 있었습니다. 다만 이
   과정에서 Prefetch(Top-10)된 결과의 **72.1%를 폐기(Drop)** 하는 비효율이 함께 발생했는데, 이는 DP2 §6.4가 지적한
   "Top-K 결과가 제거 대상 데이터로 많이 채워지면 필터링 후 결과가 부족할 수 있다"는 리스크가 실측으로 확인된
   사례입니다.

4. **Flat(무통제 baseline)**: 노출률 62.2%, 근거추적성 37.8%로 두 지표 모두 최하위입니다. 이는 "검색 정확도가 높아도
   권한/버전 통제가 없으면 시스템을 사용할 수 없다"는 DP2 §1.4의 문제의식을 정량적으로 뒷받침하는 근거로 사용할 수
   있습니다.

### 2.5 DP2 §7.1 KPI 표 대비 실측치 요약

| KPI ([Expected] 원문 기준) | DP2 문서상 예상치 (Option A / Option B) | 본 PoC 실측치 (Routed / PostFilter) |
|---|---|---|
| Unauthorized Context Exposure Rate | 0% 목표 / 0% 목표(중간 결과 통제 필요) | **0.21% / 0.25%** — 둘 다 목표치에 근접, 실사용에는 별도 로그 통제 필요 |
| P95 Permission/Version Overhead | 낮음~중간 / 중간~높음 | **79.6ms / 117.8ms** — 방향성 일치(Routed가 더 낮음) |
| Index Operation Count | 권한/버전 Scope 수에 비례 / 1개 중심 | **276개 / 4개** — 방향성 일치, Routed의 절대 규모가 예상보다 큼 |
| Post-filter Drop Rate | 해당 없음 / 30% 이하 목표 | **— / 72.1%** — 목표치(30%) 대비 실측치가 크게 초과, PostFilter 운영 시 Prefetch-K 확대 등 보완 필요 |
| Citation Trace Coverage | 90% 이상 / 95% 이상 | **99.80% / 99.75%** — 두 옵션 모두 목표치 상회 |

**핵심 시사점**: 방향성(보안성·응답성은 Routed 우위, 운영 단순성은 PostFilter/Flat 우위)은 DP2 문서의 정성적 예상과
일치합니다. 다만 **Post-filter Drop Rate가 설계 목표(30% 이하)보다 실측치(72.1%)가 훨씬 높다는 점**은, PostFilter를
보조 전략으로 채택할 경우 Prefetch-K를 더 크게 잡거나 재검색 로직을 강화해야 함을 시사합니다.

---

## 3. 원자료

전체 수치는 `src/data/dp2_backdata.json`에 조합별 상세 지표(hit/precision/recall/mrr, 지연, 노출/인용 청크 수 등)로
저장되어 있습니다. 재측정이 필요할 경우 `backend/rag/swebench_rag_engines.py`의 `ALL_ENGINES` × `ALL_STRATEGIES`
조합에 대해 동일한 방법론(§0)을 재적용하면 됩니다.
