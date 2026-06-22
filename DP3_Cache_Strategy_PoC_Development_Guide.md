# DP3 Cache Strategy PoC Development Guide

## 1. 문서 목적

이 문서는 최신 DP3 문서(`doc/04_DP3_Response_Cache_Strategy.md`) 기준으로 **Verified Answer Cache**와 **Incremental Context Cache** PoC를 준비하기 위한 개발 가이드이다.

현재 우선 구현 범위는 **Answer Cache PoC와 Context Cache PoC의 기본 골격**이다. 먼저 두 후보가 같은 LongBench 기반 EU 데이터셋을 공유하도록 만들고, 이후 테스트 결과를 보며 입력 데이터와 validation 기준을 조정한다.

DP3 PoC는 DP1의 선택안인 **SPRAG 기반 Evidence Unit RAG**를 검색 기반으로 사용한다. 현재 `src` 구현에서는 이 구조가 `roi_rag.py` / `ROIRAG` 이름으로 구현되어 있으므로, 아래 문서에서는 `ROI-RAG`라고 부른다.

PoC에서 확인할 QA와 최소 측정 지표는 다음으로 제한한다.

| QA | 핵심 질문 | PoC 지표 |
|---|---|---|
| 속도 | 반복 질문에서 응답 시간이 줄어드는가 | P95 Latency, LLM Call Count |
| 기능 적합성 | cache reuse가 답변 품질과 근거 정합성을 해치지 않는가 | Wrong Answer Reuse Rate, Wrong Context Reuse Rate, Required Answer Type Match |
| 테스트 용이성 | cache 판단과 실패 원인을 설명할 수 있는가 | Decision Reason Coverage |

RAGAS Faithfulness는 2차 평가 지표로 둔다. 1차 PoC에서는 cache routing, metadata validation, latency/LLM call 절감이 실제로 동작하는지 먼저 확인한다.

---

## 2. 비교 대상

### 2.1 Option A: Verified Answer Cache

Verified Answer Cache는 미리 정의한 `answerable_question_pool`을 통과한 질문에 한해 과거 검증 답변을 재사용한다.

초기 PoC에서는 `answerable_question_pool` routing을 **embedding similarity 기반 vector-only gate**로 구성한다. BM25 lexical matching은 near-miss 질문에서 wrong answer reuse를 줄이기 위한 2차 보강 항목으로 둔다.

```text
Query
-> Answerable Question Pool Routing
   -> Embedding Route Lookup
   -> Vector Similarity Gate Validation
-> Answer Cache Lookup
-> Metadata Validation
-> Hit: Cached Answer 반환
-> Miss: ROI-RAG Retrieval + LLM
```

특징:

- cache hit 시 ROI-RAG 검색과 LLM 생성을 모두 생략하므로 가장 빠르다.
- 질문 범위는 FAQ, 요약, 핵심 내용, 정의, 주요 사실 확인처럼 답변 형태가 안정적인 질문으로 제한한다.
- 주요 리스크는 비슷하지만 다른 질문에 기존 답변을 재사용하는 wrong answer reuse다.
- 1차 PoC에서는 구현 복잡도를 낮추기 위해 embedding similarity score가 threshold 이상일 때만 Answer Cache lookup을 수행한다.
- vector-only gate는 표현이 다른 유사 질문을 잡는 데 유리하지만, “요약”과 “예외/한계”처럼 의도가 다른 near-miss 질문에 취약할 수 있다. 이 위험은 `Wrong Answer Reuse Rate`로 측정하고, 필요하면 2차 PoC에서 BM25 gate를 추가한다.

### 2.2 Option B: Incremental Context Cache

Incremental Context Cache는 최종 답변을 재사용하지 않고, ROI-RAG가 만든 Context Pack을 재사용한다. cache hit 이후에도 LLM은 현재 질문에 맞게 답변을 새로 생성한다.

```text
Query
-> Context Cache Lookup
-> EU-level Metadata Validation
-> All Valid: Cached Context Pack + LLM
-> Partial Invalid: Delta Retrieval + Context Rebuild + LLM
-> Miss / Many Invalid: ROI-RAG Retrieval + LLM
```

특징:

- 최종 답변이 아니라 근거 묶음을 재사용한다.
- 일부 EU만 invalid하면 전체를 버리지 않고 delta retrieval로 보강할 수 있다.
- EU 단위 metadata가 남기 때문에 원인 분석과 재현에 유리하다.

Context Cache는 1차 PoC에서 `All Valid -> Cached Context Pack + LLM`, `Invalid -> ROI-RAG fallback`까지 구현하고, delta retrieval은 후속 보강으로 둔다.

---

## 3. 공통 PoC 데이터 준비

### 3.1 LongBench 공통 사전 준비

A안과 B안 PoC는 같은 LongBench 데이터를 사용한다. 원본 LongBench의 `input`, `context`, `answers`를 기반으로 PoC용 EU와 질문 세트를 만든다.

권한 분할:

```text
- LongBench 예제를 절반으로 나눈다.
- 앞 절반은 scope=A, 뒤 절반은 scope=B로 둔다.
- 사용자 요청은 A 또는 B 권한 중 하나를 가진 것으로 시뮬레이션한다.
```

버전 생성:

```text
- V1: 원본 EU 전체
- V2: V1 중 2/3를 복사하되 일부 fingerprint를 변경해 수정 버전으로 만든다.
- V3: V2 중 1/2를 다시 복사하되 일부 fingerprint를 변경한다.
```

이 구조를 통해 같은 EU가 여러 버전에 존재하거나, 특정 버전에서 fingerprint가 달라지는 상황을 만든다. A안과 B안은 모두 이 metadata로 stale answer 또는 stale context reuse를 거부할 수 있어야 한다.

### 3.2 공통 EU 데이터 구조

PoC에서는 물리적으로 DB를 권한/버전별로 나누지 않고 SQLite metadata로 구분한다. A안과 B안은 같은 `context_units`를 원천 EU 데이터로 사용한다.

```text
context_units
- logical_eu_id      -- 버전을 넘어 같은 근거임을 식별하는 ID
- version            -- V1 / V2 / V3
- fingerprint
- scope              -- A or B
- text
- source_example_id
```

`logical_eu_id`는 버전이 달라도 같은 근거임을 식별하기 위한 ID다. validation은 cache에 저장된 EU가 `requested_version`에서도 같은 `logical_eu_id`와 같은 `fingerprint`로 존재하는지 확인하는 방식으로 수행한다.

공통 EU validation 함수는 A안과 B안 모두에서 재사용한다.

```text
validate_eu(logical_eu_id, cached_fingerprint, user_scope, requested_version)

1. requested_version의 같은 logical_eu_id row를 찾는다.
2. 없으면 invalid
3. scope가 user_scope와 맞지 않으면 invalid
4. fingerprint가 cached_fingerprint와 다르면 invalid
5. 모두 통과하면 valid
```

이 방식은 다음 상황을 모두 같은 규칙으로 처리한다.

```text
- cache가 V3 근거인데 사용자가 V2를 요청하면 invalid
- cache가 V2 근거이고 사용자가 V2를 요청했으며 fingerprint가 같으면 valid
- cache가 V1 근거이고 사용자가 V2를 요청했더라도 V2의 같은 logical_eu_id fingerprint가 같으면 valid
- cache가 V1 근거이고 V2에서 같은 logical_eu_id가 수정되었거나 삭제되었으면 invalid
```

### 3.3 공통 질문 세트

A안과 B안은 같은 질문 세트를 사용한다. 질문은 LongBench 원본 질문과 context를 보고 생성하며, 예제별로 `same`, `paraphrase`, `near_miss` 유형을 붙인다.

```text
Set A. 완전 동일 질문
- Q1. V1에서 이 문서의 핵심 내용을 요약해줘.
- Q2. V1에서 이 문서의 핵심 내용을 요약해줘.

Set B. 표현만 다른 유사 질문
- Q1. V2에서 이 문서의 핵심 내용을 요약해줘.
- Q2. V2에서 주요 내용을 간단히 정리해줘.

Set C. 비슷하지만 의미가 다른 질문
- Q1. V2에서 이 문서의 핵심 내용을 요약해줘.
- Q2. V2에서 이 문서에서 주의해야 할 예외나 한계를 알려줘.
```

해석:

```text
- Set A는 cache hit 속도 개선을 본다.
- Set B는 의미적으로 안전한 cross-query reuse를 본다.
- Set C는 wrong answer reuse 또는 wrong context reuse 위험을 본다.
```

Set C는 vector-only gate의 한계를 확인하기 위한 테스트로 사용한다. embedding은 유사하게 보이지만 “요약”과 “예외/한계”처럼 답변 의도가 다른 질문에서 cache가 잘못 재사용되는지 확인한다. 이 결과가 높게 나오면 2차 PoC에서 BM25 또는 rule-based lexical gate를 추가한다.

---

## 4. Option A. Verified Answer Cache PoC 설계

### 4.1 최소 DB 구조

Answer Cache는 공통 `context_units` 위에 답변 단위 cache table을 추가한다.

```text
answerable_question_pool
- route_id
- question_text
- embedding_id
- route_type         -- summarize / definition / fact_check 등 선택적 분류
```

초기 PoC에서는 `answerable_question_pool.question_text`를 embedding하여 route 후보를 찾는다. `route_type`은 필수는 아니지만, 로그와 분석에서 어떤 질문 유형이 cache hit 또는 fallback됐는지 보기 위해 둘 수 있다.

```text
answer_cache_entries
- cache_id
- query_text
- query_embedding_id
- answer_text
- scope
- cache_version      -- 답변 생성에 사용한 EU 중 가장 높은 버전
- created_at
- ttl_seconds

answer_cache_sources
- cache_id
- logical_eu_id
- eu_version
- fingerprint
```

핵심은 `answer_cache_sources`다. Answer Cache entry가 어떤 EU 버전과 fingerprint에 근거했는지 남겨야 requested version에서 재사용 가능한지 검증할 수 있다.

BM25 index는 1차 PoC 범위에서 제외한다. 필요하면 2차 PoC에서 `answerable_question_pool.question_text`를 기반으로 실행 시 메모리 index를 구성한다.

### 4.2 처리 흐름

```text
1. 사용자 query에서 requested_version 추출
   예: "V2에서 이 내용 요약해줘" -> requested_version=V2
2. user_scope는 A/B 중 하나로 고정 또는 seeded random 생성
3. query embedding 생성
4. answerable_question_pool에서 vector-only routing 수행
   4-1. embedding similarity 기반 top route 검색
   4-2. embedding score가 threshold 이상인지 확인
   4-3. threshold 미달이면 Answer Cache lookup 없이 fallback
5. routing을 통과한 경우 Answer Cache에서 유사 query cache 후보 검색
6. Metadata Validation 수행
7. 통과하면 cached answer 반환
8. 실패하면 ROI-RAG로 검색하고 LLM 답변 생성 후 cache 저장
```

Answerable Question Pool Routing 규칙:

```text
- embedding similarity score가 threshold 이상이어야 한다.
- threshold를 만족하지 않으면 Answer Cache lookup을 수행하지 않고 ROI-RAG + LLM으로 fallback한다.
- threshold는 초기에는 보수적으로 높게 잡고, Set C의 wrong answer reuse 결과를 보며 조정한다.
```

이 routing은 Answer Cache의 wrong answer reuse를 줄이기 위한 최소 gate다. vector-only 방식은 구현이 단순하고 기존 embedding infrastructure를 재사용할 수 있지만, 질문 의도를 바꾸는 핵심 단어 차이를 충분히 구분하지 못할 수 있다. 따라서 1차 PoC의 목적은 vector-only cache의 성능 이득과 over-hit 위험을 먼저 계측하는 것이다.

Validation 규칙:

```text
- user_scope == cache.scope 이어야 한다.
- answer_cache_sources의 각 logical_eu_id에 대해 공통 validate_eu를 수행한다.
- requested_version의 같은 logical_eu_id가 없으면 실패한다.
- requested_version의 fingerprint가 cache에 저장된 fingerprint와 다르면 실패한다.
- 하나라도 validation에 실패하면 stale answer로 보고 ROI-RAG + LLM으로 fallback한다.
```

중요한 점은 `requested_version > cache_version`이라고 해서 무조건 통과시키지 않는 것이다. 더 높은 버전에서 EU가 수정됐을 수 있으므로 `answer_cache_sources`의 `logical_eu_id`와 `fingerprint`를 기준으로 requested version의 현재 EU를 재검증한다.

### 4.3 측정 로그

Answer Cache PoC에서 최소한 다음 필드는 남긴다.

```json
{
  "query_id": "B2",
  "user_scope": "A",
  "requested_version": "V2",
  "routing_passed": true,
  "embedding_route_id": "summarize_document",
  "embedding_score": 0.86,
  "routing_decision_reason": "embedding_score_above_threshold",
  "cache_hit": true,
  "validation_passed": true,
  "decision_reason": "answer_cache_hit_valid",
  "cache_id": "ans-001",
  "cache_version": "V1",
  "source_validation": "fingerprint_compatible",
  "total_ms": 320,
  "llm_call_count": 0,
  "cross_query_reuse": true,
  "wrong_answer_reuse": false
}
```

routing 실패 예시는 다음과 같다.

```json
{
  "query_id": "C2",
  "user_scope": "A",
  "requested_version": "V2",
  "routing_passed": false,
  "embedding_route_id": "summarize_document",
  "embedding_score": 0.71,
  "routing_decision_reason": "embedding_score_below_threshold",
  "cache_hit": false,
  "validation_passed": false,
  "decision_reason": "routing_failed_fallback_to_roi_rag",
  "total_ms": 1850,
  "llm_call_count": 1,
  "cross_query_reuse": false,
  "wrong_answer_reuse": false
}
```

추가 집계 지표:

```text
- Routing Lookup Pass Rate
- Cache Hit Rate
- Validation Pass Rate
- Cross-query Cache Reuse Rate
- Routing Fail by Low Similarity Count
- Near-miss Over-hit Count
```

`Cross-query Cache Reuse Rate`는 단독으로 좋고 나쁨을 판단하지 않는다.

```text
Cross-query Cache Reuse 높음 + Wrong Answer Reuse 낮음 = 좋은 semantic reuse
Cross-query Cache Reuse 높음 + Wrong Answer Reuse 높음 = 위험한 over-hit
```

---

## 5. Option B. Incremental Context Cache PoC 설계

### 5.1 최소 DB 구조

Context Cache는 공통 `context_units` 위에 Context Pack 단위 cache table을 추가한다. A안이 `질문 -> 최종 답변`을 재사용한다면, B안은 `질문 -> 검증된 근거 묶음`을 재사용한다.

```text
context_cache_entries
- context_cache_id
- anchor_query_text
- anchor_query_embedding_id
- context_pack_text
- created_version
- created_at
- ttl_seconds
- max_context_tokens

context_cache_sources
- context_cache_id
- logical_eu_id
- eu_version
- fingerprint
- source_order
```

캐시 정보는 다음 세 가지를 반드시 포함한다.

```text
1. Embedded Query
   - Context Pack을 찾기 위한 anchor query embedding

2. Context Pack
   - ROI-RAG가 검색한 EU들을 LLM 입력 직전 형태로 조합한 근거 묶음

3. Metadata for each Context
   - logical_eu_id
   - fingerprint
   - scope
   - version
```

`context_cache_sources`는 Context Pack 안의 각 EU가 어떤 version/fingerprint에 근거했는지 기록한다. 이 정보가 있어야 cache hit 이후에도 각 EU를 requested version 기준으로 다시 검증할 수 있다.

### 5.2 처리 흐름

```text
1. 사용자 query에서 requested_version 추출
   예: "V2에서 이 내용 요약해줘" -> requested_version=V2
2. user_scope는 A/B 중 하나로 고정 또는 seeded random 생성
3. query embedding 생성
4. context_cache_entries에서 vector similarity 기반 Context Pack 후보 검색
5. top context score가 threshold 이상이면 cache 후보로 판단
6. context_cache_sources의 각 EU에 대해 Metadata Validation 수행
7. 모든 EU가 valid하면 cached Context Pack을 LLM에 전달
8. 일부 EU가 invalid하면 valid EU는 유지하고 invalid EU는 delta retrieval 대상이 된다.
9. invalid EU가 너무 많거나 delta retrieval 구현 전이면 ROI-RAG full retrieval로 fallback한다.
10. LLM은 cached 또는 rebuild된 Context Pack을 기반으로 현재 질문에 맞는 답변을 새로 생성한다.
```

1차 PoC에서는 delta retrieval을 반드시 완성하지 않아도 된다. 먼저 다음 두 경로를 구현하면 B안의 기본 동작을 검증할 수 있다.

```text
- All Valid: Cached Context Pack + LLM
- Any Invalid: ROI-RAG Retrieval + LLM fallback
```

이후 2차 구현에서 invalid EU만 보강하는 delta retrieval을 추가한다.

### 5.3 Validation 규칙

B안의 validation 단위는 Context Pack 전체가 아니라 Context Pack 안의 각 EU다.

```text
- context_cache_sources의 각 logical_eu_id에 대해 공통 validate_eu를 수행한다.
- requested_version의 같은 logical_eu_id row가 없으면 해당 EU는 invalid다.
- requested_version의 scope가 user_scope와 맞지 않으면 해당 EU는 invalid다.
- requested_version의 fingerprint가 cache에 저장된 fingerprint와 다르면 해당 EU는 invalid다.
- 모든 EU가 valid하면 Context Pack을 그대로 재사용한다.
- 일부 EU만 invalid하면 partial invalid로 기록한다.
- invalid 비율이 threshold를 넘으면 Context Pack 전체를 버리고 ROI-RAG full retrieval로 fallback한다.
```

버전 검증의 핵심은 “cache가 만들어진 버전이 낮아도 requested version까지 같은 logical EU가 같은 fingerprint로 살아 있으면 valid”라는 점이다.

```text
예: 사용자가 V2를 요청
- cache source가 V3이면 invalid
- cache source가 V2이고 V2 fingerprint가 같으면 valid
- cache source가 V1이고 V2에도 같은 logical_eu_id와 같은 fingerprint가 있으면 valid
- cache source가 V1이고 V2에서 fingerprint가 달라졌으면 invalid
- cache source가 V1이고 V2에서 logical_eu_id가 사라졌으면 invalid
```

### 5.4 측정 로그

Context Cache PoC에서 최소한 다음 필드를 남긴다.

```json
{
  "query_id": "B2",
  "user_scope": "A",
  "requested_version": "V2",
  "context_cache_lookup_passed": true,
  "context_cache_id": "ctx-001",
  "context_similarity_score": 0.83,
  "validation_result": "all_valid",
  "valid_eu_count": 5,
  "invalid_eu_count": 0,
  "decision_reason": "context_cache_hit_all_sources_valid",
  "total_ms": 980,
  "llm_call_count": 1,
  "roi_rag_full_retrieval": false,
  "delta_retrieval_count": 0,
  "wrong_context_reuse": false
}
```

partial invalid 예시는 다음과 같다.

```json
{
  "query_id": "V2-change-1",
  "user_scope": "A",
  "requested_version": "V2",
  "context_cache_lookup_passed": true,
  "context_cache_id": "ctx-001",
  "context_similarity_score": 0.82,
  "validation_result": "partial_invalid",
  "valid_eu_count": 4,
  "invalid_eu_count": 1,
  "invalid_reasons": ["fingerprint_mismatch"],
  "decision_reason": "partial_invalid_fallback_to_roi_rag",
  "total_ms": 1620,
  "llm_call_count": 1,
  "roi_rag_full_retrieval": true,
  "delta_retrieval_count": 0,
  "wrong_context_reuse": false
}
```

추가 집계 지표:

```text
- Context Cache Hit Rate
- Context Validation Pass Rate
- Partial Invalid Rate
- Full RAG Fallback Rate
- Valid EU Reuse Rate
- Wrong Context Reuse Rate
- LLM Call Count
- P95 Latency
```

B안에서는 cache hit이어도 LLM 호출은 유지된다. 따라서 A안처럼 LLM call count가 크게 줄어드는 것을 기대하기보다, ROI-RAG retrieval/context build 비용 감소와 wrong answer reuse 위험 감소를 보는 것이 맞다.

---

## 6. QA 평가 방법

### 6.1 속도

| 지표 | 측정 방법 | 선택 이유 |
|---|---|---|
| P95 Latency | end-to-end 응답 시간의 95 percentile | 사용자 체감 지연을 직접 보여준다. |
| LLM Call Count | 전체 질의 중 LLM 호출 횟수 | Answer Cache의 비용 절감 효과를 쉽게 설명할 수 있다. |

### 6.2 기능 적합성

| 지표 | 측정 방법 | 선택 이유 |
|---|---|---|
| RAGAS Faithfulness | 답변이 제공된 Context에 근거하는지 평가 | cache reuse가 근거 없는 답변을 만들지 않는지 확인한다. |
| Wrong Answer Reuse Rate | 사람이 라벨링한 wrong reuse / answer cache hit | Answer Cache의 핵심 위험을 직접 측정한다. |
| Wrong Context Reuse Rate | 사람이 라벨링한 wrong context reuse / context cache hit | Context Cache가 유사하지만 부적절한 근거 묶음을 재사용하는지 확인한다. |
| Required Answer Type Match | 질문 유형에 맞는 답변 유형이 나왔는지 확인 | same/paraphrase/near_miss 질문에서 cache reuse가 질문 의도를 해치지 않는지 본다. |

RAGAS는 기능 적합성 평가에 사용할 수 있다. 단, 1차 PoC에서는 구현 부담을 줄이기 위해 `Wrong Answer Reuse Rate`, `Required Answer Type Match`, `Decision Reason Coverage`를 먼저 본다. RAGAS는 결과 답변과 retrieved context가 안정적으로 수집된 뒤 2차 평가로 붙인다.

vector-only routing의 위험은 특히 Set C에서 확인한다. embedding만으로 유사하게 보이는 near-miss 질문이 cache hit로 이어지는지, 이어진다면 사람이 라벨링한 wrong answer reuse에 해당하는지 확인한다.

### 6.3 테스트 용이성

| 지표 | 측정 방법 | 선택 이유 |
|---|---|---|
| Decision Reason Coverage | hit/miss/validation fail/fallback 사유가 로그에 남은 비율 | cache 판단을 설명할 수 있는지 확인한다. |

테스트 용이성은 RAGAS로 평가하지 않는다. PoC에서는 decision log가 빠짐없이 남는지 보는 것으로 충분하다.

vector-only routing을 사용하는 경우 routing 단계의 decision reason도 별도로 남긴다.

```text
- embedding_score_above_threshold
- embedding_score_below_threshold
- cache_candidate_not_found
```

---

## 7. LongBench와 RAGAS 적합성

DP3 PoC에서는 LongBench만 사용한다. `src/load_longbench.py`는 LongBench JSONL의 `input`, `context`, `answers`를 읽어 다음 구조로 저장한다.

```text
question = input
source context = context를 여러 message chunk로 분할
ground_truth_answers = answers
```

RAGAS 적합성:

- LongBench는 RAGAS에 비교적 잘 맞는다.
- `question`, `retrieved_contexts`, `response`, `reference/ground_truth`를 만들 수 있다.
- `Faithfulness`, `Context Recall`, `Response Relevancy` 평가에 사용할 수 있다.

주의점:

- LongBench는 긴 문서 QA 성격이 강해서 권한/버전 invalidation 시나리오를 자연스럽게 제공하지 않는다.
- 따라서 권한 A/B, 버전 V1/V2/V3, stale fingerprint, cross-query reuse는 LongBench에서 파생한 synthetic metadata와 query set으로 구성한다.

### 7.1 기존 `src`와의 연결 지점

현재 `src`에는 LongBench 로더와 ROI-RAG 구현이 이미 있으므로 PoC 자체는 가능하다. 다만 A안/B안 cache validation이 요구하는 scope/version/fingerprint metadata는 기존 결과에 충분히 포함되어 있지 않으므로 다음 보강이 필요하다.

```text
1. LongBench 로드
   - 기존 `src/load_longbench.py`로 thread와 benchmark_questions를 생성한다.

2. ROI-RAG thread index 생성
   - LongBench 로드 직후 `roi_rag.index_thread(thread_id)`를 실행하거나,
     기존 thread indexing API를 호출해 roi_indexed=1 상태를 만든다.

3. Cache PoC metadata 생성
   - context_units에 logical_eu_id/scope/version/fingerprint를 저장한다.
   - answer_cache_sources에 cache 답변이 참조한 EU와 fingerprint를 저장한다.
   - context_cache_sources에 Context Pack이 참조한 EU와 fingerprint를 저장한다.

4. A/B cache query 실행
   - routing pass와 metadata validation을 먼저 수행한다.
   - A안 miss 또는 invalid이면 기존 `roi_rag.query_thread()`로 fallback한다.
   - B안 miss 또는 invalid이면 1차 PoC에서는 `roi_rag.query_thread()`로 fallback하고, 2차에서 delta retrieval을 추가한다.
```

즉 현재 구조는 DP3 PoC가 불가능한 상태가 아니라, ROI-RAG 검색 결과를 cache 검증에 사용할 수 있도록 metadata 저장 계층을 추가해야 하는 상태다.

---

## 8. 구현 구성안

기존 `src` 구조에 최소 변경으로 붙이는 방식을 권장한다.

```text
src/backend/cache/
  answer_cache.py
  answer_cache_router.py
  context_cache.py
  cache_models.py
  cache_metrics.py

src/backend/routers/
  cache_poc.py
```

`answer_cache_router.py`는 `answerable_question_pool`에 대한 routing을 담당한다.

```text
AnswerCacheRouter
- query embedding 생성 또는 입력받기
- embedding 기반 route 후보 검색
- embedding threshold 확인
- routing result와 decision reason 반환
```

실행 모드:

```text
mode=verified_answer_cache
mode=incremental_context_cache
```

`context_cache.py`는 Context Pack lookup과 EU 단위 validation을 담당한다.

```text
ContextCache
- query embedding 생성 또는 입력받기
- context_cache_entries에서 유사 Context Pack 후보 검색
- context_cache_sources의 각 EU에 대해 공통 validate_eu 수행
- all_valid / partial_invalid / invalid decision 반환
- 1차 PoC에서는 invalid 시 ROI-RAG fallback
```

저장소:

```text
- ROI-RAG Index: 기존 ChromaDB / roi_rag.py 재사용
- Cache Metadata: SQLite
- Metrics: SQLite 또는 CSV
```

BM25 Index는 1차 범위에서 제외한다. near-miss over-hit이 높게 나오면 2차 보강으로 `rank-bm25` 기반 in-memory index를 추가한다.

Codex 구현 지시 예시는 다음과 같다.

```text
Implement AnswerCacheRouter using vector-only routing.
Use embedding similarity over answerable_question_pool.
Pass routing only when the top route score is above configured threshold.
Return routing_passed, route_id, embedding_score, and decision_reason.
```

---

## 9. 발표용 요약

```text
DP3 PoC는 DP1의 SPRAG/ROI-RAG 검색 기반 위에서 cache 전략을 비교한다.

우선 구현 범위는 Verified Answer Cache와 Incremental Context Cache의 기본 골격이다.
LongBench를 공통 `context_units`로 변환하고, scope A/B와 version V1/V2/V3 및 fingerprint를 부여한다.

Answer Cache는 제한된 질문 풀에서 검증된 답변을 재사용한다.
cache hit 시 ROI-RAG와 LLM을 모두 생략하므로 가장 빠르지만, wrong answer reuse 위험이 있다.

이를 줄이기 위해 초기 PoC에서는 answerable_question_pool routing을
embedding similarity 기반 vector-only gate로 구성한다.
top route score가 threshold 이상일 때만 Answer Cache lookup을 수행한다.
BM25 lexical gate는 near-miss over-hit이 확인된 뒤 2차 보강으로 추가한다.

Context Cache는 최종 답변이 아니라 Context Pack을 재사용한다.
cache hit 이후에도 LLM은 호출되지만, EU 단위 metadata validation을 통해 stale/unauthorized context를 거부할 수 있다.
1차 PoC에서는 all-valid Context Pack만 재사용하고, partial invalid는 ROI-RAG fallback으로 처리한다.

속도는 P95 Latency와 LLM Call Count로 본다.
기능 적합성은 Wrong Answer Reuse Rate, Wrong Context Reuse Rate, Required Answer Type Match로 먼저 본다.
RAGAS Faithfulness는 cache 로그와 retrieved context가 안정화된 뒤 2차 평가로 추가한다.
테스트 용이성은 Decision Reason Coverage로 본다.
```
