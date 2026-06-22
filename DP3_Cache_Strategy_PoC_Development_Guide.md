# DP3 Cache Strategy PoC Development Guide

## 1. 문서 목적

이 문서는 최신 DP3 문서(`doc/04_DP3_Response_Cache_Strategy.md`) 기준으로 **Verified Answer Cache**와 **Incremental Context Cache** PoC를 준비하기 위한 개발 가이드이다.

현재 우선 구현 범위는 **Answer Cache PoC**다. Context Cache는 비교 후보로 남기되, 세부 구현은 후속 정리 대상으로 둔다.

DP3 PoC는 DP1의 선택안인 **SPRAG 기반 Evidence Unit RAG**를 검색 기반으로 사용한다. 현재 `src` 구현에서는 이 구조가 `roi_rag.py` / `ROIRAG` 이름으로 구현되어 있으므로, 아래 문서에서는 `ROI-RAG`라고 부른다.

PoC에서 확인할 QA와 최소 측정 지표는 다음으로 제한한다.

| QA | 핵심 질문 | PoC 지표 |
|---|---|---|
| 속도 | 반복 질문에서 응답 시간이 줄어드는가 | P95 Latency, LLM Call Count |
| 기능 적합성 | cache reuse가 답변 품질과 근거 정합성을 해치지 않는가 | RAGAS Faithfulness, Wrong Answer Reuse Rate |
| 테스트 용이성 | cache 판단과 실패 원인을 설명할 수 있는가 | Decision Reason Coverage |

---

## 2. 비교 대상

### 2.1 Option A: Verified Answer Cache

Verified Answer Cache는 미리 정의한 `answerable_question_pool`을 통과한 질문에 한해 과거 검증 답변을 재사용한다.

초기 PoC에서는 `answerable_question_pool` routing을 **embedding similarity와 BM25 lexical matching의 hybrid gate**로 구성한다.

```text
Query
-> Answerable Question Pool Routing
   -> Embedding Route Lookup
   -> BM25 Route Lookup
   -> Hybrid Gate Validation
-> Answer Cache Lookup
-> Metadata Validation
-> Hit: Cached Answer 반환
-> Miss: ROI-RAG Retrieval + LLM
```

특징:

- cache hit 시 ROI-RAG 검색과 LLM 생성을 모두 생략하므로 가장 빠르다.
- 질문 범위는 FAQ, 요약, 핵심 내용, 정의, 주요 사실 확인처럼 답변 형태가 안정적인 질문으로 제한한다.
- 주요 리스크는 비슷하지만 다른 질문에 기존 답변을 재사용하는 wrong answer reuse다.
- embedding은 표현이 다른 유사 질문을 잡는 데 유리하고, BM25는 질문 의도를 바꾸는 핵심 단어 차이를 확인하는 데 유리하다.
- 따라서 초기 PoC에서는 embedding top route와 BM25 top route가 동일하고 각 score가 threshold 이상일 때만 Answer Cache lookup을 수행한다.

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

Context Cache의 세부 DB/정책/화면은 Answer Cache PoC 이후 별도 정리한다.

---

## 3. Answer Cache PoC 설계

### 3.1 LongBench 사전 준비

Answer Cache PoC는 LongBench만 사용한다. 원본 LongBench의 `input`, `context`, `answers`를 기반으로 PoC용 EU와 질문 세트를 만든다.

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

이 구조를 통해 같은 EU가 여러 버전에 존재하거나, 특정 버전에서 fingerprint가 달라지는 상황을 만든다. Answer Cache는 이 metadata로 stale answer를 거부할 수 있어야 한다.

### 3.2 최소 DB 구조

PoC에서는 물리적으로 DB를 권한/버전별로 나누지 않고 SQLite metadata로 구분한다.

```text
longbench_evidence_units
- eu_id
- source_example_id
- text
- scope              -- A or B
- version            -- V1 / V2 / V3
- fingerprint

answerable_question_pool
- route_id
- question_text
- embedding_id
- bm25_text          -- optional, 기본값은 question_text
```

초기 PoC에서는 `answerable_question_pool`의 `question_text` 또는 `bm25_text`를 기준으로 embedding routing과 BM25 routing을 함께 수행한다.

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
- eu_id
- eu_version
- fingerprint
```

핵심은 `answer_cache_sources`다. Answer Cache entry가 어떤 EU 버전과 fingerprint에 근거했는지 남겨야 requested version에서 재사용 가능한지 검증할 수 있다.

BM25 index는 PoC에서는 별도 영속 DB로 만들지 않고, `answerable_question_pool.question_text` 또는 `bm25_text`를 기반으로 실행 시 메모리에 구성해도 충분하다.

### 3.3 처리 흐름

```text
1. 사용자 query에서 requested_version 추출
   예: "V2에서 이 내용 요약해줘" -> requested_version=V2
2. user_scope는 A/B 중 하나로 고정 또는 seeded random 생성
3. query embedding 생성
4. answerable_question_pool에서 hybrid routing 수행
   4-1. embedding similarity 기반 top route 검색
   4-2. BM25 기반 top route 검색
   4-3. embedding top route와 BM25 top route가 동일한지 확인
   4-4. 각 score가 threshold 이상인지 확인
5. routing을 통과한 경우 Answer Cache에서 유사 query cache 후보 검색
6. Metadata Validation 수행
7. 통과하면 cached answer 반환
8. 실패하면 ROI-RAG로 검색하고 LLM 답변 생성 후 cache 저장
```

Answerable Question Pool Routing 규칙:

```text
- embedding top route와 BM25 top route가 같아야 한다.
- embedding similarity score가 threshold 이상이어야 한다.
- BM25 score가 threshold 이상이어야 한다.
- 위 조건을 만족하지 않으면 Answer Cache lookup을 수행하지 않고 ROI-RAG + LLM으로 fallback한다.
```

이 routing은 Answer Cache의 wrong answer reuse를 줄이기 위한 보수적 gate다. embedding은 표현이 다른 유사 질문을 잡는 데 유리하고, BM25는 “요약”, “예외”, “한계”, “비교”처럼 질문 의도를 바꾸는 핵심 단어 차이를 확인하는 데 유리하다.

Validation 규칙:

```text
- user_scope == cache.scope 이어야 한다.
- cache_version > requested_version 이면 실패한다.
- cache_version == requested_version 이면 fingerprint를 확인한다.
- cache_version < requested_version 이면 requested_version에 동일 EU가 존재하고 fingerprint가 동일하거나 호환 가능해야 통과한다.
- fingerprint mismatch가 있으면 stale answer로 보고 fallback한다.
```

중요한 점은 `requested_version > cache_version`이라고 해서 무조건 통과시키지 않는 것이다. 더 높은 버전에서 EU가 수정됐을 수 있으므로 `answer_cache_sources` 기준으로 fingerprint를 재검증한다.

### 3.4 질문 세트 생성

Answer Cache 효과를 보려면 질문이 모두 완전히 달라서는 안 된다. PoC 질문은 다음 3종류로 만든다.

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

이 질문 세트는 Codex가 LongBench 원본 질문과 context를 보고 생성할 수 있다. 구현 시에는 LongBench 예제별로 `same`, `paraphrase`, `near_miss` 유형을 붙여 query set을 만든다.

해석:

```text
- Set A는 cache hit 속도 개선을 본다.
- Set B는 의미적으로 안전한 cross-query reuse를 본다.
- Set C는 wrong answer reuse 위험을 본다.
```

Set C는 BM25 보조 gate의 효과를 확인하기 위한 테스트로도 사용한다. 예를 들어 embedding은 유사하게 보더라도 BM25가 “요약”과 “예외/한계”의 차이를 잡아 route mismatch를 발생시키면 Answer Cache reuse를 막을 수 있다.

### 3.5 측정 로그

Answer Cache PoC에서 최소한 다음 필드는 남긴다.

```json
{
  "query_id": "B2",
  "user_scope": "A",
  "requested_version": "V2",
  "routing_passed": true,
  "embedding_route_id": "summarize_document",
  "embedding_score": 0.86,
  "bm25_route_id": "summarize_document",
  "bm25_score": 7.2,
  "routing_decision_reason": "embedding_and_bm25_agree",
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
  "embedding_score": 0.84,
  "bm25_route_id": "caveat_document",
  "bm25_score": 6.9,
  "routing_decision_reason": "embedding_bm25_route_mismatch",
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
- Embedding/BM25 Route Agreement Rate
- Routing Fail by Route Mismatch Count
```

`Cross-query Cache Reuse Rate`는 단독으로 좋고 나쁨을 판단하지 않는다.

```text
Cross-query Cache Reuse 높음 + Wrong Answer Reuse 낮음 = 좋은 semantic reuse
Cross-query Cache Reuse 높음 + Wrong Answer Reuse 높음 = 위험한 over-hit
```

---

## 4. QA 평가 방법

### 4.1 속도

| 지표 | 측정 방법 | 선택 이유 |
|---|---|---|
| P95 Latency | end-to-end 응답 시간의 95 percentile | 사용자 체감 지연을 직접 보여준다. |
| LLM Call Count | 전체 질의 중 LLM 호출 횟수 | Answer Cache의 비용 절감 효과를 쉽게 설명할 수 있다. |

### 4.2 기능 적합성

| 지표 | 측정 방법 | 선택 이유 |
|---|---|---|
| RAGAS Faithfulness | 답변이 제공된 Context에 근거하는지 평가 | cache reuse가 근거 없는 답변을 만들지 않는지 확인한다. |
| Wrong Answer Reuse Rate | 사람이 라벨링한 wrong reuse / answer cache hit | Answer Cache의 핵심 위험을 직접 측정한다. |

RAGAS는 기능 적합성 평가에 사용한다. 단, DP3는 cache 전략 비교이므로 RAGAS 하나만으로 충분하지 않다. 특히 Answer Cache는 최종 답변 재사용이 핵심 리스크이므로 `Wrong Answer Reuse Rate`를 함께 본다.

BM25 hybrid routing의 효과는 특히 Set C에서 확인한다. embedding만으로는 유사하게 보일 수 있는 near-miss 질문이 BM25 route mismatch로 fallback되는지 확인한다.

### 4.3 테스트 용이성

| 지표 | 측정 방법 | 선택 이유 |
|---|---|---|
| Decision Reason Coverage | hit/miss/validation fail/fallback 사유가 로그에 남은 비율 | cache 판단을 설명할 수 있는지 확인한다. |

테스트 용이성은 RAGAS로 평가하지 않는다. PoC에서는 decision log가 빠짐없이 남는지 보는 것으로 충분하다.

Hybrid routing을 사용하는 경우 routing 단계의 decision reason도 별도로 남긴다.

```text
- embedding_and_bm25_agree
- embedding_score_below_threshold
- bm25_score_below_threshold
- embedding_bm25_route_mismatch
```

---

## 5. LongBench와 RAGAS 적합성

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

---

## 6. 구현 구성안

기존 `src` 구조에 최소 변경으로 붙이는 방식을 권장한다.

```text
src/backend/cache/
  answer_cache.py
  answer_cache_router.py
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
- BM25 기반 route 후보 검색
- route 일치 여부와 threshold 확인
- routing result와 decision reason 반환
```

실행 모드:

```text
mode=verified_answer_cache
```

후속으로 Context Cache PoC를 구현할 때 다음을 추가한다.

```text
src/backend/cache/context_cache.py
mode=incremental_context_cache
```

저장소:

```text
- ROI-RAG Index: 기존 ChromaDB / roi_rag.py 재사용
- Cache Metadata: SQLite
- BM25 Index: answerable_question_pool 기반 in-memory index
- Metrics: SQLite 또는 CSV
```

BM25 구현은 PoC에서는 가벼운 Python 라이브러리 사용을 권장한다.

```text
- rank-bm25
```

Codex 구현 지시 예시는 다음과 같다.

```text
Implement AnswerCacheRouter using hybrid routing.
Use embedding similarity and BM25 over answerable_question_pool.
Pass routing only when embedding top route and BM25 top route are identical
and both scores are above configured thresholds.
Return routing_passed, route_id, embedding_score, bm25_score, and decision_reason.
```

---

## 7. 발표용 요약

```text
DP3 PoC는 DP1의 SPRAG/ROI-RAG 검색 기반 위에서 cache 전략을 비교한다.

우선 구현 범위는 Verified Answer Cache다.
LongBench를 scope A/B와 version V1/V2/V3로 확장하고, cache entry가 어떤 EU 버전과 fingerprint에 근거했는지 저장한다.

Answer Cache는 제한된 질문 풀에서 검증된 답변을 재사용한다.
cache hit 시 ROI-RAG와 LLM을 모두 생략하므로 가장 빠르지만, wrong answer reuse 위험이 있다.

이를 줄이기 위해 초기 PoC에서는 answerable_question_pool routing을
embedding similarity와 BM25 lexical matching의 hybrid gate로 구성한다.
embedding top route와 BM25 top route가 동일하고, 각 score가 threshold 이상일 때만 Answer Cache lookup을 수행한다.

속도는 P95 Latency와 LLM Call Count로 본다.
기능 적합성은 RAGAS Faithfulness와 Wrong Answer Reuse Rate로 본다.
테스트 용이성은 Decision Reason Coverage로 본다.
```
