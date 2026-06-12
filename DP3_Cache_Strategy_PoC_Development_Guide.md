# DP3 Cache Strategy PoC Development Guide

## 1. 문서 목적

이 문서는 DP3 후보인 **Semantic Answer Cache**와 **Verified Context Cache**를 실제 PoC 수준에서 구현하기 위한 개발 가이드이다.

목표는 다음이다.

1. 두 후보를 동일한 샘플 데이터와 동일한 질의 세트로 비교한다.
2. Semantic Answer Cache가 cache hit 시 가장 빠르다는 점을 수치로 확인한다.
3. Verified Context Cache가 EU-only RAG 대비 RAG 비용을 줄이면서도 답변 재생성, Citation, 권한/버전 검증 가능성을 유지하는지 확인한다.
4. 룰 기반 intent 분류에 의존하지 않고, embedding similarity와 Symbol Dictionary, metadata guard 중심으로 구현한다.

---

## 2. 공통 전제

### 2.1 Baseline Pipeline

PoC의 baseline은 EU-only RAG이다.

```text
User Query
-> Query Normalization
-> Query Embedding
-> EU Index Search
-> Permission / Version Filter
-> Dedup
-> Rerank
-> Context Build
-> LLM Answer
-> Citation Attach
```

DP3의 cache 전략은 이 pipeline의 일부 또는 전체를 생략하는 방식으로 비교한다.

### 2.2 공통 데이터 모델

PoC에서는 다음 데이터를 준비한다.

#### Evidence Unit

```json
{
  "eu_id": "EU-101",
  "text": "AuthClient::init must be called before token validation.",
  "embedding_id": "emb-eu-101",
  "source_symbols": ["AuthClient", "AuthClient::init"],
  "source_metadata": {
    "file_path": "auth/AuthClient.cpp",
    "symbol": "AuthClient::init",
    "commit_sha": "abc123",
    "release": "2.1",
    "release_range": "release/2.x",
    "allowed_scopes": ["project-a-dev"],
    "source_fingerprint": "sha256:eu101"
  },
  "evidence_type": "api_signature"
}
```

#### Symbol Dictionary

Symbol Dictionary는 LLM 없이 만든다.

가능한 소스:

```text
- ctags output
- clangd / LSP symbol index
- tree-sitter parser result
- doxygen symbol list
- PoC용 수동 API 목록
```

PoC에서는 다음처럼 단순한 JSON으로 시작해도 된다.

```json
{
  "symbols": [
    "AuthClient",
    "AuthClient::init",
    "AuthClientV1",
    "TokenValidator",
    "LoginManager"
  ],
  "aliases": {
    "인증 클라이언트": "AuthClient",
    "토큰 검증": "TokenValidator"
  }
}
```

Symbol Dictionary는 intent 분류가 아니다. 질문과 cache entry가 같은 API/Symbol 영역을 다루는지 확인하기 위한 guard이다.

### 2.3 공통 Normalization

Query Normalization은 LLM을 사용하지 않는다.

권장 범위:

```text
- lower-case 변환
- 공백 정리
- punctuation 제거 또는 표준화
- 코드 symbol은 원형 보존
- alias mapping 적용
```

예시:

```text
Input: "인증 클라이언트 초기화 순서 알려줘"
Alias: "인증 클라이언트" -> "AuthClient"
Normalized: "AuthClient 초기화 순서 알려줘"
```

---

## 3. Semantic Answer Cache 구현 가이드

### 3.1 목적

Semantic Answer Cache는 유사 질문에 대해 과거 최종 답변을 재사용한다.

```text
Query
-> Answer Cache Lookup
-> Hit: Cached Answer Return
-> Miss: EU-only RAG + LLM
```

### 3.2 캐싱 대상

캐싱 대상은 최종 답변과 그 답변의 근거 metadata이다.

```json
{
  "cache_id": "ans-1001",
  "cache_type": "semantic_answer",
  "query_text": "AuthClient 초기화 순서 알려줘",
  "normalized_query": "AuthClient 초기화 순서 알려줘",
  "query_embedding": [0.01, 0.02, 0.03],
  "query_symbol_set": ["AuthClient"],
  "answer_text": "AuthClient는 init 호출 후 token validation을 수행해야 합니다...",
  "source_eu_ids": ["EU-101", "EU-205"],
  "citation_metadata": [
    {
      "eu_id": "EU-101",
      "file_path": "auth/AuthClient.cpp",
      "symbol": "AuthClient::init",
      "commit_sha": "abc123",
      "release": "2.1"
    }
  ],
  "validity_metadata": {
    "allowed_scopes": ["project-a-dev"],
    "release_range": "release/2.x",
    "source_fingerprints": ["sha256:eu101", "sha256:eu205"],
    "created_at": "2026-06-11T00:00:00Z",
    "ttl_seconds": 604800
  }
}
```

### 3.3 Lookup 절차

```text
1. Query normalize
2. Query embedding 생성
3. Symbol Dictionary로 query_symbol_set 추출
4. Answer Cache vector search
5. Top-N 후보에 대해 similarity threshold 적용
6. Symbol overlap guard 적용
7. Permission / version / freshness guard 적용
8. 통과하면 cached answer 반환
9. 실패하면 EU-only RAG 수행
```

### 3.4 유사도 측정

기본 유사도는 cosine similarity를 사용한다.

```text
similarity = cosine(query_embedding, cached_query_embedding)
```

PoC에서는 다음 threshold를 실험한다.

```text
- strict: 0.90
- normal: 0.85
- relaxed: 0.80
```

Threshold가 낮아질수록 hit rate는 올라가지만 wrong cache hit 위험도 올라간다.

### 3.5 Guard 조건

Semantic similarity만으로 hit를 결정하지 않는다.

권장 guard:

```text
- query_symbol_set ∩ cached.query_symbol_set != empty
- requested_version is compatible with cached.release_range
- user_scope is included in cached.allowed_scopes
- cached entry TTL is not expired
- source_fingerprints are still valid
```

Symbol overlap은 rule-based intent 분류가 아니라 Source domain guard이다.

### 3.6 Store 조건

모든 답변을 cache하지 않는다.

Cacheable 조건 예시:

```text
- 답변에 source_eu_ids가 존재한다.
- Citation Trace Coverage가 100%다.
- LLM 답변이 fallback 또는 uncertain 상태가 아니다.
- 사용자 개인 정보나 일회성 context가 포함되지 않는다.
- 요청 version과 permission scope가 명확하다.
```

### 3.7 Pseudocode

```python
def handle_query_with_answer_cache(query, user_scope, requested_version):
    normalized = normalize_query(query)
    query_embedding = embed(normalized)
    query_symbols = extract_symbols(normalized, symbol_dictionary)

    candidates = answer_cache.vector_search(query_embedding, top_n=5)

    for c in candidates:
        if cosine(query_embedding, c.query_embedding) < ANSWER_CACHE_THRESHOLD:
            continue
        if not symbol_overlap(query_symbols, c.query_symbol_set):
            continue
        if not is_scope_allowed(user_scope, c.validity_metadata.allowed_scopes):
            continue
        if not is_version_compatible(requested_version, c.validity_metadata.release_range):
            continue
        if is_expired(c.validity_metadata):
            continue
        if not source_fingerprints_valid(c.validity_metadata.source_fingerprints):
            continue
        return c.answer_text, c.citation_metadata, "answer_cache_hit"

    rag_result = run_eu_rag(query, user_scope, requested_version)
    answer = generate_answer(query, rag_result.context)

    if is_cacheable(answer, rag_result):
        answer_cache.store(build_answer_cache_entry(query, answer, rag_result))

    return answer.text, answer.citations, "answer_cache_miss"
```

---

## 4. Verified Context Cache 구현 가이드

### 4.1 목적

Verified Context Cache는 최종 답변이 아니라, RAG 결과로 만든 검증된 Context Pack을 재사용한다.

```text
Query
-> Context Cache Lookup
-> Hit: Cached Context Pack + LLM
-> Miss: EU-only RAG + Context Build + LLM
```

### 4.2 캐싱 대상

캐싱 대상은 LLM 입력 직전의 Context Pack이다.

```json
{
  "cache_id": "ctx-2001",
  "cache_type": "verified_context",
  "context_pack_id": "ctx-authclient-usage-2x",
  "anchor_queries": [
    "AuthClient 사용법 알려줘",
    "AuthClient 초기화 순서 알려줘"
  ],
  "context_embedding": [0.04, 0.05, 0.06],
  "source_symbol_set": [
    "AuthClient",
    "AuthClient::init",
    "AuthClientV1"
  ],
  "source_eu_ids": ["EU-101", "EU-205", "EU-330", "EU-412"],
  "context_text": "[EU-101] AuthClient::init must be called before token validation.\n[EU-205] Token validation requires...",
  "citation_metadata": [
    {
      "eu_id": "EU-101",
      "file_path": "auth/AuthClient.cpp",
      "symbol": "AuthClient::init",
      "commit_sha": "abc123",
      "release": "2.1"
    }
  ],
  "validity_metadata": {
    "allowed_scopes": ["project-a-dev"],
    "release_range": "release/2.x",
    "source_fingerprints": ["sha256:eu101", "sha256:eu205"],
    "last_verified_commit": "abc123",
    "created_at": "2026-06-11T00:00:00Z",
    "ttl_seconds": 604800
  },
  "context_quality_metadata": {
    "required_key_points": [
      "init_before_validation",
      "deprecated_v1_replacement",
      "release_2x_note"
    ],
    "evidence_types": [
      "api_signature",
      "sample_usage",
      "deprecated_note",
      "migration_note"
    ]
  }
}
```

### 4.3 Context Embedding 생성

Context Pack embedding은 다음 중 하나로 만든다.

#### 방법 A. Context Text Embedding

```text
context_embedding = embed(context_text)
```

장점:

```text
- 구현이 가장 쉽다.
- 실제 LLM에 들어갈 근거와 embedding 대상이 일치한다.
```

단점:

```text
- context_text가 길면 embedding 품질이 흐려질 수 있다.
```

#### 방법 B. Anchor Query Embedding Average

Context Pack을 만든 원인이 된 anchor query들의 embedding 평균을 사용한다.

```text
context_embedding = average(embed(anchor_query_1), embed(anchor_query_2), ...)
```

장점:

```text
- 사용자가 다시 물어볼 표현과 가까울 수 있다.
```

단점:

```text
- anchor query가 적으면 coverage가 좁다.
```

#### 방법 C. Hybrid Embedding

Context text와 anchor query embedding을 모두 저장한다.

```text
query_to_context_score = max(
  cosine(query_embedding, context_text_embedding),
  cosine(query_embedding, anchor_query_embedding_avg)
)
```

PoC에서는 C를 추천한다. 구현은 조금 늘어나지만, rule-based intent 없이도 hit 품질을 비교하기 좋다.

### 4.4 Lookup 절차

```text
1. Query normalize
2. Query embedding 생성
3. Symbol Dictionary로 query_symbol_set 추출
4. Context Cache vector search
5. Top-N 후보에 대해 similarity threshold 적용
6. Symbol overlap guard 적용
7. Permission / version / freshness guard 적용
8. 통과하면 cached context_text를 LLM Context로 사용
9. LLM은 현재 질문에 맞게 답변 생성
10. 실패하면 EU-only RAG 수행
```

### 4.5 유사도 측정

기본 유사도는 cosine similarity를 사용한다.

```text
score_text = cosine(query_embedding, context_text_embedding)
score_anchor = cosine(query_embedding, anchor_query_embedding_avg)
context_score = max(score_text, score_anchor)
```

PoC threshold 예시:

```text
- strict: 0.88
- normal: 0.82
- relaxed: 0.78
```

Verified Context Cache는 최종 답변을 재사용하지 않기 때문에 Answer Cache보다 threshold를 약간 낮게 실험할 수 있다. 다만 symbol/metadata guard는 반드시 통과해야 한다.

### 4.6 Guard 조건

```text
- query_symbol_set ∩ context.source_symbol_set != empty
- requested_version is compatible with context.release_range
- user_scope is included in context.allowed_scopes
- context TTL is not expired
- source_fingerprints are still valid
- source_eu_ids are still available
```

Symbol Dictionary를 사용하기 어려운 질문은 context cache를 보수적으로 miss 처리하고 EU-only RAG로 fallback한다.

### 4.7 Store 조건

Context Pack은 RAG 결과가 충분히 안정적일 때만 저장한다.

```text
- Context에 source_eu_ids가 존재한다.
- 최소 2개 이상의 Evidence Unit이 포함되어 있다.
- Citation metadata가 모든 EU에 연결되어 있다.
- 권한/버전 metadata가 명확하다.
- Context token 수가 max_context_tokens 이하이다.
- RAG 결과가 low-confidence가 아니다.
```

### 4.8 Pseudocode

```python
def handle_query_with_context_cache(query, user_scope, requested_version):
    normalized = normalize_query(query)
    query_embedding = embed(normalized)
    query_symbols = extract_symbols(normalized, symbol_dictionary)

    candidates = context_cache.vector_search(query_embedding, top_n=5)

    for c in candidates:
        score_text = cosine(query_embedding, c.context_text_embedding)
        score_anchor = cosine(query_embedding, c.anchor_query_embedding_avg)
        score = max(score_text, score_anchor)

        if score < CONTEXT_CACHE_THRESHOLD:
            continue
        if not symbol_overlap(query_symbols, c.source_symbol_set):
            continue
        if not is_scope_allowed(user_scope, c.validity_metadata.allowed_scopes):
            continue
        if not is_version_compatible(requested_version, c.validity_metadata.release_range):
            continue
        if is_expired(c.validity_metadata):
            continue
        if not source_fingerprints_valid(c.validity_metadata.source_fingerprints):
            continue

        answer = generate_answer(query, c.context_text)
        return answer.text, c.citation_metadata, "context_cache_hit"

    rag_result = run_eu_rag(query, user_scope, requested_version)
    answer = generate_answer(query, rag_result.context_text)

    if is_context_cacheable(rag_result):
        context_cache.store(build_context_cache_entry(query, rag_result))

    return answer.text, answer.citations, "context_cache_miss"
```

---

## 5. 공통 Invalidation 설계

### 5.1 TTL 기반 만료

가장 단순한 방식이다.

```text
created_at + ttl_seconds < now
-> invalid
```

PoC에서는 1일, 7일, 30일 TTL을 비교할 수 있다.

### 5.2 Source Fingerprint 기반 만료

EU를 만들 때 원본 Segment의 hash를 저장한다.

```text
source_fingerprint = sha256(file_path + symbol + segment_text + commit_sha)
```

Cache hit 시 현재 EU Store의 fingerprint와 비교한다.

```text
cached_fingerprint == current_fingerprint
-> valid
else
-> invalid
```

### 5.3 Version Scope 기반 만료

```text
requested_version ∈ cache.release_range
-> valid
else
-> invalid
```

### 5.4 Permission Scope 기반 거부

```text
user_scope ∈ cache.allowed_scopes
-> valid
else
-> reject
```

권한이 애매하면 cache를 쓰지 않고 EU-only RAG로 fallback한다.

---

## 6. PoC 구성안

### 6.1 Component 구성

```text
Query API
  -> Query Normalizer
  -> Embedding Client
  -> Symbol Matcher
  -> Answer Cache Store
  -> Context Cache Store
  -> EU Retriever
  -> Reranker
  -> Context Builder
  -> LLM Client
  -> Metrics Logger
```

### 6.2 Storage 구성

최소 PoC는 다음으로 충분하다.

```text
- EU Store: JSONL 또는 SQLite
- Embedding Index: FAISS 또는 Chroma
- Answer Cache: SQLite + vector index
- Context Cache: SQLite + vector index
- Metrics: CSV 또는 SQLite
```

### 6.3 실행 모드

동일한 질문 세트를 세 모드로 실행한다.

```text
mode=baseline_eu_rag
mode=semantic_answer_cache
mode=verified_context_cache
```

각 모드는 동일한 query set, 동일한 EU set, 동일한 LLM config를 사용한다.

---

## 7. 테스트 시나리오

### 7.1 Query Set

#### Set A. 완전 반복 질문

```text
A1. AuthClient 초기화 순서 알려줘.
A2. AuthClient 초기화 순서 알려줘.
A3. AuthClient 초기화 순서 알려줘.
```

#### Set B. 표현 변형 질문

```text
B1. AuthClient 초기화 순서 알려줘.
B2. AuthClient는 어떤 순서로 init 해야 해?
B3. 인증 클라이언트 초기 설정 방법 알려줘.
B4. AuthClient 사용 예제 보여줘.
```

#### Set C. 비슷하지만 다른 질문

```text
C1. AuthClient는 언제 써야 해?
C2. AuthClient를 쓰면 안 되는 경우는?
C3. AuthClientV1을 계속 써도 되는 경우는?
C4. Release 2.x에서 AuthClient 초기화 시 주의사항은?
```

#### Set D. 권한/버전 변경 질문

```text
D1. project-a-dev 권한으로 AuthClient 사용법 질문
D2. project-b-dev 권한으로 같은 질문
D3. release/2.x 기준 질문
D4. release/3.x 기준 질문
```

### 7.2 Expected Observation

| 시나리오 | Semantic Answer Cache | Verified Context Cache |
|---|---|---|
| 완전 반복 질문 | 가장 빠름 | 빠르지만 LLM 호출은 유지 |
| 표현 변형 질문 | threshold에 따라 hit/miss 또는 wrong hit 가능 | Context Pack hit 시 안정적 |
| 비슷하지만 다른 질문 | wrong answer reuse 위험 | 같은 근거를 쓰더라도 답변은 새로 생성 |
| 권한/버전 변경 | metadata guard 실패 시 fallback 필요 | metadata guard 실패 시 fallback 필요 |

---

## 8. Metrics Logging

### 8.1 공통 로그 필드

```json
{
  "query_id": "B2",
  "mode": "verified_context_cache",
  "cache_lookup_ms": 12,
  "rag_retrieval_ms": 0,
  "rerank_ms": 0,
  "context_build_ms": 0,
  "llm_ms": 1450,
  "total_ms": 1510,
  "cache_status": "hit",
  "similarity_score": 0.86,
  "symbol_overlap": true,
  "permission_valid": true,
  "version_valid": true,
  "freshness_valid": true,
  "source_eu_ids": ["EU-101", "EU-205"],
  "citation_count": 2
}
```

### 8.2 핵심 지표

| 지표 | 계산 방법 |
|---|---|
| P50/P95 Latency | total_ms percentile |
| Cache Hit Rate | hit count / total count |
| RAG Skip Rate | rag_retrieval_ms == 0 비율 |
| LLM Skip Rate | llm_ms == 0 비율 |
| Wrong Cache Hit Rate | 사람이 라벨링한 wrong hit / cache hit |
| Invalid Cache Rejection Rate | metadata guard로 reject된 count / lookup 후보 count |
| Citation Trace Coverage | citation이 source_eu_ids까지 연결된 답변 비율 |
| Repeated Answer Consistency | 같은 topic 질문의 핵심 key point 일치율 |

### 8.3 Manual Evaluation Sheet

PoC에서는 일부 항목은 사람이 라벨링해야 한다.

```text
- answer_correct: yes/no
- wrong_cache_hit: yes/no
- required_key_points_present: 0~N
- contradiction_exists: yes/no
- citation_valid: yes/no
```

룰 기반 intent 분류 대신, 평가 단계에서 사람이 expected key point를 정의하고 결과를 비교한다.

---

## 9. Implementation Notes

### 9.1 Embedding Model

PoC에서는 같은 embedding model을 세 곳에 동일하게 사용한다.

```text
- EU embedding
- Answer Cache query embedding
- Context Cache embedding
```

한국어 질의와 영어 코드 symbol이 섞일 수 있으므로 multilingual embedding model을 쓰는 것이 좋다.

### 9.2 Symbol Matching

Symbol matching은 다음 순서로 수행한다.

```text
1. Exact match: AuthClient
2. Case-sensitive code symbol match: AuthClient::init
3. Alias match: 인증 클라이언트 -> AuthClient
```

Symbol이 하나도 잡히지 않으면 cache hit를 보수적으로 막고 baseline RAG로 fallback한다.

### 9.3 Similarity Threshold 실험

두 cache 모두 threshold sweep을 수행한다.

```text
Semantic Answer Cache: 0.80 / 0.85 / 0.90
Verified Context Cache: 0.78 / 0.82 / 0.88
```

비교 관찰:

```text
- threshold 낮음: hit rate 증가, wrong hit 증가 가능
- threshold 높음: safety 증가, hit rate 감소
```

### 9.4 Fallback 원칙

애매하면 cache를 쓰지 않는다.

```text
- symbol overlap 없음
- version 불일치
- permission 불일치
- fingerprint mismatch
- TTL expired
- similarity threshold 미달
```

이 경우 EU-only RAG로 fallback한다.

---

## 10. 발표용 요약

### 10.1 한 줄 정의

```text
Semantic Answer Cache는 답을 재사용한다.
Verified Context Cache는 답변에 쓸 검증된 근거 묶음을 재사용한다.
```

### 10.2 핵심 Trade-off

| 항목 | Semantic Answer Cache | Verified Context Cache |
|---|---|---|
| 최고 속도 | 강함 | 중간 |
| RAG 생략 | 가능 | 가능 |
| LLM 생략 | 가능 | 불가능 |
| 질문별 답변 생성 | 약함 | 강함 |
| Wrong answer reuse 위험 | 높음 | 낮음 |
| Citation 유지 | 별도 관리 필요 | 구조적으로 유리 |
| 권한/버전 검증 | metadata guard 필요 | metadata guard 필요 |
| PoC 난이도 | 낮음 | 중간 |

### 10.3 선택 논리

```text
Semantic Answer Cache는 반복 질문에서 가장 빠르지만, 최종 답변 재사용 방식이므로 비슷하지만 다른 질문에 잘못된 답변을 줄 수 있다.
Verified Context Cache는 LLM 생성 비용은 남지만, 검증된 Evidence Context를 재사용해 RAG 비용을 줄이고, 현재 질문에 맞게 답변을 새로 생성할 수 있다.
따라서 코드 어시스트에서는 단순 최고 속도보다 저지연·일관성·근거 신뢰성을 균형 있게 만족하는 Verified Context Cache가 더 적합하다.
```
