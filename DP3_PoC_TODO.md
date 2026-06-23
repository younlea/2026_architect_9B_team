# DP3 PoC TODO

## 1. 시간 계측 세분화

현재 A/B안 로그는 `total_ms` 중심으로 저장되어 있어, RAG retrieval, LLM 호출, validation/cache lookup 비용을 분리해서 분석하기 어렵다.

상태: 완료. `log_json.timings_ms`와 batch summary의 `timing_avg_ms`에 주요 구간별 평균 시간이 저장/출력된다.

### 완료된 내용

- A안 Answer Cache 로그에 다음 시간 필드를 추가했다.
  - `setup_ms`
  - `embedding_ms`
  - `route_ms`
  - `cache_lookup_ms`
  - `validation_ms`
  - `rag_db_ms`
  - `rag_scoring_ms`
  - `rag_rerank_ms`
  - `rag_total_ms`
  - `prompt_build_ms`
  - `llm_ms`
  - `cache_store_ms`
  - `total_ms`
- B안 Context Cache 로그에 다음 시간 필드를 추가했다.
  - `embedding_ms`
  - `cache_lookup_ms`
  - `validation_ms`
  - `full_retrieval_db_ms`
  - `full_retrieval_scoring_ms`
  - `full_retrieval_rerank_ms`
  - `full_retrieval_total_ms`
  - `delta_retrieval_db_ms`
  - `delta_retrieval_scoring_ms`
  - `delta_retrieval_rerank_ms`
  - `delta_retrieval_filter_ms`
  - `delta_retrieval_total_ms`
  - `prompt_build_ms`
  - `llm_ms`
  - `cache_store_ms`
  - `total_ms`
- `log_json`에는 세부 시간을 저장하고, batch summary에는 평균값을 출력한다.
- 프론트엔드 통계에서 평균 시간을 다음 기준으로 볼 수 있게 했다.
  - 전체 평균
  - cache hit 평균
  - cache miss/fallback 평균
  - RAG 평균
  - LLM 평균

### 확인 기준

- A/B 100개 테스트 후 `RAG`, `LLM`, `TOTAL` 평균을 각각 계산할 수 있어야 한다.
- mock LLM 환경에서도 `llm_ms`가 분리되어 저장되어야 한다.
- 나중에 실제 LLM을 연결해도 같은 로그 구조를 그대로 사용할 수 있어야 한다.

### 후속 메모

- 현재 `rag_rerank_ms`, `full_retrieval_rerank_ms`, `delta_retrieval_rerank_ms`는 별도 reranker 모델 시간이 아니라 similarity score 정렬 시간이다.
- 실제 reranker를 추가하면 `reranker_ms` 또는 `cross_encoder_rerank_ms`를 별도 필드로 분리한다.

## 2. 현재 시간 결과 이상치 평가

현재 샘플에서는 A안 cache hit 평균 시간이 RAG fallback 평균 시간보다 크게 나오는 현상이 있었다. 이는 실제 성능 특성이라기보다 PoC 구현의 후보 검색, validation 반복, mock LLM, route fail fallback 처리 방식이 섞인 결과일 가능성이 있다.

### TODO

- A안 hit 경로가 느린 이유를 분해한다.
  - cache 후보 개수 증가에 따른 validation 비용
  - V1/V2 후보가 섞일 때 후보 순회 비용
  - route fail fallback과 일반 RAG fallback이 같은 fallback으로 집계되는 문제
- `route fail`, `cache miss`, `validation fail`, `valid hit`를 별도 카테고리로 분리해서 시간 통계를 낸다.
- A안 후보 검색 최적화가 필요한지 검토한다.
  - `cache_version` 우선 필터링 가능 여부
  - threshold 이상 후보 수 제한
  - source validation 결과 캐싱 가능 여부
- B안에서도 context cache hit가 실제로 어느 비용을 줄이는지 분리해서 확인한다.

### 확인 기준

- A안 hit가 느린 이유를 `route/cache lookup/validation/rag/llm` 중 어느 구간 때문인지 설명할 수 있어야 한다.
- 발표나 문서에는 `total_ms`만으로 성능 개선을 단정하지 않는다.
- 순수 RAG 비용 절감과 전체 요청 비용 절감을 구분해서 제시한다.

## 3. 권한/버전 혼합 테스트 추가

현재 UI 테스트는 주로 고정 scope와 V1/V2 pass를 순차 실행하는 구조다. 실제 목표는 사용자 권한과 요청 버전이 섞인 N개 요청에서 A/B안이 cache reuse와 validation을 안전하게 처리하는지 확인하는 것이다.

### TODO

- mixed workload 테스트 모드를 추가한다.
- 입력 N개에 대해 다음 값을 랜덤 또는 비율 기반으로 섞는다.
  - `scope=A`
  - `scope=B`
  - `requested_version=V1`
  - `requested_version=V2`
  - `requested_version=V3`
- 테스트 설정에 다음 옵션을 추가한다.
  - 전체 질문 수
  - scope A/B 비율
  - V1/V2/V3 비율
  - cache 초기화 여부
  - seed
- 결과 통계는 A/B안 모두 다음 기준으로 출력한다.
  - cache hit
  - cache miss
  - validation pass
  - validation fail
  - scope mismatch
  - version/fingerprint mismatch
  - full retrieval
  - delta retrieval
  - LLM calls
  - RAG/LLM/TOTAL 평균 시간

### 확인 기준

- scope가 다른 cache는 재사용되지 않아야 한다.
- 요청 version보다 최신 metadata를 가진 cache는 invalid 처리되어야 한다.
- 요청 version에서 fingerprint가 달라진 EU를 포함한 cache는 invalid 처리되어야 한다.
- B안은 invalid 비율에 따라 delta retrieval 또는 full retrieval로 분기해야 한다.
- mixed workload에서도 DP1/DP2 실행 방식과 데이터는 변경되지 않아야 한다.

## 4. Reranking 추가

현재 PoC의 retrieval은 embedding similarity score를 계산한 뒤 score sort로 top-k를 고른다. Cache 전략의 성능 개선 효과를 더 현실적으로 보려면, 일반적인 RAG pipeline에서 비용이 드는 별도 reranking 단계를 추가하는 것이 좋다.

### TODO

- 1차 retrieval에서 top-n 후보를 가져온다.
- 별도 reranker를 사용해 query-context relevance 기준으로 재정렬한다.
- 최종 top-k context를 A/B안에 전달한다.
- reranking 시간을 별도 측정한다.
  - `reranker_ms`
  - `reranker_candidate_count`
  - `reranker_top_k`
- A안/B안 cache hit 시 reranker 비용이 얼마나 절감되는지 비교한다.

### 확인 기준

- 기존 score sort 시간과 별도 reranker 시간을 구분해서 측정할 수 있어야 한다.
- A안 answer hit에서는 retrieval/reranking/LLM 비용이 줄어드는지 확인한다.
- B안 context hit에서는 full retrieval/reranking 비용이 줄어드는지 확인한다.
- reranker 추가 전후의 cache 전략 비교 결과가 설명 가능해야 한다.

## 5. BM25 및 Hybrid Retrieval 추가

현재 PoC는 vector similarity 중심으로 retrieval한다. 추후 BM25 또는 vector+BM25 hybrid retrieval을 추가하면 실제 RAG 환경에 더 가까운 실험이 가능하다.

### TODO

- BM25 index를 DP3 전용 EU 기준으로 생성한다.
- BM25 검색 결과와 vector 검색 결과를 비교한다.
- 필요하면 hybrid score를 적용한다.
  - vector score
  - BM25 score
  - weighted hybrid score
- A안 route filtering과 B안 context retrieval에 BM25/hybrid 옵션을 적용할지 검토한다.
- BM25/hybrid retrieval 시간을 별도 측정한다.
  - `bm25_ms`
  - `hybrid_merge_ms`
  - `hybrid_total_ms`

### 확인 기준

- vector-only, BM25-only, hybrid 결과를 같은 질문 세트에서 비교할 수 있어야 한다.
- cache 전략 비교와 retrieval 방식 비교가 섞여 해석되지 않도록 실험 조건을 분리한다.
- BM25 추가가 DP1/DP2 실행 방식에 영향을 주지 않아야 한다.
