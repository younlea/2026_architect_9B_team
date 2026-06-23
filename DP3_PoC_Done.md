# DP3 PoC Done

이 문서는 DP3 Cache PoC에서 이미 구현했거나 1차 확인이 끝난 작업을 정리한다. 앞으로 남은 작업은 `DP3_PoC_TODO.md`에서 관리한다.

## 1. 시간 계측 세분화

상태: 완료

A/B안 모두 `log_json.timings_ms`에 세부 구간별 시간을 저장하고, batch/test-suite summary에서 `avg/min/max/count` 형태로 확인할 수 있게 했다.

### A안 Answer Cache

측정 가능한 주요 구간은 다음과 같다.

- `embedding_ms`
- `route_ms`
- `cache_lookup_ms`
- `validation_ms`
- `rag_db_ms`
- `rag_scoring_ms`
- `rag_score_sort_ms`
- `rag_rerank_ms`
- `rag_total_ms`
- `prompt_build_ms`
- `llm_ms`
- `cache_store_ms`
- `total_ms`

### B안 Context Cache

측정 가능한 주요 구간은 다음과 같다.

- `embedding_ms`
- `cache_lookup_db_ms`
- `cache_lookup_scoring_ms`
- `cache_lookup_ms`
- `validation_ms`
- `valid_current_lookup_ms`
- `full_retrieval_db_ms`
- `full_retrieval_scoring_ms`
- `full_retrieval_score_sort_ms`
- `full_retrieval_rerank_ms`
- `full_retrieval_total_ms`
- `delta_retrieval_db_ms`
- `delta_retrieval_scoring_ms`
- `delta_retrieval_score_sort_ms`
- `delta_retrieval_rerank_ms`
- `delta_retrieval_filter_ms`
- `delta_retrieval_total_ms`
- `prompt_build_ms`
- `llm_ms`
- `cache_store_ms`
- `total_ms`

## 2. 시간 결과 이상치 1차 평가

상태: 완료

A안에서 cache hit 경로가 예상보다 느리게 보이던 현상은 per-query setup 비용과 route pool DB load/embedding JSON decode 비용이 online latency에 포함된 영향으로 확인했다.

조치한 내용:

- DP3 metadata/setup 비용을 preflight로 분리했다.
- route pool을 메모리 캐시로 전환했다.
- 테스트 통계에서 setup 비용과 online query latency를 구분하도록 정리했다.

해석 기준:

- 최종 비교는 N개 query의 online 처리 평균을 기준으로 본다.
- embedding model, reranker model cold start는 warm-up query로 제외한다.
- wall-clock 전체 실행 시간과 결과 표의 per-query timing은 구분해서 해석한다.

## 3. 권한/버전 혼합 테스트

상태: 완료

HTML test suite에 `TC2. 권한/버전 혼합 수행시간`을 추가했다.

구현 내용:

- 동일 질문 리스트에 대해 A/B안을 각각 실행한다.
- 질문별로 `scope=A/B`, `requested_version=V1/V2/V3`를 섞어 요청한다.
- A안은 route/cache/validator/fallback을 측정한다.
- B안은 cache hit, full validation, partial validation, full RAG, delta RAG를 측정한다.
- 결과는 A/B 각각 `avg/min/max/count` timing 표로 확인한다.

## 4. Cross-Encoder Reranking

상태: 완료

`cross-encoder/ms-marco-MiniLM-L-6-v2` 기반 reranking 옵션을 추가했다.

구현 내용:

- 1차 vector retrieval 후보를 가져온다.
- 선택 시 cross-encoder가 `query-context` pair를 재평가한다.
- rerank score 기준으로 재정렬한 뒤 최종 top-k를 선택한다.
- HTML에서 reranker 사용 여부와 후보 수를 조정할 수 있다.
- A/B안 모두 rerank 시간을 별도 timing field로 저장한다.

현재 기본 해석:

- CPU 환경에서는 reranking 비용이 크게 나타날 수 있다.
- cache hit 시 reranking 비용을 회피하는 효과를 확인할 수 있다.
- 기능 테스트에서는 후보 수 10, 기본 비교에서는 10~30 범위를 권장한다.

## 5. Groq LLM 연동

상태: 완료

Mock LLM 외에 Groq API를 선택할 수 있게 했다.

현재 UI 선택지:

- Mock
- `llama-3.1-8b-instant`
- `llama-3.3-70b-versatile`
- `qwen/qwen3.6-27b`

기본 후보는 `llama-3.1-8b-instant`로 본다. API key는 `GROQ_API_KEY` 환경변수로 읽는다.

## 6. Test Suite UI

상태: 완료

`src/frontend/dp3.html`에서 다음 3개 테스트 케이스를 선택 실행할 수 있게 했다.

- TC1. Cache 동작 확인
- TC2. 권한/버전 혼합 수행시간
- TC3. 데이터셋 확장성

결과 표시:

- A안 route 통과 비율
- A/B cache hit 비율
- validator 통과 비율
- B안 partial validation 비율
- full RAG / delta RAG 비율
- embedding, routing, cache, validation, RAG, reranking, LLM timing의 `avg/min/max/count`

## 7. Background Job 및 진행률 표시

상태: 완료

긴 테스트 실행 중 화면이 멈춘 것처럼 보이지 않도록 background job 방식으로 바꿨다.

구현 내용:

- `/api/dp3/test-suite/start`
- `/api/dp3/test-suite/jobs/{job_id}`
- HTML에서 1초 주기 polling
- 진행 단계, 완료 개수, 전체 개수, progress bar 표시

제약:

- LongBench/EU 준비 단계 내부는 아직 세부 progress가 없다.
- 준비 단계가 끝난 뒤에는 query/pass 단위로 진행률이 갱신된다.

## 8. DP1/DP2 영향 분리

상태: 완료

DP3 PoC는 기존 DP1/DP2 실행 방식을 직접 변경하지 않도록 구성했다.

분리한 내용:

- DP3 전용 LongBench loader
- DP3 전용 SQLite table
- DP3 전용 HTML
- DP3 전용 cache metadata
- DP3 전용 query/test runner

주의:

- local DB, LongBench 원본, 전처리 결과는 로컬 실행 산출물이다.
- DP3 테스트용 산출물은 DP1/DP2 benchmark 실행 흐름과 분리해서 관리한다.
