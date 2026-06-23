# DP3 Cache PoC README

이 문서는 현재 구현된 DP3 Cache PoC를 다른 PC 또는 새 환경에서 실행하기 위한 사용 가이드다.

## 1. 현재 구현 범위

현재 PoC에는 다음 기능이 포함되어 있다.

- DP3 전용 LongBench/RAGBench 다운로드
- LongBench/RAGBench 예제 기반 ROI-RAG Evidence Unit 생성
- DP3 전용 원본 EU 저장
- `scope=A/B`, `version=V1/V2/V3`, `fingerprint` metadata 생성
- A안 Verified Answer Cache 테스트
- B안 Incremental Context Cache 테스트
- LongBench/RAGBench 질문풀 생성
- 웹 UI 기반 A/B/No-cache 테스트 실행

DP3 PoC는 DP1/DP2의 기존 실행 방식과 데이터를 직접 수정하지 않도록 DP3 전용 loader, DB table, UI를 사용한다.

## 2. 기능 스펙

현재 DP3 PoC의 주요 기능 스펙은 다음과 같다.

| 항목 | 현재 구현 | 비고 |
|---|---|---|
| Dataset | LongBench, RAGBench | LongBench는 개발 baseline, RAGBench는 평가용 |
| Evidence Unit | ROI-RAG 기반 EU | 실패 시 fallback chunk 생성 가능 |
| Embedding | `all-MiniLM-L6-v2` | sentence-transformers 기반, 미설치 시 hash embedding fallback |
| A안 Routing | 질문풀 vector 유사도 | TODO: BM25 또는 hybrid routing 검토 |
| A안 Cache | Verified Answer Cache | answer + source metadata 저장 |
| B안 Cache | Incremental Context Cache | context pack + source metadata 저장 |
| Metadata | `scope`, `version`, `fingerprint`, `logical_eu_id` | 권한/버전 validation 기준 |
| RAG | ROI-RAG EU 대상 vector retrieval | DP3 전용 versioned EU table 기준 |
| RAG Ranking | similarity score sort + optional cross-encoder reranking | reranker 옵션은 UI에서 켜고 끌 수 있음 |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` | 기본 후보 수 30, 최종 top-k 5 |
| Retrieval 확장 | vector-only | TODO: BM25, vector+BM25 hybrid 추가 |
| LLM | Mock 또는 Groq API | 기본 실험 후보: `llama-3.1-8b-instant` |
| Timing | 구간별 `timings_ms` 저장 | embedding/routing/cache/validation/RAG/LLM/total |

## 3. Test Case 기본 구성

최종 DP3 평가는 다음 Test Case를 기준으로 한다. HTML에서 Test Case를 고르면 아래 값이 기본값으로 적용된다.

| TC | 목적 | 기본 Dataset | 기본 질문 수 |
|---|---|---|---:|
| TC1 Cache Benefit | Cache off / miss / hit 비용 비교 | RAGBench `techqa` | 50 |
| TC2 Mixed Workload Performance | 권한/버전 혼합 + 유사질문 set workload | RAGBench `emanual` | 32 |
| TC3 Scale Cost | 데이터셋 규모 증가에 따른 RAG 비용 증가 확인 | RAGBench `techqa` | 30 |
| TC4 Similar Query Pair Quality | cache-hit pair에서 A/B 답변 재사용 방식 비교 | RAGBench `emanual` | 18 |

TC2는 eManual test split에서 생성한 유사질문 set을 사용한다. 현재 생성 기준으로 8개 set, 총 32개 질문이 만들어진다.

```text
same
paraphrase
near_miss
random
```

TC4는 eManual test split에서 생성한 cache-hit pair를 사용한다. 현재 생성 기준으로 9개 pair, 총 18개 질문이 만들어진다. 각 pair는 질문 embedding 유사도가 높지만 reference answer가 서로 달라야 한다. TC4의 목적은 A안이 answer-level cache reuse로 동일 답변을 반환할 위험이 있고, B안은 context-level cache reuse 후 질문별로 답변을 다시 만들 수 있음을 확인하는 것이다.

### RAG 세부 단계

현재 retrieval은 다음 단계로 측정한다.

```text
DB 후보 조회
-> embedding cosine scoring
-> score sort
-> optional cross-encoder reranking
-> top-k 선택
```

현재 timing 필드는 다음 의미로 사용한다.

```text
rag_db_ms                     후보 EU 조회 시간
rag_scoring_ms                query-context cosine score 계산 시간
rag_score_sort_ms             vector score 기준 정렬 시간
rag_rerank_ms                 cross-encoder reranking 시간
rag_total_ms                  위 RAG 단계 합산 시간
full_retrieval_*              B안 full retrieval 구간 시간
delta_retrieval_*             B안 delta retrieval 구간 시간
llm_ms                        LLM 호출 시간
total_ms                      요청 전체 처리 시간
```

주의: reranker 옵션을 끄면 `rerank_ms`는 0에 가깝고, score sort 시간은 `score_sort_ms` 계열 필드에 기록된다.

## 4. 필요 환경

권장 환경은 다음과 같다.

```text
Python 3.11
네트워크 연결
src/data/ 디렉터리 쓰기 권한
```

최초 실행 시 다음 항목이 자동으로 다운로드 또는 생성될 수 있다.

- LongBench 원본 데이터
- sentence-transformers embedding model
- DP3 전용 SQLite DB 데이터
- ROI-RAG 기반 Evidence Unit
- DP3 versioned metadata

## 5. 최초 환경 설정

프로젝트 루트에서 실행한다.

```powershell
uv venv .venv311 --python 3.11
.venv311\Scripts\python -m pip install -r src\requirements.txt
```

`uv`가 없다면 Python 3.11 venv를 직접 만들어도 된다.

```powershell
python -m venv .venv311
.venv311\Scripts\python -m pip install -r src\requirements.txt
```

## 6. 환경 변수

기본값은 `src/.env.example`을 참고한다.

DP3 PoC는 기본적으로 mock LLM을 사용한다.

```text
DP3_MOCK_LLM=true
DP3_LLM_PROVIDER=mock
SQLITE_DB_PATH=./data/poc.db
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

현재 테스트 목적이라면 `DP3_MOCK_LLM=true`를 유지하면 된다. 실제 LLM을 연결할 때는 기존 DP1/DP2 방식의 LLM 설정을 사용하되, DP3 mock 설정을 끄면 된다.

```text
DP3_MOCK_LLM=false
DP3_LLM_PROVIDER=default
```

Groq API를 사용할 경우 서버 환경에 다음 값을 설정한다. Groq는 OpenAI-compatible endpoint를 사용한다.

```text
DP3_LLM_PROVIDER=groq
GROQ_API_KEY=your_groq_api_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama-3.1-8b-instant
```

웹 UI의 `LLM 설정`에서 Mock 또는 Groq 모델을 선택할 수 있다. API key는 브라우저에 입력하지 않고 서버의 `.env` 또는 환경 변수에만 둔다.

현재 UI의 LLM 선택지는 다음 네 가지다.

```text
Mock
Llama 8B - llama-3.1-8b-instant
Llama 70B - llama-3.3-70b-versatile
Qwen 27B - qwen/qwen3.6-27b
```

기본값은 `Mock`이다. 실제 Groq latency를 측정할 때는 `Llama 8B - llama-3.1-8b-instant`를 우선 사용한다.

## 7. 서버 실행

프로젝트 루트에서 다음 명령을 실행한다.

```powershell
cd src
..\ .venv311\Scripts\python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

PowerShell에서 위 명령의 공백이 불편하면 다음처럼 실행한다.

```powershell
cd src
..\ .venv311\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

정상 실행 후 브라우저에서 접속한다.

```text
http://127.0.0.1:8000/dp3
```

VSCode Live Preview로 `dp3.html` 파일을 직접 열면 API 경로가 다르게 잡힐 수 있으므로, 가능하면 FastAPI 서버 주소로 접속한다.

## 8. 웹 UI 사용 순서

권장 실행 순서는 다음과 같다.

1. `Prepare`
   - LongBench 데이터가 없으면 다운로드한다.
   - 선택한 dataset과 예제 수 기준으로 DP3 전용 EU를 생성한다.
   - ROI-RAG 사용 가능 환경이면 ROI-RAG 기반 EU를 만든다.
   - 필요한 경우 V1/V2/V3 metadata를 생성한다.

2. `Seed Pool`
   - LongBench 질문에서 A안 route filtering용 질문풀을 생성한다.
   - 질문풀 비율과 seed를 조정할 수 있다.

3. `Run A Test`
   - A안 Verified Answer Cache를 실행한다.
   - route threshold와 cache threshold를 사용한다.
   - V1 첫 실행, V1 반복, V2 검증 pass를 수행한다.

4. `Run B Test`
   - B안 Incremental Context Cache를 실행한다.
   - route filtering 없이 cache threshold만 사용한다.
   - V1 첫 실행, V1 반복, V2 검증 pass를 수행한다.

5. `Run A+B`
   - 동일한 질문 샘플로 A안과 B안을 연속 실행한다.

## 9. A안 동작 요약

A안은 Answer Cache 전략이다.

```text
질문 입력
-> 질문풀 route filtering
-> route threshold 미만이면 fallback
-> Answer Cache 후보 검색
-> cache threshold 이상 후보 validation
-> scope/version/fingerprint valid이면 answer 재사용
-> invalid이면 ROI-RAG fallback 후 새 answer cache 저장
```

A안에서 `Cache Hit`은 단순 후보 발견이 아니라, validation까지 통과해서 answer를 재사용한 경우를 의미한다.

## 10. B안 동작 요약

B안은 Context Cache 전략이다.

```text
질문 입력
-> route filtering 없이 Context Cache 후보 검색
-> cache threshold 이상 후보 validation
-> 모든 context source가 valid이면 context pack 재사용
-> 일부 invalid이면 delta retrieval 또는 full retrieval
-> context pack 기반으로 LLM 호출
```

B안은 context pack을 재사용하더라도 최종 answer 생성을 위해 LLM 호출은 수행한다.

Delta retrieval 정책은 현재 다음과 같다.

```text
top-k 중 invalid source가 절반 이상이면 full retrieval
invalid source가 절반 미만이면 invalid_count * 2 만큼 후보 검색
기존 valid/invalid logical_eu_id와 중복 제거
필요 개수만큼 보충되면 rebuilt context pack 생성
보충이 부족하면 full retrieval fallback
```

## 11. 생성되는 로컬 파일

다음 파일과 디렉터리는 로컬 실행 산출물이다.

```text
src/data/longbench/
src/data/longbench.zip
src/data/poc.db
```

이 파일들은 git에 포함하지 않는 것을 전제로 한다. 다른 PC에서 처음 실행하면 다시 생성된다.

## 12. 현재 제약과 주의점

- 시간 로그는 `log_json.timings_ms`와 batch summary의 `timing_avg_ms`에 저장된다.
- 현재 `rerank_ms`는 score sort 시간이며, 별도 reranker 모델은 아직 적용하지 않았다.
- mock LLM 환경에서는 실제 LLM latency를 평가할 수 없다.
- B안 delta retrieval은 데이터와 mutation 조건에 따라 full fallback으로 많이 떨어질 수 있다.
- A안 route threshold가 낮으면 route pass가 과도하게 많아질 수 있다.
- 질문풀이 LongBench 전체 질문의 일부 샘플이므로, 최종 실험 전에는 sampling 비율과 seed를 고정해야 한다.

추가 작업 목록은 `DP3_PoC_TODO.md`를 참고한다.
