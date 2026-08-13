# Workflow.md - RAG Compare PoC 구현 지시서

## 목차

1. [문서 목적](#1-문서-목적)
2. [과제 목표](#2-과제-목표)
3. [구현 범위](#3-구현-범위)
4. [기술 스택](#4-기술-스택)
   - 4.1 [실제 LLM 사용 방식](#41-실제-llm-사용-방식)
5. [디렉터리 구조](#5-디렉터리-구조)
6. [전체 아키텍처](#6-전체-아키텍처)
7. [핵심 설계 원칙](#7-핵심-설계-원칙)
8. [사용자 시나리오](#8-사용자-시나리오)
9. [주요 화면 설계](#9-주요-화면-설계)
10. [백엔드 API 설계](#10-백엔드-api-설계)
11. [데이터 모델](#11-데이터-모델)
12. [RAG 파이프라인 설계](#12-rag-파이프라인-설계)
13. [SWE-bench 검색 설계](#13-swe-bench-검색-설계)
14. [DP3 Cache 설계](#14-dp3-cache-설계)
15. [환경변수](#15-환경변수)
16. [구현 순서](#16-구현-순서)
17. [오류 처리 기준](#17-오류-처리-기준)
18. [성능 및 품질 지표](#18-성능-및-품질-지표)
19. [실행 방법](#19-실행-방법)
20. [완료 기준](#20-완료-기준)
21. [LLM 구현 지시 요약](#21-llm-구현-지시-요약)
22. [상세 구현 파라미터](#22-상세-구현-파라미터)
23. [실제 사용 LLM 및 모델 명세](#23-실제-사용-llm-및-모델-명세)
24. [API Request/Response 상세 명세](#24-api-requestresponse-상세-명세)
25. [화면별 UI 구성 상세](#25-화면별-ui-구성-상세)
26. [DP3 Test Case 상세 로직](#26-dp3-test-case-상세-로직)
27. [현재 src 기준 파일별 구현 책임](#27-현재-src-기준-파일별-구현-책임)
28. [재현 정확도 기준](#28-재현-정확도-기준)

## 1. 문서 목적

이 문서는 `src/` 하위에 구현할 RAG 비교 PoC 웹 애플리케이션의 요구사항, 아키텍처, 사용자 시나리오, API, 데이터 모델, 구현 순서를 한 파일로 정리한 작업 지시서이다.

LLM 기반 개발 에이전트는 이 문서를 기준으로 프론트엔드와 백엔드를 구현해야 한다. 구현 결과는 사용자가 대화 데이터를 만들고, 여러 RAG 전략을 인덱싱하고, 질문별 검색/생성 품질과 지연 시간을 비교할 수 있는 로컬 웹 애플리케이션이어야 한다.

## 2. 과제 목표

본 과제의 목표는 단순한 채팅 앱이 아니라, RAG 아키텍처 설계 의사결정을 실험할 수 있는 PoC 도구를 만드는 것이다.

핵심 목표는 다음과 같다.

- 사용자가 직접 입력하거나 AI가 생성한 대화 데이터를 SQLite에 저장한다.
- 저장된 대화 데이터를 세션 단위 또는 주제 스레드 단위로 묶는다.
- 동일한 질문에 대해 Basic RAG, RAPTOR RAG, ROI-RAG 결과를 나란히 비교한다.
- 각 RAG 방식의 검색 결과, 최종 답변, retrieval latency, generation latency, total latency를 확인한다.
- LongBench 기반 문답 평가와 SWE-bench Lite 기반 코드 검색 평가를 수행한다.
- DP3 Cache PoC 화면에서 Answer Cache, Context Cache, 유사 질문 pair 품질, RAGAS 평가를 실험한다.
- 모든 화면은 FastAPI가 정적 HTML로 제공하며, 프론트엔드는 별도 빌드 도구 없이 vanilla HTML/CSS/JavaScript로 구현한다.

## 3. 구현 범위

### 3.1 포함 범위

- FastAPI 백엔드 서버
- SQLite 기반 영속 데이터 저장
- ChromaDB 기반 벡터 인덱스 저장
- OpenAI 또는 Ollama 기반 LLM 호출 추상화
- sentence-transformers 기반 로컬 임베딩
- 채팅/세션 관리 화면
- 스레드 자동 그룹핑 및 스레드 인덱싱
- 3-way RAG 비교 화면
- 멀티 모델 비교 화면
- LongBench 데이터셋 로드 및 벤치마크 화면
- SWE-bench Lite 인덱싱/검색/평가 화면
- DP3 Cache PoC 실험 화면
- 실행 스크립트와 `.env.example`

### 3.2 제외 범위

- 운영 배포용 인증/인가
- 대규모 사용자 동시 접속 처리
- 프론트엔드 SPA 프레임워크 도입
- 외부 클라우드 DB 연동
- 프로덕션 수준의 마이그레이션 도구

## 4. 기술 스택

| 영역 | 선택 기술 | 목적 |
|---|---|---|
| Backend | FastAPI | API 서버 및 HTML 정적 파일 라우팅 |
| Frontend | HTML, CSS, vanilla JavaScript | 빌드 없는 단일 파일 UI |
| DB | SQLite | 세션, 메시지, 평가 결과 저장 |
| Vector DB | ChromaDB | RAG별 벡터 컬렉션 저장 |
| Embedding | sentence-transformers | 로컬 텍스트 임베딩 |
| LLM | OpenAI, Ollama, Groq, Mock | 답변 생성 및 PoC 실험 |
| Benchmark Data | datasets, JSON/JSONL | LongBench, RAGBench, SWE-bench Lite |
| Evaluation | 자체 metric, RAGAS optional | 정확도/검색 품질/캐시 효과 측정 |

## 4.1 실제 LLM 사용 방식

이 PoC의 구현 및 실험 기준은 다음과 같이 LLM provider를 분리한다.

| 사용 영역 | 실제 사용 방식 | 설명 |
|---|---|---|
| 일반 RAG 비교 | Ollama 로컬 모델 | `/chat`, `/compare`, Basic RAG, RAPTOR RAG, ROI-RAG의 답변 생성은 로컬 PC 또는 서버에 Ollama로 로드한 모델을 사용한다. |
| 멀티모델 비교 | Ollama 로컬 모델 목록 | `/api/models`에서 Ollama에 등록된 모델 목록을 가져오고, 선택한 복수 모델로 동일 질문을 반복 실행한다. |
| LongBench 평가 | Ollama 로컬 모델 | 벤치마크 질문에 대한 답변 생성도 기본적으로 Ollama 로컬 모델을 사용한다. |
| SWE-bench PoC | LLM 생성보다 검색 평가 중심 | SWE-bench 화면은 코드 chunk 검색과 hit/context metric 비교가 중심이며, RAG 방식과 검색 전략 조합을 로컬 인덱스 기준으로 평가한다. |
| DP3 Cache PoC | Groq 또는 Mock | DP3의 Answer Cache/Context Cache 실험에서 실제 외부 LLM 호출이 필요한 경우 Groq를 사용한다. 대량 실험과 반복 검증은 비용과 rate limit을 피하기 위해 Mock provider를 사용할 수 있다. |
| OpenAI | 옵션/대체 provider | 코드상 지원은 남겨두되, 본 PoC 실험 기준에서는 주된 실행 경로가 아니다. |

즉, 본 과제에서 “일반 RAG 기능”은 로컬 Ollama 모델 기반으로 테스트하고, “DP3 Cache PoC”만 Groq 기반 외부 LLM 호출을 별도 사용한 것으로 정리한다.

## 5. 디렉터리 구조

```text
src/
  backend/
    main.py
    config.py
    db/
      database.py
    routers/
      chat.py
      threads.py
      rag_compare.py
      benchmark.py
      swebench.py
      cache_poc.py
      agent.py
    rag/
      basic_rag.py
      raptor_rag.py
      roi_rag.py
      swebench_rag_engines.py
      llm_client.py
      _ef.py
    cache/
      answer_cache.py
      context_cache.py
      cache_llm.py
    agent/
      conversation_gen.py
    scripts/
      load_swebench.py
  frontend/
    index.html
    chat.html
    compare.html
    swebench.html
    dp3.html
    static/css/style.css
  data/
  .env.example
  requirements.txt
  run.sh
```

## 6. 전체 아키텍처

```text
[Browser]
  |-- /                 -> index.html
  |-- /chat             -> chat.html
  |-- /compare          -> compare.html
  |-- /swebench         -> swebench.html
  |-- /dp3              -> dp3.html
  |
  | fetch('/api/...')
  v
[FastAPI backend.main]
  |-- chat router       -> session/message CRUD
  |-- threads router    -> thread grouping/indexing
  |-- rag router        -> Basic/RAPTOR/ROI compare
  |-- benchmark router  -> LongBench load/evaluate
  |-- swebench router   -> SWE-bench index/retrieve/evaluate
  |-- cache_poc router  -> DP3 cache experiments
  |-- agent router      -> AI conversation generation
  |
  +--> [SQLite: src/data/poc.db]
  +--> [ChromaDB: src/data/chroma/]
  +--> [LLM Provider: OpenAI/Ollama/Groq/Mock]
  +--> [Embedding Model: all-MiniLM-L6-v2]
```

## 7. 핵심 설계 원칙

1. 로컬 실행 우선
   - 사용자는 `cd src && ./run.sh`로 서버를 실행할 수 있어야 한다.
   - 기본 포트는 `8000`이다.

2. 화면별 독립 HTML
   - `/chat`, `/compare`, `/swebench`, `/dp3`는 각각 독립 HTML 파일로 구현한다.
   - 공통 다크 테마는 `frontend/static/css/style.css`를 사용한다.
   - DP3 화면은 실험 화면 특성상 독립 스타일을 허용한다.

3. API-first 구조
   - 프론트엔드는 `fetch()`로 FastAPI API만 호출한다.
   - 화면 상태는 JavaScript에서 관리하되, 영속 상태는 SQLite와 ChromaDB에 저장한다.

4. RAG 방식 분리
   - Basic RAG, RAPTOR RAG, ROI-RAG는 별도 모듈로 분리한다.
   - 인덱싱 컬렉션명은 방식과 대상 단위를 포함한다.
   - 예: `basic_s_{session_id}`, `raptor_t_{thread_id}`, `roi_t_{thread_id}`

5. 실험 가능성 우선
   - 지연 시간, 참조 청크, 정답 여부, cache hit 여부 등 실험 지표를 화면에 노출한다.
   - 대량 실험에서는 Mock LLM을 사용할 수 있게 한다.

## 8. 사용자 시나리오

### 8.1 시나리오 A - 대화 데이터 생성 및 RAG 비교

1. 사용자는 `/chat` 화면에 접속한다.
2. `새 세션`을 눌러 2인 대화 또는 그룹 채팅 세션을 만든다.
3. 사용자는 메시지를 직접 입력하거나 `AI 대화 생성`을 사용해 주제 기반 대화를 자동 생성한다.
4. 세션을 선택한 후 `RAG 인덱싱`을 실행한다.
5. 백엔드는 해당 세션 메시지를 Basic RAG와 RAPTOR RAG로 인덱싱한다.
6. 사용자는 `/compare` 화면으로 이동한다.
7. 세션을 선택하고 질문을 입력한다.
8. 화면은 Basic RAG와 RAPTOR RAG 답변, 참조 문서, 지연 시간을 표시한다.

### 8.2 시나리오 B - 여러 세션을 스레드로 묶어 3-way 비교

1. 사용자는 `/chat` 화면에서 여러 세션을 만든다.
2. `자동 생성`을 눌러 `[DayN]` 패턴의 세션을 Day, Week, 전체 월간 스레드로 자동 그룹핑한다.
3. 특정 스레드를 선택하고 `RAG 인덱싱`을 실행한다.
4. 백엔드는 스레드에 포함된 모든 세션 메시지를 합친다.
5. Basic RAG, RAPTOR RAG, ROI-RAG를 모두 인덱싱한다.
6. 사용자는 `/compare`에서 스레드를 선택하고 질문을 입력한다.
7. 화면은 Basic/RAPTOR/ROI 3-way 결과를 보여준다.
8. ROI-RAG는 Evidence Unit 수, regime, retrieval/generation latency를 함께 보여준다.

### 8.3 시나리오 C - 멀티 모델 비교

1. 사용자는 `/compare` 화면의 `멀티모델 비교` 탭을 연다.
2. Ollama에서 조회된 모델 목록 중 복수 모델을 선택한다.
3. 동일한 컨텍스트와 질문으로 Basic/RAPTOR/ROI 결과를 모델별로 실행한다.
4. 결과 카드는 모델명, RAG 방식, 답변, 지연 시간을 표시한다.

### 8.4 시나리오 D - LongBench 기반 평가

1. 사용자는 `/compare` 화면의 `LongBench 평가` 탭을 연다.
2. 데이터셋 관리에서 LongBench 데이터셋 목록을 확인한다.
3. 필요한 데이터셋을 SQLite에 로드한다.
4. 로드된 데이터셋을 스레드로 인덱싱한다.
5. 벤치마크를 실행한다.
6. 백엔드는 ground truth와 생성 답변을 비교하여 Basic/RAPTOR/ROI별 정답 여부와 지연 시간을 저장한다.
7. 화면은 summary와 상세 결과를 표시한다.

### 8.5 시나리오 E - SWE-bench Lite 코드 검색 평가

1. 사용자는 `/swebench` 화면에 접속한다.
2. `인덱싱 시작`을 눌러 SWE-bench Lite 이슈/패치 데이터를 로드하고 인덱싱한다.
3. 인덱싱 방식은 Legacy, Basic RAG, RAPTOR RAG, ROI-RAG를 지원한다.
4. 검색 전략은 Flat, Post-Filter, Routed를 지원한다.
5. 사용자는 이슈를 선택하고 검색을 실행한다.
6. 화면은 각 조합별 Top-K 코드 청크, hit 여부, latency를 표시한다.
7. 사용자는 평가 실행으로 context precision, recall, MRR, hit rate를 비교한다.

### 8.6 시나리오 F - DP3 Cache PoC

1. 사용자는 `/dp3` 화면에 접속한다.
2. Dataset Family를 LongBench 또는 RAGBench로 선택한다.
3. Test Case를 선택한다.
   - TC1: Cache 동작 확인
   - TC2: 데이터셋 확장성
   - TC3: 유사질문 Pair 품질
4. 데이터셋 prepare, question pool seed를 실행한다.
5. Test Suite를 실행한다.
6. 백엔드는 Answer Cache 또는 Context Cache 경로를 실행하고 cache hit, route pass, validation, fallback, LLM 호출 수, 지연 시간을 기록한다.
7. 화면은 pass별 summary card, timing table, decision table, 로그를 표시한다.
8. 필요한 경우 Proxy RAGAS 또는 Official RAGAS를 실행해 품질 지표를 확인한다.

## 9. 주요 화면 설계

### 9.1 Home - `/`

목적:
- PoC의 진입점이다.
- 상단 네비게이션으로 주요 기능 화면에 접근한다.

필수 메뉴:
- 홈
- 채팅
- RAG 비교
- SWE-bench PoC
- DP3 Cache PoC
- GitHub 외부 링크

### 9.2 Chat - `/chat`

주요 기능:
- 세션 목록 조회
- 세션 생성/삭제
- 메시지 조회/추가
- AI 대화 생성
- 스레드 목록 조회
- Day/Week/전체 월간 기준 자동 스레드 생성
- 세션 또는 스레드 인덱싱

상태 처리:
- 세션 선택 시 메시지 입력 가능
- 스레드 선택 시 메시지는 읽기 전용으로 표시
- 인덱싱 중 버튼 비활성화
- 성공/실패 toast 표시

### 9.3 Compare - `/compare`

탭 구성:
- 3-Way 비교
- 멀티모델 비교
- LongBench 평가

3-Way 비교:
- 컨텍스트 선택 드롭다운에서 세션 또는 스레드를 선택한다.
- 세션은 Basic/RAPTOR만 표시하고 ROI는 미지원 상태로 처리한다.
- 스레드는 Basic/RAPTOR/ROI를 모두 표시한다.
- 참조 문서는 접기/펼치기와 더 보기 기능을 제공한다.

멀티모델 비교:
- `/api/models`로 모델 목록을 가져온다.
- 선택 모델별 결과를 카드로 표시한다.

LongBench 평가:
- 데이터셋 목록, 로드, 인덱싱, 미리보기, 벤치마크 실행, 이전 결과 조회를 제공한다.

### 9.4 SWE-bench - `/swebench`

주요 기능:
- 인덱싱 상태 확인
- 인덱싱 시작/중지
- 전체 삭제
- RAG 방식 선택: Legacy, BasicRAG, RaptorRAG, ROIRAG
- 검색 전략 선택: Flat, PostFilter, Routed
- 단일 이슈 검색
- 일괄 평가
- 조합별 동작 흐름 모달 표시

검색 결과:
- 조합별 컬럼/카드
- Hit 여부
- latency
- answer file과 매칭되는 chunk 표시
- route/filter 조건 표시

### 9.5 DP3 Cache PoC - `/dp3`

주요 기능:
- 테스트 케이스 선택
- Dataset Family, dataset, split, row count 설정
- route threshold, cache threshold 설정
- answer cache/context cache 모드 실행
- reranker 사용 여부 및 device/model 설정
- job start 후 polling으로 진행률 표시
- RAGAS 입력 파일 생성 및 평가 실행

## 10. 백엔드 API 설계

### 10.1 HTML 라우트

| Method | Path | 반환 |
|---|---|---|
| GET | `/` | `frontend/index.html` |
| GET | `/chat` | `frontend/chat.html` |
| GET | `/compare` | `frontend/compare.html` |
| GET | `/swebench` | `frontend/swebench.html` |
| GET | `/dp3` | `frontend/dp3.html` |

HTML 응답은 브라우저 캐시 문제를 줄이기 위해 `Cache-Control: no-cache` 헤더를 포함한다.

### 10.2 Chat API

| Method | Path | 설명 |
|---|---|---|
| GET | `/api/sessions` | 세션 목록 조회 |
| POST | `/api/sessions` | 세션 생성 |
| GET | `/api/sessions/{session_id}/messages` | 세션 메시지 조회 |
| POST | `/api/sessions/{session_id}/messages` | 메시지 추가 |
| DELETE | `/api/sessions/{session_id}` | 세션 삭제 |
| POST | `/api/sessions/{session_id}/index` | 세션 단위 Basic/RAPTOR 인덱싱 |
| POST | `/api/agent/generate` | AI 대화 생성 |

### 10.3 Thread API

| Method | Path | 설명 |
|---|---|---|
| GET | `/api/threads` | 스레드 목록 조회 |
| POST | `/api/threads` | 스레드 생성 |
| GET | `/api/threads/{thread_id}` | 스레드 상세 조회 |
| GET | `/api/threads/{thread_id}/messages` | 스레드 내 전체 메시지 조회 |
| POST | `/api/threads/{thread_id}/index` | Basic/RAPTOR/ROI 인덱싱 |
| POST | `/api/threads/auto-group` | Day/Week/전체 월간 자동 그룹핑 |

### 10.4 RAG Compare API

| Method | Path | 설명 |
|---|---|---|
| GET | `/api/models` | Ollama 모델 목록 조회 |
| POST | `/api/rag/compare` | Basic/RAPTOR/ROI 비교 실행 |
| POST | `/api/rag/multimodel` | 여러 모델에 대해 RAG 비교 실행 |
| GET | `/api/rag/results/{session_id}` | 세션 비교 이력 |
| GET | `/api/rag/results/thread/{thread_id}` | 스레드 비교 이력 |
| GET | `/api/rag/multimodel/results/{session_id}` | 멀티모델 비교 이력 |

### 10.5 LongBench Benchmark API

| Method | Path | 설명 |
|---|---|---|
| GET | `/api/benchmark/datasets` | 사용 가능한 데이터셋 목록 |
| GET | `/api/benchmark/datasets/{dataset_name}/view` | 데이터셋 미리보기 |
| POST | `/api/benchmark/load` | 데이터셋을 SQLite/스레드로 로드 |
| GET | `/api/threads/{thread_id}/benchmark/questions` | 벤치마크 질문 조회 |
| POST | `/api/threads/{thread_id}/benchmark/run` | 벤치마크 실행 |
| GET | `/api/threads/{thread_id}/benchmark/results` | 벤치마크 결과 조회 |

### 10.6 SWE-bench API

| Method | Path | 설명 |
|---|---|---|
| POST | `/api/swebench/clear` | SWE-bench 데이터 및 선택 인덱스 삭제 |
| GET | `/api/swebench/status` | 이슈/인덱스 상태 조회 |
| GET | `/api/swebench/issues` | 이슈 목록 |
| GET | `/api/swebench/issues/{instance_id}` | 이슈 상세 |
| POST | `/api/swebench/index` | SWE-bench 데이터 인덱싱 |
| POST | `/api/swebench/retrieve` | 단일 이슈 검색 |
| POST | `/api/swebench/evaluate` | 일괄 평가 |

### 10.7 DP3 Cache PoC API

| Method | Path | 설명 |
|---|---|---|
| POST | `/api/dp3/answer-cache/setup` | Answer Cache 메타데이터 초기화 |
| POST | `/api/dp3/answer-cache/run` | 단일 Answer Cache query 실행 |
| GET | `/api/dp3/longbench/datasets` | LongBench DP3 데이터셋 목록 |
| POST | `/api/dp3/longbench/prepare` | LongBench DP3 자산 준비 |
| GET | `/api/dp3/ragbench/datasets` | RAGBench 데이터셋 목록 |
| POST | `/api/dp3/ragbench/prepare` | RAGBench DP3 자산 준비 |
| GET | `/api/dp3/question-pool/stats` | route question pool 통계 |
| POST | `/api/dp3/question-pool/seed` | route question pool seed |
| POST | `/api/dp3/answer-cache/batch` | Answer Cache batch 실행 |
| POST | `/api/dp3/context-cache/batch` | Context Cache batch 실행 |
| POST | `/api/dp3/test-suite/run` | 동기 test suite 실행 |
| POST | `/api/dp3/test-suite/start` | 비동기 test suite 시작 |
| GET | `/api/dp3/test-suite/jobs/{job_id}` | test suite job 상태 조회 |
| POST | `/api/dp3/ragas/run` | RAGAS 평가 실행 |

## 11. 데이터 모델

### 11.1 기본 테이블

```sql
sessions(id, title, mode, created_at, is_indexed)
messages(id, session_id, speaker, content, timestamp)
rag_results(id, session_id, query, basic_rag_answer, basic_rag_latency_ms,
            raptor_rag_answer, raptor_rag_latency_ms, created_at)
model_compare_results(id, session_id, query, rag_type, model_name,
                      answer, latency_ms, created_at)
```

### 11.2 스레드 및 3-way 비교 테이블

```sql
threads(id, title, description, created_at,
        basic_indexed, raptor_indexed,
        basic_chunk_count, raptor_node_count,
        roi_indexed, roi_eu_count, roi_regime)

thread_sessions(thread_id, session_id, sort_order)

thread_rag_results(id, thread_id, query,
                   basic_rag_answer, basic_rag_latency_ms,
                   raptor_rag_answer, raptor_rag_latency_ms,
                   roi_rag_answer, roi_rag_latency_ms,
                   model_name, created_at,
                   basic_rag_retrieval_ms, basic_rag_generation_ms,
                   raptor_rag_retrieval_ms, raptor_rag_generation_ms,
                   roi_rag_retrieval_ms, roi_rag_generation_ms)
```

### 11.3 평가 테이블

```sql
benchmark_questions(id, thread_id, question, ground_truth_answers,
                    dataset_name, source_id, created_at)

benchmark_results(id, question_id, thread_id,
                  basic_rag_answer, basic_rag_latency_ms, basic_correct,
                  raptor_rag_answer, raptor_rag_latency_ms, raptor_correct,
                  roi_rag_answer, roi_rag_latency_ms, roi_correct,
                  model_name, created_at)

swebench_issues(instance_id, repo, version, problem_statement,
                answer_files, created_at)
```

### 11.4 DP3 Cache 테이블

DP3 Cache PoC는 별도 cache schema를 사용한다.

```sql
dp3_context_units
dp3_evidence_units
dp3_versioned_evidence_units
dp3_answerable_question_pool
dp3_answer_cache_entries
dp3_answer_cache_sources
dp3_answer_cache_logs
dp3_query_sets
dp3_context_cache_entries
dp3_context_cache_sources
```

## 12. RAG 파이프라인 설계

### 12.1 Basic RAG

목적:
- 기준선 RAG를 제공한다.

동작:
1. 메시지 또는 스레드 텍스트를 고정 크기 chunk로 분할한다.
2. ChromaDB collection에 chunk를 저장한다.
3. 질문 입력 시 top-k chunk를 cosine similarity로 검색한다.
4. 검색 chunk를 context로 LLM에 전달해 답변을 생성한다.
5. answer, references, retrieval_ms, generation_ms, latency_ms를 반환한다.

### 12.2 RAPTOR RAG

목적:
- 세부 chunk와 상위 요약 노드를 함께 검색해 포괄 질문 대응력을 높인다.

동작:
1. 원문을 chunk로 나눈다.
2. chunk embedding을 생성한다.
3. K-Means 또는 클러스터링으로 묶는다.
4. 클러스터별 LLM 요약 노드를 만든다.
5. 최대 level까지 재귀적으로 트리를 만든다.
6. leaf와 summary node를 모두 ChromaDB에 저장한다.
7. 질문 시 전체 level에서 검색하고, level metadata를 포함해 답변을 생성한다.

### 12.3 ROI-RAG

목적:
- 중복 정보가 많은 대화 데이터에서 Evidence Unit 단위로 정보 밀도를 높인다.

동작:
1. 스레드 텍스트를 segment로 나눈다.
2. segment embedding을 생성한다.
3. kNN neighborhood를 구성한다.
4. redundancy entropy(RE)와 diversity entropy(DE)를 계산한다.
5. Greedy 방식으로 Evidence Unit(EU)을 구성한다.
6. corpus-level regime을 LOW/MID/HIGH로 분류한다.
7. regime에 따라 요약 강도를 조절한다.
8. EU embedding을 ChromaDB에 저장한다.
9. 질문 시 단일 ANN 조회로 EU를 검색하고 답변을 생성한다.

제약:
- ROI-RAG는 세션 단위가 아니라 스레드 단위에서만 지원한다.

## 13. SWE-bench 검색 설계

SWE-bench는 일반 대화 RAG와 별도 코드 검색 실험이다.

RAG 방식:
- Legacy: 기존 patch chunk 방식
- BasicRAG: 고정 코드 chunk
- RaptorRAG: leaf chunk와 cluster node
- ROIRAG: 코드 segment 기반 Evidence Unit

검색 전략:
- Flat: 전체 collection에서 바로 검색
- PostFilter: 전체 검색 후 repo/version 기준으로 후처리 필터링
- Routed: repo/version partition collection 또는 route 조건을 먼저 적용

평가 지표:
- Hit: answer file이 top-k 결과에 포함되는지
- Context Precision@K
- Context Recall@K
- MRR
- Latency

## 14. DP3 Cache 설계

DP3 Cache PoC는 RAG 응답 비용과 지연 시간을 줄이기 위한 캐시 전략 실험이다.

### 14.1 Answer Cache

동작:
1. query embedding을 생성한다.
2. answerable question pool에서 route를 찾는다.
3. route score가 threshold 미만이면 RAG fallback을 수행한다.
4. route가 통과되면 동일 route 내 answer cache 후보를 찾는다.
5. cache similarity와 version match를 확인한다.
6. cache source Evidence Unit fingerprint를 검증한다.
7. 검증 통과 시 cached answer를 반환한다.
8. 실패 시 RAG fallback으로 새 답변을 생성하고 cache에 저장한다.

핵심 로그:
- route_ms
- cache_lookup_ms
- validation_passed
- cache_hit
- decision_reason
- llm_call_count
- total_ms

### 14.2 Context Cache

동작:
1. query와 유사한 cached context pack을 찾는다.
2. cache source가 현재 Evidence Unit과 유효한지 검증한다.
3. 모두 유효하면 cached context로 답변을 생성한다.
4. 일부 source가 invalid이면 delta retrieval로 부족한 context만 보강한다.
5. invalid 비율이 높거나 delta가 부족하면 full retrieval로 fallback한다.

### 14.3 Reranker

옵션:
- `use_reranker`
- `rerank_candidates`
- `rerank_model`
- `rerank_device`

목적:
- vector search 후보를 cross-encoder reranker로 재정렬해 cache fallback retrieval 품질을 높인다.

## 15. 환경변수

`.env.example`에는 다음 값을 제공한다.

```env
OPENAI_API_KEY=your_openai_api_key_here
LLM_PROVIDER=openai
OPENAI_MODEL=gpt-4o-mini

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3

EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./data/chroma
SQLITE_DB_PATH=./data/poc.db

DP3_MOCK_LLM=true
DP3_LLM_PROVIDER=mock
GROQ_API_KEY=your_groq_api_key_here
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=llama-3.1-8b-instant
GROQ_MIN_INTERVAL_SECONDS=2.2
GROQ_MAX_RETRIES=5
GROQ_MAX_OUTPUT_TOKENS=256
GROQ_RATE_LIMIT_SAFETY=0.85
```

규칙:
- 일반 RAG 비교, 멀티모델 비교, LongBench 평가는 `LLM_PROVIDER=ollama` 기준으로 로컬 Ollama 모델을 사용한다.
- `/api/models`는 Ollama에 로드된 모델 목록을 조회해 프론트엔드 선택지로 제공한다.
- `LLM_PROVIDER=openai`는 대체 실행 옵션으로만 둔다.
- DP3 Cache PoC에서 외부 LLM이 필요한 실험은 `DP3_LLM_PROVIDER=groq`와 `GROQ_API_KEY`를 사용한다.
- DP3 대량 실험은 기본적으로 `DP3_MOCK_LLM=true` 또는 `DP3_LLM_PROVIDER=mock`을 사용해 비용과 rate limit을 피한다.
- 따라서 발표/보고 시에는 “DP3는 Groq, 그 외 RAG 비교 및 평가 기능은 로컬 Ollama 모델 기반으로 테스트”했다고 설명한다.

## 16. 구현 순서

### Phase 1 - 서버와 데이터 기반

1. `backend/config.py` 구현
2. `backend/db/database.py` 구현
3. FastAPI `backend/main.py` 구현
4. 정적 HTML 라우팅과 `/static` mount 구현
5. `.env.example`, `requirements.txt`, `run.sh` 작성

완료 기준:
- `cd src && ./run.sh`로 서버가 실행된다.
- `/`, `/chat`, `/compare` HTML이 열리는 상태가 된다.

### Phase 2 - 채팅과 세션

1. `sessions`, `messages` 테이블 구현
2. 세션 CRUD API 구현
3. 메시지 조회/추가 API 구현
4. `chat.html` UI 구현
5. AI 대화 생성 API와 modal UI 구현

완료 기준:
- 브라우저에서 세션을 만들고 메시지를 저장할 수 있다.
- 생성된 메시지가 새로고침 후에도 유지된다.

### Phase 3 - Basic/RAPTOR RAG

1. `basic_rag.py` 구현
2. `raptor_rag.py` 구현
3. 세션 인덱싱 API 구현
4. `/api/rag/compare` 구현
5. `compare.html` 3-way 비교 탭의 Basic/RAPTOR 패널 구현

완료 기준:
- 세션을 인덱싱한 뒤 질문하면 Basic/RAPTOR 답변과 latency가 표시된다.

### Phase 4 - 스레드와 ROI-RAG

1. `threads`, `thread_sessions`, `thread_rag_results` 테이블 추가
2. 스레드 CRUD와 자동 그룹핑 구현
3. `roi_rag.py` 구현
4. 스레드 인덱싱에서 Basic/RAPTOR/ROI를 모두 실행
5. `/compare`에서 thread context 선택 시 ROI 패널 표시

완료 기준:
- 스레드 단위 질문에서 Basic/RAPTOR/ROI 3-way 비교가 가능하다.

### Phase 5 - LongBench 평가

1. benchmark dataset 목록 조회 구현
2. dataset preview 구현
3. dataset load 구현
4. benchmark question/result 테이블 구현
5. benchmark run/result API 구현
6. `compare.html`의 LongBench 평가 탭 구현

완료 기준:
- 데이터셋 로드, 인덱싱, 벤치마크 실행, 결과 조회가 가능하다.

### Phase 6 - SWE-bench PoC

1. SWE-bench loader 구현
2. `swebench_issues` 테이블 구현
3. `swebench_rag_engines.py` 구현
4. `/api/swebench/index`, `/retrieve`, `/evaluate` 구현
5. `swebench.html` UI 구현

완료 기준:
- 이슈별 Top-K 검색과 조합별 평가 결과가 표시된다.

### Phase 7 - DP3 Cache PoC

1. `answer_cache.py` schema와 query flow 구현
2. `context_cache.py` schema와 query flow 구현
3. `cache_llm.py` provider abstraction 구현
4. `cache_poc.py`에 LongBench/RAGBench prepare, question pool, batch, suite, RAGAS API 구현
5. `dp3.html` UI 구현

완료 기준:
- DP3 화면에서 TC1/TC2/TC3 suite를 실행하고 결과 summary와 로그를 확인할 수 있다.

## 17. 오류 처리 기준

- 존재하지 않는 세션/스레드는 404를 반환한다.
- 메시지가 없는 세션/스레드 인덱싱은 400을 반환한다.
- 인덱싱되지 않은 context에 대한 비교 요청은 400을 반환한다.
- 외부 LLM 오류는 화면에 toast 또는 결과 카드 오류 메시지로 표시한다.
- 대량 작업은 가능한 경우 job id를 반환하고 polling으로 상태를 조회한다.

## 18. 성능 및 품질 지표

공통 지표:
- `retrieval_ms`
- `generation_ms`
- `latency_ms`
- reference count
- model name

RAG 평가 지표:
- answer correctness
- hit rate
- context precision
- context recall
- MRR

Cache 평가 지표:
- route pass ratio
- cache hit ratio
- validation pass ratio
- fallback ratio
- LLM call count
- estimated total latency
- mocked/real LLM provider 구분

## 19. 실행 방법

```bash
cd src
cp .env.example .env
./run.sh
```

수동 실행:

```bash
cd src
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python seed_data.py
PYTHONPATH=$(pwd) uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

Windows PowerShell에서 직접 실행하는 경우:

```powershell
cd src
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python seed_data.py
$env:PYTHONPATH = (Get-Location).Path
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## 20. 완료 기준

아래 조건을 만족하면 구현 완료로 본다.

- `/chat`에서 세션 생성, 메시지 입력, AI 대화 생성, 인덱싱이 가능하다.
- `/compare`에서 세션/스레드를 선택하고 RAG 비교가 가능하다.
- 스레드 인덱싱 시 Basic/RAPTOR/ROI 결과가 모두 생성된다.
- `/compare`의 LongBench 탭에서 데이터셋 관리와 평가가 가능하다.
- `/swebench`에서 인덱싱, 단일 검색, 일괄 평가가 가능하다.
- `/dp3`에서 DP3 Cache PoC test suite와 RAGAS 평가를 실행할 수 있다.
- SQLite DB와 ChromaDB는 `src/data/` 아래에 저장된다.
- `.env.example`만으로 필요한 설정 항목을 파악할 수 있다.
- 화면 상단 네비게이션에서 모든 주요 페이지에 접근할 수 있다.

## 21. LLM 구현 지시 요약

LLM 개발 에이전트는 다음 원칙으로 구현한다.

1. 먼저 `backend/main.py`, `config.py`, `database.py`로 실행 가능한 골격을 만든다.
2. 세션/메시지 CRUD를 완성한 뒤 Basic/RAPTOR RAG를 붙인다.
3. 스레드와 ROI-RAG는 세션 기능이 동작한 뒤 확장한다.
4. 벤치마크와 SWE-bench는 일반 RAG 비교 기능과 독립된 실험 화면으로 구현한다.
5. DP3 Cache PoC는 Answer Cache와 Context Cache의 의사결정 로그를 가장 중요하게 다룬다.
6. 프론트엔드는 빌드 도구 없이 HTML 파일마다 필요한 JavaScript를 포함한다.
7. 구현 중 확정되지 않은 수치나 외부 데이터셋 경로는 `TODO` 또는 명시적 기본값으로 남긴다.
8. 변경 후에는 최소한 라우트 접근, API 호출, 주요 버튼 동작을 수동 검증한다.

## 22. 상세 구현 파라미터

이 섹션의 값은 현재 PoC 재현을 위해 기본값으로 사용한다. 별도 튜닝 요청이 없으면 LLM 개발 에이전트는 아래 값을 그대로 구현한다.

### 22.1 Basic RAG

| 항목 | 값 |
|---|---|
| Chunk size | `512` characters |
| Chunk overlap | `80` characters |
| Top-K | `5` |
| Collection prefix(session) | `basic_s_{session_id}` |
| Collection prefix(thread) | `basic_t_{thread_id}` |
| Chunking rule | 마침표/줄바꿈 단위 우선 분리, 초과 시 고정 길이 강제 분할 |
| Output | `answer`, `references`, `retrieval_ms`, `generation_ms`, `latency_ms` |

### 22.2 RAPTOR RAG

| 항목 | 값 |
|---|---|
| Chunk size | `512` characters |
| Chunk overlap | `80` characters |
| Top-K | `5` |
| Max levels | `3` |
| Min cluster size | `2` |
| Collection prefix(session) | `raptor_s_{session_id}` |
| Collection prefix(thread) | `raptor_t_{thread_id}` |
| Tree node | leaf chunk + cluster summary node |
| Summary generation | LLM으로 cluster 요약 생성 |

RAPTOR 인덱싱은 chunk embedding → cluster label 생성 → cluster별 summary 생성 → summary를 상위 level 입력으로 재귀 처리하는 방식이다. 검색 시 leaf와 summary node가 모두 후보가 된다.

### 22.3 ROI-RAG

| 항목 | 값 |
|---|---|
| Chunk size | `300` characters |
| Chunk overlap | `50` characters |
| kNN K | `10` |
| Max segments per EU | `6` |
| Top-K | `5` |
| High RE threshold | `0.01` |
| Mid RE threshold | `0.003` |
| Redundancy tau | `0.6` |
| Collection prefix(thread) | `roi_t_{thread_id}` |

ROI-RAG regime 분류:

| 조건 | Regime |
|---|---|
| `RE >= 0.01` | `HIGH` |
| `0.003 <= RE < 0.01` | `MID` |
| `RE < 0.003` | `LOW` |

ROI-RAG는 thread 단위에서만 지원한다. session 단위 비교에서는 ROI 패널을 비활성 또는 미지원 상태로 처리한다.

### 22.4 SWE-bench RAG Engine

| 항목 | 값 |
|---|---|
| Top-K | `3` |
| PostFilter prefetch K | `10` |
| Chunk size | `512` |
| Chunk overlap | `80` |
| Engines | `Legacy`, `BasicRAG`, `RaptorRAG`, `ROIRAG` |
| Strategies | `Flat`, `PostFilter`, `Routed` |

SWE-bench collection naming:

| Engine | Flat collection |
|---|---|
| `Legacy` | `swebench_flat` |
| `BasicRAG` | `sweb_basic` |
| `RaptorRAG` | `sweb_raptor` |
| `ROIRAG` | `sweb_roi` |

### 22.5 DP3 Cache

| 항목 | 값 |
|---|---|
| Default route threshold | `0.70` |
| Default cache threshold | `0.86` |
| Top-K sources | `5` |
| Default rerank candidates | `30` |
| Default reranker model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Default reranker device | `auto` |
| Default user scope | `A` |
| Default route pool mode | `sampled` |
| Default sample rate | `0.10` |
| Default pool seed | `42` |
| Default test seed | `7` |
| Default warmup count | `3` |

## 23. 실제 사용 LLM 및 모델 명세

본 PoC는 일반 RAG 실험과 DP3 Cache 실험의 LLM provider를 의도적으로 분리한다.

### 23.1 일반 RAG, LongBench, 멀티모델 비교

| 항목 | 값 |
|---|---|
| Provider | `Ollama` |
| Base URL | `http://localhost:11434` |
| Default model env | `OLLAMA_MODEL=llama3` |
| Model list source | `/api/models`가 Ollama 로컬 모델 목록을 조회 |
| 사용 방식 | 로컬에 미리 pull/load된 Ollama 모델을 선택해 테스트 |

일반 RAG 비교에서 OpenAI는 코드상 fallback/provider 옵션으로 남겨두지만, 과제 설명에서는 “로컬 Ollama 모델 기반으로 테스트했다”고 설명한다.

### 23.2 DP3 Cache PoC

| 항목 | 값 |
|---|---|
| Provider | `Groq` 또는 `Mock` |
| Groq base URL | `https://api.groq.com/openai/v1` |
| 기본 Groq model | `llama-3.1-8b-instant` |
| 추가 후보 | `llama-3.3-70b-versatile`, `qwen/qwen3.6-27b` |
| Mock 사용 목적 | 대량 batch, 반복 테스트, rate limit 회피 |
| 실제 latency 검증 | Groq `llama-3.1-8b-instant` 기준 |

DP3 RAGAS evaluator 후보:

| 용도 | 모델 |
|---|---|
| Official RAGAS 기본 evaluator | `meta-llama/llama-4-scout-17b-16e-instruct` |
| Fallback 1 | `llama-3.3-70b-versatile` |
| Fallback 2 | `qwen/qwen3-32b` |
| Fallback 3 | `qwen/qwen3.6-27b` |

보고서/발표 문구:

```text
일반 RAG 비교와 LongBench 평가는 로컬 Ollama에 로드한 모델들로 수행했고,
DP3 Cache PoC의 실제 외부 LLM latency 검증은 Groq의 llama-3.1-8b-instant를 사용했다.
대량 반복 실험은 비용과 rate limit 영향을 제거하기 위해 Mock LLM 경로를 병행했다.
```

## 24. API Request/Response 상세 명세

### 24.1 Chat

`POST /api/sessions`

Request:

```json
{
  "title": "Day 1 - 온보딩 대화",
  "mode": "2person"
}
```

Response:

```json
{
  "id": "uuid",
  "title": "Day 1 - 온보딩 대화",
  "mode": "2person"
}
```

`POST /api/sessions/{session_id}/messages`

Request:

```json
{
  "speaker": "A",
  "content": "오늘 온보딩에서 가장 헷갈린 부분은 무엇인가요?"
}
```

Response:

```json
{ "ok": true }
```

`POST /api/agent/generate`

Request:

```json
{
  "mode": "group",
  "topic": "신입 구성원의 첫 달 업무 적응 과정",
  "turns": 20,
  "speakers": ["PM", "개발자", "멘토"],
  "model": "llama3"
}
```

Response:

```json
{
  "session_id": "uuid",
  "ok": true
}
```

### 24.2 Thread

`POST /api/threads`

Request:

```json
{
  "title": "Week 1 - 온보딩",
  "description": "Day 1~5 세션 통합",
  "session_ids": ["uuid-1", "uuid-2"]
}
```

Response:

```json
{
  "id": "uuid",
  "title": "Week 1 - 온보딩"
}
```

`POST /api/threads/{thread_id}/index`

Request:

```json
{
  "model": "llama3"
}
```

Response:

```json
{
  "ok": true,
  "message_count": 120,
  "basic_chunk_count": 35,
  "raptor_node_count": 48,
  "roi_eu_count": 22,
  "roi_regime": "MID"
}
```

### 24.3 RAG Compare

`POST /api/rag/compare`

Request for session:

```json
{
  "session_id": "uuid",
  "query": "이 대화에서 주요 의사결정은 무엇인가요?",
  "model": "llama3"
}
```

Request for thread:

```json
{
  "thread_id": "uuid",
  "query": "Week 1의 핵심 이슈와 해결 방향을 요약해줘.",
  "model": "llama3"
}
```

Response:

```json
{
  "query": "질문",
  "model": "llama3",
  "context_type": "thread",
  "basic_rag": {
    "answer": "...",
    "latency_ms": 1200,
    "retrieval_ms": 40,
    "generation_ms": 1160,
    "references": []
  },
  "raptor_rag": {
    "answer": "...",
    "latency_ms": 1500,
    "retrieval_ms": 60,
    "generation_ms": 1440,
    "references": []
  },
  "roi_rag": {
    "answer": "...",
    "latency_ms": 900,
    "retrieval_ms": 35,
    "generation_ms": 865,
    "references": [],
    "regime": "MID",
    "eu_count": 22
  }
}
```

`POST /api/rag/multimodel`

Request:

```json
{
  "thread_id": "uuid",
  "query": "핵심 이슈를 비교해줘.",
  "models": ["llama3", "qwen-local"]
}
```

Response:

```json
{
  "query": "핵심 이슈를 비교해줘.",
  "results": [
    {
      "rag_type": "basic",
      "model": "llama3",
      "answer": "...",
      "latency_ms": 1000,
      "references": []
    }
  ]
}
```

### 24.4 LongBench

`POST /api/benchmark/load`

Request:

```json
{
  "dataset_name": "multifieldqa_en",
  "num_examples": 5
}
```

Expected response:

```json
{
  "ok": true,
  "thread_id": "uuid",
  "dataset_name": "multifieldqa_en",
  "loaded": 5
}
```

`POST /api/threads/{thread_id}/benchmark/run`

Request:

```json
{
  "model": "llama3"
}
```

Response는 question별 Basic/RAPTOR/ROI 답변, latency, correctness와 summary를 포함한다.

### 24.5 SWE-bench

`POST /api/swebench/retrieve`

Request:

```json
{
  "instance_id": "django__django-12345",
  "query": "problem statement text",
  "rag_methods": ["Legacy", "BasicRAG", "RaptorRAG", "ROIRAG"],
  "strategies": ["Flat", "PostFilter", "Routed"]
}
```

Response:

```json
{
  "results": [
    {
      "retriever": "BasicRAG + Routed",
      "chunks": [],
      "hit": true,
      "latency_ms": 12.3,
      "criteria": "repo/version routed collection"
    }
  ]
}
```

### 24.6 DP3 Cache

`POST /api/dp3/test-suite/start`

Request:

```json
{
  "test_case": "cache",
  "dataset_family": "longbench",
  "dataset_name": "multifieldqa_en",
  "dataset_split": "test",
  "num_examples": 5,
  "query_count": 100,
  "seed": 7,
  "warmup_count": 3,
  "user_scope": "A",
  "route_threshold": 0.70,
  "cache_threshold": 0.86,
  "route_pool_mode": "sampled",
  "sample_rate": 0.10,
  "min_per_dataset": 5,
  "pool_seed": 42,
  "llm_provider": "mock",
  "model": "llama-3.1-8b-instant",
  "use_reranker": false,
  "rerank_candidates": 30,
  "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
  "reset_metadata": false,
  "max_scale": 5,
  "include_smoke": false
}
```

Response:

```json
{
  "job_id": "uuid",
  "status": "running"
}
```

`GET /api/dp3/test-suite/jobs/{job_id}`

Response:

```json
{
  "job_id": "uuid",
  "status": "running|succeeded|failed",
  "progress": {
    "current": 10,
    "total": 100,
    "label": "TC1 실행 중"
  },
  "result": {}
}
```

## 25. 화면별 UI 구성 상세

### 25.1 공통 네비게이션

모든 주요 화면은 상단에서 다음 순서로 접근 가능해야 한다.

```text
홈 | 채팅 | RAG 비교 | SWE-bench PoC | DP3 Cache PoC | GitHub
```

GitHub는 외부 링크이며, 내부 기능 메뉴 오른쪽에 배치한다.

### 25.2 Chat UI

필수 영역:
- 좌측 스레드 목록
- 좌측 세션 목록
- 세션/스레드 상세 메시지 영역
- 메시지 입력 영역
- 새 세션 modal
- AI 대화 생성 modal

필수 버튼:
- `자동 생성`: Day/Week/전체 월간 스레드 자동 생성
- `+ 새 세션`: 세션 생성 modal open
- `AI 대화 생성`: LLM 기반 대화 생성 modal open
- `RAG 인덱싱`: 선택한 세션 또는 스레드 인덱싱
- `전송`: 현재 세션에 메시지 추가

### 25.3 Compare UI

필수 영역:
- context select: session과 thread를 함께 표시
- model select: Ollama model 목록
- query input
- 실행 버튼
- tab bar: `3-Way 비교`, `멀티모델 비교`, `LongBench 평가`

3-Way 비교 패널:
- Basic RAG column
- RAPTOR RAG column
- ROI-RAG column
- 각 column은 answer, latency, retrieval/generation 분리 지표, references를 표시한다.

LongBench 탭:
- dataset manager
- dataset preview modal
- benchmark run button
- previous result button
- result summary table

### 25.4 SWE-bench UI

필수 영역:
- 상태 패널
- 인덱싱 컨트롤
- RAG method chip group
- strategy chip group
- issue selector
- 단일 검색 결과 grid
- 평가 결과 summary
- 동작 흐름 modal

조합 수는 최대 `4 engines x 3 strategies = 12`개이다.

### 25.5 DP3 UI

필수 영역:
- Test Case select
- Dataset Family select
- Dataset/Split/Row count inputs
- Route threshold slider
- Cache threshold slider
- LLM provider/model 선택
- Reranker toggle/candidates/model/device 설정
- Prepare button
- Seed Pool button
- Test Suite start button
- Progress/status card
- Summary cards
- Timing table
- Decision table
- RAGAS action panel
- Raw log panel

## 26. DP3 Test Case 상세 로직

### 26.1 TC1 - Cache 동작 확인

목적:
- Answer Cache와 Context Cache가 동일/유사 질문에서 cache hit를 만드는지 확인한다.

기본 흐름:
1. source dataset을 prepare한다.
2. route pool을 seed한다.
3. warmup query를 실행해 embedding/reranker cold start 영향을 줄인다.
4. Pass A를 실행한다.
   - cache reset 상태에서 query batch를 실행한다.
   - 대부분 fallback path가 발생하고 cache entry가 저장된다.
5. Pass B를 실행한다.
   - 동일 또는 유사 query batch를 다시 실행한다.
   - route 통과, cache lookup, validation을 거쳐 cache hit가 증가해야 한다.
6. Answer Cache와 Context Cache 결과를 각각 summary로 비교한다.

핵심 확인 항목:
- Pass B의 cache hit ratio가 Pass A보다 높아야 한다.
- fallback이 발생한 경우 decision reason을 로그로 남긴다.
- Mock LLM 실행 시에도 estimated Groq latency를 함께 계산할 수 있다.

### 26.2 TC2 - 데이터셋 확장성

목적:
- dataset row 수 또는 scale이 증가할 때 route/cache/retrieval latency가 어떻게 변하는지 확인한다.

기본 흐름:
1. `max_scale`까지 scale별 source를 구성한다.
2. 각 scale마다 metadata prepare와 route pool seed를 수행한다.
3. 동일 query count 기준으로 Answer Cache와 Context Cache batch를 실행한다.
4. scale별 total_ms, route_ms, cache_lookup_ms, retrieval_ms, generation_ms를 집계한다.

핵심 확인 항목:
- scale 증가에 따른 latency 변화
- cache hit ratio 변화
- reranker 사용 여부에 따른 추가 latency

### 26.3 TC3 - 유사질문 Pair 품질

목적:
- 의미적으로 유사한 질문 pair에서 cache reuse가 답변 품질을 해치지 않는지 확인한다.

제약:
- `dataset_family=ragbench`
- `dataset_name=techqa` 또는 `emanual`
- query pair asset이 사전에 준비되어 있어야 한다.

기본 흐름:
1. RAGBench pair asset을 읽는다.
2. left/right query를 pair로 구성한다.
3. Pass A에서 left/right를 각각 실행해 baseline answer를 만든다.
4. Pass B에서 cache reuse 경로를 실행한다.
5. left/right 답변 동등성, context overlap, proxy RAGAS metric을 계산한다.
6. Official RAGAS는 선택 실행으로 둔다.

핵심 확인 항목:
- similar pair cache hit 여부
- answer equality 또는 proxy similarity
- context precision/faithfulness 경향
- RAGAS evaluator token usage

## 27. 현재 src 기준 파일별 구현 책임

| 파일 | 구현 책임 |
|---|---|
| `src/backend/main.py` | FastAPI app 생성, CORS, DB 초기화, router 등록, HTML route, `/static` mount |
| `src/backend/config.py` | `.env` 및 Windows user env fallback, LLM/DB/Chroma/Groq 설정 |
| `src/backend/db/database.py` | SQLite schema, migration, `get_conn()`, thread text 조립 |
| `src/backend/routers/chat.py` | session/message CRUD, session indexing |
| `src/backend/routers/threads.py` | thread CRUD, auto-group, thread indexing |
| `src/backend/routers/rag_compare.py` | model list, RAG compare, multimodel compare, result history |
| `src/backend/routers/benchmark.py` | LongBench dataset list/view/load/run/results |
| `src/backend/routers/swebench.py` | SWE-bench clear/status/issues/index/retrieve/evaluate |
| `src/backend/routers/cache_poc.py` | DP3 prepare, question pool, batch, suite job, RAGAS |
| `src/backend/routers/agent.py` | AI conversation generation endpoint |
| `src/backend/rag/basic_rag.py` | Basic chunking/index/query |
| `src/backend/rag/raptor_rag.py` | RAPTOR tree build/index/query |
| `src/backend/rag/roi_rag.py` | ROI Evidence Unit build/index/query |
| `src/backend/rag/swebench_rag_engines.py` | SWE-bench RAG engine and strategy matrix |
| `src/backend/rag/llm_client.py` | OpenAI/Ollama LLM abstraction |
| `src/backend/cache/answer_cache.py` | DP3 Answer Cache schema and query flow |
| `src/backend/cache/context_cache.py` | DP3 Context Cache schema and delta retrieval flow |
| `src/backend/cache/cache_llm.py` | DP3 Mock/Groq LLM abstraction, rate limit handling |
| `src/backend/agent/conversation_gen.py` | topic/speaker/turn 기반 대화 생성 |
| `src/frontend/index.html` | 홈 화면 |
| `src/frontend/chat.html` | chat/session/thread UI |
| `src/frontend/compare.html` | 3-way/multimodel/LongBench UI |
| `src/frontend/swebench.html` | SWE-bench UI |
| `src/frontend/dp3.html` | DP3 Cache PoC UI |
| `src/frontend/static/css/style.css` | 공통 dark theme |
| `src/run.sh` | 로컬 실행 스크립트 |
| `src/.env.example` | 실행 환경변수 템플릿 |

## 28. 재현 정확도 기준

`workflow.md`만으로 구현하는 LLM 에이전트는 다음 수준까지 재현해야 한다.

- 동일한 URL 구조와 주요 화면을 제공한다.
- 동일한 API path와 핵심 request/response field를 제공한다.
- Basic/RAPTOR/ROI/SWE-bench/DP3의 기본 파라미터를 동일하게 사용한다.
- SQLite와 ChromaDB 저장 위치를 동일하게 사용한다.
- 일반 RAG는 Ollama 로컬 모델, DP3는 Groq/Mock provider 분리 구조를 따른다.
- DP3 test suite는 TC1/TC2/TC3의 목적과 pass 구조를 유지한다.

단, 다음은 구현 환경에 따라 달라질 수 있다.

- 로컬 Ollama에 실제 설치된 모델 이름
- 외부 dataset 다운로드 가능 여부
- Groq rate limit과 queue latency
- RAGAS evaluator 모델 fallback 결과
- ChromaDB 내부 저장 형식
