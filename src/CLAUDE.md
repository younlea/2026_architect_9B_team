# CLAUDE.md — RAG Compare POC 프로젝트 가이드

## 프로젝트 개요

Basic RAG vs RAPTOR RAG 성능 비교 POC 웹 애플리케이션.
모든 소스코드는 `src/` 디렉터리에 위치합니다.

## 핵심 명령어

```bash
# 서버 실행 (권장)
cd src && ./run.sh

# 서버 직접 실행
cd src && PYTHONPATH=$(pwd) uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload

# 샘플 데이터 로드만 실행
cd src && python seed_data.py
```

## 코드 수정 시 주의사항

- `PYTHONPATH`를 `src/`로 설정해야 `backend.*` import가 동작합니다.
- DB 파일: `src/data/poc.db` (SQLite), ChromaDB: `src/data/chroma/`
- `.env` 파일은 `src/` 내에 위치합니다 (`src/.env`).

## 주요 파일 역할

| 파일 | 역할 |
|------|------|
| `src/backend/main.py` | FastAPI 앱 + 정적 파일 서빙 |
| `src/backend/rag/basic_rag.py` | Basic RAG: 고정 청킹, ChromaDB cosine 검색 |
| `src/backend/rag/raptor_rag.py` | RAPTOR: 재귀 요약 트리, K-Means 클러스터링, 다층 검색 |
| `src/backend/rag/llm_client.py` | OpenAI / Ollama 공통 인터페이스 |
| `src/backend/db/database.py` | SQLite 초기화, `get_conn()` context manager |
| `src/backend/routers/chat.py` | 세션·메시지 CRUD, `/index` 엔드포인트 |
| `src/backend/routers/rag_compare.py` | `/rag/compare` — Basic + RAPTOR 비동기 동시 실행 |
| `src/backend/routers/agent.py` | AI 에이전트 대화 자동 생성 |
| `src/frontend/compare.html` | 좌(Basic) / 우(RAPTOR) 분할 비교 UI |
| `src/frontend/chat.html` | 채팅 + 세션 관리 UI |

## DB 스키마

```sql
sessions  (id, title, mode, created_at, is_indexed)
messages  (id, session_id, speaker, content, timestamp)
rag_results (id, session_id, query, basic_rag_answer, basic_rag_latency_ms, raptor_rag_answer, raptor_rag_latency_ms, created_at)
```

## RAG 파이프라인 흐름

```
[채팅 입력] → SQLite 저장
     ↓
[/index 호출]
  ├── basic_rag.index_session()  → 고정 청킹 → ChromaDB (basic_{session_id})
  └── raptor_rag.index_session() → 청킹 → K-Means 클러스터링 → LLM 요약 → 트리 구성 → ChromaDB (raptor_{session_id})
     ↓
[/rag/compare 호출] (asyncio.gather로 병렬 실행)
  ├── basic_rag.query()   → top-5 검색 → LLM 답변
  └── raptor_rag.query()  → 전체 레벨 검색 → 레벨별 컨텍스트 조합 → LLM 답변
     ↓
결과를 rag_results 테이블에 저장 + 프론트엔드에 반환
```

## 환경변수 (`src/.env`)

```
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai          # openai | ollama
OPENAI_MODEL=gpt-4o-mini
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./data/chroma
SQLITE_DB_PATH=./data/poc.db
```

## 의존성 주요 패키지

- `fastapi` + `uvicorn` — 웹 서버
- `chromadb` — 벡터 저장소
- `sentence-transformers` — 로컬 임베딩 (`all-MiniLM-L6-v2`)
- `scikit-learn` — K-Means 클러스터링 (RAPTOR 트리 구성)
- `openai` — LLM API 클라이언트

## 자주 발생하는 문제

| 문제 | 원인 | 해결 |
|------|------|------|
| `ModuleNotFoundError: backend` | PYTHONPATH 미설정 | `PYTHONPATH=$(pwd) uvicorn ...` |
| 인덱싱 중 느림 | RAPTOR가 LLM을 재귀 호출 | 정상 동작 (긴 대화는 2~3분 소요) |
| ChromaDB 충돌 | 동시에 두 프로세스 실행 | 서버 하나만 실행 |
| `.env` 인식 안 됨 | 실행 위치 문제 | `cd src` 후 실행 |
