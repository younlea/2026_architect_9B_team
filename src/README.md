# 2026 Architect 9B Team — RAG Compare POC

Basic RAG와 RAPTOR RAG의 성능을 나란히 비교하는 POC 웹 애플리케이션입니다.
대화 데이터를 수집하고, 두 RAG 방식의 검색 품질과 응답 속도를 실시간으로 비교합니다.

---

## 프로젝트 구조

```
src/
├── backend/
│   ├── main.py                   # FastAPI 앱 진입점
│   ├── config.py                 # 환경변수 로드
│   ├── routers/
│   │   ├── chat.py               # 세션/메시지 CRUD, RAG 인덱싱 트리거
│   │   ├── rag_compare.py        # Basic + RAPTOR 동시 비교 API
│   │   └── agent.py              # AI 에이전트 대화 자동 생성 API
│   ├── rag/
│   │   ├── basic_rag.py          # 고정 청킹 + cosine 검색 RAG
│   │   ├── raptor_rag.py         # 재귀 요약 트리 + 다층 검색 RAPTOR
│   │   └── llm_client.py         # OpenAI / Ollama 공통 LLM 클라이언트
│   ├── db/
│   │   └── database.py           # SQLite 초기화 및 연결 관리
│   └── agent/
│       └── conversation_gen.py   # AI 에이전트 대화 생성 로직
├── frontend/
│   ├── index.html                # 홈 (랜딩 페이지)
│   ├── chat.html                 # 채팅 UI + 세션 관리
│   ├── compare.html              # RAG 좌/우 비교 UI
│   └── static/css/style.css      # 다크 테마 스타일시트
├── data/
│   ├── sample_2person.json       # 2인 대화 샘플 (3개 세션)
│   └── sample_group.json         # 그룹 대화 샘플 (2개 세션)
├── seed_data.py                  # 샘플 데이터 DB 로드 스크립트
├── run.sh                        # 서버 실행 스크립트
├── requirements.txt              # Python 의존성
└── .env.example                  # 환경변수 템플릿
```

---

## 빠른 시작

### 1. 환경 설정

```bash
cd src
cp .env.example .env
```

`.env` 파일을 열어 `OPENAI_API_KEY`를 입력합니다.

```env
OPENAI_API_KEY=sk-...
LLM_PROVIDER=openai          # openai 또는 ollama
OPENAI_MODEL=gpt-4o-mini
```

Ollama 로컬 LLM을 사용하는 경우:

```env
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3
```

### 2. 서버 실행

```bash
cd src
./run.sh
```

스크립트가 자동으로 수행하는 작업:
- Python 가상환경 생성 (`.venv/`)
- `requirements.txt` 패키지 설치
- 샘플 대화 데이터 DB 로드 (`seed_data.py`)
- FastAPI 서버 구동 (`0.0.0.0:8000`)

### 3. 수동 실행 (run.sh 없이)

```bash
cd src
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python seed_data.py
PYTHONPATH=$(pwd) uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 접속 주소

| 환경 | URL |
|------|-----|
| 로컬 | `http://localhost:8000` |
| 외부 | `http://<서버IP>:8000` |
| 채팅 | `http://<서버IP>:8000/chat` |
| RAG 비교 | `http://<서버IP>:8000/compare` |

---

## 사용 흐름

```
1. /chat  →  새 세션 생성 또는 AI 에이전트로 대화 자동 생성
2. /chat  →  [RAG 인덱싱] 버튼 클릭 (Basic + RAPTOR 동시 인덱싱)
3. /compare  →  세션 선택 → 질문 입력 → [비교 실행]
4. 좌(Basic RAG) / 우(RAPTOR RAG) 결과 동시 확인
```

---

## 주요 기능

### 채팅 앱 (`/chat`)
- 2인 대화 / 그룹 채팅 세션 생성
- 화자 선택 후 메시지 입력
- AI 에이전트 대화 자동 생성 (주제/화자/턴 수 설정)
- 세션별 RAG 인덱싱 트리거

### RAG 비교 (`/compare`)
- 좌/우 분할 화면으로 두 RAG 결과 나란히 비교
- 응답 지연시간(ms) 실시간 표시
- 참조 청크 토글 (어떤 데이터를 참조했는지 확인)
- 비교 결과 이력 조회

### AI 에이전트 대화 생성
- LLM이 지정된 주제로 대화를 자동 생성
- 2인 대화 (페르소나 기반) / 그룹 대화 지원
- 생성 즉시 DB 저장 및 RAG 인덱싱 가능

---

## RAG 방식 비교

| 항목 | Basic RAG | RAPTOR RAG |
|------|-----------|------------|
| 구조 | 1차원 평면 청크 | 재귀 요약 트리 (최대 3단계) |
| 청킹 | 고정 크기 (300자, 50자 오버랩) | 동일 + 상위 노드에 LLM 요약 추가 |
| 검색 | cosine similarity top-5 | 전체 트리 레벨 동시 검색 |
| 포괄 질문 | 취약 | 강점 (전체 요약 노드 활용) |
| 인덱싱 비용 | 낮음 | 높음 (LLM 재귀 호출) |

---

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| GET | `/api/sessions` | 세션 목록 |
| POST | `/api/sessions` | 세션 생성 |
| GET | `/api/sessions/{id}/messages` | 메시지 조회 |
| POST | `/api/sessions/{id}/messages` | 메시지 추가 |
| POST | `/api/sessions/{id}/index` | RAG 인덱싱 실행 |
| POST | `/api/rag/compare` | Basic + RAPTOR 비교 실행 |
| GET | `/api/rag/results/{session_id}` | 비교 결과 이력 |
| POST | `/api/agent/generate` | AI 대화 자동 생성 |

---

## 관련 자료

- [RAPTOR 리뷰](homework/0.RAPTOR_Hierarchical_리뷰.md)
- [데이터셋 분석](homework/0.dataset.md)
- [서비스 주제](homework/1.주제_service.md)
