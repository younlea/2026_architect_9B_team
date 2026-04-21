# RAG Compare POC — 사용 가이드 & 비교 방법

## 목차

1. [앱 전체 구조](#1-앱-전체-구조)
2. [서버 실행](#2-서버-실행)
3. [채팅 페이지 사용법](#3-채팅-페이지-사용법)
4. [RAG 비교 페이지 사용법](#4-rag-비교-페이지-사용법)
5. [LongBench 벤치마크 평가](#5-longbench-벤치마크-평가)
6. [효과적인 비교를 위한 질문 전략](#6-효과적인-비교를-위한-질문-전략)
7. [결과 해석 방법](#7-결과-해석-방법)
8. [현재 데이터 현황](#8-현재-데이터-현황)

---

## 1. 앱 전체 구조

```
채팅 페이지 (/chat)          RAG 비교 페이지 (/compare)
     │                              │
     ▼                              ▼
 대화 데이터 입력        Basic RAG vs RAPTOR RAG 답변 비교
 세션/스레드 관리         멀티모델 비교 / LongBench 평가
     │
     ▼
 RAG 인덱싱 (벡터DB 구축)
  ├── Basic RAG  → ChromaDB (고정 청크)
  └── RAPTOR RAG → ChromaDB (요약 트리)
```

| 페이지 | 주소 | 용도 |
|--------|------|------|
| 홈 | `/` | 앱 개요 |
| 채팅 | `/chat` | 대화 데이터 관리, 인덱싱 |
| RAG 비교 | `/compare` | 질문 입력 후 Basic vs RAPTOR 비교 |

---

## 2. 서버 실행

```bash
cd src
./run.sh
# 또는
PYTHONPATH=$(pwd) uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

접속: `http://localhost:8000`

---

## 3. 채팅 페이지 사용법

### 3-1. 스레드 vs 세션 개념

| 개념 | 설명 |
|------|------|
| **세션** | 단일 대화. 화자 2인 또는 그룹. |
| **스레드** | 여러 세션을 하나의 컨텍스트로 묶은 것. RAG 입력이 더 풍부해짐. |

스레드를 쓰면 예: "Week 1 — 온보딩" 에 속한 6개 세션 전체(140개 메시지)가 하나의 벡터DB로 인덱싱되어 더 맥락이 풍부한 검색이 가능합니다.

### 3-2. 스레드 자동 생성

1. `/chat` 접속 → 왼쪽 사이드바 상단 **[자동 생성]** 버튼 클릭
2. Day별(2세션 이상인 날), Week별(5주), 전체 월간 스레드가 자동 생성됩니다.

```
생성되는 스레드 예시:
  Day 1 — 전체 대화       (2개 세션, 48개 메시지)
  Week 1 — 온보딩         (6개 세션, 140개 메시지)
  전체 월간 대화 통합      (28개 세션, 577개 메시지)
  [LongBench] multifieldqa_en (5개 예제, 518개 메시지)
```

### 3-3. RAG 인덱싱

스레드 또는 세션을 선택 → **[📥 RAG 인덱싱]** 버튼 클릭

- 인덱싱 완료 후 하단 패널에 청크 수 / 노드 수 표시
- **Basic RAG**: 청크 수가 많을수록 세밀한 검색 가능
- **RAPTOR RAG**: 노드 수 = 원본 청크 + 요약 노드 (항상 청크 수보다 많음)

```
예시 (전체 월간 스레드, 577개 메시지):
  🔵 Basic RAG  → 약 210개 청크
  🟣 RAPTOR RAG → 약 290개 노드 (요약 계층 포함)
```

인덱싱 후 **[📊 비교]** 버튼이 활성화되고 클릭 시 해당 스레드가 선택된 비교 페이지로 이동합니다.

---

## 4. RAG 비교 페이지 사용법

### 4-1. 컨텍스트 선택

상단 드롭다운에서 스레드 또는 세션 선택:

```
📚 주제별 스레드 (인덱싱됨)  ← 이 항목만 선택 가능
  [LongBench] multifieldqa_en
  Day 1 — 전체 대화
  전체 월간 대화 통합

💬 개별 세션 (인덱싱됨)
  [Day1] 신입사원 OJT 첫날
  ...
```

- `?thread=<id>` 또는 `?session=<id>` URL 파라미터로 직접 접근 가능

### 4-2. 탭 1 — Basic vs RAPTOR 비교

```
[컨텍스트 선택] → [LLM 모델 선택] → [질문 입력] → [⚡ 비교 실행]
```

- 좌: 🔵 Basic RAG 답변 + 지연시간(ms)
- 우: 🟣 RAPTOR RAG 답변 + 지연시간(ms)
- 하단 **[▶ 참조 청크 보기]** 클릭 → 각 방식이 실제로 꺼낸 컨텍스트 확인 가능
- **[📋 결과 이력]** 클릭 → 이전 질문/답변 테이블 확인

### 4-3. 탭 2 — 멀티모델 비교

여러 LLM 모델로 동시에 Basic + RAPTOR 실행:

```
모델 체크박스에서 비교할 모델 선택 (예: gemma2:9b, qwen3:8b, llama3)
→ [⚡ 비교 실행]
→ 모델 × RAG기법 조합의 결과 카드 그리드로 표시
```

사용 가능한 모델:
- `gemma2:9b` — 한국어 이해 양호
- `qwen3:8b` — 한국어 대화 데이터에 적합
- `llama3:latest` — 기본값
- `llama3.1:8b`, `gpt-oss:20b` 등

### 4-4. 탭 3 — LongBench 평가

ground truth가 있는 벤치마크 질문으로 **정확도**를 수치로 비교:

```
[LongBench 스레드 선택] → [▶ 벤치마크 실행]
→ 정확도 요약: 🔵 Basic 20% vs 🟣 RAPTOR 20%
→ 질문별 Ground Truth / Basic 답변 / RAPTOR 답변 / ✓✗ 표시
```

---

## 5. LongBench 벤치마크 평가

### 5-1. 현재 로드된 데이터

**데이터셋**: `multifieldqa_en` (5개 예제)

| 예제 | 도메인 | 컨텍스트 길이 | 벤치마크 질문 |
|------|--------|--------------|--------------|
| 예제 1 | 스포츠 (FC Urartu) | ~5,000자 | What is the name of the most active fan club? |
| 예제 2 | 생물학 (ISR/단백질 제한) | ~45,000자 | Is the ISR necessary for transgene reactivation? |
| 예제 3 | 물리학 (양자점 나노튜브) | ~31,000자 | What experimental techniques were used...? |
| 예제 4 | 의학 (ICD 심장 장치) | ~54,000자 | What is the purpose of an ICD? |
| 예제 5 | 항공 (동체 설계) | ~34,000자 | Why is it important for the sides of the fuselage to be sloped...? |

### 5-2. 추가 데이터 로드

```bash
# hotpotqa 10개 (멀티홉 추론 — RAPTOR에 유리)
python load_longbench.py hotpotqa 10

# narrativeqa 5개 (장편 서사 — 긴 문서 이해)
python load_longbench.py narrativeqa 5

# 사용 가능한 데이터셋 목록 (/tmp/longbench/extracted/data/)
ls /tmp/longbench/extracted/data/*.jsonl
```

### 5-3. 정확도 측정 방식

현재: **부분 문자열 포함 여부** (case-insensitive contains)
- LLM 답변 안에 ground truth 문자열이 들어있으면 정답 처리
- 긴 ground truth이거나 LLM이 같은 내용을 다른 표현으로 답하면 오탐 가능
- 모델 변경(특히 영어 특화 모델)으로 정확도 향상 가능

---

## 6. 효과적인 비교를 위한 질문 전략

### 6-1. Basic RAG가 유리한 질문 (단순 사실 검색)

청크 하나에 답이 있는 질문. 빠르고 직접적입니다.

```
예제 1 (FC Urartu):
  "What is the name of the most active fan club?"
  "What year did Dzhevan Cheloyants become co-owner of the club?"

예제 4 (ICD):
  "What does ICD stand for?"

예제 5 (동체 설계):
  "What tool is recommended for checking fuselage alignment?"
```

### 6-2. RAPTOR RAG가 유리한 질문 (전체 문서 이해 필요)

여러 청크에 흩어진 정보를 조합하거나 문서 전체 맥락이 필요한 질문.

```
예제 2 (ISR, 45,000자):
  "What is the overall relationship between amino acid restriction and lifespan?"
  "Summarize the main findings about ISR and its downstream effects."

예제 3 (양자점, 31,000자):
  "What were the main conclusions of this research and why are they significant?"

예제 4 (ICD, 54,000자 — 가장 극적인 차이 기대):
  "How does an ICD differ from a pacemaker, and what conditions require each?"
  "Summarize the key differences between all the cardiac devices described in this text."

예제 5 (동체 설계, 34,000자):
  "What are all the major steps described for building a straight fuselage?"
```

### 6-3. 두 기법 차이가 가장 극명한 질문 (강력 추천)

```
예제 4 (ICD, 54,000자):
"Summarize the key differences between all the cardiac devices described in this text."

예제 2 (ISR, 45,000자):
"What is the main finding of this study and how does it connect
 protein restriction to ISR activation?"
```

> **왜 극적인가**: Basic RAG는 청크 5개(~1,500자)만 보지만, RAPTOR RAG는 전체 문서를 압축한 요약 노드들을 트리에서 꺼내기 때문에 긴 문서에서 조합 질문에 훨씬 유리합니다.

### 6-4. 월간 대화 데이터 (신입사원 OJT) 비교 질문

```
스레드: "전체 월간 대화 통합" (577개 메시지, 28일치)

단순 사실 (Basic 유리):
  "김재원이 처음 입사한 날 받은 OJT 과제는 무엇인가요?"
  "팀장 박민지가 처음 언급한 프로젝트 이름은 무엇인가요?"

종합 분석 (RAPTOR 유리):
  "김재원이 한 달 동안 성장한 과정을 단계별로 요약해주세요."
  "팀 내에서 반복적으로 등장한 기술적 갈등이나 문제는 무엇이었나요?"
  "이현수 멘토의 코칭 스타일은 어떻게 변화했나요?"
```

---

## 7. 결과 해석 방법

### 7-1. 참조 청크 분석 (가장 중요)

**[▶ 참조 청크 보기]** 를 펼치면 각 방식이 꺼낸 컨텍스트가 보입니다.

```
Basic RAG 참조:  원문 청크 그대로 → "어디서 찾았는지" 명확
RAPTOR RAG 참조: 요약 노드 포함   → "어느 레벨 요약인지" 레이블 표시
                  예) [전체 요약 (레벨 2)] / [세부 내용 (레벨 0)]
```

**체크 포인트:**
- Basic이 엉뚱한 청크를 꺼냈다면 → 질문 키워드와 청크 내용의 어휘 불일치
- RAPTOR 참조에 레벨 1~2 요약이 많다면 → 문서 전체 맥락을 활용한 것
- 두 방식이 같은 청크를 꺼냈다면 → 그 질문은 단순 검색으로 충분한 것

### 7-2. 지연시간 해석

| 상황 | 의미 |
|------|------|
| Basic이 훨씬 빠름 | 청크 검색이 단순, RAPTOR 트리 탐색 오버헤드 있음 |
| RAPTOR가 더 빠름 | 요약 노드가 효율적으로 압축되어 LLM 프롬프트가 짧아짐 |
| 둘 다 느림 (10초+) | LLM 응답 시간 지배적 (모델 크기 영향) |

### 7-3. 멀티모델 탭에서 볼 포인트

```
같은 질문, 같은 RAG 방식인데 모델마다 결과가 크게 다르다
  → RAG 방식보다 LLM 자체가 더 큰 변수임을 확인

같은 모델인데 Basic vs RAPTOR 답변이 크게 다르다
  → 검색된 컨텍스트 차이가 결과에 영향을 줌 (RAG 방식이 의미있는 변수)
```

### 7-4. LongBench 정확도 해석

| 결과 패턴 | 해석 |
|-----------|------|
| Basic ✓ / RAPTOR ✗ | 정답이 특정 청크에 집중된 단순 팩트 질문 |
| Basic ✗ / RAPTOR ✓ | 문서 전반 이해가 필요한 종합 질문 |
| 둘 다 ✗ | LLM이 다른 표현을 사용 (정답이 포함되어 있을 수 있음) |
| 둘 다 ✓ | 두 방식 모두 충분한 컨텍스트를 찾은 경우 |

> **주의**: 현재 정확도 측정은 "ground truth 문자열 포함 여부"로 단순 체크합니다. 영어 모델(llama3 계열)이 영어 ground truth와 더 잘 매칭되므로, 정확도 비교는 같은 모델 조건에서 실행하세요.

---

## 8. 현재 데이터 현황

### 스레드 (인덱싱 완료)

| 스레드 | 메시지 수 | Basic 청크 | RAPTOR 노드 |
|--------|----------|-----------|------------|
| `[LongBench] multifieldqa_en` | 518 | 748 | 1,002 |
| `전체 월간 대화 통합` | 577 | - | - |
| `Day 1 — 전체 대화` | 48 | 11 | 13 |

### 나머지 스레드 인덱싱 방법

```
/chat → 스레드 선택 → [📥 RAG 인덱싱] 클릭
  또는
curl -X POST http://localhost:8000/api/threads/<thread_id>/index
```

### 데이터 생성 스크립트

```bash
# 월간 대화 데이터 재생성 (qwen3:8b 모델 사용, 한국어)
python gen_monthly_data.py qwen3:8b

# LongBench 추가 데이터 로드
python load_longbench.py hotpotqa 10
python load_longbench.py narrativeqa 5
```
