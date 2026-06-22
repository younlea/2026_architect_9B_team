# DP3 Cache PoC README

이 문서는 현재 구현된 DP3 Cache PoC를 다른 PC 또는 새 환경에서 실행하기 위한 사용 가이드다.

## 1. 현재 구현 범위

현재 PoC에는 다음 기능이 포함되어 있다.

- DP3 전용 LongBench 다운로드
- LongBench 예제 기반 ROI-RAG Evidence Unit 생성
- DP3 전용 원본 EU 저장
- `scope=A/B`, `version=V1/V2/V3`, `fingerprint` metadata 생성
- A안 Verified Answer Cache 테스트
- B안 Incremental Context Cache 테스트
- LongBench 질문풀 생성
- 웹 UI 기반 A/B 테스트 실행

DP3 PoC는 DP1/DP2의 기존 실행 방식과 데이터를 직접 수정하지 않도록 DP3 전용 loader, DB table, UI를 사용한다.

## 2. 필요 환경

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

## 3. 최초 환경 설정

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

## 4. 환경 변수

기본값은 `src/.env.example`을 참고한다.

DP3 PoC는 기본적으로 mock LLM을 사용한다.

```text
DP3_MOCK_LLM=true
SQLITE_DB_PATH=./data/poc.db
EMBEDDING_MODEL=all-MiniLM-L6-v2
```

현재 테스트 목적이라면 `DP3_MOCK_LLM=true`를 유지하면 된다. 실제 LLM을 연결할 때는 기존 DP1/DP2 방식의 LLM 설정을 사용하되, DP3 mock 설정을 끄면 된다.

```text
DP3_MOCK_LLM=false
```

## 5. 서버 실행

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

## 6. 웹 UI 사용 순서

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

## 7. A안 동작 요약

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

## 8. B안 동작 요약

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

## 9. 생성되는 로컬 파일

다음 파일과 디렉터리는 로컬 실행 산출물이다.

```text
src/data/longbench/
src/data/longbench.zip
src/data/poc.db
```

이 파일들은 git에 포함하지 않는 것을 전제로 한다. 다른 PC에서 처음 실행하면 다시 생성된다.

## 10. 현재 제약과 주의점

- 현재 시간 로그는 주로 `total_ms` 중심이다.
- RAG, LLM, validation 시간을 분리 저장하는 작업은 TODO로 남아 있다.
- mock LLM 환경에서는 실제 LLM latency를 평가할 수 없다.
- B안 delta retrieval은 데이터와 mutation 조건에 따라 full fallback으로 많이 떨어질 수 있다.
- A안 route threshold가 낮으면 route pass가 과도하게 많아질 수 있다.
- 질문풀이 LongBench 전체 질문의 일부 샘플이므로, 최종 실험 전에는 sampling 비율과 seed를 고정해야 한다.

추가 작업 목록은 `DP3_PoC_TODO.md`를 참고한다.

