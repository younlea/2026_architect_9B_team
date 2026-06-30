# DP3 TC3 Official RAGAS Export

이 폴더는 DP3 PoC의 TC3 Similar Query Pair Quality 결과를 다른 환경에서 official RAGAS로 다시 평가하기 위한 독립 실행 패키지다.

## 포함 파일

| 파일 | 설명 |
|---|---|
| `dp3_tc3_ragas_input_techqa_20rows.jsonl` | RAGAS 입력 데이터. 10개 pair x A/B 2개 mode = 20 rows |
| `run_official_ragas_slow.py` | 프로젝트 backend 없이 실행 가능한 standalone RAGAS runner |
| `requirements-ragas-runner.txt` | 실행 환경 설치용 requirements |
| `previous_partial_official_ragas_scores_17rows.json` | 기존 Desktop 환경에서 얻은 partial/degraded 참고 결과. 최종 근거로 사용하지 않는 것을 권장 |

## 입력 데이터 의미

각 JSONL row는 다음 필드를 포함한다.

| 필드 | 의미 |
|---|---|
| `pair_id` | 유사질문 pair 식별자 |
| `mode` | `A` 또는 `B` |
| `question` | 평가 대상 질문 |
| `answer` | 해당 안의 답변 |
| `contexts` | RAGAS retrieved_contexts |
| `ground_truth` | RAGBench reference answer |
| `cache_hit` | 해당 mode의 cache hit 여부 |
| `decision_reason` | cache/RAG 분기 사유 |
| `llm_usage` | LLM 호출 및 latency metadata |

## 권장 실행 환경

- Python 3.10 이상
- 안정적인 네트워크
- Groq 또는 OpenAI API key
- 긴 실행 시간을 허용할 수 있는 터미널 세션

## 설치

```bash
cd dp3_tc3_ragas_techqa_20260630
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-ragas-runner.txt
```

## Groq로 실행

```bash
export GROQ_API_KEY="여기에_Groq_API_Key"

python run_official_ragas_slow.py \
  --provider groq \
  --model meta-llama/llama-4-scout-17b-16e-instruct \
  --input dp3_tc3_ragas_input_techqa_20rows.jsonl \
  --output-dir official_ragas_output \
  --chunk-size 1 \
  --seconds-per-request 65 \
  --timeout 300 \
  --max-retries 2
```

## OpenAI로 실행

```bash
export OPENAI_API_KEY="여기에_OpenAI_API_Key"

python run_official_ragas_slow.py \
  --provider openai \
  --model gpt-4o-mini \
  --input dp3_tc3_ragas_input_techqa_20rows.jsonl \
  --output-dir official_ragas_output \
  --chunk-size 1 \
  --seconds-per-request 5 \
  --timeout 300 \
  --max-retries 2
```

## 산출물

실행 중간부터 다음 파일이 생성된다.

| 경로 | 설명 |
|---|---|
| `official_ragas_output/chunks/chunk_000_000.json` | chunk별 score. 실패해도 완료된 chunk는 유지됨 |
| `official_ragas_output/official_ragas_scores_combined.json` | 완료된 chunk를 합친 전체 score |
| `official_ragas_output/summary.json` | mode별 metric 평균 및 valid/missing 수 |

## 재시작

중간에 끊기면 같은 명령을 다시 실행하면 된다. 기본값으로 이미 있는 chunk는 건너뛴다.

```bash
python run_official_ragas_slow.py \
  --provider groq \
  --model meta-llama/llama-4-scout-17b-16e-instruct \
  --input dp3_tc3_ragas_input_techqa_20rows.jsonl \
  --output-dir official_ragas_output
```

## Context Precision만 계속 실패할 때

Desktop 환경에서는 `context_precision` evaluator job이 timeout되어 모든 값이 `NaN`이 되었다. 같은 문제가 반복되면 우선 아래처럼 나머지 3개 metric만 안정적으로 계산한다.

```bash
python run_official_ragas_slow.py \
  --provider groq \
  --input dp3_tc3_ragas_input_techqa_20rows.jsonl \
  --output-dir official_ragas_output_no_context_precision \
  --metrics faithfulness,answer_relevancy,context_recall
```

이 경우 QA 근거에는 `context_precision`을 제외했다고 명시해야 한다.

## 결과 해석 기준

최종 QA 문서에 반영할 때는 다음 조건을 먼저 확인한다.

| 체크 | 권장 기준 |
|---|---|
| row 수 | `rows_scored=20`이면 전체 완료. 20 미만이면 partial result |
| A/B 균형 | A 10 rows, B 10 rows가 모두 있어야 공정 비교 가능 |
| metric valid 수 | 각 metric의 valid 수가 mode별 N과 같아야 신뢰 가능 |
| NaN | 특정 metric이 NaN이면 해당 metric은 해석에서 제외 |

Desktop partial 결과는 20 rows 중 17 rows만 남고 `context_precision`이 모두 NaN이었으므로, 최종 결론에는 직접 사용하지 않는 것이 안전하다.
