# DP3 PoC Result Raw

이 문서는 DP3 PoC 실험 중 별도로 측정한 원시 결과를 기록한다. 수치는 네트워크 상태, Groq queue 상태, rate limit window, 로컬 실행 환경에 따라 달라질 수 있다.

## 1. Groq LLM Latency

### 목적

DP3 PoC에서 실제 LLM 호출이 전체 수행시간에 어느 정도 영향을 주는지 확인하기 위해 Groq API 응답 시간을 별도로 측정했다.

### 공통 조건

| 항목 | 값 |
|---|---|
| Provider | Groq |
| Endpoint | OpenAI-compatible Chat Completions |
| 기준 모델 | `llama-3.1-8b-instant` |
| 측정 방식 | HTTP 요청 시작부터 응답 수신까지 |
| 포함 | Groq queue time, model processing time, HTTP 왕복 |
| 제외 | 요청 사이 rate-limit 대기 sleep |

## 2. 짧은 프롬프트

짧은 DP3/RAG 형태의 프롬프트를 사용해 `llama-3.1-8b-instant`를 10회 호출했다.

| 항목 | 값 |
|---|---:|
| Runs | 10 |
| Success | 10 |
| Error | 0 |
| Min | 246.5 ms |
| Max | 375.7 ms |
| Avg | 319.9 ms |

### 요청별 응답 시간

| # | Latency |
|---:|---:|
| 1 | 375.7 ms |
| 2 | 265.0 ms |
| 3 | 246.5 ms |
| 4 | 368.4 ms |
| 5 | 355.3 ms |
| 6 | 351.5 ms |
| 7 | 366.7 ms |
| 8 | 261.4 ms |
| 9 | 251.9 ms |
| 10 | 357.0 ms |

## 3. 긴 프롬프트, 약 6K tokens 수준

Groq의 `llama-3.1-8b-instant` TPM 제한이 6,000 tokens/minute이므로, 요청당 약 5.6K tokens가 되도록 프롬프트를 조정했다.

| 항목 | 값 |
|---|---:|
| Runs | 10 |
| Success | 10 |
| Error | 0 |
| Total tokens/request | 약 5,580~5,590 |
| Request interval | 약 62 sec |
| Min | 533.0 ms |
| Max | 850.5 ms |
| Avg | 700.5 ms |
| Avg Groq queue time | 0.099 sec |
| Max Groq queue time | 0.169 sec |
| Avg Groq model total time | 0.404 sec |
| Max Groq model total time | 0.579 sec |

### 요청별 응답 시간

| # | Latency |
|---:|---:|
| 1 | 739.2 ms |
| 2 | 795.6 ms |
| 3 | 682.5 ms |
| 4 | 698.0 ms |
| 5 | 701.7 ms |
| 6 | 719.6 ms |
| 7 | 533.0 ms |
| 8 | 700.6 ms |
| 9 | 584.3 ms |
| 10 | 850.5 ms |

10번째 보충 요청은 기존 9회 측정과 동일한 프롬프트 생성 방식으로 수행했다. `prompt_chars=32,544`, `prompt_words=4,115`, `total_tokens=5,590`, HTTP 왕복 `850.5 ms`, Groq usage 기준 `queue_time=0.077 sec`, `total_time=0.579 sec`로 측정되었다.

## 4. 해석

- 짧은 프롬프트에서는 평균 약 320 ms 수준으로 응답했다.
- 약 5.6K tokens/request의 긴 프롬프트에서는 평균 약 701 ms 수준으로 응답했다.
- 긴 프롬프트에서도 개별 응답 시간 자체는 대체로 1초 내외로 관측되었지만, throughput은 응답 시간이 아니라 TPM 제한에 의해 결정된다.
- `llama-3.1-8b-instant`의 6,000 TPM 조건에서는 5.5K tokens급 요청을 안정적으로 보내려면 65~70초 간격이 더 안전하다.
- DP3 대량 테스트에서는 mock LLM을 사용하고, Groq는 대표 샘플 검증 또는 소규모 latency 확인에 사용하는 편이 현실적이다.
