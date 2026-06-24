# DP3 PoC Result Raw

이 문서는 DP3 PoC 실험 중 별도로 측정한 원시 결과를 기록한다. 수치는 로컬 실행 환경, 네트워크 상태, Groq queue 상태, rate limit window, 로컬 DB 상태에 따라 달라질 수 있다.

## 1. Groq LLM Latency

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

### 요약

| Prompt | Runs | Success | Error | Tokens/request | Min | Max | Avg |
|---|---:|---:|---:|---:|---:|---:|---:|
| 짧은 DP3/RAG 형태 prompt | 10 | 10 | 0 | 소량 | 246.5 ms | 375.7 ms | 319.9 ms |
| 긴 prompt | 10 | 10 | 0 | 약 5,580-5,590 | 533.0 ms | 850.5 ms | 700.5 ms |

긴 prompt는 Groq `llama-3.1-8b-instant`의 6,000 TPM 제한을 고려해 요청당 약 5.6K tokens가 되도록 구성했다. 요청 간격은 약 62초였고, 이 sleep 시간은 latency에서 제외했다.

### 짧은 Prompt 요청별 응답 시간

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

### 긴 Prompt 요청별 응답 시간

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

### 긴 Prompt 내부 측정

| 항목 | 값 |
|---|---:|
| Avg Groq queue time | 0.099 sec |
| Max Groq queue time | 0.169 sec |
| Avg Groq model total time | 0.404 sec |
| Max Groq model total time | 0.579 sec |

긴 prompt 추가 1회 측정은 기존 9회와 동일한 prompt 생성 방식으로 수행했다. 해당 요청은 `prompt_chars=32,544`, `prompt_words=4,115`, `total_tokens=5,590`, HTTP 왕복 `850.5 ms`, Groq usage 기준 `queue_time=0.077 sec`, `total_time=0.579 sec`로 측정되었다.

### 해석

- 짧은 prompt에서는 평균 약 320 ms 수준으로 응답했다.
- 약 5.6K tokens/request의 긴 prompt에서는 평균 약 701 ms 수준으로 응답했다.
- 긴 prompt에서도 개별 응답 시간은 대체로 1초 이내였지만, throughput은 응답 시간이 아니라 TPM 제한에 의해 결정된다.
- `llama-3.1-8b-instant`의 6,000 TPM 조건에서는 5.5K tokens급 요청을 안정적으로 보내려면 65-70초 간격이 안전하다.
- DP3 대량 테스트에서는 mock LLM을 사용하고, Groq는 소규모 latency 검증 또는 샘플 검증에 사용하는 편이 현실적이다.
- HTML의 `Est. Total+LLM`은 긴 prompt 평균에 가까운 `700 ms`를 mock LLM 호출 1회당 더하는 방식으로 계산한다.

## 2. TC1 Cache Benefit - TechQA 100 Rows

사용자가 HTML에서 새로 실행한 TC1 결과를 DB 로그 기준으로 정리했다.

### 테스트 설정

| 항목 | 값 |
|---|---|
| Test Case | TC1 Cache Benefit |
| Dataset | RAGBench `techqa` |
| Split | `test` |
| Source ID | `dp3_ragbench_techqa_test_100` |
| RAG corpus row 수 | 100 |
| Base EU rows | 1,874 |
| Versioned EU rows | 4,062 |
| Test query 수 | 50 |
| Requested version | V1 |
| User scope | A |
| LLM | Mock |
| Reranker | Off |
| Route threshold | 0.60 |
| Cache hit threshold | 0.86 |
| Route pool | `techqa` 질문 15개 + 기본 route 4개 |
| 로그 생성 시각 | 2026-06-24 04:22:02 - 04:23:06 |

기본 route 4개는 `summary`, `summarize`, `definition`, `fact_check`이다. TC1 해석에서는 `techqa` route pool 15개가 주 route 후보이며, 기본 route는 fallback 성격의 일반 route 후보로 남아 있다.

### 전체 요약

| Mode | Total | Cache hit | Validation passed | Full/RAG fallback | LLM calls | Total avg |
|---|---:|---:|---:|---:|---:|---:|
| NoCache | 50 | 0 | 0 | 50 | 50 | 300.320 ms |
| A first | 50 | 3 | 3 | 47 | 47 | 313.220 ms |
| A repeat | 50 | 14 | 14 | 36 | 36 | 257.960 ms |
| B first | 50 | 5 | 5 | 45 | 50 | 296.660 ms |
| B repeat | 50 | 50 | 50 | 0 | 50 | 66.340 ms |

주의: B안은 context cache이므로 cache hit 후에도 LLM은 호출한다. 반면 A안은 answer cache hit이면 LLM 호출까지 생략한다.

### LLM 700 ms 가정 예상 총 시간

mock LLM은 실제 LLM 시간을 거의 쓰지 않는다. 아래 값은 Groq `llama-3.1-8b-instant`, 약 5.6K tokens 기준 평균 `700 ms`를 LLM call 1회당 더한 예상치다.

| Mode | Mock total avg | LLM calls/query | Est. Total+LLM avg | NoCache 대비 |
|---|---:|---:|---:|---:|
| NoCache | 300.320 ms | 1.00 | 1,000.320 ms | baseline |
| A first | 313.220 ms | 0.94 | 971.220 ms | -29.100 ms |
| A repeat | 257.960 ms | 0.72 | 761.960 ms | -238.360 ms |
| B first | 296.660 ms | 1.00 | 996.660 ms | -3.660 ms |
| B repeat | 66.340 ms | 1.00 | 766.340 ms | -233.980 ms |

### NoCache

| Metric | Min | Max | Avg | N |
|---|---:|---:|---:|---:|
| Total | 260.000 ms | 347.000 ms | 300.320 ms | 50 |
| Embedding | 9.646 ms | 34.377 ms | 17.219 ms | 50 |
| Full RAG total | 241.580 ms | 320.692 ms | 279.694 ms | 50 |
| Full RAG DB | 47.979 ms | 101.748 ms | 71.710 ms | 50 |
| Full RAG scoring | 177.591 ms | 234.762 ms | 207.703 ms | 50 |
| Full RAG score sort | 0.139 ms | 0.494 ms | 0.281 ms | 50 |
| Full RAG rerank | 0.000 ms | 0.000 ms | 0.000 ms | 50 |
| Prompt build | 0.009 ms | 0.053 ms | 0.018 ms | 50 |
| Mock LLM | 0.063 ms | 0.334 ms | 0.112 ms | 50 |

Decision reason:

| Reason | Count |
|---|---:|
| `no_cache_full_rag` | 50 |

### A안 결과

| Pass | Total | Route passed | Cache hit | Validation passed | RAG fallback | LLM calls |
|---|---:|---:|---:|---:|---:|---:|
| V1 첫 실행 | 50 | 14 | 3 | 3 | 47 | 47 |
| V1 반복 | 50 | 14 | 14 | 14 | 36 | 36 |

#### A안 decision reasons

| Pass | Reason | Count |
|---|---|---:|
| V1 첫 실행 | `embedding_score_below_threshold` | 36 |
| V1 첫 실행 | `cache_candidate_not_found_fallback_to_roi_rag` | 11 |
| V1 첫 실행 | `answer_cache_hit_valid` | 3 |
| V1 반복 | `embedding_score_below_threshold` | 36 |
| V1 반복 | `answer_cache_hit_valid` | 14 |

#### A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| V1 첫 실행 | Total | 43.000 ms | 376.000 ms | 313.220 ms | 50 |
| V1 첫 실행 | Embedding | 8.659 ms | 47.063 ms | 17.706 ms | 50 |
| V1 첫 실행 | Route | 2.161 ms | 5.967 ms | 3.906 ms | 50 |
| V1 첫 실행 | Cache lookup | 2.245 ms | 5.223 ms | 3.782 ms | 14 |
| V1 첫 실행 | Validation | 0.000 ms | 18.151 ms | 3.701 ms | 14 |
| V1 첫 실행 | RAG total | 247.572 ms | 320.439 ms | 288.019 ms | 47 |
| V1 첫 실행 | RAG DB | 51.492 ms | 87.370 ms | 73.994 ms | 47 |
| V1 첫 실행 | RAG scoring | 189.950 ms | 243.912 ms | 213.735 ms | 47 |
| V1 첫 실행 | RAG score sort | 0.143 ms | 0.574 ms | 0.289 ms | 47 |
| V1 첫 실행 | RAG rerank | 0.000 ms | 0.000 ms | 0.000 ms | 47 |
| V1 반복 | Total | 28.000 ms | 445.000 ms | 257.960 ms | 50 |
| V1 반복 | Embedding | 8.216 ms | 51.334 ms | 18.724 ms | 50 |
| V1 반복 | Route | 2.577 ms | 10.966 ms | 3.937 ms | 50 |
| V1 반복 | Cache lookup | 2.638 ms | 14.227 ms | 4.999 ms | 14 |
| V1 반복 | Validation | 10.186 ms | 44.730 ms | 19.011 ms | 14 |
| V1 반복 | RAG total | 251.306 ms | 398.920 ms | 292.535 ms | 36 |
| V1 반복 | RAG DB | 51.553 ms | 229.087 ms | 79.343 ms | 36 |
| V1 반복 | RAG scoring | 169.417 ms | 250.377 ms | 212.879 ms | 36 |
| V1 반복 | RAG score sort | 0.160 ms | 0.554 ms | 0.313 ms | 36 |
| V1 반복 | RAG rerank | 0.000 ms | 0.000 ms | 0.000 ms | 36 |

### B안 결과

| Pass | Total | Cache hit | Validation passed | Full RAG | Delta RAG | LLM calls |
|---|---:|---:|---:|---:|---:|---:|
| V1 첫 실행 | 50 | 5 | 5 | 45 | 0 | 50 |
| V1 반복 | 50 | 50 | 50 | 0 | 0 | 50 |

#### B안 decision reasons

| Pass | Reason | Count |
|---|---|---:|
| V1 첫 실행 | `context_cache_similarity_below_threshold` | 45 |
| V1 첫 실행 | `context_cache_hit_all_valid` | 5 |
| V1 반복 | `context_cache_hit_all_valid` | 50 |

#### B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| V1 첫 실행 | Total | 51.000 ms | 439.000 ms | 296.660 ms | 50 |
| V1 첫 실행 | Embedding | 7.122 ms | 69.504 ms | 22.945 ms | 50 |
| V1 첫 실행 | Cache lookup | 9.337 ms | 52.139 ms | 23.752 ms | 50 |
| V1 첫 실행 | Validation | 11.895 ms | 25.086 ms | 16.544 ms | 5 |
| V1 첫 실행 | Full RAG total | 227.009 ms | 360.648 ms | 249.810 ms | 45 |
| V1 첫 실행 | Full RAG DB | 46.109 ms | 122.046 ms | 62.537 ms | 45 |
| V1 첫 실행 | Full RAG scoring | 154.662 ms | 238.380 ms | 187.078 ms | 45 |
| V1 첫 실행 | Full RAG score sort | 0.165 ms | 0.321 ms | 0.195 ms | 45 |
| V1 첫 실행 | Full RAG rerank | 0.000 ms | 0.000 ms | 0.000 ms | 45 |
| V1 반복 | Total | 50.000 ms | 87.000 ms | 66.340 ms | 50 |
| V1 반복 | Embedding | 6.714 ms | 34.180 ms | 17.000 ms | 50 |
| V1 반복 | Cache lookup | 21.277 ms | 29.589 ms | 24.805 ms | 50 |
| V1 반복 | Validation | 11.914 ms | 23.252 ms | 17.822 ms | 50 |

### 중간 해석

- NoCache의 평균 RAG 비용은 `Full RAG total` 기준 약 `279.694 ms`, 전체 평균은 `300.320 ms`였다.
- A안은 route threshold `0.60`에서 50개 중 14개가 route를 통과했다. 반복 실행에서는 route 통과 14개가 모두 answer cache hit이 되었지만, route fail 36개는 계속 RAG fallback으로 처리되었다.
- B안은 첫 실행에서 5개만 context cache hit이었고, 반복 실행에서는 50개 모두 context cache hit이 되었다. 이때 Full RAG는 0회가 되어 평균 total이 `66.340 ms`까지 낮아졌다.
- mock 기준에서는 cache lookup/store 오버헤드 때문에 A first는 NoCache보다 느리다. 실제 LLM을 `700 ms/call`로 가정하면 A repeat와 B repeat 모두 NoCache 대비 약 `230 ms/query` 이상 절감되는 것으로 추정된다.
- B안은 context cache이므로 반복 실행에서도 LLM 호출은 유지된다. B안의 절감은 주로 RAG 비용 제거에서 발생한다.

## 3. TC2 Scale Cost - TechQA 100/200/300 Rows

TC2는 동일한 `techqa` 기반 workload에서 corpus row 수를 100, 200, 300으로 늘렸을 때 RAG 비용이 어떻게 증가하는지 확인하기 위한 테스트다. 각 scale마다 NoCache, A first, A repeat, B first, B repeat를 순서대로 실행했다.

### 테스트 설정

| 항목 | 값 |
|---|---|
| Test Case | TC2 Scale Cost |
| Dataset | RAGBench `techqa` |
| Split | `test` |
| Scale rows | 100 / 200 / 300 |
| Test query 수 | scale별 50개 |
| Requested version | V1 |
| User scope | A |
| LLM | Mock |
| Reranker | Off |
| Route threshold | 0.60 |
| Cache hit threshold | 0.86 |
| LLM 예상치 | Groq `llama-3.1-8b-instant`, 약 5.6K tokens 기준 `700 ms/call` |

각 scale은 독립적인 source로 실행했다. NoCache 이후 A안을 실행하기 전에 answer cache를 비우고, B안 첫 실행 전에는 context cache를 비워 scale 간 cache 오염을 줄였다. 단, 같은 scale 내부에서 반복되거나 유사한 질문은 첫 실행 중에도 cache hit이 발생할 수 있다.

### Corpus 규모

| Rows | Source ID | Base EU | Versioned EU | Log window |
|---:|---|---:|---:|---|
| 100 | `dp3_ragbench_techqa_test_100` | 1,874 | 4,062 | 2026-06-24 04:29:21 - 04:30:21 |
| 200 | `dp3_ragbench_techqa_test_200` | 3,882 | 8,411 | 2026-06-24 04:30:26 - 04:32:02 |
| 300 | `dp3_ragbench_techqa_test_300` | 5,701 | 12,353 | 2026-06-24 04:32:08 - 04:34:24 |

### NoCache Scale Trend

| Rows | Total avg | Full RAG avg | Scoring avg | DB avg | LLM calls | Est. Total+LLM avg |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 310.500 ms | 282.657 ms | 198.235 ms | 84.166 ms | 50 | 1,010.500 ms |
| 200 | 454.460 ms | 425.143 ms | 339.965 ms | 84.708 ms | 50 | 1,154.460 ms |
| 300 | 734.060 ms | 707.365 ms | 604.166 ms | 102.319 ms | 50 | 1,434.060 ms |

NoCache 기준으로 Full RAG 평균은 row 100에서 `282.657 ms`, row 300에서 `707.365 ms`까지 증가했다. 증가분은 주로 scoring 단계에서 발생했다.

### A안 Scale Trend

| Rows | Pass | Route passed | Cache hit | Validation passed | RAG fallback | LLM calls | Total avg | RAG avg | Est. Total+LLM avg |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | A first | 14 | 3 | 3 | 47 | 47 | 302.620 ms | 269.663 ms | 960.620 ms |
| 100 | A repeat | 14 | 14 | 14 | 36 | 36 | 221.960 ms | 248.211 ms | 725.960 ms |
| 200 | A first | 16 | 3 | 3 | 47 | 47 | 494.020 ms | 471.574 ms | 1,152.020 ms |
| 200 | A repeat | 16 | 16 | 16 | 34 | 34 | 376.700 ms | 483.114 ms | 852.700 ms |
| 300 | A first | 19 | 2 | 2 | 48 | 48 | 711.600 ms | 689.528 ms | 1,383.600 ms |
| 300 | A repeat | 19 | 19 | 19 | 31 | 31 | 490.240 ms | 713.224 ms | 924.240 ms |

A안은 route를 통과한 질문만 answer cache 대상이 된다. 반복 실행에서는 route 통과 질문이 모두 cache hit이 되었지만, route fail 질문은 계속 RAG fallback을 탄다. 따라서 row scale이 커질수록 A repeat의 평균 total도 함께 증가한다.

### B안 Scale Trend

| Rows | Pass | Cache hit | Validation passed | Full RAG | Delta RAG | LLM calls | Total avg | Full RAG avg | Est. Total+LLM avg |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 100 | B first | 5 | 5 | 45 | 0 | 50 | 275.680 ms | 244.396 ms | 975.680 ms |
| 100 | B repeat | 50 | 50 | 0 | 0 | 50 | 50.800 ms | 0.000 ms | 750.800 ms |
| 200 | B first | 7 | 7 | 43 | 0 | 50 | 485.680 ms | 489.510 ms | 1,185.680 ms |
| 200 | B repeat | 50 | 50 | 0 | 0 | 50 | 57.880 ms | 0.000 ms | 757.880 ms |
| 300 | B first | 8 | 8 | 42 | 0 | 50 | 678.880 ms | 738.188 ms | 1,378.880 ms |
| 300 | B repeat | 50 | 50 | 0 | 0 | 50 | 55.160 ms | 0.000 ms | 755.160 ms |

B안은 context cache hit 이후에도 LLM 호출은 유지된다. 그래서 `Est. Total+LLM`은 반복 실행에서도 약 `750 ms/query` 수준으로 남지만, Full RAG 비용은 0에 가까워져 row scale 증가 영향을 거의 받지 않는다.

### 주요 Timing Detail

| Rows | Mode | Total min | Total max | Total avg | Retrieval/DB avg | Scoring avg | Cache lookup avg | Validation avg |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 100 | NoCache | 246.000 ms | 402.000 ms | 310.500 ms | 84.166 ms | 198.235 ms | - | - |
| 100 | A repeat | 27.000 ms | 336.000 ms | 221.960 ms | 63.129 ms | 184.891 ms | 5.642 ms | 16.029 ms |
| 100 | B repeat | 34.000 ms | 72.000 ms | 50.800 ms | - | - | 12.846 ms | 15.720 ms |
| 200 | NoCache | 392.000 ms | 579.000 ms | 454.460 ms | 84.708 ms | 339.965 ms | - | - |
| 200 | A repeat | 28.000 ms | 611.000 ms | 376.700 ms | 90.862 ms | 391.575 ms | 5.413 ms | 18.579 ms |
| 200 | B repeat | 47.000 ms | 82.000 ms | 57.880 ms | - | - | 13.261 ms | 19.517 ms |
| 300 | NoCache | 635.000 ms | 953.000 ms | 734.060 ms | 102.319 ms | 604.166 ms | - | - |
| 300 | A repeat | 29.000 ms | 881.000 ms | 490.240 ms | 105.834 ms | 606.428 ms | 4.620 ms | 18.570 ms |
| 300 | B repeat | 40.000 ms | 77.000 ms | 55.160 ms | - | - | 12.905 ms | 18.573 ms |

### 중간 해석

- TC2의 핵심 관찰은 corpus row가 커질수록 NoCache의 Full RAG 비용이 증가한다는 점이다. `100 -> 200 -> 300 rows`에서 Full RAG 평균은 `282.657 -> 425.143 -> 707.365 ms`로 증가했다.
- 증가분은 DB read보다 scoring 단계에서 더 크게 나타났다. scoring 평균은 `198.235 -> 339.965 -> 604.166 ms`로 증가했다.
- A안은 answer cache 구조라 hit된 질문은 LLM까지 생략하지만, route를 통과하지 못한 질문은 계속 RAG fallback을 탄다. 따라서 A repeat는 NoCache보다 낮아지지만 scale 증가 영향은 여전히 받는다.
- B안은 반복 실행에서 50개 모두 context cache hit이 되어 Full RAG가 0회가 되었다. 이 경우 scale이 커져도 평균 total은 `50-58 ms` 수준으로 유지되었다.
- B안은 context pack을 재사용한 뒤 LLM은 다시 호출하는 구조이므로, 실제 LLM을 붙이면 반복 실행의 예상 총 시간은 약 `750 ms/query` 수준이다.
- 이 결과는 cache가 RAG 비용을 줄이는 효과를 보여준다. 다만 mock LLM 기준에서는 LLM 절감 효과가 보이지 않으므로, 최종 해석에서는 `Est. Total+LLM`을 함께 보는 편이 적절하다.
