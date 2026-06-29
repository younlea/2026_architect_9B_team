# DP3 PoC Result Raw - Desktop

이 문서는 Desktop 환경에서 DP3 PoC를 처음부터 다시 실행하며 기록하는 Raw Data 문서다.

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

## 2. TC1 Cache Benefit

TC1은 RAGBench `techqa` 100 rows 기준으로 NoCache, A안 Verified Answer Cache, B안 Incremental Context Cache의 cache off / miss / hit 수행시간을 비교한다.

### 공통 세팅

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
| Warm-up query 수 | 3 |
| Requested version | V1 |
| User scope | A |
| LLM | Mock |
| Route threshold | 0.70 |
| Cache hit threshold | 0.86 |
| Estimated long LLM latency | 700.5 ms/call |
| Route pool mode | sampled |
| Route pool sample rate | 10% |
| Route pool min per dataset | 5 |
| Route pool seed | 42 |
| Query seed | 7 |

`Total (RAG+LLM) avg = Total avg + (LLM calls / Total) * 700.5 ms`

### 2-1. Reranker Off

#### 세팅

| 항목 | 값 |
|---|---|
| Reranker | Off |
| Reranker requested device | N/A |
| Reranker resolved device | N/A |
| Rerank model | N/A |
| Rerank candidates | N/A |
| Route threshold | 0.7 |
| Route pool sample rate | 10% |
| Route pool min per dataset | 5 |
| Route pool seed | 42 |
| Query seed | 7 |
| Route pool | `ragbench:techqa:test` 10개 / 100 |
| Route pool indexes | `3, 13, 14, 17, 28, 31, 35, 81, 86, 94` |
| Run log file | `20260629_135829_cache_techqa_rerank-off_na_5dd2177d.json` |
| Job ID | `5dd2177d-53c3-49cb-a98c-3e449dbe6814` |
| Saved at | 20260629_135829 |

#### 전체 요약

| Mode | Total | Route passed | Cache hit | Validation passed | RAG fallback | LLM calls | Total avg | Total (RAG+LLM) avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 114.120 ms | 814.620 ms |
| A first | 50 | 12 | 2 | 2 | 48 | 48 | 120.507 ms | 792.987 ms |
| A repeat | 50 | 12 | 12 | 12 | 38 | 38 | 98.665 ms | 631.045 ms |
| B first | 50 | 0 | 3 | 3 | 47 | 50 | 124.686 ms | 825.186 ms |
| B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 24.170 ms | 724.670 ms |

#### Decision reasons

| Mode | Reason | Count |
|---|---|---:|
| NoCache | `no_cache_full_rag` | 50 |
| A first | `embedding_score_below_threshold` | 38 |
| A first | `cache_candidate_not_found_fallback_to_roi_rag` | 10 |
| A first | `answer_cache_hit_valid` | 2 |
| A repeat | `embedding_score_below_threshold` | 38 |
| A repeat | `answer_cache_hit_valid` | 12 |
| B first | `context_cache_candidate_not_found_full_fallback` | 1 |
| B first | `context_cache_similarity_below_threshold` | 46 |
| B first | `context_cache_hit_all_valid` | 3 |
| B repeat | `context_cache_hit_all_valid` | 50 |

#### NoCache timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| NoCache | Total | 105.919 ms | 200.038 ms | 114.120 ms | 50 |
| NoCache | Total (RAG+LLM) | 806.419 ms | 900.538 ms | 814.620 ms | 50 |
| NoCache | Embedding | 4.943 ms | 16.317 ms | 8.099 ms | 50 |
| NoCache | Full retrieval DB | 13.734 ms | 16.117 ms | 14.453 ms | 50 |
| NoCache | Full retrieval scoring | 83.361 ms | 176.960 ms | 90.377 ms | 50 |
| NoCache | Full retrieval score sort | 0.114 ms | 0.157 ms | 0.134 ms | 50 |
| NoCache | Full retrieval rerank | 0.000 ms | 0.000 ms | 0.000 ms | 50 |
| NoCache | Full retrieval total | 98.062 ms | 191.253 ms | 104.964 ms | 50 |
| NoCache | Prompt build | 0.009 ms | 0.024 ms | 0.013 ms | 50 |
| NoCache | Mock LLM | 0.151 ms | 0.363 ms | 0.257 ms | 50 |

#### A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 13.438 ms | 136.079 ms | 120.507 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 792.987 ms | 50 |
| A first | Embedding | 4.812 ms | 14.409 ms | 8.014 ms | 50 |
| A first | Route | 0.837 ms | 1.225 ms | 0.935 ms | 50 |
| A first | Cache lookup | 0.513 ms | 2.333 ms | 1.102 ms | 12 |
| A first | Validation | 0.000 ms | 2.868 ms | 0.462 ms | 12 |
| A first | RAG DB | 13.594 ms | 17.234 ms | 14.549 ms | 48 |
| A first | RAG scoring | 82.460 ms | 93.973 ms | 87.786 ms | 48 |
| A first | RAG score sort | 0.124 ms | 0.179 ms | 0.149 ms | 48 |
| A first | RAG rerank | 0.000 ms | 0.000 ms | 0.000 ms | 48 |
| A first | RAG total | 96.987 ms | 109.772 ms | 102.484 ms | 48 |
| A first | Prompt build | 0.009 ms | 0.023 ms | 0.013 ms | 48 |
| A first | Mock LLM | 0.160 ms | 0.329 ms | 0.250 ms | 48 |
| A first | Cache store | 6.541 ms | 19.768 ms | 12.383 ms | 48 |
| A repeat | Total | 10.311 ms | 147.238 ms | 98.665 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 631.045 ms | 50 |
| A repeat | Embedding | 4.968 ms | 14.812 ms | 7.952 ms | 50 |
| A repeat | Route | 0.836 ms | 1.353 ms | 0.922 ms | 50 |
| A repeat | Cache lookup | 0.693 ms | 3.484 ms | 2.048 ms | 12 |
| A repeat | Validation | 2.555 ms | 3.144 ms | 2.767 ms | 12 |
| A repeat | RAG DB | 13.611 ms | 18.989 ms | 14.565 ms | 38 |
| A repeat | RAG scoring | 82.589 ms | 101.214 ms | 88.383 ms | 38 |
| A repeat | RAG score sort | 0.126 ms | 0.168 ms | 0.144 ms | 38 |
| A repeat | RAG rerank | 0.000 ms | 0.000 ms | 0.000 ms | 38 |
| A repeat | RAG total | 97.239 ms | 119.728 ms | 103.093 ms | 38 |
| A repeat | Prompt build | 0.010 ms | 0.015 ms | 0.012 ms | 38 |
| A repeat | Mock LLM | 0.159 ms | 0.329 ms | 0.251 ms | 38 |
| A repeat | Cache store | 7.008 ms | 21.262 ms | 12.590 ms | 38 |

#### B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 16.698 ms | 155.218 ms | 124.686 ms | 50 |
| B first | Total (RAG+LLM) | 717.198 ms | 855.718 ms | 825.186 ms | 50 |
| B first | Embedding | 5.057 ms | 14.423 ms | 8.130 ms | 50 |
| B first | Cache lookup DB | 0.313 ms | 2.010 ms | 1.219 ms | 50 |
| B first | Cache lookup scoring | 0.000 ms | 5.130 ms | 2.772 ms | 50 |
| B first | Cache lookup total | 0.313 ms | 7.126 ms | 3.991 ms | 50 |
| B first | Validation | 2.500 ms | 2.608 ms | 2.555 ms | 3 |
| B first | Full retrieval DB | 12.730 ms | 24.252 ms | 14.210 ms | 47 |
| B first | Full retrieval scoring | 82.739 ms | 106.725 ms | 88.741 ms | 47 |
| B first | Full retrieval score sort | 0.120 ms | 0.156 ms | 0.139 ms | 47 |
| B first | Full retrieval rerank | 0.000 ms | 0.000 ms | 0.000 ms | 47 |
| B first | Full retrieval total | 97.798 ms | 131.128 ms | 103.090 ms | 47 |
| B first | Prompt build | 0.008 ms | 0.020 ms | 0.014 ms | 50 |
| B first | Mock LLM | 0.160 ms | 0.511 ms | 0.255 ms | 50 |
| B first | Cache store | 6.872 ms | 28.711 ms | 14.243 ms | 47 |
| B repeat | Total | 15.846 ms | 143.091 ms | 24.170 ms | 50 |
| B repeat | Total (RAG+LLM) | 716.346 ms | 843.591 ms | 724.670 ms | 50 |
| B repeat | Embedding | 4.474 ms | 126.327 ms | 12.267 ms | 50 |
| B repeat | Cache lookup DB | 2.044 ms | 6.182 ms | 2.338 ms | 50 |
| B repeat | Cache lookup scoring | 4.733 ms | 6.901 ms | 5.301 ms | 50 |
| B repeat | Cache lookup total | 6.803 ms | 12.683 ms | 7.639 ms | 50 |
| B repeat | Validation | 2.133 ms | 4.810 ms | 2.738 ms | 50 |
| B repeat | Prompt build | 0.006 ms | 0.017 ms | 0.008 ms | 50 |
| B repeat | Mock LLM | 0.157 ms | 0.486 ms | 0.274 ms | 50 |

### 2-2. Reranker On (CPU)

#### 세팅

| 항목 | 값 |
|---|---|
| Reranker | On |
| Reranker requested device | cpu |
| Reranker resolved device | cpu |
| Rerank model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Rerank candidates | 30 |
| Route threshold | 0.7 |
| Route pool sample rate | 10% |
| Route pool min per dataset | 5 |
| Route pool seed | 42 |
| Query seed | 7 |
| Route pool | `ragbench:techqa:test` 10개 / 100 |
| Route pool indexes | `3, 13, 14, 17, 28, 31, 35, 81, 86, 94` |
| Run log file | `20260629_142123_cache_techqa_rerank-on_cpu_0862699a.json` |
| Job ID | `0862699a-dbdc-4528-b03b-9fe19d202c7d` |
| Saved at | 20260629_142123 |

#### 전체 요약

| Mode | Total | Route passed | Cache hit | Validation passed | RAG fallback | LLM calls | Total avg | Total (RAG+LLM) avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 863.417 ms | 1563.917 ms |
| A first | 50 | 12 | 2 | 2 | 48 | 48 | 869.471 ms | 1541.951 ms |
| A repeat | 50 | 12 | 12 | 12 | 38 | 38 | 679.858 ms | 1212.238 ms |
| B first | 50 | 0 | 3 | 3 | 47 | 50 | 849.708 ms | 1550.208 ms |
| B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 21.373 ms | 721.873 ms |

#### Decision reasons

| Mode | Reason | Count |
|---|---|---:|
| NoCache | `no_cache_full_rag` | 50 |
| A first | `embedding_score_below_threshold` | 38 |
| A first | `cache_candidate_not_found_fallback_to_roi_rag` | 10 |
| A first | `answer_cache_hit_valid` | 2 |
| A repeat | `embedding_score_below_threshold` | 38 |
| A repeat | `answer_cache_hit_valid` | 12 |
| B first | `context_cache_candidate_not_found_full_fallback` | 1 |
| B first | `context_cache_similarity_below_threshold` | 46 |
| B first | `context_cache_hit_all_valid` | 3 |
| B repeat | `context_cache_hit_all_valid` | 50 |

#### NoCache timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| NoCache | Total | 826.424 ms | 947.934 ms | 863.417 ms | 50 |
| NoCache | Total (RAG+LLM) | 1526.924 ms | 1648.434 ms | 1563.917 ms | 50 |
| NoCache | Embedding | 5.651 ms | 28.338 ms | 10.073 ms | 50 |
| NoCache | Full retrieval DB | 14.286 ms | 25.394 ms | 18.183 ms | 50 |
| NoCache | Full retrieval scoring | 88.701 ms | 114.285 ms | 96.462 ms | 50 |
| NoCache | Full retrieval score sort | 0.138 ms | 0.298 ms | 0.162 ms | 50 |
| NoCache | Full retrieval rerank | 702.636 ms | 814.212 ms | 736.928 ms | 50 |
| NoCache | Full retrieval total | 813.190 ms | 933.984 ms | 851.735 ms | 50 |
| NoCache | Prompt build | 0.025 ms | 0.075 ms | 0.031 ms | 50 |
| NoCache | Mock LLM | 0.125 ms | 0.478 ms | 0.320 ms | 50 |

#### A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 15.891 ms | 1088.323 ms | 869.471 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 1541.951 ms | 50 |
| A first | Embedding | 5.304 ms | 17.620 ms | 9.270 ms | 50 |
| A first | Route | 0.730 ms | 1.234 ms | 0.975 ms | 50 |
| A first | Cache lookup | 0.536 ms | 3.048 ms | 1.283 ms | 12 |
| A first | Validation | 0.000 ms | 3.363 ms | 0.537 ms | 12 |
| A first | RAG DB | 13.947 ms | 107.961 ms | 19.500 ms | 48 |
| A first | RAG scoring | 89.382 ms | 112.809 ms | 95.566 ms | 48 |
| A first | RAG score sort | 0.131 ms | 0.390 ms | 0.164 ms | 48 |
| A first | RAG rerank | 713.703 ms | 817.050 ms | 739.201 ms | 48 |
| A first | RAG total | 818.438 ms | 946.588 ms | 854.430 ms | 48 |
| A first | Prompt build | 0.026 ms | 0.041 ms | 0.033 ms | 48 |
| A first | Mock LLM | 0.139 ms | 0.657 ms | 0.356 ms | 48 |
| A first | Cache store | 6.637 ms | 150.769 ms | 38.559 ms | 48 |
| A repeat | Total | 12.315 ms | 940.703 ms | 679.858 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 1212.238 ms | 50 |
| A repeat | Embedding | 4.993 ms | 18.060 ms | 9.705 ms | 50 |
| A repeat | Route | 0.732 ms | 1.241 ms | 1.028 ms | 50 |
| A repeat | Cache lookup | 0.851 ms | 5.057 ms | 2.678 ms | 12 |
| A repeat | Validation | 2.713 ms | 4.560 ms | 3.639 ms | 12 |
| A repeat | RAG DB | 14.391 ms | 24.921 ms | 18.199 ms | 38 |
| A repeat | RAG scoring | 91.740 ms | 103.892 ms | 95.183 ms | 38 |
| A repeat | RAG score sort | 0.131 ms | 0.178 ms | 0.153 ms | 38 |
| A repeat | RAG rerank | 725.047 ms | 787.518 ms | 749.932 ms | 38 |
| A repeat | RAG total | 833.928 ms | 907.646 ms | 863.468 ms | 38 |
| A repeat | Prompt build | 0.027 ms | 0.040 ms | 0.033 ms | 38 |
| A repeat | Mock LLM | 0.191 ms | 0.713 ms | 0.375 ms | 38 |
| A repeat | Cache store | 6.670 ms | 20.726 ms | 13.376 ms | 38 |

#### B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 24.959 ms | 995.805 ms | 849.708 ms | 50 |
| B first | Total (RAG+LLM) | 725.459 ms | 1696.305 ms | 1550.208 ms | 50 |
| B first | Embedding | 5.029 ms | 18.949 ms | 9.206 ms | 50 |
| B first | Cache lookup DB | 0.314 ms | 3.389 ms | 1.590 ms | 50 |
| B first | Cache lookup scoring | 0.000 ms | 6.818 ms | 3.399 ms | 50 |
| B first | Cache lookup total | 0.322 ms | 10.023 ms | 4.989 ms | 50 |
| B first | Validation | 4.287 ms | 4.432 ms | 4.381 ms | 3 |
| B first | Full retrieval DB | 13.375 ms | 24.904 ms | 16.253 ms | 47 |
| B first | Full retrieval scoring | 89.809 ms | 104.604 ms | 95.375 ms | 47 |
| B first | Full retrieval score sort | 0.130 ms | 0.289 ms | 0.157 ms | 47 |
| B first | Full retrieval rerank | 729.515 ms | 793.318 ms | 751.593 ms | 47 |
| B first | Full retrieval total | 835.580 ms | 920.678 ms | 863.378 ms | 47 |
| B first | Prompt build | 0.009 ms | 0.048 ms | 0.036 ms | 50 |
| B first | Mock LLM | 0.141 ms | 0.573 ms | 0.382 ms | 50 |
| B first | Cache store | 7.126 ms | 124.613 ms | 21.979 ms | 47 |
| B repeat | Total | 17.021 ms | 45.123 ms | 21.373 ms | 50 |
| B repeat | Total (RAG+LLM) | 717.521 ms | 745.623 ms | 721.873 ms | 50 |
| B repeat | Embedding | 5.078 ms | 32.971 ms | 8.767 ms | 50 |
| B repeat | Cache lookup DB | 2.078 ms | 3.598 ms | 2.361 ms | 50 |
| B repeat | Cache lookup scoring | 4.819 ms | 7.220 ms | 5.563 ms | 50 |
| B repeat | Cache lookup total | 6.953 ms | 10.804 ms | 7.924 ms | 50 |
| B repeat | Validation | 2.569 ms | 4.579 ms | 3.019 ms | 50 |
| B repeat | Prompt build | 0.003 ms | 0.014 ms | 0.008 ms | 50 |
| B repeat | Mock LLM | 0.135 ms | 0.487 ms | 0.307 ms | 50 |

### 2-3. Reranker On (GPU)

#### 세팅

| 항목 | 값 |
|---|---|
| Reranker | On |
| Reranker requested device | cuda |
| Reranker resolved device | cuda |
| Rerank model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Rerank candidates | 30 |
| Route threshold | 0.7 |
| Route pool sample rate | 10% |
| Route pool min per dataset | 5 |
| Route pool seed | 42 |
| Query seed | 7 |
| Route pool | `ragbench:techqa:test` 10개 / 100 |
| Route pool indexes | `3, 13, 14, 17, 28, 31, 35, 81, 86, 94` |
| Run log file | `20260629_141731_cache_techqa_rerank-on_cuda_d7b12f84.json` |
| Job ID | `d7b12f84-795a-4068-bb32-abdbb48a8a94` |
| Saved at | 20260629_141731 |

#### Auto 실행 참고

| 항목 | 값 |
|---|---|
| Run log file | `20260629_140140_cache_techqa_rerank-on_auto_b38d8e96.json` |
| Reranker requested device | auto |
| Post-hoc resolved device | cuda |
| NoCache full_retrieval_rerank avg | 162.366 ms |
| 판단 근거 | `CrossEncoder._target_device` 및 `predict()` 동작 확인 |

#### 전체 요약

| Mode | Total | Route passed | Cache hit | Validation passed | RAG fallback | LLM calls | Total avg | Total (RAG+LLM) avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 281.960 ms | 982.460 ms |
| A first | 50 | 12 | 2 | 2 | 48 | 48 | 350.393 ms | 1022.873 ms |
| A repeat | 50 | 12 | 12 | 12 | 38 | 38 | 252.324 ms | 784.704 ms |
| B first | 50 | 0 | 3 | 3 | 47 | 50 | 305.130 ms | 1005.630 ms |
| B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 20.838 ms | 721.338 ms |

#### Decision reasons

| Mode | Reason | Count |
|---|---|---:|
| NoCache | `no_cache_full_rag` | 50 |
| A first | `embedding_score_below_threshold` | 38 |
| A first | `cache_candidate_not_found_fallback_to_roi_rag` | 10 |
| A first | `answer_cache_hit_valid` | 2 |
| A repeat | `embedding_score_below_threshold` | 38 |
| A repeat | `answer_cache_hit_valid` | 12 |
| B first | `context_cache_candidate_not_found_full_fallback` | 1 |
| B first | `context_cache_similarity_below_threshold` | 46 |
| B first | `context_cache_hit_all_valid` | 3 |
| B repeat | `context_cache_hit_all_valid` | 50 |

#### NoCache timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| NoCache | Total | 271.653 ms | 296.447 ms | 281.960 ms | 50 |
| NoCache | Total (RAG+LLM) | 972.153 ms | 996.947 ms | 982.460 ms | 50 |
| NoCache | Embedding | 5.132 ms | 15.030 ms | 8.267 ms | 50 |
| NoCache | Full retrieval DB | 14.371 ms | 23.488 ms | 16.323 ms | 50 |
| NoCache | Full retrieval scoring | 87.959 ms | 102.959 ms | 93.340 ms | 50 |
| NoCache | Full retrieval score sort | 0.131 ms | 0.195 ms | 0.153 ms | 50 |
| NoCache | Full retrieval rerank | 158.124 ms | 167.510 ms | 162.663 ms | 50 |
| NoCache | Full retrieval total | 263.473 ms | 288.817 ms | 272.478 ms | 50 |
| NoCache | Prompt build | 0.023 ms | 0.041 ms | 0.029 ms | 50 |
| NoCache | Mock LLM | 0.118 ms | 0.380 ms | 0.253 ms | 50 |

#### A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 14.756 ms | 540.826 ms | 350.393 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 1022.873 ms | 50 |
| A first | Embedding | 5.251 ms | 15.027 ms | 8.440 ms | 50 |
| A first | Route | 0.734 ms | 1.215 ms | 0.929 ms | 50 |
| A first | Cache lookup | 0.536 ms | 3.072 ms | 1.260 ms | 12 |
| A first | Validation | 0.000 ms | 3.203 ms | 0.531 ms | 12 |
| A first | RAG DB | 14.267 ms | 24.583 ms | 17.064 ms | 48 |
| A first | RAG scoring | 88.077 ms | 111.967 ms | 96.064 ms | 48 |
| A first | RAG score sort | 0.128 ms | 0.282 ms | 0.156 ms | 48 |
| A first | RAG rerank | 158.179 ms | 172.616 ms | 163.129 ms | 48 |
| A first | RAG total | 263.970 ms | 296.217 ms | 276.412 ms | 48 |
| A first | Prompt build | 0.022 ms | 0.052 ms | 0.030 ms | 48 |
| A first | Mock LLM | 0.132 ms | 0.473 ms | 0.256 ms | 48 |
| A first | Cache store | 6.720 ms | 256.282 ms | 77.223 ms | 48 |
| A repeat | Total | 10.717 ms | 442.020 ms | 252.324 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 784.704 ms | 50 |
| A repeat | Embedding | 5.355 ms | 21.604 ms | 8.846 ms | 50 |
| A repeat | Route | 0.727 ms | 1.178 ms | 0.896 ms | 50 |
| A repeat | Cache lookup | 0.741 ms | 3.476 ms | 2.278 ms | 12 |
| A repeat | Validation | 2.695 ms | 3.713 ms | 3.049 ms | 12 |
| A repeat | RAG DB | 14.191 ms | 22.608 ms | 16.288 ms | 38 |
| A repeat | RAG scoring | 88.122 ms | 103.069 ms | 93.123 ms | 38 |
| A repeat | RAG score sort | 0.130 ms | 0.181 ms | 0.152 ms | 38 |
| A repeat | RAG rerank | 157.069 ms | 174.836 ms | 163.494 ms | 38 |
| A repeat | RAG total | 263.399 ms | 293.546 ms | 273.057 ms | 38 |
| A repeat | Prompt build | 0.020 ms | 0.042 ms | 0.028 ms | 38 |
| A repeat | Mock LLM | 0.128 ms | 0.352 ms | 0.250 ms | 38 |
| A repeat | Cache store | 7.351 ms | 159.554 ms | 43.311 ms | 38 |

#### B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 22.041 ms | 428.437 ms | 305.130 ms | 50 |
| B first | Total (RAG+LLM) | 722.541 ms | 1128.937 ms | 1005.630 ms | 50 |
| B first | Embedding | 5.263 ms | 15.223 ms | 8.363 ms | 50 |
| B first | Cache lookup DB | 0.313 ms | 2.820 ms | 1.352 ms | 50 |
| B first | Cache lookup scoring | 0.000 ms | 6.559 ms | 3.171 ms | 50 |
| B first | Cache lookup total | 0.313 ms | 8.796 ms | 4.522 ms | 50 |
| B first | Validation | 4.271 ms | 4.908 ms | 4.542 ms | 3 |
| B first | Full retrieval DB | 12.946 ms | 24.188 ms | 15.696 ms | 47 |
| B first | Full retrieval scoring | 87.847 ms | 107.133 ms | 93.571 ms | 47 |
| B first | Full retrieval score sort | 0.132 ms | 0.328 ms | 0.154 ms | 47 |
| B first | Full retrieval rerank | 157.549 ms | 170.948 ms | 162.836 ms | 47 |
| B first | Full retrieval total | 261.836 ms | 290.427 ms | 272.258 ms | 47 |
| B first | Prompt build | 0.007 ms | 0.041 ms | 0.030 ms | 50 |
| B first | Mock LLM | 0.140 ms | 0.534 ms | 0.270 ms | 50 |
| B first | Cache store | 6.742 ms | 134.182 ms | 35.804 ms | 47 |
| B repeat | Total | 16.420 ms | 29.926 ms | 20.838 ms | 50 |
| B repeat | Total (RAG+LLM) | 716.920 ms | 730.426 ms | 721.338 ms | 50 |
| B repeat | Embedding | 5.114 ms | 16.012 ms | 8.392 ms | 50 |
| B repeat | Cache lookup DB | 2.072 ms | 3.539 ms | 2.312 ms | 50 |
| B repeat | Cache lookup scoring | 4.789 ms | 6.955 ms | 5.556 ms | 50 |
| B repeat | Cache lookup total | 6.891 ms | 10.399 ms | 7.869 ms | 50 |
| B repeat | Validation | 2.542 ms | 4.981 ms | 2.930 ms | 50 |
| B repeat | Prompt build | 0.006 ms | 0.032 ms | 0.009 ms | 50 |
| B repeat | Mock LLM | 0.132 ms | 0.503 ms | 0.286 ms | 50 |

## 3. TC2 Scale Cost

TC2는 RAGBench `techqa`의 row 수를 100, 200, 300으로 늘리며 NoCache, A안, B안의 수행시간 변화를 비교한다.

### 공통 세팅

| 항목 | 값 |
|---|---|
| Test Case | TC2 Scale Cost |
| Dataset | RAGBench `techqa` |
| Split | `test` |
| Scale rows | `100, 200, 300` |
| Test query 수 | 50 |
| Warm-up query 수 | 3 |
| Requested version | V1 |
| User scope | A |
| LLM | Mock |
| Route threshold | 0.70 |
| Cache hit threshold | 0.86 |
| Estimated long LLM latency | 700.5 ms/call |
| Route pool mode | sampled |
| Route pool sample rate | 10% |
| Route pool min per dataset | 5 |
| Route pool seed | 42 |
| Query seed | 7 |

`Total (RAG+LLM) avg = Total avg + (LLM calls / Total) * 700.5 ms`

### 3-1. Reranker Off

#### 세팅

| 항목 | 값 |
|---|---|
| Reranker | Off |
| Reranker requested device | N/A |
| Reranker resolved device | N/A |
| Rerank model | N/A |
| Rerank candidates | N/A |
| Scale rows | `100, 200, 300` |
| Route threshold | 0.7 |
| Route pool sample rate | 10% |
| Route pool min per dataset | 5 |
| Route pool seed | 42 |
| Query seed | 7 |
| Run log file | `20260629_142743_scalability_techqa_rerank-off_na_1e0884ae.json` |
| Job ID | `1e0884ae-8530-4c12-a1f9-b24ebc203636` |
| Saved at | 20260629_142743 |

#### Scale별 route pool

| Scale | Row count | Source ID | Base EU rows | Versioned EU rows | Route pool | Route pool indexes |
|---:|---:|---|---:|---:|---:|---|
| 1 | 100 | `dp3_ragbench_techqa_test_100` | 1,874 | 4,062 | 10 / 100 | `3, 13, 14, 17, 28, 31, 35, 81, 86, 94` |
| 2 | 200 | `dp3_ragbench_techqa_test_200` | 3,882 | 8,411 | 20 / 200 | `6, 7, 8, 22, 23, 26, 28, 35, 55, 57, 59, 62, 70, 108, 139, 151, 163, 173, 188, 189` |
| 3 | 300 | `dp3_ragbench_techqa_test_300` | 5,701 | 12,353 | 30 / 300 | `3, 12, 13, 15, 16, 44, 47, 52, 57, 71, 79, 81, 101, 110, 111, 112, 114, 119, 125, 140, 142, 172, 174, 194, 214, 216, 229, 258, 279, 287` |

#### Scale별 전체 요약

| Scale | Row count | Mode | Total | Route passed | Cache hit | Validation passed | RAG fallback | LLM calls | Total avg | Total (RAG+LLM) avg |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 117.902 ms | 818.402 ms |
| 1 | 100 | A first | 50 | 12 | 2 | 2 | 48 | 48 | 133.986 ms | 806.466 ms |
| 1 | 100 | A repeat | 50 | 12 | 12 | 12 | 38 | 38 | 105.757 ms | 638.137 ms |
| 1 | 100 | B first | 50 | 0 | 3 | 3 | 47 | 50 | 133.422 ms | 833.922 ms |
| 1 | 100 | B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 21.511 ms | 722.011 ms |
| 2 | 200 | NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 205.063 ms | 905.563 ms |
| 2 | 200 | A first | 50 | 14 | 1 | 1 | 49 | 49 | 237.848 ms | 924.338 ms |
| 2 | 200 | A repeat | 50 | 14 | 14 | 14 | 36 | 36 | 195.543 ms | 699.903 ms |
| 2 | 200 | B first | 50 | 0 | 5 | 5 | 45 | 50 | 201.426 ms | 901.926 ms |
| 2 | 200 | B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 22.391 ms | 722.891 ms |
| 3 | 300 | NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 304.907 ms | 1005.407 ms |
| 3 | 300 | A first | 50 | 17 | 3 | 3 | 47 | 47 | 327.020 ms | 985.490 ms |
| 3 | 300 | A repeat | 50 | 17 | 17 | 17 | 33 | 33 | 217.593 ms | 679.923 ms |
| 3 | 300 | B first | 50 | 0 | 5 | 5 | 45 | 50 | 289.881 ms | 990.381 ms |
| 3 | 300 | B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 20.898 ms | 721.398 ms |

#### Scale별 Decision reasons

| Scale | Row count | Mode | Reason | Count |
|---:|---:|---|---|---:|
| 1 | 100 | NoCache | `no_cache_full_rag` | 50 |
| 1 | 100 | A first | `embedding_score_below_threshold` | 38 |
| 1 | 100 | A first | `cache_candidate_not_found_fallback_to_roi_rag` | 10 |
| 1 | 100 | A first | `answer_cache_hit_valid` | 2 |
| 1 | 100 | A repeat | `embedding_score_below_threshold` | 38 |
| 1 | 100 | A repeat | `answer_cache_hit_valid` | 12 |
| 1 | 100 | B first | `context_cache_candidate_not_found_full_fallback` | 1 |
| 1 | 100 | B first | `context_cache_similarity_below_threshold` | 46 |
| 1 | 100 | B first | `context_cache_hit_all_valid` | 3 |
| 1 | 100 | B repeat | `context_cache_hit_all_valid` | 50 |
| 2 | 200 | NoCache | `no_cache_full_rag` | 50 |
| 2 | 200 | A first | `embedding_score_below_threshold` | 36 |
| 2 | 200 | A first | `cache_candidate_not_found_fallback_to_roi_rag` | 13 |
| 2 | 200 | A first | `answer_cache_hit_valid` | 1 |
| 2 | 200 | A repeat | `embedding_score_below_threshold` | 36 |
| 2 | 200 | A repeat | `answer_cache_hit_valid` | 14 |
| 2 | 200 | B first | `context_cache_candidate_not_found_full_fallback` | 1 |
| 2 | 200 | B first | `context_cache_similarity_below_threshold` | 44 |
| 2 | 200 | B first | `context_cache_hit_all_valid` | 5 |
| 2 | 200 | B repeat | `context_cache_hit_all_valid` | 50 |
| 3 | 300 | NoCache | `no_cache_full_rag` | 50 |
| 3 | 300 | A first | `embedding_score_below_threshold` | 33 |
| 3 | 300 | A first | `cache_candidate_not_found_fallback_to_roi_rag` | 14 |
| 3 | 300 | A first | `answer_cache_hit_valid` | 3 |
| 3 | 300 | A repeat | `embedding_score_below_threshold` | 33 |
| 3 | 300 | A repeat | `answer_cache_hit_valid` | 17 |
| 3 | 300 | B first | `context_cache_candidate_not_found_full_fallback` | 1 |
| 3 | 300 | B first | `context_cache_similarity_below_threshold` | 44 |
| 3 | 300 | B first | `context_cache_hit_all_valid` | 5 |
| 3 | 300 | B repeat | `context_cache_hit_all_valid` | 50 |

#### Scale 1 (100 rows) NoCache timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| NoCache | Total | 108.047 ms | 185.321 ms | 117.902 ms | 50 |
| NoCache | Total (RAG+LLM) | 808.547 ms | 885.821 ms | 818.402 ms | 50 |
| NoCache | Embedding | 5.031 ms | 69.504 ms | 9.474 ms | 50 |
| NoCache | Full retrieval DB | 14.051 ms | 19.618 ms | 15.414 ms | 50 |
| NoCache | Full retrieval scoring | 85.268 ms | 121.435 ms | 91.698 ms | 50 |
| NoCache | Full retrieval score sort | 0.129 ms | 0.195 ms | 0.152 ms | 50 |
| NoCache | Full retrieval rerank | 0.000 ms | 0.000 ms | 0.000 ms | 50 |
| NoCache | Full retrieval total | 100.170 ms | 136.152 ms | 107.264 ms | 50 |
| NoCache | Prompt build | 0.019 ms | 0.047 ms | 0.024 ms | 50 |
| NoCache | Mock LLM | 0.168 ms | 0.386 ms | 0.267 ms | 50 |

#### Scale 1 (100 rows) A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 14.602 ms | 186.189 ms | 133.986 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 806.466 ms | 50 |
| A first | Embedding | 5.095 ms | 15.126 ms | 8.426 ms | 50 |
| A first | Route | 0.731 ms | 1.217 ms | 0.874 ms | 50 |
| A first | Cache lookup | 0.535 ms | 3.061 ms | 1.214 ms | 12 |
| A first | Validation | 0.000 ms | 3.225 ms | 0.528 ms | 12 |
| A first | RAG DB | 14.251 ms | 22.468 ms | 16.287 ms | 48 |
| A first | RAG scoring | 87.681 ms | 143.904 ms | 97.376 ms | 48 |
| A first | RAG score sort | 0.131 ms | 0.310 ms | 0.158 ms | 48 |
| A first | RAG rerank | 0.000 ms | 0.000 ms | 0.000 ms | 48 |
| A first | RAG total | 105.068 ms | 163.757 ms | 113.821 ms | 48 |
| A first | Prompt build | 0.018 ms | 0.037 ms | 0.025 ms | 48 |
| A first | Mock LLM | 0.167 ms | 0.632 ms | 0.283 ms | 48 |
| A first | Cache store | 6.478 ms | 38.923 ms | 14.568 ms | 48 |
| A repeat | Total | 10.637 ms | 154.555 ms | 105.757 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 638.137 ms | 50 |
| A repeat | Embedding | 4.759 ms | 18.413 ms | 8.281 ms | 50 |
| A repeat | Route | 0.732 ms | 1.115 ms | 0.834 ms | 50 |
| A repeat | Cache lookup | 0.721 ms | 4.539 ms | 2.294 ms | 12 |
| A repeat | Validation | 2.747 ms | 4.625 ms | 3.119 ms | 12 |
| A repeat | RAG DB | 14.003 ms | 24.307 ms | 15.268 ms | 38 |
| A repeat | RAG scoring | 87.142 ms | 99.990 ms | 93.119 ms | 38 |
| A repeat | RAG score sort | 0.125 ms | 0.172 ms | 0.149 ms | 38 |
| A repeat | RAG rerank | 0.000 ms | 0.000 ms | 0.000 ms | 38 |
| A repeat | RAG total | 101.901 ms | 120.736 ms | 108.536 ms | 38 |
| A repeat | Prompt build | 0.018 ms | 0.028 ms | 0.022 ms | 38 |
| A repeat | Mock LLM | 0.174 ms | 0.389 ms | 0.268 ms | 38 |
| A repeat | Cache store | 7.011 ms | 26.099 ms | 15.891 ms | 38 |

#### Scale 1 (100 rows) B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 17.880 ms | 168.564 ms | 133.422 ms | 50 |
| B first | Total (RAG+LLM) | 718.380 ms | 869.064 ms | 833.922 ms | 50 |
| B first | Embedding | 5.058 ms | 15.397 ms | 8.357 ms | 50 |
| B first | Cache lookup DB | 0.271 ms | 3.123 ms | 1.354 ms | 50 |
| B first | Cache lookup scoring | 0.000 ms | 6.194 ms | 3.113 ms | 50 |
| B first | Cache lookup total | 0.271 ms | 8.889 ms | 4.468 ms | 50 |
| B first | Validation | 2.620 ms | 3.361 ms | 2.885 ms | 3 |
| B first | Full retrieval DB | 13.304 ms | 24.247 ms | 15.491 ms | 47 |
| B first | Full retrieval scoring | 90.604 ms | 104.559 ms | 95.860 ms | 47 |
| B first | Full retrieval score sort | 0.136 ms | 0.326 ms | 0.157 ms | 47 |
| B first | Full retrieval rerank | 0.000 ms | 0.000 ms | 0.000 ms | 47 |
| B first | Full retrieval total | 105.297 ms | 124.414 ms | 111.508 ms | 47 |
| B first | Prompt build | 0.006 ms | 0.034 ms | 0.025 ms | 50 |
| B first | Mock LLM | 0.168 ms | 0.431 ms | 0.280 ms | 50 |
| B first | Cache store | 6.876 ms | 30.457 ms | 14.124 ms | 47 |
| B repeat | Total | 16.743 ms | 33.910 ms | 21.511 ms | 50 |
| B repeat | Total (RAG+LLM) | 717.243 ms | 734.410 ms | 722.011 ms | 50 |
| B repeat | Embedding | 5.123 ms | 19.638 ms | 8.778 ms | 50 |
| B repeat | Cache lookup DB | 2.111 ms | 3.705 ms | 2.444 ms | 50 |
| B repeat | Cache lookup scoring | 4.687 ms | 6.834 ms | 5.597 ms | 50 |
| B repeat | Cache lookup total | 6.890 ms | 10.471 ms | 8.042 ms | 50 |
| B repeat | Validation | 2.588 ms | 4.693 ms | 3.012 ms | 50 |
| B repeat | Prompt build | 0.004 ms | 0.011 ms | 0.007 ms | 50 |
| B repeat | Mock LLM | 0.185 ms | 0.411 ms | 0.299 ms | 50 |

#### Scale 2 (200 rows) NoCache timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| NoCache | Total | 191.725 ms | 221.087 ms | 205.063 ms | 50 |
| NoCache | Total (RAG+LLM) | 892.225 ms | 921.587 ms | 905.563 ms | 50 |
| NoCache | Embedding | 4.985 ms | 14.545 ms | 8.030 ms | 50 |
| NoCache | Full retrieval DB | 17.536 ms | 25.633 ms | 18.922 ms | 50 |
| NoCache | Full retrieval scoring | 165.838 ms | 188.077 ms | 176.371 ms | 50 |
| NoCache | Full retrieval score sort | 0.245 ms | 0.310 ms | 0.276 ms | 50 |
| NoCache | Full retrieval rerank | 0.000 ms | 0.000 ms | 0.000 ms | 50 |
| NoCache | Full retrieval total | 183.940 ms | 206.807 ms | 195.570 ms | 50 |
| NoCache | Prompt build | 0.018 ms | 0.035 ms | 0.022 ms | 50 |
| NoCache | Mock LLM | 0.175 ms | 0.363 ms | 0.272 ms | 50 |

#### Scale 2 (200 rows) A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 12.649 ms | 327.395 ms | 237.848 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 924.338 ms | 50 |
| A first | Embedding | 5.106 ms | 66.526 ms | 9.341 ms | 50 |
| A first | Route | 1.143 ms | 1.948 ms | 1.362 ms | 50 |
| A first | Cache lookup | 0.547 ms | 1.645 ms | 0.889 ms | 14 |
| A first | Validation | 0.000 ms | 3.515 ms | 0.251 ms | 14 |
| A first | RAG DB | 17.131 ms | 26.416 ms | 18.880 ms | 49 |
| A first | RAG scoring | 168.059 ms | 185.409 ms | 175.775 ms | 49 |
| A first | RAG score sort | 0.246 ms | 0.332 ms | 0.291 ms | 49 |
| A first | RAG rerank | 0.000 ms | 0.000 ms | 0.000 ms | 49 |
| A first | RAG total | 185.553 ms | 207.889 ms | 194.946 ms | 49 |
| A first | Prompt build | 0.017 ms | 0.042 ms | 0.022 ms | 49 |
| A first | Mock LLM | 0.182 ms | 0.367 ms | 0.279 ms | 49 |
| A first | Cache store | 6.424 ms | 112.799 ms | 35.119 ms | 49 |
| A repeat | Total | 11.463 ms | 390.242 ms | 195.543 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 699.903 ms | 50 |
| A repeat | Embedding | 4.926 ms | 117.815 ms | 10.661 ms | 50 |
| A repeat | Route | 1.150 ms | 1.925 ms | 1.526 ms | 50 |
| A repeat | Cache lookup | 0.853 ms | 2.138 ms | 1.402 ms | 14 |
| A repeat | Validation | 2.771 ms | 3.338 ms | 3.099 ms | 14 |
| A repeat | RAG DB | 17.498 ms | 123.968 ms | 23.407 ms | 36 |
| A repeat | RAG scoring | 169.839 ms | 251.312 ms | 180.316 ms | 36 |
| A repeat | RAG score sort | 0.275 ms | 0.712 ms | 0.317 ms | 36 |
| A repeat | RAG rerank | 0.000 ms | 0.000 ms | 0.000 ms | 36 |
| A repeat | RAG total | 188.217 ms | 298.644 ms | 204.040 ms | 36 |
| A repeat | Prompt build | 0.018 ms | 0.031 ms | 0.022 ms | 36 |
| A repeat | Mock LLM | 0.179 ms | 0.428 ms | 0.271 ms | 36 |
| A repeat | Cache store | 14.747 ms | 174.618 ms | 47.394 ms | 36 |

#### Scale 2 (200 rows) B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 14.836 ms | 242.407 ms | 201.426 ms | 50 |
| B first | Total (RAG+LLM) | 715.336 ms | 942.907 ms | 901.926 ms | 50 |
| B first | Embedding | 4.984 ms | 15.063 ms | 8.024 ms | 50 |
| B first | Cache lookup DB | 0.269 ms | 2.323 ms | 1.291 ms | 50 |
| B first | Cache lookup scoring | 0.000 ms | 6.433 ms | 2.981 ms | 50 |
| B first | Cache lookup total | 0.269 ms | 8.756 ms | 4.272 ms | 50 |
| B first | Validation | 2.475 ms | 4.076 ms | 2.953 ms | 5 |
| B first | Full retrieval DB | 16.230 ms | 31.309 ms | 18.289 ms | 45 |
| B first | Full retrieval scoring | 162.405 ms | 188.747 ms | 172.517 ms | 45 |
| B first | Full retrieval score sort | 0.253 ms | 0.351 ms | 0.297 ms | 45 |
| B first | Full retrieval rerank | 0.000 ms | 0.000 ms | 0.000 ms | 45 |
| B first | Full retrieval total | 178.954 ms | 212.717 ms | 191.103 ms | 45 |
| B first | Prompt build | 0.006 ms | 0.031 ms | 0.023 ms | 50 |
| B first | Mock LLM | 0.179 ms | 0.440 ms | 0.276 ms | 50 |
| B first | Cache store | 10.630 ms | 19.701 ms | 15.873 ms | 45 |
| B repeat | Total | 15.865 ms | 109.189 ms | 22.391 ms | 50 |
| B repeat | Total (RAG+LLM) | 716.365 ms | 809.689 ms | 722.891 ms | 50 |
| B repeat | Embedding | 4.989 ms | 92.382 ms | 9.765 ms | 50 |
| B repeat | Cache lookup DB | 2.021 ms | 3.537 ms | 2.382 ms | 50 |
| B repeat | Cache lookup scoring | 4.586 ms | 7.054 ms | 5.394 ms | 50 |
| B repeat | Cache lookup total | 6.620 ms | 10.188 ms | 7.776 ms | 50 |
| B repeat | Validation | 2.426 ms | 5.490 ms | 3.156 ms | 50 |
| B repeat | Prompt build | 0.006 ms | 0.030 ms | 0.008 ms | 50 |
| B repeat | Mock LLM | 0.186 ms | 0.586 ms | 0.313 ms | 50 |

#### Scale 3 (300 rows) NoCache timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| NoCache | Total | 287.406 ms | 333.551 ms | 304.907 ms | 50 |
| NoCache | Total (RAG+LLM) | 987.906 ms | 1034.051 ms | 1005.407 ms | 50 |
| NoCache | Embedding | 4.838 ms | 15.178 ms | 7.718 ms | 50 |
| NoCache | Full retrieval DB | 21.110 ms | 36.302 ms | 24.308 ms | 50 |
| NoCache | Full retrieval scoring | 254.461 ms | 289.717 ms | 270.524 ms | 50 |
| NoCache | Full retrieval score sort | 0.362 ms | 0.516 ms | 0.417 ms | 50 |
| NoCache | Full retrieval rerank | 0.000 ms | 0.000 ms | 0.000 ms | 50 |
| NoCache | Full retrieval total | 277.192 ms | 322.119 ms | 295.249 ms | 50 |
| NoCache | Prompt build | 0.020 ms | 0.029 ms | 0.024 ms | 50 |
| NoCache | Mock LLM | 0.162 ms | 0.396 ms | 0.266 ms | 50 |

#### Scale 3 (300 rows) A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 11.330 ms | 447.051 ms | 327.020 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 985.490 ms | 50 |
| A first | Embedding | 4.715 ms | 28.118 ms | 8.394 ms | 50 |
| A first | Route | 1.452 ms | 2.518 ms | 1.753 ms | 50 |
| A first | Cache lookup | 0.544 ms | 1.056 ms | 0.765 ms | 17 |
| A first | Validation | 0.000 ms | 3.146 ms | 0.535 ms | 17 |
| A first | RAG DB | 20.733 ms | 118.595 ms | 25.461 ms | 47 |
| A first | RAG scoring | 254.808 ms | 288.519 ms | 273.592 ms | 47 |
| A first | RAG score sort | 0.375 ms | 0.535 ms | 0.446 ms | 47 |
| A first | RAG rerank | 0.000 ms | 0.000 ms | 0.000 ms | 47 |
| A first | RAG total | 277.372 ms | 383.689 ms | 299.499 ms | 47 |
| A first | Prompt build | 0.020 ms | 0.035 ms | 0.025 ms | 47 |
| A first | Mock LLM | 0.138 ms | 0.504 ms | 0.271 ms | 47 |
| A first | Cache store | 9.397 ms | 128.909 ms | 35.256 ms | 47 |
| A repeat | Total | 9.641 ms | 357.856 ms | 217.593 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 679.923 ms | 50 |
| A repeat | Embedding | 4.703 ms | 15.823 ms | 7.732 ms | 50 |
| A repeat | Route | 1.447 ms | 2.421 ms | 1.673 ms | 50 |
| A repeat | Cache lookup | 0.716 ms | 1.480 ms | 1.020 ms | 17 |
| A repeat | Validation | 2.583 ms | 3.783 ms | 2.946 ms | 17 |
| A repeat | RAG DB | 21.307 ms | 42.012 ms | 23.713 ms | 33 |
| A repeat | RAG scoring | 253.018 ms | 290.131 ms | 272.941 ms | 33 |
| A repeat | RAG score sort | 0.386 ms | 1.221 ms | 0.477 ms | 33 |
| A repeat | RAG rerank | 0.000 ms | 0.000 ms | 0.000 ms | 33 |
| A repeat | RAG total | 275.153 ms | 331.128 ms | 297.132 ms | 33 |
| A repeat | Prompt build | 0.020 ms | 0.033 ms | 0.024 ms | 33 |
| A repeat | Mock LLM | 0.164 ms | 0.468 ms | 0.282 ms | 33 |
| A repeat | Cache store | 6.817 ms | 32.411 ms | 14.330 ms | 33 |

#### Scale 3 (300 rows) B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 14.532 ms | 421.885 ms | 289.881 ms | 50 |
| B first | Total (RAG+LLM) | 715.032 ms | 1122.385 ms | 990.381 ms | 50 |
| B first | Embedding | 4.585 ms | 14.218 ms | 7.354 ms | 50 |
| B first | Cache lookup DB | 0.307 ms | 2.869 ms | 1.244 ms | 50 |
| B first | Cache lookup scoring | 0.000 ms | 7.046 ms | 2.828 ms | 50 |
| B first | Cache lookup total | 0.307 ms | 9.915 ms | 4.072 ms | 50 |
| B first | Validation | 2.636 ms | 2.739 ms | 2.679 ms | 5 |
| B first | Full retrieval DB | 20.000 ms | 26.886 ms | 21.873 ms | 45 |
| B first | Full retrieval scoring | 250.421 ms | 351.993 ms | 265.103 ms | 45 |
| B first | Full retrieval score sort | 0.390 ms | 0.524 ms | 0.447 ms | 45 |
| B first | Full retrieval rerank | 0.000 ms | 0.000 ms | 0.000 ms | 45 |
| B first | Full retrieval total | 271.897 ms | 373.791 ms | 287.424 ms | 45 |
| B first | Prompt build | 0.007 ms | 0.034 ms | 0.025 ms | 50 |
| B first | Mock LLM | 0.158 ms | 0.368 ms | 0.263 ms | 50 |
| B first | Cache store | 9.542 ms | 33.998 ms | 18.459 ms | 45 |
| B repeat | Total | 15.551 ms | 32.939 ms | 20.898 ms | 50 |
| B repeat | Total (RAG+LLM) | 716.051 ms | 733.439 ms | 721.398 ms | 50 |
| B repeat | Embedding | 4.908 ms | 17.264 ms | 8.005 ms | 50 |
| B repeat | Cache lookup DB | 2.035 ms | 3.585 ms | 2.498 ms | 50 |
| B repeat | Cache lookup scoring | 4.621 ms | 7.014 ms | 5.518 ms | 50 |
| B repeat | Cache lookup total | 6.697 ms | 10.377 ms | 8.016 ms | 50 |
| B repeat | Validation | 2.478 ms | 5.115 ms | 3.205 ms | 50 |
| B repeat | Prompt build | 0.006 ms | 0.012 ms | 0.008 ms | 50 |
| B repeat | Mock LLM | 0.159 ms | 0.533 ms | 0.304 ms | 50 |

### 3-2. Reranker On (CPU)

#### 세팅

| 항목 | 값 |
|---|---|
| Reranker | On |
| Reranker requested device | cpu |
| Reranker resolved device | cpu |
| Rerank model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Rerank candidates | 30 |
| Scale rows | `100, 200, 300` |
| Route threshold | 0.7 |
| Route pool sample rate | 10% |
| Route pool min per dataset | 5 |
| Route pool seed | 42 |
| Query seed | 7 |
| Run log file | `20260629_143838_scalability_techqa_rerank-on_cpu_97eba4b8.json` |
| Job ID | `97eba4b8-b06a-4fc6-a4f0-0d2182f6435b` |
| Saved at | 20260629_143838 |

#### Scale별 route pool

| Scale | Row count | Source ID | Base EU rows | Versioned EU rows | Route pool | Route pool indexes |
|---:|---:|---|---:|---:|---:|---|
| 1 | 100 | `dp3_ragbench_techqa_test_100` | 1,874 | 4,062 | 10 / 100 | `3, 13, 14, 17, 28, 31, 35, 81, 86, 94` |
| 2 | 200 | `dp3_ragbench_techqa_test_200` | 3,882 | 8,411 | 20 / 200 | `6, 7, 8, 22, 23, 26, 28, 35, 55, 57, 59, 62, 70, 108, 139, 151, 163, 173, 188, 189` |
| 3 | 300 | `dp3_ragbench_techqa_test_300` | 5,701 | 12,353 | 30 / 300 | `3, 12, 13, 15, 16, 44, 47, 52, 57, 71, 79, 81, 101, 110, 111, 112, 114, 119, 125, 140, 142, 172, 174, 194, 214, 216, 229, 258, 279, 287` |

#### Scale별 전체 요약

| Scale | Row count | Mode | Total | Route passed | Cache hit | Validation passed | RAG fallback | LLM calls | Total avg | Total (RAG+LLM) avg |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 878.852 ms | 1579.352 ms |
| 1 | 100 | A first | 50 | 12 | 2 | 2 | 48 | 48 | 907.015 ms | 1579.495 ms |
| 1 | 100 | A repeat | 50 | 12 | 12 | 12 | 38 | 38 | 691.510 ms | 1223.890 ms |
| 1 | 100 | B first | 50 | 0 | 3 | 3 | 47 | 50 | 871.865 ms | 1572.365 ms |
| 1 | 100 | B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 22.971 ms | 723.471 ms |
| 2 | 200 | NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 962.775 ms | 1663.275 ms |
| 2 | 200 | A first | 50 | 14 | 1 | 1 | 49 | 49 | 975.253 ms | 1661.743 ms |
| 2 | 200 | A repeat | 50 | 14 | 14 | 14 | 36 | 36 | 716.781 ms | 1221.141 ms |
| 2 | 200 | B first | 50 | 0 | 5 | 5 | 45 | 50 | 896.763 ms | 1597.263 ms |
| 2 | 200 | B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 19.894 ms | 720.394 ms |
| 3 | 300 | NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 1066.503 ms | 1767.003 ms |
| 3 | 300 | A first | 50 | 17 | 3 | 3 | 47 | 47 | 1020.138 ms | 1678.608 ms |
| 3 | 300 | A repeat | 50 | 17 | 17 | 17 | 33 | 33 | 720.493 ms | 1182.823 ms |
| 3 | 300 | B first | 50 | 0 | 5 | 5 | 45 | 50 | 983.359 ms | 1683.859 ms |
| 3 | 300 | B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 20.818 ms | 721.318 ms |

#### Scale별 Decision reasons

| Scale | Row count | Mode | Reason | Count |
|---:|---:|---|---|---:|
| 1 | 100 | NoCache | `no_cache_full_rag` | 50 |
| 1 | 100 | A first | `embedding_score_below_threshold` | 38 |
| 1 | 100 | A first | `cache_candidate_not_found_fallback_to_roi_rag` | 10 |
| 1 | 100 | A first | `answer_cache_hit_valid` | 2 |
| 1 | 100 | A repeat | `embedding_score_below_threshold` | 38 |
| 1 | 100 | A repeat | `answer_cache_hit_valid` | 12 |
| 1 | 100 | B first | `context_cache_candidate_not_found_full_fallback` | 1 |
| 1 | 100 | B first | `context_cache_similarity_below_threshold` | 46 |
| 1 | 100 | B first | `context_cache_hit_all_valid` | 3 |
| 1 | 100 | B repeat | `context_cache_hit_all_valid` | 50 |
| 2 | 200 | NoCache | `no_cache_full_rag` | 50 |
| 2 | 200 | A first | `embedding_score_below_threshold` | 36 |
| 2 | 200 | A first | `cache_candidate_not_found_fallback_to_roi_rag` | 13 |
| 2 | 200 | A first | `answer_cache_hit_valid` | 1 |
| 2 | 200 | A repeat | `embedding_score_below_threshold` | 36 |
| 2 | 200 | A repeat | `answer_cache_hit_valid` | 14 |
| 2 | 200 | B first | `context_cache_candidate_not_found_full_fallback` | 1 |
| 2 | 200 | B first | `context_cache_similarity_below_threshold` | 44 |
| 2 | 200 | B first | `context_cache_hit_all_valid` | 5 |
| 2 | 200 | B repeat | `context_cache_hit_all_valid` | 50 |
| 3 | 300 | NoCache | `no_cache_full_rag` | 50 |
| 3 | 300 | A first | `embedding_score_below_threshold` | 33 |
| 3 | 300 | A first | `cache_candidate_not_found_fallback_to_roi_rag` | 14 |
| 3 | 300 | A first | `answer_cache_hit_valid` | 3 |
| 3 | 300 | A repeat | `embedding_score_below_threshold` | 33 |
| 3 | 300 | A repeat | `answer_cache_hit_valid` | 17 |
| 3 | 300 | B first | `context_cache_candidate_not_found_full_fallback` | 1 |
| 3 | 300 | B first | `context_cache_similarity_below_threshold` | 44 |
| 3 | 300 | B first | `context_cache_hit_all_valid` | 5 |
| 3 | 300 | B repeat | `context_cache_hit_all_valid` | 50 |

#### Scale 1 (100 rows) NoCache timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| NoCache | Total | 855.847 ms | 993.710 ms | 878.852 ms | 50 |
| NoCache | Total (RAG+LLM) | 1556.347 ms | 1694.210 ms | 1579.352 ms | 50 |
| NoCache | Embedding | 5.305 ms | 18.067 ms | 9.263 ms | 50 |
| NoCache | Full retrieval DB | 14.534 ms | 25.150 ms | 18.201 ms | 50 |
| NoCache | Full retrieval scoring | 89.001 ms | 110.157 ms | 95.224 ms | 50 |
| NoCache | Full retrieval score sort | 0.126 ms | 0.184 ms | 0.151 ms | 50 |
| NoCache | Full retrieval rerank | 728.937 ms | 876.899 ms | 754.473 ms | 50 |
| NoCache | Full retrieval total | 845.474 ms | 986.428 ms | 868.048 ms | 50 |
| NoCache | Prompt build | 0.021 ms | 0.057 ms | 0.031 ms | 50 |
| NoCache | Mock LLM | 0.124 ms | 0.567 ms | 0.264 ms | 50 |

#### Scale 1 (100 rows) A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 19.477 ms | 1020.670 ms | 907.015 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 1579.495 ms | 50 |
| A first | Embedding | 5.501 ms | 17.675 ms | 9.457 ms | 50 |
| A first | Route | 0.751 ms | 1.278 ms | 1.006 ms | 50 |
| A first | Cache lookup | 0.615 ms | 3.062 ms | 1.343 ms | 12 |
| A first | Validation | 0.000 ms | 4.484 ms | 0.632 ms | 12 |
| A first | RAG DB | 14.509 ms | 27.116 ms | 17.372 ms | 48 |
| A first | RAG scoring | 89.006 ms | 108.950 ms | 96.331 ms | 48 |
| A first | RAG score sort | 0.117 ms | 0.181 ms | 0.154 ms | 48 |
| A first | RAG rerank | 743.051 ms | 880.612 ms | 775.789 ms | 48 |
| A first | RAG total | 853.057 ms | 995.381 ms | 889.646 ms | 48 |
| A first | Prompt build | 0.022 ms | 0.050 ms | 0.033 ms | 48 |
| A first | Mock LLM | 0.126 ms | 0.519 ms | 0.284 ms | 48 |
| A first | Cache store | 10.330 ms | 115.432 ms | 42.325 ms | 48 |
| A repeat | Total | 13.896 ms | 971.731 ms | 691.510 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 1223.890 ms | 50 |
| A repeat | Embedding | 5.410 ms | 17.025 ms | 9.104 ms | 50 |
| A repeat | Route | 0.741 ms | 1.263 ms | 1.004 ms | 50 |
| A repeat | Cache lookup | 0.929 ms | 4.543 ms | 2.699 ms | 12 |
| A repeat | Validation | 2.387 ms | 4.534 ms | 3.854 ms | 12 |
| A repeat | RAG DB | 14.347 ms | 25.524 ms | 17.317 ms | 38 |
| A repeat | RAG scoring | 88.890 ms | 102.728 ms | 94.102 ms | 38 |
| A repeat | RAG score sort | 0.123 ms | 0.180 ms | 0.153 ms | 38 |
| A repeat | RAG rerank | 747.854 ms | 838.737 ms | 767.382 ms | 38 |
| A repeat | RAG total | 854.603 ms | 954.133 ms | 878.954 ms | 38 |
| A repeat | Prompt build | 0.028 ms | 0.058 ms | 0.035 ms | 38 |
| A repeat | Mock LLM | 0.134 ms | 0.511 ms | 0.316 ms | 38 |
| A repeat | Cache store | 6.574 ms | 46.871 ms | 14.078 ms | 38 |

#### Scale 1 (100 rows) B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 21.056 ms | 1001.395 ms | 871.865 ms | 50 |
| B first | Total (RAG+LLM) | 721.556 ms | 1701.895 ms | 1572.365 ms | 50 |
| B first | Embedding | 5.187 ms | 15.598 ms | 8.949 ms | 50 |
| B first | Cache lookup DB | 0.265 ms | 3.282 ms | 1.512 ms | 50 |
| B first | Cache lookup scoring | 0.000 ms | 6.779 ms | 3.432 ms | 50 |
| B first | Cache lookup total | 0.265 ms | 9.057 ms | 4.943 ms | 50 |
| B first | Validation | 2.686 ms | 4.480 ms | 3.728 ms | 3 |
| B first | Full retrieval DB | 13.695 ms | 24.889 ms | 16.587 ms | 47 |
| B first | Full retrieval scoring | 86.785 ms | 113.737 ms | 94.259 ms | 47 |
| B first | Full retrieval score sort | 0.122 ms | 0.366 ms | 0.152 ms | 47 |
| B first | Full retrieval rerank | 745.099 ms | 856.006 ms | 779.722 ms | 47 |
| B first | Full retrieval total | 856.407 ms | 976.161 ms | 890.721 ms | 47 |
| B first | Prompt build | 0.008 ms | 0.084 ms | 0.037 ms | 50 |
| B first | Mock LLM | 0.130 ms | 0.507 ms | 0.330 ms | 50 |
| B first | Cache store | 6.794 ms | 49.372 ms | 18.725 ms | 47 |
| B repeat | Total | 16.340 ms | 100.910 ms | 22.971 ms | 50 |
| B repeat | Total (RAG+LLM) | 716.840 ms | 801.410 ms | 723.471 ms | 50 |
| B repeat | Embedding | 5.257 ms | 85.579 ms | 9.980 ms | 50 |
| B repeat | Cache lookup DB | 2.094 ms | 3.762 ms | 2.515 ms | 50 |
| B repeat | Cache lookup scoring | 4.808 ms | 7.028 ms | 5.725 ms | 50 |
| B repeat | Cache lookup total | 6.960 ms | 10.654 ms | 8.241 ms | 50 |
| B repeat | Validation | 2.428 ms | 5.686 ms | 3.082 ms | 50 |
| B repeat | Prompt build | 0.006 ms | 0.013 ms | 0.008 ms | 50 |
| B repeat | Mock LLM | 0.121 ms | 0.460 ms | 0.278 ms | 50 |

#### Scale 2 (200 rows) NoCache timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| NoCache | Total | 935.805 ms | 988.464 ms | 962.775 ms | 50 |
| NoCache | Total (RAG+LLM) | 1636.305 ms | 1688.964 ms | 1663.275 ms | 50 |
| NoCache | Embedding | 5.004 ms | 15.948 ms | 8.464 ms | 50 |
| NoCache | Full retrieval DB | 17.707 ms | 31.807 ms | 20.355 ms | 50 |
| NoCache | Full retrieval scoring | 168.810 ms | 182.300 ms | 174.795 ms | 50 |
| NoCache | Full retrieval score sort | 0.224 ms | 0.321 ms | 0.275 ms | 50 |
| NoCache | Full retrieval rerank | 722.463 ms | 775.644 ms | 757.085 ms | 50 |
| NoCache | Full retrieval total | 918.310 ms | 980.283 ms | 952.510 ms | 50 |
| NoCache | Prompt build | 0.021 ms | 0.044 ms | 0.028 ms | 50 |
| NoCache | Mock LLM | 0.130 ms | 0.513 ms | 0.269 ms | 50 |

#### Scale 2 (200 rows) A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 11.607 ms | 1108.484 ms | 975.253 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 1661.743 ms | 50 |
| A first | Embedding | 5.188 ms | 16.141 ms | 8.558 ms | 50 |
| A first | Route | 1.060 ms | 1.801 ms | 1.372 ms | 50 |
| A first | Cache lookup | 0.538 ms | 1.519 ms | 0.918 ms | 14 |
| A first | Validation | 0.000 ms | 3.499 ms | 0.250 ms | 14 |
| A first | RAG DB | 17.768 ms | 29.867 ms | 20.680 ms | 49 |
| A first | RAG scoring | 169.310 ms | 183.398 ms | 175.476 ms | 49 |
| A first | RAG score sort | 0.245 ms | 0.691 ms | 0.301 ms | 49 |
| A first | RAG rerank | 742.631 ms | 823.232 ms | 769.508 ms | 49 |
| A first | RAG total | 941.642 ms | 1019.535 ms | 965.965 ms | 49 |
| A first | Prompt build | 0.024 ms | 0.074 ms | 0.032 ms | 49 |
| A first | Mock LLM | 0.140 ms | 0.470 ms | 0.298 ms | 49 |
| A first | Cache store | 6.506 ms | 136.907 ms | 16.869 ms | 49 |
| A repeat | Total | 11.061 ms | 1091.177 ms | 716.781 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 1221.141 ms | 50 |
| A repeat | Embedding | 5.222 ms | 17.784 ms | 8.698 ms | 50 |
| A repeat | Route | 1.058 ms | 1.950 ms | 1.374 ms | 50 |
| A repeat | Cache lookup | 0.781 ms | 2.574 ms | 1.591 ms | 14 |
| A repeat | Validation | 3.040 ms | 4.390 ms | 3.527 ms | 14 |
| A repeat | RAG DB | 17.856 ms | 31.828 ms | 20.854 ms | 36 |
| A repeat | RAG scoring | 166.965 ms | 271.749 ms | 176.707 ms | 36 |
| A repeat | RAG score sort | 0.255 ms | 0.359 ms | 0.293 ms | 36 |
| A repeat | RAG rerank | 733.754 ms | 798.628 ms | 766.013 ms | 36 |
| A repeat | RAG total | 926.244 ms | 1062.590 ms | 963.868 ms | 36 |
| A repeat | Prompt build | 0.023 ms | 0.036 ms | 0.029 ms | 36 |
| A repeat | Mock LLM | 0.141 ms | 0.460 ms | 0.294 ms | 36 |
| A repeat | Cache store | 8.208 ms | 27.960 ms | 13.808 ms | 36 |

#### Scale 2 (200 rows) B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 14.143 ms | 1059.276 ms | 896.763 ms | 50 |
| B first | Total (RAG+LLM) | 714.643 ms | 1759.776 ms | 1597.263 ms | 50 |
| B first | Embedding | 5.118 ms | 15.996 ms | 8.461 ms | 50 |
| B first | Cache lookup DB | 0.407 ms | 2.920 ms | 1.443 ms | 50 |
| B first | Cache lookup scoring | 0.001 ms | 6.691 ms | 3.290 ms | 50 |
| B first | Cache lookup total | 0.408 ms | 9.001 ms | 4.733 ms | 50 |
| B first | Validation | 2.579 ms | 4.295 ms | 3.043 ms | 5 |
| B first | Full retrieval DB | 16.787 ms | 31.338 ms | 19.036 ms | 45 |
| B first | Full retrieval scoring | 169.361 ms | 181.892 ms | 174.970 ms | 45 |
| B first | Full retrieval score sort | 0.252 ms | 0.350 ms | 0.294 ms | 45 |
| B first | Full retrieval rerank | 736.650 ms | 815.801 ms | 766.198 ms | 45 |
| B first | Full retrieval total | 933.442 ms | 1026.092 ms | 960.498 ms | 45 |
| B first | Prompt build | 0.007 ms | 0.049 ms | 0.030 ms | 50 |
| B first | Mock LLM | 0.138 ms | 0.505 ms | 0.289 ms | 50 |
| B first | Cache store | 9.568 ms | 66.311 ms | 17.559 ms | 45 |
| B repeat | Total | 15.967 ms | 30.750 ms | 19.894 ms | 50 |
| B repeat | Total (RAG+LLM) | 716.467 ms | 731.250 ms | 720.394 ms | 50 |
| B repeat | Embedding | 4.851 ms | 15.727 ms | 8.064 ms | 50 |
| B repeat | Cache lookup DB | 1.996 ms | 2.780 ms | 2.268 ms | 50 |
| B repeat | Cache lookup scoring | 4.621 ms | 6.767 ms | 5.259 ms | 50 |
| B repeat | Cache lookup total | 6.746 ms | 9.210 ms | 7.527 ms | 50 |
| B repeat | Validation | 2.502 ms | 4.234 ms | 2.727 ms | 50 |
| B repeat | Prompt build | 0.006 ms | 0.011 ms | 0.007 ms | 50 |
| B repeat | Mock LLM | 0.133 ms | 0.400 ms | 0.269 ms | 50 |

#### Scale 3 (300 rows) NoCache timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| NoCache | Total | 1027.348 ms | 1148.540 ms | 1066.503 ms | 50 |
| NoCache | Total (RAG+LLM) | 1727.848 ms | 1849.040 ms | 1767.003 ms | 50 |
| NoCache | Embedding | 4.652 ms | 14.989 ms | 8.196 ms | 50 |
| NoCache | Full retrieval DB | 21.314 ms | 39.433 ms | 25.430 ms | 50 |
| NoCache | Full retrieval scoring | 259.351 ms | 308.694 ms | 269.811 ms | 50 |
| NoCache | Full retrieval score sort | 0.342 ms | 0.511 ms | 0.424 ms | 50 |
| NoCache | Full retrieval rerank | 725.397 ms | 841.490 ms | 760.198 ms | 50 |
| NoCache | Full retrieval total | 1019.571 ms | 1137.895 ms | 1055.863 ms | 50 |
| NoCache | Prompt build | 0.023 ms | 0.053 ms | 0.030 ms | 50 |
| NoCache | Mock LLM | 0.144 ms | 0.549 ms | 0.264 ms | 50 |

#### Scale 3 (300 rows) A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 11.297 ms | 1184.359 ms | 1020.138 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 1678.608 ms | 50 |
| A first | Embedding | 5.368 ms | 56.162 ms | 8.768 ms | 50 |
| A first | Route | 1.365 ms | 2.619 ms | 1.698 ms | 50 |
| A first | Cache lookup | 0.560 ms | 1.205 ms | 0.836 ms | 17 |
| A first | Validation | 0.000 ms | 3.227 ms | 0.531 ms | 17 |
| A first | RAG DB | 20.995 ms | 36.351 ms | 23.825 ms | 47 |
| A first | RAG scoring | 258.091 ms | 366.072 ms | 271.427 ms | 47 |
| A first | RAG score sort | 0.392 ms | 0.561 ms | 0.450 ms | 47 |
| A first | RAG rerank | 727.079 ms | 801.887 ms | 753.183 ms | 47 |
| A first | RAG total | 1014.486 ms | 1141.021 ms | 1048.884 ms | 47 |
| A first | Prompt build | 0.021 ms | 0.040 ms | 0.028 ms | 47 |
| A first | Mock LLM | 0.172 ms | 0.521 ms | 0.263 ms | 47 |
| A first | Cache store | 6.672 ms | 88.337 ms | 22.473 ms | 47 |
| A repeat | Total | 9.975 ms | 1162.480 ms | 720.493 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 1182.823 ms | 50 |
| A repeat | Embedding | 4.739 ms | 15.301 ms | 7.945 ms | 50 |
| A repeat | Route | 1.354 ms | 2.322 ms | 1.808 ms | 50 |
| A repeat | Cache lookup | 0.746 ms | 1.571 ms | 1.241 ms | 17 |
| A repeat | Validation | 2.662 ms | 4.475 ms | 3.584 ms | 17 |
| A repeat | RAG DB | 21.116 ms | 31.057 ms | 23.392 ms | 33 |
| A repeat | RAG scoring | 255.810 ms | 275.467 ms | 265.468 ms | 33 |
| A repeat | RAG score sort | 0.382 ms | 0.834 ms | 0.472 ms | 33 |
| A repeat | RAG rerank | 730.994 ms | 835.722 ms | 761.851 ms | 33 |
| A repeat | RAG total | 1021.475 ms | 1139.077 ms | 1051.183 ms | 33 |
| A repeat | Prompt build | 0.022 ms | 0.050 ms | 0.030 ms | 33 |
| A repeat | Mock LLM | 0.185 ms | 0.377 ms | 0.261 ms | 33 |
| A repeat | Cache store | 10.494 ms | 88.041 ms | 20.808 ms | 33 |

#### Scale 3 (300 rows) B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 17.010 ms | 1214.338 ms | 983.359 ms | 50 |
| B first | Total (RAG+LLM) | 717.510 ms | 1914.838 ms | 1683.859 ms | 50 |
| B first | Embedding | 5.077 ms | 15.170 ms | 7.761 ms | 50 |
| B first | Cache lookup DB | 0.274 ms | 2.821 ms | 1.409 ms | 50 |
| B first | Cache lookup scoring | 0.000 ms | 6.743 ms | 3.258 ms | 50 |
| B first | Cache lookup total | 0.274 ms | 9.288 ms | 4.667 ms | 50 |
| B first | Validation | 2.581 ms | 4.392 ms | 3.403 ms | 5 |
| B first | Full retrieval DB | 19.879 ms | 38.231 ms | 22.844 ms | 45 |
| B first | Full retrieval scoring | 253.939 ms | 354.277 ms | 265.808 ms | 45 |
| B first | Full retrieval score sort | 0.403 ms | 0.546 ms | 0.464 ms | 45 |
| B first | Full retrieval rerank | 718.351 ms | 798.845 ms | 755.147 ms | 45 |
| B first | Full retrieval total | 1007.108 ms | 1094.405 ms | 1044.262 ms | 45 |
| B first | Prompt build | 0.007 ms | 0.045 ms | 0.029 ms | 50 |
| B first | Mock LLM | 0.177 ms | 0.401 ms | 0.266 ms | 50 |
| B first | Cache store | 7.032 ms | 117.309 ms | 30.267 ms | 45 |
| B repeat | Total | 15.930 ms | 32.021 ms | 20.818 ms | 50 |
| B repeat | Total (RAG+LLM) | 716.430 ms | 732.521 ms | 721.318 ms | 50 |
| B repeat | Embedding | 5.022 ms | 16.799 ms | 8.000 ms | 50 |
| B repeat | Cache lookup DB | 2.006 ms | 3.581 ms | 2.465 ms | 50 |
| B repeat | Cache lookup scoring | 4.559 ms | 7.309 ms | 5.625 ms | 50 |
| B repeat | Cache lookup total | 6.735 ms | 10.721 ms | 8.090 ms | 50 |
| B repeat | Validation | 2.441 ms | 4.536 ms | 3.053 ms | 50 |
| B repeat | Prompt build | 0.006 ms | 0.048 ms | 0.009 ms | 50 |
| B repeat | Mock LLM | 0.178 ms | 0.581 ms | 0.283 ms | 50 |

### 3-3. Reranker On (GPU)

#### 세팅

| 항목 | 값 |
|---|---|
| Reranker | On |
| Reranker requested device | cuda |
| Reranker resolved device | cuda |
| Rerank model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Rerank candidates | 30 |
| Scale rows | `100, 200, 300` |
| Route threshold | 0.7 |
| Route pool sample rate | 10% |
| Route pool min per dataset | 5 |
| Route pool seed | 42 |
| Query seed | 7 |
| Run log file | `20260629_145410_scalability_techqa_rerank-on_cuda_3c731308.json` |
| Job ID | `3c731308-d312-4708-9677-19903d777563` |
| Saved at | 20260629_145410 |

#### Scale별 route pool

| Scale | Row count | Source ID | Base EU rows | Versioned EU rows | Route pool | Route pool indexes |
|---:|---:|---|---:|---:|---:|---|
| 1 | 100 | `dp3_ragbench_techqa_test_100` | 1,874 | 4,062 | 10 / 100 | `3, 13, 14, 17, 28, 31, 35, 81, 86, 94` |
| 2 | 200 | `dp3_ragbench_techqa_test_200` | 3,882 | 8,411 | 20 / 200 | `6, 7, 8, 22, 23, 26, 28, 35, 55, 57, 59, 62, 70, 108, 139, 151, 163, 173, 188, 189` |
| 3 | 300 | `dp3_ragbench_techqa_test_300` | 5,701 | 12,353 | 30 / 300 | `3, 12, 13, 15, 16, 44, 47, 52, 57, 71, 79, 81, 101, 110, 111, 112, 114, 119, 125, 140, 142, 172, 174, 194, 214, 216, 229, 258, 279, 287` |

#### Scale별 전체 요약

| Scale | Row count | Mode | Total | Route passed | Cache hit | Validation passed | RAG fallback | LLM calls | Total avg | Total (RAG+LLM) avg |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 284.295 ms | 984.795 ms |
| 1 | 100 | A first | 50 | 12 | 2 | 2 | 48 | 48 | 290.151 ms | 962.631 ms |
| 1 | 100 | A repeat | 50 | 12 | 12 | 12 | 38 | 38 | 232.962 ms | 765.342 ms |
| 1 | 100 | B first | 50 | 0 | 3 | 3 | 47 | 50 | 286.939 ms | 987.439 ms |
| 1 | 100 | B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 21.690 ms | 722.190 ms |
| 2 | 200 | NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 357.349 ms | 1057.849 ms |
| 2 | 200 | A first | 50 | 14 | 1 | 1 | 49 | 49 | 415.767 ms | 1102.257 ms |
| 2 | 200 | A repeat | 50 | 14 | 14 | 14 | 36 | 36 | 306.025 ms | 810.385 ms |
| 2 | 200 | B first | 50 | 0 | 5 | 5 | 45 | 50 | 347.850 ms | 1048.350 ms |
| 2 | 200 | B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 19.353 ms | 719.853 ms |
| 3 | 300 | NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 445.118 ms | 1145.618 ms |
| 3 | 300 | A first | 50 | 17 | 3 | 3 | 47 | 47 | 438.694 ms | 1097.164 ms |
| 3 | 300 | A repeat | 50 | 17 | 17 | 17 | 33 | 33 | 312.205 ms | 774.535 ms |
| 3 | 300 | B first | 50 | 0 | 5 | 5 | 45 | 50 | 428.211 ms | 1128.711 ms |
| 3 | 300 | B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 18.406 ms | 718.906 ms |

#### Scale별 Decision reasons

| Scale | Row count | Mode | Reason | Count |
|---:|---:|---|---|---:|
| 1 | 100 | NoCache | `no_cache_full_rag` | 50 |
| 1 | 100 | A first | `embedding_score_below_threshold` | 38 |
| 1 | 100 | A first | `cache_candidate_not_found_fallback_to_roi_rag` | 10 |
| 1 | 100 | A first | `answer_cache_hit_valid` | 2 |
| 1 | 100 | A repeat | `embedding_score_below_threshold` | 38 |
| 1 | 100 | A repeat | `answer_cache_hit_valid` | 12 |
| 1 | 100 | B first | `context_cache_candidate_not_found_full_fallback` | 1 |
| 1 | 100 | B first | `context_cache_similarity_below_threshold` | 46 |
| 1 | 100 | B first | `context_cache_hit_all_valid` | 3 |
| 1 | 100 | B repeat | `context_cache_hit_all_valid` | 50 |
| 2 | 200 | NoCache | `no_cache_full_rag` | 50 |
| 2 | 200 | A first | `embedding_score_below_threshold` | 36 |
| 2 | 200 | A first | `cache_candidate_not_found_fallback_to_roi_rag` | 13 |
| 2 | 200 | A first | `answer_cache_hit_valid` | 1 |
| 2 | 200 | A repeat | `embedding_score_below_threshold` | 36 |
| 2 | 200 | A repeat | `answer_cache_hit_valid` | 14 |
| 2 | 200 | B first | `context_cache_candidate_not_found_full_fallback` | 1 |
| 2 | 200 | B first | `context_cache_similarity_below_threshold` | 44 |
| 2 | 200 | B first | `context_cache_hit_all_valid` | 5 |
| 2 | 200 | B repeat | `context_cache_hit_all_valid` | 50 |
| 3 | 300 | NoCache | `no_cache_full_rag` | 50 |
| 3 | 300 | A first | `embedding_score_below_threshold` | 33 |
| 3 | 300 | A first | `cache_candidate_not_found_fallback_to_roi_rag` | 14 |
| 3 | 300 | A first | `answer_cache_hit_valid` | 3 |
| 3 | 300 | A repeat | `embedding_score_below_threshold` | 33 |
| 3 | 300 | A repeat | `answer_cache_hit_valid` | 17 |
| 3 | 300 | B first | `context_cache_candidate_not_found_full_fallback` | 1 |
| 3 | 300 | B first | `context_cache_similarity_below_threshold` | 44 |
| 3 | 300 | B first | `context_cache_hit_all_valid` | 5 |
| 3 | 300 | B repeat | `context_cache_hit_all_valid` | 50 |

#### Scale 1 (100 rows) NoCache timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| NoCache | Total | 270.697 ms | 323.671 ms | 284.295 ms | 50 |
| NoCache | Total (RAG+LLM) | 971.197 ms | 1024.171 ms | 984.795 ms | 50 |
| NoCache | Embedding | 4.876 ms | 14.755 ms | 8.568 ms | 50 |
| NoCache | Full retrieval DB | 14.360 ms | 24.578 ms | 17.284 ms | 50 |
| NoCache | Full retrieval scoring | 87.976 ms | 123.547 ms | 94.564 ms | 50 |
| NoCache | Full retrieval score sort | 0.129 ms | 0.193 ms | 0.154 ms | 50 |
| NoCache | Full retrieval rerank | 156.581 ms | 169.391 ms | 162.443 ms | 50 |
| NoCache | Full retrieval total | 261.567 ms | 310.703 ms | 274.444 ms | 50 |
| NoCache | Prompt build | 0.012 ms | 0.367 ms | 0.035 ms | 50 |
| NoCache | Mock LLM | 0.130 ms | 0.530 ms | 0.253 ms | 50 |

#### Scale 1 (100 rows) A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 13.659 ms | 336.036 ms | 290.151 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 962.631 ms | 50 |
| A first | Embedding | 5.546 ms | 34.008 ms | 9.503 ms | 50 |
| A first | Route | 0.746 ms | 1.236 ms | 0.940 ms | 50 |
| A first | Cache lookup | 0.517 ms | 3.023 ms | 1.232 ms | 12 |
| A first | Validation | 0.000 ms | 3.015 ms | 0.477 ms | 12 |
| A first | RAG DB | 14.060 ms | 24.666 ms | 18.252 ms | 48 |
| A first | RAG scoring | 87.041 ms | 116.557 ms | 94.533 ms | 48 |
| A first | RAG score sort | 0.127 ms | 0.196 ms | 0.151 ms | 48 |
| A first | RAG rerank | 158.336 ms | 168.209 ms | 161.335 ms | 48 |
| A first | RAG total | 263.090 ms | 305.841 ms | 274.272 ms | 48 |
| A first | Prompt build | 0.018 ms | 0.045 ms | 0.028 ms | 48 |
| A first | Mock LLM | 0.114 ms | 0.494 ms | 0.249 ms | 48 |
| A first | Cache store | 8.131 ms | 26.173 ms | 15.509 ms | 48 |
| A repeat | Total | 11.778 ms | 315.530 ms | 232.962 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 765.342 ms | 50 |
| A repeat | Embedding | 4.856 ms | 15.805 ms | 8.583 ms | 50 |
| A repeat | Route | 0.738 ms | 1.260 ms | 0.910 ms | 50 |
| A repeat | Cache lookup | 0.721 ms | 3.698 ms | 2.327 ms | 12 |
| A repeat | Validation | 2.703 ms | 3.222 ms | 2.914 ms | 12 |
| A repeat | RAG DB | 13.901 ms | 24.963 ms | 16.705 ms | 38 |
| A repeat | RAG scoring | 89.896 ms | 103.674 ms | 94.711 ms | 38 |
| A repeat | RAG score sort | 0.135 ms | 0.172 ms | 0.152 ms | 38 |
| A repeat | RAG rerank | 160.143 ms | 170.294 ms | 163.190 ms | 38 |
| A repeat | RAG total | 265.464 ms | 288.987 ms | 274.758 ms | 38 |
| A repeat | Prompt build | 0.018 ms | 0.036 ms | 0.028 ms | 38 |
| A repeat | Mock LLM | 0.112 ms | 0.411 ms | 0.242 ms | 38 |
| A repeat | Cache store | 10.879 ms | 22.880 ms | 16.471 ms | 38 |

#### Scale 1 (100 rows) B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 17.713 ms | 325.487 ms | 286.939 ms | 50 |
| B first | Total (RAG+LLM) | 718.213 ms | 1025.987 ms | 987.439 ms | 50 |
| B first | Embedding | 5.098 ms | 15.189 ms | 8.359 ms | 50 |
| B first | Cache lookup DB | 0.304 ms | 2.453 ms | 1.345 ms | 50 |
| B first | Cache lookup scoring | 0.001 ms | 6.308 ms | 3.214 ms | 50 |
| B first | Cache lookup total | 0.305 ms | 8.552 ms | 4.560 ms | 50 |
| B first | Validation | 2.493 ms | 2.677 ms | 2.589 ms | 3 |
| B first | Full retrieval DB | 13.179 ms | 24.823 ms | 15.726 ms | 47 |
| B first | Full retrieval scoring | 88.307 ms | 105.239 ms | 93.513 ms | 47 |
| B first | Full retrieval score sort | 0.134 ms | 0.305 ms | 0.157 ms | 47 |
| B first | Full retrieval rerank | 159.533 ms | 170.768 ms | 163.791 ms | 47 |
| B first | Full retrieval total | 264.473 ms | 288.518 ms | 273.187 ms | 47 |
| B first | Prompt build | 0.008 ms | 0.052 ms | 0.030 ms | 50 |
| B first | Mock LLM | 0.125 ms | 0.346 ms | 0.247 ms | 50 |
| B first | Cache store | 11.070 ms | 20.706 ms | 15.629 ms | 47 |
| B repeat | Total | 16.254 ms | 50.067 ms | 21.690 ms | 50 |
| B repeat | Total (RAG+LLM) | 716.754 ms | 750.567 ms | 722.190 ms | 50 |
| B repeat | Embedding | 5.151 ms | 37.793 ms | 9.059 ms | 50 |
| B repeat | Cache lookup DB | 2.088 ms | 3.599 ms | 2.393 ms | 50 |
| B repeat | Cache lookup scoring | 4.803 ms | 7.849 ms | 5.652 ms | 50 |
| B repeat | Cache lookup total | 6.911 ms | 10.726 ms | 8.045 ms | 50 |
| B repeat | Validation | 2.491 ms | 4.368 ms | 2.918 ms | 50 |
| B repeat | Prompt build | 0.006 ms | 0.032 ms | 0.008 ms | 50 |
| B repeat | Mock LLM | 0.129 ms | 0.470 ms | 0.284 ms | 50 |

#### Scale 2 (200 rows) NoCache timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| NoCache | Total | 341.792 ms | 381.647 ms | 357.349 ms | 50 |
| NoCache | Total (RAG+LLM) | 1042.292 ms | 1082.147 ms | 1057.849 ms | 50 |
| NoCache | Embedding | 5.233 ms | 16.068 ms | 8.274 ms | 50 |
| NoCache | Full retrieval DB | 17.407 ms | 27.995 ms | 19.843 ms | 50 |
| NoCache | Full retrieval scoring | 164.202 ms | 197.233 ms | 172.268 ms | 50 |
| NoCache | Full retrieval score sort | 0.244 ms | 0.332 ms | 0.277 ms | 50 |
| NoCache | Full retrieval rerank | 149.787 ms | 169.548 ms | 155.100 ms | 50 |
| NoCache | Full retrieval total | 333.045 ms | 370.087 ms | 347.488 ms | 50 |
| NoCache | Prompt build | 0.023 ms | 0.035 ms | 0.027 ms | 50 |
| NoCache | Mock LLM | 0.126 ms | 0.328 ms | 0.252 ms | 50 |

#### Scale 2 (200 rows) A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 10.854 ms | 534.397 ms | 415.767 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 1102.257 ms | 50 |
| A first | Embedding | 5.132 ms | 18.206 ms | 8.159 ms | 50 |
| A first | Route | 1.039 ms | 1.792 ms | 1.267 ms | 50 |
| A first | Cache lookup | 0.525 ms | 1.658 ms | 0.891 ms | 14 |
| A first | Validation | 0.000 ms | 3.132 ms | 0.224 ms | 14 |
| A first | RAG DB | 17.151 ms | 28.331 ms | 19.495 ms | 49 |
| A first | RAG scoring | 165.947 ms | 180.508 ms | 172.313 ms | 49 |
| A first | RAG score sort | 0.240 ms | 0.338 ms | 0.291 ms | 49 |
| A first | RAG rerank | 149.818 ms | 170.296 ms | 157.822 ms | 49 |
| A first | RAG total | 337.128 ms | 367.887 ms | 349.921 ms | 49 |
| A first | Prompt build | 0.021 ms | 0.042 ms | 0.028 ms | 49 |
| A first | Mock LLM | 0.132 ms | 0.334 ms | 0.257 ms | 49 |
| A first | Cache store | 9.443 ms | 165.495 ms | 62.859 ms | 49 |
| A repeat | Total | 10.892 ms | 562.902 ms | 306.025 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 810.385 ms | 50 |
| A repeat | Embedding | 5.346 ms | 14.568 ms | 8.177 ms | 50 |
| A repeat | Route | 1.053 ms | 1.789 ms | 1.326 ms | 50 |
| A repeat | Cache lookup | 0.851 ms | 2.696 ms | 1.497 ms | 14 |
| A repeat | Validation | 2.738 ms | 4.361 ms | 3.378 ms | 14 |
| A repeat | RAG DB | 17.180 ms | 119.844 ms | 23.499 ms | 36 |
| A repeat | RAG scoring | 164.979 ms | 183.657 ms | 172.169 ms | 36 |
| A repeat | RAG score sort | 0.265 ms | 0.590 ms | 0.315 ms | 36 |
| A repeat | RAG rerank | 150.017 ms | 162.643 ms | 155.285 ms | 36 |
| A repeat | RAG total | 336.368 ms | 457.651 ms | 351.268 ms | 36 |
| A repeat | Prompt build | 0.023 ms | 0.047 ms | 0.028 ms | 36 |
| A repeat | Mock LLM | 0.132 ms | 0.581 ms | 0.263 ms | 36 |
| A repeat | Cache store | 11.561 ms | 129.196 ms | 57.125 ms | 36 |

#### Scale 2 (200 rows) B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 14.940 ms | 428.814 ms | 347.850 ms | 50 |
| B first | Total (RAG+LLM) | 715.440 ms | 1129.314 ms | 1048.350 ms | 50 |
| B first | Embedding | 4.997 ms | 15.283 ms | 7.941 ms | 50 |
| B first | Cache lookup DB | 0.307 ms | 2.255 ms | 1.297 ms | 50 |
| B first | Cache lookup scoring | 0.000 ms | 6.053 ms | 3.162 ms | 50 |
| B first | Cache lookup total | 0.307 ms | 8.308 ms | 4.459 ms | 50 |
| B first | Validation | 2.904 ms | 4.215 ms | 3.420 ms | 5 |
| B first | Full retrieval DB | 16.279 ms | 29.325 ms | 18.452 ms | 45 |
| B first | Full retrieval scoring | 161.291 ms | 178.855 ms | 169.066 ms | 45 |
| B first | Full retrieval score sort | 0.248 ms | 1.367 ms | 0.325 ms | 45 |
| B first | Full retrieval rerank | 148.285 ms | 165.286 ms | 154.240 ms | 45 |
| B first | Full retrieval total | 329.555 ms | 366.212 ms | 342.083 ms | 45 |
| B first | Prompt build | 0.006 ms | 0.052 ms | 0.029 ms | 50 |
| B first | Mock LLM | 0.142 ms | 0.363 ms | 0.254 ms | 50 |
| B first | Cache store | 7.121 ms | 64.806 ms | 27.328 ms | 45 |
| B repeat | Total | 15.579 ms | 26.506 ms | 19.353 ms | 50 |
| B repeat | Total (RAG+LLM) | 716.079 ms | 727.006 ms | 719.853 ms | 50 |
| B repeat | Embedding | 4.944 ms | 14.718 ms | 7.740 ms | 50 |
| B repeat | Cache lookup DB | 1.972 ms | 2.854 ms | 2.182 ms | 50 |
| B repeat | Cache lookup scoring | 4.612 ms | 6.176 ms | 5.221 ms | 50 |
| B repeat | Cache lookup total | 6.622 ms | 8.494 ms | 7.403 ms | 50 |
| B repeat | Validation | 2.427 ms | 3.100 ms | 2.678 ms | 50 |
| B repeat | Prompt build | 0.007 ms | 0.009 ms | 0.007 ms | 50 |
| B repeat | Mock LLM | 0.126 ms | 0.549 ms | 0.284 ms | 50 |

#### Scale 3 (300 rows) NoCache timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| NoCache | Total | 430.214 ms | 462.549 ms | 445.118 ms | 50 |
| NoCache | Total (RAG+LLM) | 1130.714 ms | 1163.049 ms | 1145.618 ms | 50 |
| NoCache | Embedding | 4.730 ms | 14.602 ms | 7.481 ms | 50 |
| NoCache | Full retrieval DB | 20.831 ms | 37.593 ms | 23.097 ms | 50 |
| NoCache | Full retrieval scoring | 247.339 ms | 269.269 ms | 257.732 ms | 50 |
| NoCache | Full retrieval score sort | 0.372 ms | 0.505 ms | 0.423 ms | 50 |
| NoCache | Full retrieval rerank | 149.985 ms | 162.633 ms | 154.359 ms | 50 |
| NoCache | Full retrieval total | 420.252 ms | 453.193 ms | 435.611 ms | 50 |
| NoCache | Prompt build | 0.022 ms | 0.043 ms | 0.030 ms | 50 |
| NoCache | Mock LLM | 0.158 ms | 0.368 ms | 0.255 ms | 50 |

#### Scale 3 (300 rows) A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 11.186 ms | 571.790 ms | 438.694 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 1097.164 ms | 50 |
| A first | Embedding | 4.568 ms | 15.370 ms | 7.481 ms | 50 |
| A first | Route | 1.348 ms | 2.300 ms | 1.659 ms | 50 |
| A first | Cache lookup | 0.540 ms | 1.043 ms | 0.763 ms | 17 |
| A first | Validation | 0.000 ms | 4.166 ms | 0.554 ms | 17 |
| A first | RAG DB | 20.699 ms | 37.957 ms | 24.301 ms | 47 |
| A first | RAG scoring | 250.063 ms | 358.358 ms | 262.463 ms | 47 |
| A first | RAG score sort | 0.358 ms | 0.987 ms | 0.451 ms | 47 |
| A first | RAG rerank | 150.187 ms | 164.660 ms | 154.467 ms | 47 |
| A first | RAG total | 423.502 ms | 544.592 ms | 441.681 ms | 47 |
| A first | Prompt build | 0.024 ms | 0.038 ms | 0.029 ms | 47 |
| A first | Mock LLM | 0.163 ms | 0.423 ms | 0.257 ms | 47 |
| A first | Cache store | 6.903 ms | 31.780 ms | 12.868 ms | 47 |
| A repeat | Total | 11.248 ms | 496.153 ms | 312.205 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 774.535 ms | 50 |
| A repeat | Embedding | 4.935 ms | 14.770 ms | 7.624 ms | 50 |
| A repeat | Route | 1.319 ms | 2.383 ms | 1.684 ms | 50 |
| A repeat | Cache lookup | 0.760 ms | 1.769 ms | 1.134 ms | 17 |
| A repeat | Validation | 2.649 ms | 4.238 ms | 3.177 ms | 17 |
| A repeat | RAG DB | 20.899 ms | 31.476 ms | 23.093 ms | 33 |
| A repeat | RAG scoring | 253.541 ms | 275.667 ms | 261.701 ms | 33 |
| A repeat | RAG score sort | 0.407 ms | 0.538 ms | 0.460 ms | 33 |
| A repeat | RAG rerank | 150.303 ms | 163.191 ms | 154.331 ms | 33 |
| A repeat | RAG total | 429.781 ms | 461.673 ms | 439.586 ms | 33 |
| A repeat | Prompt build | 0.024 ms | 0.034 ms | 0.029 ms | 33 |
| A repeat | Mock LLM | 0.204 ms | 0.354 ms | 0.262 ms | 33 |
| A repeat | Cache store | 6.960 ms | 50.495 ms | 15.145 ms | 33 |

#### Scale 3 (300 rows) B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 14.447 ms | 543.970 ms | 428.211 ms | 50 |
| B first | Total (RAG+LLM) | 714.947 ms | 1244.470 ms | 1128.711 ms | 50 |
| B first | Embedding | 4.944 ms | 14.110 ms | 7.553 ms | 50 |
| B first | Cache lookup DB | 0.325 ms | 2.260 ms | 1.248 ms | 50 |
| B first | Cache lookup scoring | 0.001 ms | 5.691 ms | 2.854 ms | 50 |
| B first | Cache lookup total | 0.413 ms | 7.585 ms | 4.102 ms | 50 |
| B first | Validation | 2.539 ms | 4.251 ms | 2.993 ms | 5 |
| B first | Full retrieval DB | 20.127 ms | 33.912 ms | 22.586 ms | 45 |
| B first | Full retrieval scoring | 252.608 ms | 348.415 ms | 262.955 ms | 45 |
| B first | Full retrieval score sort | 0.408 ms | 0.520 ms | 0.454 ms | 45 |
| B first | Full retrieval rerank | 149.455 ms | 163.836 ms | 154.184 ms | 45 |
| B first | Full retrieval total | 428.243 ms | 519.698 ms | 440.179 ms | 45 |
| B first | Prompt build | 0.007 ms | 0.047 ms | 0.030 ms | 50 |
| B first | Mock LLM | 0.171 ms | 0.357 ms | 0.259 ms | 50 |
| B first | Cache store | 6.934 ms | 53.581 ms | 18.956 ms | 45 |
| B repeat | Total | 15.029 ms | 25.370 ms | 18.406 ms | 50 |
| B repeat | Total (RAG+LLM) | 715.529 ms | 725.870 ms | 718.906 ms | 50 |
| B repeat | Embedding | 4.543 ms | 14.157 ms | 7.148 ms | 50 |
| B repeat | Cache lookup DB | 1.961 ms | 3.433 ms | 2.117 ms | 50 |
| B repeat | Cache lookup scoring | 4.543 ms | 6.759 ms | 4.988 ms | 50 |
| B repeat | Cache lookup total | 6.548 ms | 10.192 ms | 7.104 ms | 50 |
| B repeat | Validation | 2.378 ms | 4.271 ms | 2.668 ms | 50 |
| B repeat | Prompt build | 0.007 ms | 0.012 ms | 0.007 ms | 50 |
| B repeat | Mock LLM | 0.160 ms | 0.448 ms | 0.265 ms | 50 |

## 4. TC3 Mixed Workload Performance

TBD

## 5. TC4 Similar Query Pair Quality

TBD
