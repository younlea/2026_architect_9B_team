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
| LLM | mock |
| Route threshold | 0.70 |
| Cache hit threshold | 0.86 |
| LLM latency basis | Mock LLM: `700.5 ms/call` 가상 보정 |
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
| Run log file | `20260629_201730_cache_techqa_rerank-off_na_a9652eb0.json` |
| Job ID | `a9652eb0-ffcd-4958-8549-6a65bdb5948b` |
| Saved at | 20260629_201730 |

#### 전체 요약

| Mode | Total | Route passed | Cache hit | Validation passed | RAG fallback | LLM calls | Total avg | Total (RAG+LLM) avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 120.509 ms | 821.009 ms |
| A first | 50 | 12 | 2 | 2 | 48 | 48 | 129.516 ms | 801.996 ms |
| A repeat | 50 | 12 | 12 | 12 | 38 | 38 | 107.227 ms | 639.607 ms |
| B first | 50 | 0 | 3 | 3 | 47 | 50 | 134.291 ms | 834.791 ms |
| B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 20.021 ms | 720.521 ms |

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
| NoCache | Total | 111.434 ms | 138.600 ms | 120.509 ms | 50 |
| NoCache | Total (RAG+LLM) | 811.934 ms | 839.100 ms | 821.009 ms | 50 |
| NoCache | Embedding | 4.999 ms | 16.876 ms | 8.301 ms | 50 |
| NoCache | Full retrieval DB | 18.326 ms | 27.002 ms | 20.341 ms | 50 |
| NoCache | Full retrieval scoring | 83.847 ms | 111.914 ms | 90.668 ms | 50 |
| NoCache | Full retrieval score sort | 0.109 ms | 0.252 ms | 0.134 ms | 50 |
| NoCache | Full retrieval rerank | 0.000 ms | 0.000 ms | 0.000 ms | 50 |
| NoCache | Full retrieval total | 103.899 ms | 131.648 ms | 111.142 ms | 50 |
| NoCache | Prompt build | 0.010 ms | 0.017 ms | 0.013 ms | 50 |
| NoCache | LLM request | 0.170 ms | 0.351 ms | 0.256 ms | 50 |

#### A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 13.264 ms | 224.128 ms | 129.516 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 801.996 ms | 50 |
| A first | Embedding | 5.074 ms | 14.436 ms | 7.973 ms | 50 |
| A first | Route | 0.848 ms | 1.262 ms | 0.935 ms | 50 |
| A first | Cache lookup | 0.547 ms | 2.397 ms | 1.128 ms | 12 |
| A first | Validation | 0.000 ms | 2.739 ms | 0.456 ms | 12 |
| A first | RAG DB | 18.051 ms | 23.908 ms | 19.364 ms | 48 |
| A first | RAG scoring | 83.975 ms | 185.950 ms | 90.644 ms | 48 |
| A first | RAG score sort | 0.114 ms | 0.170 ms | 0.140 ms | 48 |
| A first | RAG rerank | 0.000 ms | 0.000 ms | 0.000 ms | 48 |
| A first | RAG total | 102.812 ms | 204.804 ms | 110.148 ms | 48 |
| A first | Prompt build | 0.009 ms | 0.021 ms | 0.013 ms | 48 |
| A first | LLM request | 0.145 ms | 0.343 ms | 0.250 ms | 48 |
| A first | Cache store | 7.108 ms | 33.147 ms | 14.145 ms | 48 |
| A repeat | Total | 12.058 ms | 153.924 ms | 107.227 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 639.607 ms | 50 |
| A repeat | Embedding | 4.700 ms | 14.788 ms | 8.191 ms | 50 |
| A repeat | Route | 0.839 ms | 1.354 ms | 0.978 ms | 50 |
| A repeat | Cache lookup | 0.726 ms | 3.887 ms | 2.259 ms | 12 |
| A repeat | Validation | 2.565 ms | 4.649 ms | 3.113 ms | 12 |
| A repeat | RAG DB | 18.422 ms | 27.910 ms | 20.128 ms | 38 |
| A repeat | RAG scoring | 83.786 ms | 95.107 ms | 89.042 ms | 38 |
| A repeat | RAG score sort | 0.130 ms | 0.193 ms | 0.151 ms | 38 |
| A repeat | RAG rerank | 0.000 ms | 0.000 ms | 0.000 ms | 38 |
| A repeat | RAG total | 102.691 ms | 120.012 ms | 109.321 ms | 38 |
| A repeat | Prompt build | 0.009 ms | 0.025 ms | 0.013 ms | 38 |
| A repeat | LLM request | 0.160 ms | 0.476 ms | 0.259 ms | 38 |
| A repeat | Cache store | 9.447 ms | 31.016 ms | 17.017 ms | 38 |

#### B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 17.770 ms | 158.090 ms | 134.291 ms | 50 |
| B first | Total (RAG+LLM) | 718.270 ms | 858.590 ms | 834.791 ms | 50 |
| B first | Embedding | 4.930 ms | 14.327 ms | 7.993 ms | 50 |
| B first | Cache lookup DB | 0.823 ms | 2.850 ms | 1.748 ms | 50 |
| B first | Cache lookup scoring | 0.001 ms | 6.334 ms | 2.752 ms | 50 |
| B first | Cache lookup total | 0.921 ms | 9.184 ms | 4.501 ms | 50 |
| B first | Validation | 2.487 ms | 3.107 ms | 2.718 ms | 3 |
| B first | Full retrieval DB | 17.182 ms | 22.439 ms | 18.566 ms | 47 |
| B first | Full retrieval scoring | 83.896 ms | 96.171 ms | 90.105 ms | 47 |
| B first | Full retrieval score sort | 0.127 ms | 0.179 ms | 0.154 ms | 47 |
| B first | Full retrieval rerank | 0.000 ms | 0.000 ms | 0.000 ms | 47 |
| B first | Full retrieval total | 101.686 ms | 116.050 ms | 108.825 ms | 47 |
| B first | Prompt build | 0.007 ms | 0.022 ms | 0.015 ms | 50 |
| B first | LLM request | 0.163 ms | 0.355 ms | 0.258 ms | 50 |
| B first | Cache store | 10.160 ms | 35.574 ms | 18.280 ms | 47 |
| B repeat | Total | 15.980 ms | 26.163 ms | 20.021 ms | 50 |
| B repeat | Total (RAG+LLM) | 716.480 ms | 726.663 ms | 720.521 ms | 50 |
| B repeat | Embedding | 4.610 ms | 14.751 ms | 7.978 ms | 50 |
| B repeat | Cache lookup DB | 2.480 ms | 3.198 ms | 2.673 ms | 50 |
| B repeat | Cache lookup scoring | 4.568 ms | 7.244 ms | 5.136 ms | 50 |
| B repeat | Cache lookup total | 7.210 ms | 9.783 ms | 7.809 ms | 50 |
| B repeat | Validation | 2.489 ms | 3.985 ms | 2.720 ms | 50 |
| B repeat | Prompt build | 0.005 ms | 0.009 ms | 0.007 ms | 50 |
| B repeat | LLM request | 0.172 ms | 0.539 ms | 0.271 ms | 50 |

### 2-2. Reranker On (CPU)

#### 세팅

| 항목 | 값 |
|---|---|
| Reranker | On |
| Reranker requested device | cpu |
| Reranker resolved device | TBD |
| Rerank model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Rerank candidates | 30 |
| Route threshold | 0.7 |
| Route pool sample rate | 10% |
| Route pool min per dataset | 5 |
| Route pool seed | 42 |
| Query seed | 7 |
| Route pool | `ragbench:techqa:test` 10개 / 100 |
| Route pool indexes | `3, 13, 14, 17, 28, 31, 35, 81, 86, 94` |
| Run log file | `20260629_202439_cache_techqa_rerank-on_cpu_7d1d17e6.json` |
| Job ID | `7d1d17e6-f5b3-4b07-96df-0977735dcd5a` |
| Saved at | 20260629_202439 |

#### 전체 요약

| Mode | Total | Route passed | Cache hit | Validation passed | RAG fallback | LLM calls | Total avg | Total (RAG+LLM) avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 846.606 ms | 1547.106 ms |
| A first | 50 | 12 | 2 | 2 | 48 | 48 | 836.266 ms | 1508.746 ms |
| A repeat | 50 | 12 | 12 | 12 | 38 | 38 | 672.947 ms | 1205.327 ms |
| B first | 50 | 0 | 3 | 3 | 47 | 50 | 834.933 ms | 1535.433 ms |
| B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 23.012 ms | 723.512 ms |

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
| NoCache | Total | 824.314 ms | 962.826 ms | 846.606 ms | 50 |
| NoCache | Total (RAG+LLM) | 1524.814 ms | 1663.326 ms | 1547.106 ms | 50 |
| NoCache | Embedding | 4.788 ms | 24.181 ms | 9.051 ms | 50 |
| NoCache | Full retrieval DB | 18.796 ms | 29.310 ms | 21.101 ms | 50 |
| NoCache | Full retrieval scoring | 86.997 ms | 101.591 ms | 91.483 ms | 50 |
| NoCache | Full retrieval score sort | 0.128 ms | 0.271 ms | 0.152 ms | 50 |
| NoCache | Full retrieval rerank | 705.469 ms | 821.640 ms | 723.477 ms | 50 |
| NoCache | Full retrieval total | 816.687 ms | 946.523 ms | 836.212 ms | 50 |
| NoCache | Prompt build | 0.021 ms | 0.048 ms | 0.028 ms | 50 |
| NoCache | LLM request | 0.129 ms | 0.422 ms | 0.249 ms | 50 |

#### A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 13.740 ms | 902.650 ms | 836.266 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 1508.746 ms | 50 |
| A first | Embedding | 5.456 ms | 15.484 ms | 8.441 ms | 50 |
| A first | Route | 0.739 ms | 1.140 ms | 0.892 ms | 50 |
| A first | Cache lookup | 0.567 ms | 3.477 ms | 1.352 ms | 12 |
| A first | Validation | 0.000 ms | 2.855 ms | 0.463 ms | 12 |
| A first | RAG DB | 18.655 ms | 23.953 ms | 20.344 ms | 48 |
| A first | RAG scoring | 89.094 ms | 101.357 ms | 92.660 ms | 48 |
| A first | RAG score sort | 0.131 ms | 0.169 ms | 0.146 ms | 48 |
| A first | RAG rerank | 704.812 ms | 753.986 ms | 732.152 ms | 48 |
| A first | RAG total | 817.857 ms | 871.400 ms | 845.302 ms | 48 |
| A first | Prompt build | 0.020 ms | 0.052 ms | 0.029 ms | 48 |
| A first | LLM request | 0.105 ms | 0.431 ms | 0.255 ms | 48 |
| A first | Cache store | 9.279 ms | 23.105 ms | 14.458 ms | 48 |
| A repeat | Total | 12.112 ms | 906.109 ms | 672.947 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 1205.327 ms | 50 |
| A repeat | Embedding | 4.918 ms | 15.345 ms | 8.733 ms | 50 |
| A repeat | Route | 0.725 ms | 1.230 ms | 0.942 ms | 50 |
| A repeat | Cache lookup | 0.778 ms | 5.165 ms | 2.648 ms | 12 |
| A repeat | Validation | 2.677 ms | 4.161 ms | 3.364 ms | 12 |
| A repeat | RAG DB | 18.691 ms | 31.199 ms | 21.182 ms | 38 |
| A repeat | RAG scoring | 88.210 ms | 102.912 ms | 92.655 ms | 38 |
| A repeat | RAG score sort | 0.125 ms | 0.199 ms | 0.148 ms | 38 |
| A repeat | RAG rerank | 726.611 ms | 771.235 ms | 741.753 ms | 38 |
| A repeat | RAG total | 841.717 ms | 880.616 ms | 855.738 ms | 38 |
| A repeat | Prompt build | 0.028 ms | 0.085 ms | 0.033 ms | 38 |
| A repeat | LLM request | 0.129 ms | 0.442 ms | 0.290 ms | 38 |
| A repeat | Cache store | 8.623 ms | 23.418 ms | 13.762 ms | 38 |

#### B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 19.182 ms | 926.425 ms | 834.933 ms | 50 |
| B first | Total (RAG+LLM) | 719.682 ms | 1626.925 ms | 1535.433 ms | 50 |
| B first | Embedding | 4.847 ms | 15.254 ms | 8.453 ms | 50 |
| B first | Cache lookup DB | 0.863 ms | 3.405 ms | 1.974 ms | 50 |
| B first | Cache lookup scoring | 0.001 ms | 5.766 ms | 3.087 ms | 50 |
| B first | Cache lookup total | 0.864 ms | 8.559 ms | 5.061 ms | 50 |
| B first | Validation | 2.603 ms | 4.400 ms | 3.223 ms | 3 |
| B first | Full retrieval DB | 18.117 ms | 24.898 ms | 19.805 ms | 47 |
| B first | Full retrieval scoring | 88.236 ms | 98.719 ms | 92.519 ms | 47 |
| B first | Full retrieval score sort | 0.124 ms | 0.164 ms | 0.143 ms | 47 |
| B first | Full retrieval rerank | 722.957 ms | 767.171 ms | 743.168 ms | 47 |
| B first | Full retrieval total | 836.338 ms | 877.437 ms | 855.636 ms | 47 |
| B first | Prompt build | 0.008 ms | 0.051 ms | 0.035 ms | 50 |
| B first | LLM request | 0.141 ms | 0.518 ms | 0.318 ms | 50 |
| B first | Cache store | 7.462 ms | 41.032 ms | 15.257 ms | 47 |
| B repeat | Total | 16.459 ms | 32.286 ms | 23.012 ms | 50 |
| B repeat | Total (RAG+LLM) | 716.959 ms | 732.786 ms | 723.512 ms | 50 |
| B repeat | Embedding | 4.967 ms | 15.250 ms | 8.902 ms | 50 |
| B repeat | Cache lookup DB | 2.566 ms | 4.340 ms | 3.194 ms | 50 |
| B repeat | Cache lookup scoring | 4.703 ms | 7.592 ms | 5.897 ms | 50 |
| B repeat | Cache lookup total | 7.319 ms | 11.373 ms | 9.091 ms | 50 |
| B repeat | Validation | 2.538 ms | 4.247 ms | 3.236 ms | 50 |
| B repeat | Prompt build | 0.004 ms | 0.015 ms | 0.008 ms | 50 |
| B repeat | LLM request | 0.155 ms | 0.603 ms | 0.331 ms | 50 |

### 2-3. Reranker On (GPU)

#### 세팅

| 항목 | 값 |
|---|---|
| Reranker | On |
| Reranker requested device | cuda |
| Reranker resolved device | TBD |
| Rerank model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Rerank candidates | 30 |
| Route threshold | 0.7 |
| Route pool sample rate | 10% |
| Route pool min per dataset | 5 |
| Route pool seed | 42 |
| Query seed | 7 |
| Route pool | `ragbench:techqa:test` 10개 / 100 |
| Route pool indexes | `3, 13, 14, 17, 28, 31, 35, 81, 86, 94` |
| Run log file | `20260629_202133_cache_techqa_rerank-on_cuda_cb4e9044.json` |
| Job ID | `cb4e9044-7ac7-4233-bdfb-1146d0250214` |
| Saved at | 20260629_202133 |

#### 전체 요약

| Mode | Total | Route passed | Cache hit | Validation passed | RAG fallback | LLM calls | Total avg | Total (RAG+LLM) avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 274.457 ms | 974.957 ms |
| A first | 50 | 12 | 2 | 2 | 48 | 48 | 277.819 ms | 950.299 ms |
| A repeat | 50 | 12 | 12 | 12 | 38 | 38 | 228.377 ms | 760.757 ms |
| B first | 50 | 0 | 3 | 3 | 47 | 50 | 282.831 ms | 983.331 ms |
| B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 21.851 ms | 722.351 ms |

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
| NoCache | Total | 261.040 ms | 294.999 ms | 274.457 ms | 50 |
| NoCache | Total (RAG+LLM) | 961.540 ms | 995.499 ms | 974.957 ms | 50 |
| NoCache | Embedding | 5.029 ms | 17.681 ms | 8.183 ms | 50 |
| NoCache | Full retrieval DB | 18.570 ms | 26.606 ms | 20.317 ms | 50 |
| NoCache | Full retrieval scoring | 83.744 ms | 106.198 ms | 90.055 ms | 50 |
| NoCache | Full retrieval score sort | 0.131 ms | 0.324 ms | 0.153 ms | 50 |
| NoCache | Full retrieval rerank | 149.406 ms | 168.918 ms | 154.591 ms | 50 |
| NoCache | Full retrieval total | 254.200 ms | 281.072 ms | 265.116 ms | 50 |
| NoCache | Prompt build | 0.015 ms | 0.026 ms | 0.021 ms | 50 |
| NoCache | LLM request | 0.124 ms | 0.356 ms | 0.257 ms | 50 |

#### A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 13.444 ms | 314.942 ms | 277.819 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 950.299 ms | 50 |
| A first | Embedding | 4.850 ms | 14.647 ms | 8.183 ms | 50 |
| A first | Route | 0.708 ms | 1.178 ms | 0.814 ms | 50 |
| A first | Cache lookup | 0.562 ms | 3.707 ms | 1.322 ms | 12 |
| A first | Validation | 0.000 ms | 4.120 ms | 0.572 ms | 12 |
| A first | RAG DB | 18.423 ms | 33.443 ms | 20.475 ms | 48 |
| A first | RAG scoring | 84.320 ms | 104.117 ms | 89.590 ms | 48 |
| A first | RAG score sort | 0.119 ms | 0.316 ms | 0.149 ms | 48 |
| A first | RAG rerank | 148.341 ms | 167.511 ms | 154.912 ms | 48 |
| A first | RAG total | 254.356 ms | 293.732 ms | 265.126 ms | 48 |
| A first | Prompt build | 0.014 ms | 0.032 ms | 0.022 ms | 48 |
| A first | LLM request | 0.124 ms | 0.388 ms | 0.252 ms | 48 |
| A first | Cache store | 7.094 ms | 25.062 ms | 13.364 ms | 48 |
| A repeat | Total | 10.660 ms | 338.581 ms | 228.377 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 760.757 ms | 50 |
| A repeat | Embedding | 4.921 ms | 15.584 ms | 8.096 ms | 50 |
| A repeat | Route | 0.725 ms | 1.145 ms | 0.806 ms | 50 |
| A repeat | Cache lookup | 0.728 ms | 3.681 ms | 2.132 ms | 12 |
| A repeat | Validation | 2.656 ms | 4.422 ms | 2.907 ms | 12 |
| A repeat | RAG DB | 18.331 ms | 24.359 ms | 19.926 ms | 38 |
| A repeat | RAG scoring | 85.861 ms | 119.583 ms | 90.532 ms | 38 |
| A repeat | RAG score sort | 0.127 ms | 0.164 ms | 0.143 ms | 38 |
| A repeat | RAG rerank | 149.952 ms | 164.757 ms | 155.024 ms | 38 |
| A repeat | RAG total | 255.625 ms | 299.242 ms | 265.624 ms | 38 |
| A repeat | Prompt build | 0.016 ms | 0.031 ms | 0.021 ms | 38 |
| A repeat | LLM request | 0.121 ms | 0.359 ms | 0.251 ms | 38 |
| A repeat | Cache store | 9.462 ms | 70.042 ms | 20.507 ms | 38 |

#### B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 16.969 ms | 357.718 ms | 282.831 ms | 50 |
| B first | Total (RAG+LLM) | 717.469 ms | 1058.218 ms | 983.331 ms | 50 |
| B first | Embedding | 5.090 ms | 14.675 ms | 8.027 ms | 50 |
| B first | Cache lookup DB | 0.939 ms | 3.383 ms | 1.812 ms | 50 |
| B first | Cache lookup scoring | 0.001 ms | 6.279 ms | 2.800 ms | 50 |
| B first | Cache lookup total | 0.993 ms | 8.608 ms | 4.612 ms | 50 |
| B first | Validation | 2.603 ms | 2.674 ms | 2.639 ms | 3 |
| B first | Full retrieval DB | 17.116 ms | 22.267 ms | 18.706 ms | 47 |
| B first | Full retrieval scoring | 84.692 ms | 94.406 ms | 89.402 ms | 47 |
| B first | Full retrieval score sort | 0.119 ms | 0.196 ms | 0.141 ms | 47 |
| B first | Full retrieval rerank | 149.938 ms | 167.117 ms | 155.663 ms | 47 |
| B first | Full retrieval total | 253.232 ms | 274.139 ms | 263.912 ms | 47 |
| B first | Prompt build | 0.007 ms | 0.033 ms | 0.023 ms | 50 |
| B first | LLM request | 0.124 ms | 0.361 ms | 0.257 ms | 50 |
| B first | Cache store | 7.322 ms | 77.506 ms | 20.969 ms | 47 |
| B repeat | Total | 16.165 ms | 109.791 ms | 21.851 ms | 50 |
| B repeat | Total (RAG+LLM) | 716.665 ms | 810.291 ms | 722.351 ms | 50 |
| B repeat | Embedding | 4.642 ms | 93.456 ms | 9.678 ms | 50 |
| B repeat | Cache lookup DB | 2.548 ms | 4.288 ms | 2.766 ms | 50 |
| B repeat | Cache lookup scoring | 4.635 ms | 6.741 ms | 5.052 ms | 50 |
| B repeat | Cache lookup total | 7.191 ms | 11.029 ms | 7.818 ms | 50 |
| B repeat | Validation | 2.548 ms | 4.086 ms | 2.843 ms | 50 |
| B repeat | Prompt build | 0.005 ms | 0.012 ms | 0.007 ms | 50 |
| B repeat | LLM request | 0.117 ms | 0.498 ms | 0.268 ms | 50 |

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
| LLM | mock |
| Route threshold | 0.70 |
| Cache hit threshold | 0.86 |
| LLM latency basis | Mock LLM: `700.5 ms/call` 가상 보정 |
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
| Run log file | `20260629_202725_scalability_techqa_rerank-off_na_59ec7e7d.json` |
| Job ID | `59ec7e7d-2d76-48e7-b24f-ed19eab9bec4` |
| Saved at | 20260629_202725 |

#### Scale별 route pool

| Scale | Row count | Source ID | Base EU rows | Versioned EU rows | Route pool | Route pool indexes |
|---:|---:|---|---:|---:|---:|---|
| 1 | 100 | `dp3_ragbench_techqa_test_100` | 1,874 | 4,062 | 10 / 100 | `3, 13, 14, 17, 28, 31, 35, 81, 86, 94` |
| 2 | 200 | `dp3_ragbench_techqa_test_200` | 3,882 | 8,411 | 20 / 200 | `6, 7, 8, 22, 23, 26, 28, 35, 55, 57, 59, 62, 70, 108, 139, 151, 163, 173, 188, 189` |
| 3 | 300 | `dp3_ragbench_techqa_test_300` | 5,701 | 12,353 | 30 / 300 | `3, 12, 13, 15, 16, 44, 47, 52, 57, 71, 79, 81, 101, 110, 111, 112, 114, 119, 125, 140, 142, 172, 174, 194, 214, 216, 229, 258, 279, 287` |

#### Scale별 전체 요약

| Scale | Row count | Mode | Total | Route passed | Cache hit | Validation passed | RAG fallback | LLM calls | Total avg | Total (RAG+LLM) avg |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 118.612 ms | 819.112 ms |
| 1 | 100 | A first | 50 | 12 | 2 | 2 | 48 | 48 | 126.858 ms | 799.338 ms |
| 1 | 100 | A repeat | 50 | 12 | 12 | 12 | 38 | 38 | 107.715 ms | 640.095 ms |
| 1 | 100 | B first | 50 | 0 | 3 | 3 | 47 | 50 | 131.864 ms | 832.364 ms |
| 1 | 100 | B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 19.860 ms | 720.360 ms |
| 2 | 200 | NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 200.798 ms | 901.298 ms |
| 2 | 200 | A first | 50 | 14 | 1 | 1 | 49 | 49 | 215.200 ms | 901.690 ms |
| 2 | 200 | A repeat | 50 | 14 | 14 | 14 | 36 | 36 | 161.017 ms | 665.377 ms |
| 2 | 200 | B first | 50 | 0 | 5 | 5 | 45 | 50 | 198.981 ms | 899.481 ms |
| 2 | 200 | B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 19.872 ms | 720.372 ms |
| 3 | 300 | NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 295.225 ms | 995.725 ms |
| 3 | 300 | A first | 50 | 17 | 3 | 3 | 47 | 47 | 292.481 ms | 950.951 ms |
| 3 | 300 | A repeat | 50 | 17 | 17 | 17 | 33 | 33 | 212.212 ms | 674.542 ms |
| 3 | 300 | B first | 50 | 0 | 5 | 5 | 45 | 50 | 282.086 ms | 982.586 ms |
| 3 | 300 | B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 18.427 ms | 718.927 ms |

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
| NoCache | Total | 111.870 ms | 124.999 ms | 118.612 ms | 50 |
| NoCache | Total (RAG+LLM) | 812.370 ms | 825.499 ms | 819.112 ms | 50 |
| NoCache | Embedding | 4.817 ms | 14.927 ms | 8.138 ms | 50 |
| NoCache | Full retrieval DB | 18.458 ms | 21.533 ms | 19.843 ms | 50 |
| NoCache | Full retrieval scoring | 86.223 ms | 94.634 ms | 89.445 ms | 50 |
| NoCache | Full retrieval score sort | 0.125 ms | 0.165 ms | 0.143 ms | 50 |
| NoCache | Full retrieval rerank | 0.000 ms | 0.000 ms | 0.000 ms | 50 |
| NoCache | Full retrieval total | 105.608 ms | 115.223 ms | 109.431 ms | 50 |
| NoCache | Prompt build | 0.008 ms | 0.022 ms | 0.010 ms | 50 |
| NoCache | LLM request | 0.158 ms | 0.353 ms | 0.251 ms | 50 |

#### Scale 1 (100 rows) A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 13.174 ms | 145.719 ms | 126.858 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 799.338 ms | 50 |
| A first | Embedding | 5.062 ms | 14.772 ms | 8.022 ms | 50 |
| A first | Route | 0.715 ms | 1.145 ms | 0.771 ms | 50 |
| A first | Cache lookup | 0.522 ms | 2.394 ms | 1.034 ms | 12 |
| A first | Validation | 0.000 ms | 2.801 ms | 0.463 ms | 12 |
| A first | RAG DB | 18.163 ms | 22.159 ms | 19.235 ms | 48 |
| A first | RAG scoring | 83.802 ms | 95.186 ms | 89.685 ms | 48 |
| A first | RAG score sort | 0.120 ms | 0.176 ms | 0.140 ms | 48 |
| A first | RAG rerank | 0.000 ms | 0.000 ms | 0.000 ms | 48 |
| A first | RAG total | 105.103 ms | 116.970 ms | 109.060 ms | 48 |
| A first | Prompt build | 0.008 ms | 0.019 ms | 0.010 ms | 48 |
| A first | LLM request | 0.164 ms | 0.348 ms | 0.252 ms | 48 |
| A first | Cache store | 6.810 ms | 17.700 ms | 12.624 ms | 48 |
| A repeat | Total | 11.914 ms | 214.610 ms | 107.715 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 640.095 ms | 50 |
| A repeat | Embedding | 4.962 ms | 14.994 ms | 8.416 ms | 50 |
| A repeat | Route | 0.719 ms | 1.163 ms | 0.871 ms | 50 |
| A repeat | Cache lookup | 0.704 ms | 3.956 ms | 2.424 ms | 12 |
| A repeat | Validation | 2.696 ms | 4.442 ms | 3.270 ms | 12 |
| A repeat | RAG DB | 18.488 ms | 27.392 ms | 20.721 ms | 38 |
| A repeat | RAG scoring | 85.317 ms | 175.714 ms | 92.262 ms | 38 |
| A repeat | RAG score sort | 0.124 ms | 0.171 ms | 0.145 ms | 38 |
| A repeat | RAG rerank | 0.000 ms | 0.000 ms | 0.000 ms | 38 |
| A repeat | RAG total | 104.475 ms | 194.449 ms | 113.128 ms | 38 |
| A repeat | Prompt build | 0.008 ms | 0.019 ms | 0.010 ms | 38 |
| A repeat | LLM request | 0.172 ms | 0.444 ms | 0.259 ms | 38 |
| A repeat | Cache store | 6.510 ms | 19.927 ms | 13.606 ms | 38 |

#### Scale 1 (100 rows) B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 20.297 ms | 165.576 ms | 131.864 ms | 50 |
| B first | Total (RAG+LLM) | 720.797 ms | 866.076 ms | 832.364 ms | 50 |
| B first | Embedding | 4.843 ms | 14.801 ms | 8.173 ms | 50 |
| B first | Cache lookup DB | 0.262 ms | 3.363 ms | 1.278 ms | 50 |
| B first | Cache lookup scoring | 0.000 ms | 6.446 ms | 3.025 ms | 50 |
| B first | Cache lookup total | 0.262 ms | 9.717 ms | 4.302 ms | 50 |
| B first | Validation | 3.390 ms | 4.522 ms | 4.010 ms | 3 |
| B first | Full retrieval DB | 17.755 ms | 27.520 ms | 19.694 ms | 47 |
| B first | Full retrieval scoring | 86.921 ms | 105.896 ms | 90.789 ms | 47 |
| B first | Full retrieval score sort | 0.125 ms | 0.299 ms | 0.150 ms | 47 |
| B first | Full retrieval rerank | 0.000 ms | 0.000 ms | 0.000 ms | 47 |
| B first | Full retrieval total | 105.137 ms | 124.829 ms | 110.633 ms | 47 |
| B first | Prompt build | 0.010 ms | 0.023 ms | 0.013 ms | 50 |
| B first | LLM request | 0.156 ms | 0.394 ms | 0.261 ms | 50 |
| B first | Cache store | 6.843 ms | 24.100 ms | 13.824 ms | 47 |
| B repeat | Total | 15.962 ms | 27.127 ms | 19.860 ms | 50 |
| B repeat | Total (RAG+LLM) | 716.462 ms | 727.627 ms | 720.360 ms | 50 |
| B repeat | Embedding | 4.935 ms | 14.338 ms | 8.064 ms | 50 |
| B repeat | Cache lookup DB | 2.050 ms | 3.574 ms | 2.242 ms | 50 |
| B repeat | Cache lookup scoring | 4.792 ms | 6.730 ms | 5.323 ms | 50 |
| B repeat | Cache lookup total | 6.856 ms | 10.283 ms | 7.565 ms | 50 |
| B repeat | Validation | 2.496 ms | 4.053 ms | 2.714 ms | 50 |
| B repeat | Prompt build | 0.006 ms | 0.017 ms | 0.008 ms | 50 |
| B repeat | LLM request | 0.155 ms | 0.397 ms | 0.272 ms | 50 |

#### Scale 2 (200 rows) NoCache timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| NoCache | Total | 192.315 ms | 211.573 ms | 200.798 ms | 50 |
| NoCache | Total (RAG+LLM) | 892.815 ms | 912.073 ms | 901.298 ms | 50 |
| NoCache | Embedding | 4.993 ms | 14.841 ms | 7.831 ms | 50 |
| NoCache | Full retrieval DB | 21.676 ms | 26.672 ms | 23.023 ms | 50 |
| NoCache | Full retrieval scoring | 160.773 ms | 182.346 ms | 168.350 ms | 50 |
| NoCache | Full retrieval score sort | 0.212 ms | 0.295 ms | 0.252 ms | 50 |
| NoCache | Full retrieval rerank | 0.000 ms | 0.000 ms | 0.000 ms | 50 |
| NoCache | Full retrieval total | 182.817 ms | 204.696 ms | 191.625 ms | 50 |
| NoCache | Prompt build | 0.009 ms | 0.017 ms | 0.012 ms | 50 |
| NoCache | LLM request | 0.179 ms | 0.352 ms | 0.260 ms | 50 |

#### Scale 2 (200 rows) A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 9.763 ms | 239.876 ms | 215.200 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 901.690 ms | 50 |
| A first | Embedding | 4.965 ms | 14.843 ms | 7.693 ms | 50 |
| A first | Route | 1.140 ms | 1.652 ms | 1.223 ms | 50 |
| A first | Cache lookup | 0.536 ms | 1.345 ms | 0.800 ms | 14 |
| A first | Validation | 0.000 ms | 2.736 ms | 0.195 ms | 14 |
| A first | RAG DB | 21.526 ms | 28.158 ms | 22.627 ms | 49 |
| A first | RAG scoring | 161.985 ms | 180.512 ms | 170.117 ms | 49 |
| A first | RAG score sort | 0.235 ms | 0.299 ms | 0.263 ms | 49 |
| A first | RAG rerank | 0.000 ms | 0.000 ms | 0.000 ms | 49 |
| A first | RAG total | 184.006 ms | 204.332 ms | 193.007 ms | 49 |
| A first | Prompt build | 0.009 ms | 0.020 ms | 0.012 ms | 49 |
| A first | LLM request | 0.180 ms | 0.373 ms | 0.262 ms | 49 |
| A first | Cache store | 6.767 ms | 26.110 ms | 15.950 ms | 49 |
| A repeat | Total | 9.855 ms | 237.547 ms | 161.017 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 665.377 ms | 50 |
| A repeat | Embedding | 4.929 ms | 15.355 ms | 7.940 ms | 50 |
| A repeat | Route | 1.143 ms | 1.912 ms | 1.305 ms | 50 |
| A repeat | Cache lookup | 0.702 ms | 1.677 ms | 1.256 ms | 14 |
| A repeat | Validation | 2.716 ms | 3.566 ms | 2.936 ms | 14 |
| A repeat | RAG DB | 21.653 ms | 33.086 ms | 23.780 ms | 36 |
| A repeat | RAG scoring | 161.770 ms | 184.455 ms | 169.001 ms | 36 |
| A repeat | RAG score sort | 0.229 ms | 0.537 ms | 0.288 ms | 36 |
| A repeat | RAG rerank | 0.000 ms | 0.000 ms | 0.000 ms | 36 |
| A repeat | RAG total | 183.749 ms | 210.361 ms | 193.069 ms | 36 |
| A repeat | Prompt build | 0.009 ms | 0.017 ms | 0.012 ms | 36 |
| A repeat | LLM request | 0.170 ms | 0.381 ms | 0.259 ms | 36 |
| A repeat | Cache store | 8.537 ms | 24.361 ms | 14.776 ms | 36 |

#### Scale 2 (200 rows) B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 14.891 ms | 251.958 ms | 198.981 ms | 50 |
| B first | Total (RAG+LLM) | 715.391 ms | 952.458 ms | 899.481 ms | 50 |
| B first | Embedding | 4.904 ms | 14.812 ms | 7.661 ms | 50 |
| B first | Cache lookup DB | 0.262 ms | 2.200 ms | 1.191 ms | 50 |
| B first | Cache lookup scoring | 0.000 ms | 5.627 ms | 2.753 ms | 50 |
| B first | Cache lookup total | 0.262 ms | 7.827 ms | 3.945 ms | 50 |
| B first | Validation | 2.504 ms | 3.042 ms | 2.652 ms | 5 |
| B first | Full retrieval DB | 20.770 ms | 37.952 ms | 22.704 ms | 45 |
| B first | Full retrieval scoring | 161.289 ms | 180.113 ms | 169.017 ms | 45 |
| B first | Full retrieval score sort | 0.235 ms | 0.366 ms | 0.273 ms | 45 |
| B first | Full retrieval rerank | 0.000 ms | 0.000 ms | 0.000 ms | 45 |
| B first | Full retrieval total | 182.782 ms | 214.304 ms | 191.994 ms | 45 |
| B first | Prompt build | 0.006 ms | 0.023 ms | 0.014 ms | 50 |
| B first | LLM request | 0.164 ms | 0.335 ms | 0.247 ms | 50 |
| B first | Cache store | 6.883 ms | 27.982 ms | 13.326 ms | 45 |
| B repeat | Total | 15.336 ms | 27.381 ms | 19.872 ms | 50 |
| B repeat | Total (RAG+LLM) | 715.836 ms | 727.881 ms | 720.372 ms | 50 |
| B repeat | Embedding | 4.768 ms | 15.351 ms | 7.846 ms | 50 |
| B repeat | Cache lookup DB | 1.976 ms | 3.460 ms | 2.253 ms | 50 |
| B repeat | Cache lookup scoring | 4.581 ms | 7.554 ms | 5.355 ms | 50 |
| B repeat | Cache lookup total | 6.581 ms | 10.506 ms | 7.608 ms | 50 |
| B repeat | Validation | 2.460 ms | 4.113 ms | 2.876 ms | 50 |
| B repeat | Prompt build | 0.006 ms | 0.015 ms | 0.007 ms | 50 |
| B repeat | LLM request | 0.181 ms | 0.412 ms | 0.272 ms | 50 |

#### Scale 3 (300 rows) NoCache timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| NoCache | Total | 283.573 ms | 398.393 ms | 295.225 ms | 50 |
| NoCache | Total (RAG+LLM) | 984.073 ms | 1098.893 ms | 995.725 ms | 50 |
| NoCache | Embedding | 4.752 ms | 14.509 ms | 7.304 ms | 50 |
| NoCache | Full retrieval DB | 25.061 ms | 121.652 ms | 28.619 ms | 50 |
| NoCache | Full retrieval scoring | 245.790 ms | 271.021 ms | 257.133 ms | 50 |
| NoCache | Full retrieval score sort | 0.344 ms | 0.527 ms | 0.406 ms | 50 |
| NoCache | Full retrieval rerank | 0.000 ms | 0.000 ms | 0.000 ms | 50 |
| NoCache | Full retrieval total | 275.841 ms | 391.954 ms | 286.158 ms | 50 |
| NoCache | Prompt build | 0.012 ms | 0.021 ms | 0.015 ms | 50 |
| NoCache | LLM request | 0.148 ms | 0.354 ms | 0.248 ms | 50 |

#### Scale 3 (300 rows) A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 9.963 ms | 325.368 ms | 292.481 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 950.951 ms | 50 |
| A first | Embedding | 4.660 ms | 13.610 ms | 7.239 ms | 50 |
| A first | Route | 1.436 ms | 2.408 ms | 1.588 ms | 50 |
| A first | Cache lookup | 0.528 ms | 1.210 ms | 0.721 ms | 17 |
| A first | Validation | 0.000 ms | 3.127 ms | 0.501 ms | 17 |
| A first | RAG DB | 24.570 ms | 32.170 ms | 26.676 ms | 47 |
| A first | RAG scoring | 248.471 ms | 270.454 ms | 259.276 ms | 47 |
| A first | RAG score sort | 0.381 ms | 0.479 ms | 0.428 ms | 47 |
| A first | RAG rerank | 0.000 ms | 0.000 ms | 0.000 ms | 47 |
| A first | RAG total | 276.373 ms | 298.569 ms | 286.380 ms | 47 |
| A first | Prompt build | 0.012 ms | 0.021 ms | 0.015 ms | 47 |
| A first | LLM request | 0.151 ms | 0.383 ms | 0.255 ms | 47 |
| A first | Cache store | 6.339 ms | 21.093 ms | 13.246 ms | 47 |
| A repeat | Total | 9.623 ms | 421.291 ms | 212.212 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 674.542 ms | 50 |
| A repeat | Embedding | 4.761 ms | 14.584 ms | 7.555 ms | 50 |
| A repeat | Route | 1.445 ms | 2.590 ms | 1.777 ms | 50 |
| A repeat | Cache lookup | 0.701 ms | 1.619 ms | 1.098 ms | 17 |
| A repeat | Validation | 2.626 ms | 4.218 ms | 3.215 ms | 17 |
| A repeat | RAG DB | 25.290 ms | 120.534 ms | 30.213 ms | 33 |
| A repeat | RAG scoring | 247.167 ms | 274.986 ms | 259.233 ms | 33 |
| A repeat | RAG score sort | 0.393 ms | 0.509 ms | 0.449 ms | 33 |
| A repeat | RAG rerank | 0.000 ms | 0.000 ms | 0.000 ms | 33 |
| A repeat | RAG total | 274.652 ms | 395.981 ms | 289.895 ms | 33 |
| A repeat | Prompt build | 0.011 ms | 0.021 ms | 0.015 ms | 33 |
| A repeat | LLM request | 0.160 ms | 0.358 ms | 0.265 ms | 33 |
| A repeat | Cache store | 7.435 ms | 29.056 ms | 13.507 ms | 33 |

#### Scale 3 (300 rows) B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 15.128 ms | 330.126 ms | 282.086 ms | 50 |
| B first | Total (RAG+LLM) | 715.628 ms | 1030.626 ms | 982.586 ms | 50 |
| B first | Embedding | 4.468 ms | 13.544 ms | 7.210 ms | 50 |
| B first | Cache lookup DB | 0.295 ms | 2.063 ms | 1.180 ms | 50 |
| B first | Cache lookup scoring | 0.001 ms | 5.533 ms | 2.728 ms | 50 |
| B first | Cache lookup total | 0.384 ms | 7.445 ms | 3.908 ms | 50 |
| B first | Validation | 2.529 ms | 3.042 ms | 2.668 ms | 5 |
| B first | Full retrieval DB | 24.357 ms | 41.476 ms | 26.289 ms | 45 |
| B first | Full retrieval scoring | 245.115 ms | 269.572 ms | 258.168 ms | 45 |
| B first | Full retrieval score sort | 0.391 ms | 0.536 ms | 0.455 ms | 45 |
| B first | Full retrieval rerank | 0.000 ms | 0.000 ms | 0.000 ms | 45 |
| B first | Full retrieval total | 270.638 ms | 297.932 ms | 284.912 ms | 45 |
| B first | Prompt build | 0.006 ms | 0.024 ms | 0.017 ms | 50 |
| B first | LLM request | 0.150 ms | 0.378 ms | 0.270 ms | 50 |
| B first | Cache store | 7.050 ms | 19.764 ms | 12.806 ms | 45 |
| B repeat | Total | 15.002 ms | 26.663 ms | 18.427 ms | 50 |
| B repeat | Total (RAG+LLM) | 715.502 ms | 727.163 ms | 718.927 ms | 50 |
| B repeat | Embedding | 4.486 ms | 15.089 ms | 7.195 ms | 50 |
| B repeat | Cache lookup DB | 1.948 ms | 2.722 ms | 2.118 ms | 50 |
| B repeat | Cache lookup scoring | 4.519 ms | 6.132 ms | 4.964 ms | 50 |
| B repeat | Cache lookup total | 6.599 ms | 8.518 ms | 7.083 ms | 50 |
| B repeat | Validation | 2.469 ms | 3.414 ms | 2.675 ms | 50 |
| B repeat | Prompt build | 0.006 ms | 0.010 ms | 0.007 ms | 50 |
| B repeat | LLM request | 0.152 ms | 0.486 ms | 0.286 ms | 50 |

### 3-2. Reranker On (CPU)

#### 세팅

| 항목 | 값 |
|---|---|
| Reranker | On |
| Reranker requested device | cpu |
| Reranker resolved device | TBD |
| Rerank model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Rerank candidates | 30 |
| Scale rows | `100, 200, 300` |
| Route threshold | 0.7 |
| Route pool sample rate | 10% |
| Route pool min per dataset | 5 |
| Route pool seed | 42 |
| Query seed | 7 |
| Run log file | `20260629_205801_scalability_techqa_rerank-on_cpu_4da17acc.json` |
| Job ID | `4da17acc-4843-48aa-862b-73ab55f1c50a` |
| Saved at | 20260629_205801 |

#### Scale별 route pool

| Scale | Row count | Source ID | Base EU rows | Versioned EU rows | Route pool | Route pool indexes |
|---:|---:|---|---:|---:|---:|---|
| 1 | 100 | `dp3_ragbench_techqa_test_100` | 1,874 | 4,062 | 10 / 100 | `3, 13, 14, 17, 28, 31, 35, 81, 86, 94` |
| 2 | 200 | `dp3_ragbench_techqa_test_200` | 3,882 | 8,411 | 20 / 200 | `6, 7, 8, 22, 23, 26, 28, 35, 55, 57, 59, 62, 70, 108, 139, 151, 163, 173, 188, 189` |
| 3 | 300 | `dp3_ragbench_techqa_test_300` | 5,701 | 12,353 | 30 / 300 | `3, 12, 13, 15, 16, 44, 47, 52, 57, 71, 79, 81, 101, 110, 111, 112, 114, 119, 125, 140, 142, 172, 174, 194, 214, 216, 229, 258, 279, 287` |

#### Scale별 전체 요약

| Scale | Row count | Mode | Total | Route passed | Cache hit | Validation passed | RAG fallback | LLM calls | Total avg | Total (RAG+LLM) avg |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 847.123 ms | 1547.623 ms |
| 1 | 100 | A first | 50 | 12 | 2 | 2 | 48 | 48 | 841.308 ms | 1513.788 ms |
| 1 | 100 | A repeat | 50 | 12 | 12 | 12 | 38 | 38 | 675.670 ms | 1208.050 ms |
| 1 | 100 | B first | 50 | 0 | 3 | 3 | 47 | 50 | 839.858 ms | 1540.358 ms |
| 1 | 100 | B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 21.508 ms | 722.008 ms |
| 2 | 200 | NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 943.998 ms | 1644.498 ms |
| 2 | 200 | A first | 50 | 14 | 1 | 1 | 49 | 49 | 947.782 ms | 1634.272 ms |
| 2 | 200 | A repeat | 50 | 14 | 14 | 14 | 36 | 36 | 692.543 ms | 1196.903 ms |
| 2 | 200 | B first | 50 | 0 | 5 | 5 | 45 | 50 | 874.879 ms | 1575.379 ms |
| 2 | 200 | B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 20.940 ms | 721.440 ms |
| 3 | 300 | NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 1038.390 ms | 1738.890 ms |
| 3 | 300 | A first | 50 | 17 | 3 | 3 | 47 | 47 | 996.320 ms | 1654.790 ms |
| 3 | 300 | A repeat | 50 | 17 | 17 | 17 | 33 | 33 | 699.688 ms | 1162.018 ms |
| 3 | 300 | B first | 50 | 0 | 5 | 5 | 45 | 50 | 959.899 ms | 1660.399 ms |
| 3 | 300 | B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 21.287 ms | 721.787 ms |

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
| NoCache | Total | 825.031 ms | 880.507 ms | 847.123 ms | 50 |
| NoCache | Total (RAG+LLM) | 1525.531 ms | 1581.007 ms | 1547.623 ms | 50 |
| NoCache | Embedding | 4.877 ms | 17.856 ms | 8.825 ms | 50 |
| NoCache | Full retrieval DB | 19.034 ms | 31.044 ms | 21.622 ms | 50 |
| NoCache | Full retrieval scoring | 88.552 ms | 103.130 ms | 91.675 ms | 50 |
| NoCache | Full retrieval score sort | 0.134 ms | 0.168 ms | 0.149 ms | 50 |
| NoCache | Full retrieval rerank | 702.029 ms | 744.765 ms | 723.449 ms | 50 |
| NoCache | Full retrieval total | 811.314 ms | 860.707 ms | 836.895 ms | 50 |
| NoCache | Prompt build | 0.022 ms | 0.051 ms | 0.030 ms | 50 |
| NoCache | LLM request | 0.118 ms | 0.441 ms | 0.261 ms | 50 |

#### Scale 1 (100 rows) A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 15.579 ms | 962.308 ms | 841.308 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 1513.788 ms | 50 |
| A first | Embedding | 4.834 ms | 15.545 ms | 8.553 ms | 50 |
| A first | Route | 0.732 ms | 1.237 ms | 0.933 ms | 50 |
| A first | Cache lookup | 0.824 ms | 2.795 ms | 1.383 ms | 12 |
| A first | Validation | 0.000 ms | 3.506 ms | 0.560 ms | 12 |
| A first | RAG DB | 19.050 ms | 108.633 ms | 22.560 ms | 48 |
| A first | RAG scoring | 87.062 ms | 99.521 ms | 91.649 ms | 48 |
| A first | RAG score sort | 0.114 ms | 0.168 ms | 0.144 ms | 48 |
| A first | RAG rerank | 716.599 ms | 766.965 ms | 734.022 ms | 48 |
| A first | RAG total | 827.564 ms | 936.650 ms | 848.375 ms | 48 |
| A first | Prompt build | 0.023 ms | 0.045 ms | 0.031 ms | 48 |
| A first | LLM request | 0.123 ms | 0.612 ms | 0.281 ms | 48 |
| A first | Cache store | 9.454 ms | 25.295 ms | 16.339 ms | 48 |
| A repeat | Total | 11.995 ms | 938.391 ms | 675.670 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 1208.050 ms | 50 |
| A repeat | Embedding | 4.782 ms | 16.483 ms | 8.603 ms | 50 |
| A repeat | Route | 0.742 ms | 1.267 ms | 0.953 ms | 50 |
| A repeat | Cache lookup | 0.859 ms | 3.996 ms | 2.515 ms | 12 |
| A repeat | Validation | 2.572 ms | 4.556 ms | 3.600 ms | 12 |
| A repeat | RAG DB | 18.952 ms | 29.979 ms | 21.283 ms | 38 |
| A repeat | RAG scoring | 88.479 ms | 98.271 ms | 91.855 ms | 38 |
| A repeat | RAG score sort | 0.122 ms | 0.168 ms | 0.147 ms | 38 |
| A repeat | RAG rerank | 725.345 ms | 799.450 ms | 743.737 ms | 38 |
| A repeat | RAG total | 834.081 ms | 915.244 ms | 857.022 ms | 38 |
| A repeat | Prompt build | 0.028 ms | 0.040 ms | 0.033 ms | 38 |
| A repeat | LLM request | 0.137 ms | 0.490 ms | 0.301 ms | 38 |
| A repeat | Cache store | 10.510 ms | 20.409 ms | 16.167 ms | 38 |

#### Scale 1 (100 rows) B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 19.324 ms | 919.939 ms | 839.858 ms | 50 |
| B first | Total (RAG+LLM) | 719.824 ms | 1620.439 ms | 1540.358 ms | 50 |
| B first | Embedding | 5.229 ms | 16.837 ms | 8.904 ms | 50 |
| B first | Cache lookup DB | 0.259 ms | 3.228 ms | 1.546 ms | 50 |
| B first | Cache lookup scoring | 0.000 ms | 6.817 ms | 3.351 ms | 50 |
| B first | Cache lookup total | 0.259 ms | 9.544 ms | 4.896 ms | 50 |
| B first | Validation | 2.645 ms | 3.084 ms | 2.794 ms | 3 |
| B first | Full retrieval DB | 18.101 ms | 33.556 ms | 20.505 ms | 47 |
| B first | Full retrieval scoring | 86.900 ms | 100.072 ms | 92.845 ms | 47 |
| B first | Full retrieval score sort | 0.121 ms | 0.193 ms | 0.146 ms | 47 |
| B first | Full retrieval rerank | 731.594 ms | 766.001 ms | 746.141 ms | 47 |
| B first | Full retrieval total | 844.119 ms | 881.812 ms | 859.637 ms | 47 |
| B first | Prompt build | 0.008 ms | 0.050 ms | 0.036 ms | 50 |
| B first | LLM request | 0.156 ms | 0.484 ms | 0.326 ms | 50 |
| B first | Cache store | 9.615 ms | 28.250 ms | 16.091 ms | 47 |
| B repeat | Total | 16.374 ms | 32.835 ms | 21.508 ms | 50 |
| B repeat | Total (RAG+LLM) | 716.874 ms | 733.335 ms | 722.008 ms | 50 |
| B repeat | Embedding | 4.863 ms | 15.202 ms | 8.482 ms | 50 |
| B repeat | Cache lookup DB | 2.084 ms | 3.655 ms | 2.513 ms | 50 |
| B repeat | Cache lookup scoring | 4.731 ms | 7.109 ms | 5.795 ms | 50 |
| B repeat | Cache lookup total | 6.924 ms | 10.764 ms | 8.308 ms | 50 |
| B repeat | Validation | 2.474 ms | 5.170 ms | 3.019 ms | 50 |
| B repeat | Prompt build | 0.005 ms | 0.012 ms | 0.008 ms | 50 |
| B repeat | LLM request | 0.120 ms | 0.485 ms | 0.301 ms | 50 |

#### Scale 2 (200 rows) NoCache timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| NoCache | Total | 916.714 ms | 990.852 ms | 943.998 ms | 50 |
| NoCache | Total (RAG+LLM) | 1617.214 ms | 1691.352 ms | 1644.498 ms | 50 |
| NoCache | Embedding | 5.143 ms | 17.579 ms | 9.127 ms | 50 |
| NoCache | Full retrieval DB | 22.682 ms | 38.021 ms | 26.702 ms | 50 |
| NoCache | Full retrieval scoring | 165.285 ms | 181.627 ms | 172.536 ms | 50 |
| NoCache | Full retrieval score sort | 0.243 ms | 0.318 ms | 0.283 ms | 50 |
| NoCache | Full retrieval rerank | 713.175 ms | 785.706 ms | 733.169 ms | 50 |
| NoCache | Full retrieval total | 910.052 ms | 980.189 ms | 932.690 ms | 50 |
| NoCache | Prompt build | 0.029 ms | 0.057 ms | 0.037 ms | 50 |
| NoCache | LLM request | 0.129 ms | 0.557 ms | 0.346 ms | 50 |

#### Scale 2 (200 rows) A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 11.617 ms | 1055.220 ms | 947.782 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 1634.272 ms | 50 |
| A first | Embedding | 5.114 ms | 17.120 ms | 9.096 ms | 50 |
| A first | Route | 1.048 ms | 1.823 ms | 1.436 ms | 50 |
| A first | Cache lookup | 0.535 ms | 1.478 ms | 0.891 ms | 14 |
| A first | Validation | 0.000 ms | 3.203 ms | 0.229 ms | 14 |
| A first | RAG DB | 22.192 ms | 38.432 ms | 26.012 ms | 49 |
| A first | RAG scoring | 168.838 ms | 183.125 ms | 173.761 ms | 49 |
| A first | RAG score sort | 0.227 ms | 0.343 ms | 0.291 ms | 49 |
| A first | RAG rerank | 711.539 ms | 765.788 ms | 732.902 ms | 49 |
| A first | RAG total | 908.444 ms | 970.287 ms | 932.966 ms | 49 |
| A first | Prompt build | 0.029 ms | 0.053 ms | 0.037 ms | 49 |
| A first | LLM request | 0.192 ms | 0.515 ms | 0.341 ms | 49 |
| A first | Cache store | 10.539 ms | 79.193 ms | 20.980 ms | 49 |
| A repeat | Total | 11.101 ms | 979.934 ms | 692.543 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 1196.903 ms | 50 |
| A repeat | Embedding | 5.292 ms | 15.644 ms | 8.728 ms | 50 |
| A repeat | Route | 1.072 ms | 1.866 ms | 1.418 ms | 50 |
| A repeat | Cache lookup | 1.008 ms | 2.713 ms | 1.604 ms | 14 |
| A repeat | Validation | 2.673 ms | 4.398 ms | 3.672 ms | 14 |
| A repeat | RAG DB | 22.278 ms | 29.353 ms | 25.162 ms | 36 |
| A repeat | RAG scoring | 167.245 ms | 182.813 ms | 174.142 ms | 36 |
| A repeat | RAG score sort | 0.268 ms | 0.339 ms | 0.298 ms | 36 |
| A repeat | RAG rerank | 704.920 ms | 752.594 ms | 727.901 ms | 36 |
| A repeat | RAG total | 900.037 ms | 952.000 ms | 927.503 ms | 36 |
| A repeat | Prompt build | 0.030 ms | 0.052 ms | 0.037 ms | 36 |
| A repeat | LLM request | 0.136 ms | 0.547 ms | 0.352 ms | 36 |
| A repeat | Cache store | 11.664 ms | 20.702 ms | 16.203 ms | 36 |

#### Scale 2 (200 rows) B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 15.081 ms | 1043.633 ms | 874.879 ms | 50 |
| B first | Total (RAG+LLM) | 715.581 ms | 1744.133 ms | 1575.379 ms | 50 |
| B first | Embedding | 5.397 ms | 16.294 ms | 8.512 ms | 50 |
| B first | Cache lookup DB | 0.395 ms | 3.539 ms | 1.464 ms | 50 |
| B first | Cache lookup scoring | 0.001 ms | 6.461 ms | 3.329 ms | 50 |
| B first | Cache lookup total | 0.401 ms | 8.987 ms | 4.793 ms | 50 |
| B first | Validation | 2.522 ms | 4.152 ms | 3.045 ms | 5 |
| B first | Full retrieval DB | 21.618 ms | 29.991 ms | 24.135 ms | 45 |
| B first | Full retrieval scoring | 167.209 ms | 189.853 ms | 174.135 ms | 45 |
| B first | Full retrieval score sort | 0.254 ms | 0.364 ms | 0.299 ms | 45 |
| B first | Full retrieval rerank | 705.917 ms | 768.713 ms | 729.755 ms | 45 |
| B first | Full retrieval total | 905.279 ms | 964.895 ms | 928.325 ms | 45 |
| B first | Prompt build | 0.007 ms | 0.067 ms | 0.038 ms | 50 |
| B first | LLM request | 0.150 ms | 0.455 ms | 0.334 ms | 50 |
| B first | Cache store | 13.045 ms | 85.135 ms | 24.897 ms | 45 |
| B repeat | Total | 15.739 ms | 31.286 ms | 20.940 ms | 50 |
| B repeat | Total (RAG+LLM) | 716.239 ms | 731.786 ms | 721.440 ms | 50 |
| B repeat | Embedding | 4.975 ms | 15.739 ms | 8.400 ms | 50 |
| B repeat | Cache lookup DB | 2.012 ms | 3.518 ms | 2.435 ms | 50 |
| B repeat | Cache lookup scoring | 4.684 ms | 7.388 ms | 5.478 ms | 50 |
| B repeat | Cache lookup total | 6.731 ms | 10.839 ms | 7.913 ms | 50 |
| B repeat | Validation | 2.253 ms | 4.225 ms | 2.951 ms | 50 |
| B repeat | Prompt build | 0.006 ms | 0.014 ms | 0.008 ms | 50 |
| B repeat | LLM request | 0.132 ms | 0.491 ms | 0.289 ms | 50 |

#### Scale 3 (300 rows) NoCache timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| NoCache | Total | 1007.583 ms | 1090.540 ms | 1038.390 ms | 50 |
| NoCache | Total (RAG+LLM) | 1708.083 ms | 1791.040 ms | 1738.890 ms | 50 |
| NoCache | Embedding | 4.908 ms | 15.412 ms | 8.379 ms | 50 |
| NoCache | Full retrieval DB | 26.071 ms | 40.192 ms | 29.087 ms | 50 |
| NoCache | Full retrieval scoring | 256.133 ms | 280.654 ms | 264.789 ms | 50 |
| NoCache | Full retrieval score sort | 0.392 ms | 0.894 ms | 0.452 ms | 50 |
| NoCache | Full retrieval rerank | 708.694 ms | 774.197 ms | 732.977 ms | 50 |
| NoCache | Full retrieval total | 998.685 ms | 1073.306 ms | 1027.305 ms | 50 |
| NoCache | Prompt build | 0.031 ms | 0.064 ms | 0.038 ms | 50 |
| NoCache | LLM request | 0.169 ms | 0.467 ms | 0.306 ms | 50 |

#### Scale 3 (300 rows) A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 10.082 ms | 1153.319 ms | 996.320 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 1654.790 ms | 50 |
| A first | Embedding | 4.753 ms | 15.397 ms | 8.124 ms | 50 |
| A first | Route | 1.340 ms | 2.372 ms | 1.815 ms | 50 |
| A first | Cache lookup | 0.576 ms | 1.221 ms | 0.835 ms | 17 |
| A first | Validation | 0.000 ms | 2.778 ms | 0.479 ms | 17 |
| A first | RAG DB | 25.853 ms | 35.739 ms | 28.678 ms | 47 |
| A first | RAG scoring | 254.892 ms | 351.448 ms | 265.522 ms | 47 |
| A first | RAG score sort | 0.374 ms | 0.532 ms | 0.450 ms | 47 |
| A first | RAG rerank | 713.061 ms | 782.946 ms | 734.743 ms | 47 |
| A first | RAG total | 999.436 ms | 1123.250 ms | 1029.393 ms | 47 |
| A first | Prompt build | 0.033 ms | 0.048 ms | 0.039 ms | 47 |
| A first | LLM request | 0.167 ms | 0.529 ms | 0.312 ms | 47 |
| A first | Cache store | 12.071 ms | 21.453 ms | 16.798 ms | 47 |
| A repeat | Total | 10.397 ms | 1107.749 ms | 699.688 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 1162.018 ms | 50 |
| A repeat | Embedding | 4.882 ms | 14.889 ms | 8.102 ms | 50 |
| A repeat | Route | 1.361 ms | 2.393 ms | 1.787 ms | 50 |
| A repeat | Cache lookup | 0.789 ms | 1.723 ms | 1.177 ms | 17 |
| A repeat | Validation | 2.671 ms | 4.198 ms | 3.309 ms | 17 |
| A repeat | RAG DB | 25.894 ms | 34.742 ms | 28.339 ms | 33 |
| A repeat | RAG scoring | 254.971 ms | 275.791 ms | 264.055 ms | 33 |
| A repeat | RAG score sort | 0.431 ms | 0.576 ms | 0.477 ms | 33 |
| A repeat | RAG rerank | 710.868 ms | 745.287 ms | 724.543 ms | 33 |
| A repeat | RAG total | 994.722 ms | 1039.911 ms | 1017.414 ms | 33 |
| A repeat | Prompt build | 0.031 ms | 0.049 ms | 0.037 ms | 33 |
| A repeat | LLM request | 0.179 ms | 0.449 ms | 0.281 ms | 33 |
| A repeat | Cache store | 13.216 ms | 68.573 ms | 22.975 ms | 33 |

#### Scale 3 (300 rows) B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 16.106 ms | 1122.730 ms | 959.899 ms | 50 |
| B first | Total (RAG+LLM) | 716.606 ms | 1823.230 ms | 1660.399 ms | 50 |
| B first | Embedding | 4.770 ms | 15.113 ms | 8.282 ms | 50 |
| B first | Cache lookup DB | 0.277 ms | 3.350 ms | 1.489 ms | 50 |
| B first | Cache lookup scoring | 0.000 ms | 6.666 ms | 3.295 ms | 50 |
| B first | Cache lookup total | 0.277 ms | 10.016 ms | 4.784 ms | 50 |
| B first | Validation | 2.592 ms | 4.461 ms | 3.850 ms | 5 |
| B first | Full retrieval DB | 25.558 ms | 47.422 ms | 28.691 ms | 45 |
| B first | Full retrieval scoring | 254.614 ms | 279.096 ms | 265.417 ms | 45 |
| B first | Full retrieval score sort | 0.407 ms | 0.550 ms | 0.487 ms | 45 |
| B first | Full retrieval rerank | 711.093 ms | 774.869 ms | 730.683 ms | 45 |
| B first | Full retrieval total | 998.307 ms | 1070.443 ms | 1025.277 ms | 45 |
| B first | Prompt build | 0.007 ms | 0.064 ms | 0.039 ms | 50 |
| B first | LLM request | 0.167 ms | 0.577 ms | 0.296 ms | 50 |
| B first | Cache store | 11.576 ms | 68.688 ms | 21.935 ms | 45 |
| B repeat | Total | 15.795 ms | 26.953 ms | 21.287 ms | 50 |
| B repeat | Total (RAG+LLM) | 716.295 ms | 727.453 ms | 721.787 ms | 50 |
| B repeat | Embedding | 4.863 ms | 14.435 ms | 8.034 ms | 50 |
| B repeat | Cache lookup DB | 2.026 ms | 3.607 ms | 2.475 ms | 50 |
| B repeat | Cache lookup scoring | 4.637 ms | 6.911 ms | 5.814 ms | 50 |
| B repeat | Cache lookup total | 6.663 ms | 10.255 ms | 8.289 ms | 50 |
| B repeat | Validation | 2.505 ms | 4.421 ms | 3.208 ms | 50 |
| B repeat | Prompt build | 0.006 ms | 0.014 ms | 0.008 ms | 50 |
| B repeat | LLM request | 0.167 ms | 0.525 ms | 0.300 ms | 50 |

### 3-3. Reranker On (GPU)

#### 세팅

| 항목 | 값 |
|---|---|
| Reranker | On |
| Reranker requested device | cuda |
| Reranker resolved device | TBD |
| Rerank model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Rerank candidates | 30 |
| Scale rows | `100, 200, 300` |
| Route threshold | 0.7 |
| Route pool sample rate | 10% |
| Route pool min per dataset | 5 |
| Route pool seed | 42 |
| Query seed | 7 |
| Run log file | `20260629_204632_scalability_techqa_rerank-on_cuda_2509da26.json` |
| Job ID | `2509da26-47f7-48d3-b724-279de500ba80` |
| Saved at | 20260629_204632 |

#### Scale별 route pool

| Scale | Row count | Source ID | Base EU rows | Versioned EU rows | Route pool | Route pool indexes |
|---:|---:|---|---:|---:|---:|---|
| 1 | 100 | `dp3_ragbench_techqa_test_100` | 1,874 | 4,062 | 10 / 100 | `3, 13, 14, 17, 28, 31, 35, 81, 86, 94` |
| 2 | 200 | `dp3_ragbench_techqa_test_200` | 3,882 | 8,411 | 20 / 200 | `6, 7, 8, 22, 23, 26, 28, 35, 55, 57, 59, 62, 70, 108, 139, 151, 163, 173, 188, 189` |
| 3 | 300 | `dp3_ragbench_techqa_test_300` | 5,701 | 12,353 | 30 / 300 | `3, 12, 13, 15, 16, 44, 47, 52, 57, 71, 79, 81, 101, 110, 111, 112, 114, 119, 125, 140, 142, 172, 174, 194, 214, 216, 229, 258, 279, 287` |

#### Scale별 전체 요약

| Scale | Row count | Mode | Total | Route passed | Cache hit | Validation passed | RAG fallback | LLM calls | Total avg | Total (RAG+LLM) avg |
|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 100 | NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 275.942 ms | 976.442 ms |
| 1 | 100 | A first | 50 | 12 | 2 | 2 | 48 | 48 | 285.567 ms | 958.047 ms |
| 1 | 100 | A repeat | 50 | 12 | 12 | 12 | 38 | 38 | 227.573 ms | 759.953 ms |
| 1 | 100 | B first | 50 | 0 | 3 | 3 | 47 | 50 | 282.055 ms | 982.555 ms |
| 1 | 100 | B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 21.025 ms | 721.525 ms |
| 2 | 200 | NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 356.386 ms | 1056.886 ms |
| 2 | 200 | A first | 50 | 14 | 1 | 1 | 49 | 49 | 371.834 ms | 1058.324 ms |
| 2 | 200 | A repeat | 50 | 14 | 14 | 14 | 36 | 36 | 274.023 ms | 778.383 ms |
| 2 | 200 | B first | 50 | 0 | 5 | 5 | 45 | 50 | 342.641 ms | 1043.141 ms |
| 2 | 200 | B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 19.850 ms | 720.350 ms |
| 3 | 300 | NoCache | 50 | 0 | 0 | 0 | 50 | 50 | 450.950 ms | 1151.450 ms |
| 3 | 300 | A first | 50 | 17 | 3 | 3 | 47 | 47 | 445.401 ms | 1103.871 ms |
| 3 | 300 | A repeat | 50 | 17 | 17 | 17 | 33 | 33 | 314.680 ms | 777.010 ms |
| 3 | 300 | B first | 50 | 0 | 5 | 5 | 45 | 50 | 428.013 ms | 1128.513 ms |
| 3 | 300 | B repeat | 50 | 0 | 50 | 50 | 0 | 50 | 18.738 ms | 719.238 ms |

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
| NoCache | Total | 262.209 ms | 342.412 ms | 275.942 ms | 50 |
| NoCache | Total (RAG+LLM) | 962.709 ms | 1042.912 ms | 976.442 ms | 50 |
| NoCache | Embedding | 5.391 ms | 65.444 ms | 9.327 ms | 50 |
| NoCache | Full retrieval DB | 18.403 ms | 30.655 ms | 21.769 ms | 50 |
| NoCache | Full retrieval scoring | 85.952 ms | 96.596 ms | 89.478 ms | 50 |
| NoCache | Full retrieval score sort | 0.120 ms | 0.172 ms | 0.145 ms | 50 |
| NoCache | Full retrieval rerank | 148.612 ms | 161.504 ms | 154.033 ms | 50 |
| NoCache | Full retrieval total | 255.610 ms | 278.294 ms | 265.426 ms | 50 |
| NoCache | Prompt build | 0.015 ms | 0.033 ms | 0.019 ms | 50 |
| NoCache | LLM request | 0.107 ms | 0.365 ms | 0.241 ms | 50 |

#### Scale 1 (100 rows) A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 13.790 ms | 357.415 ms | 285.567 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 958.047 ms | 50 |
| A first | Embedding | 5.084 ms | 15.209 ms | 8.121 ms | 50 |
| A first | Route | 0.727 ms | 1.140 ms | 0.838 ms | 50 |
| A first | Cache lookup | 0.521 ms | 2.389 ms | 1.155 ms | 12 |
| A first | Validation | 0.000 ms | 3.135 ms | 0.481 ms | 12 |
| A first | RAG DB | 18.309 ms | 23.453 ms | 19.954 ms | 48 |
| A first | RAG scoring | 83.327 ms | 94.422 ms | 88.624 ms | 48 |
| A first | RAG score sort | 0.114 ms | 0.168 ms | 0.142 ms | 48 |
| A first | RAG rerank | 149.507 ms | 162.988 ms | 154.206 ms | 48 |
| A first | RAG total | 251.953 ms | 274.335 ms | 262.926 ms | 48 |
| A first | Prompt build | 0.015 ms | 0.042 ms | 0.019 ms | 48 |
| A first | LLM request | 0.124 ms | 0.348 ms | 0.236 ms | 48 |
| A first | Cache store | 10.780 ms | 83.191 ms | 23.754 ms | 48 |
| A repeat | Total | 11.642 ms | 386.092 ms | 227.573 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 759.953 ms | 50 |
| A repeat | Embedding | 4.740 ms | 14.792 ms | 8.328 ms | 50 |
| A repeat | Route | 0.730 ms | 1.197 ms | 0.878 ms | 50 |
| A repeat | Cache lookup | 0.816 ms | 3.844 ms | 2.421 ms | 12 |
| A repeat | Validation | 2.692 ms | 3.958 ms | 3.183 ms | 12 |
| A repeat | RAG DB | 18.290 ms | 114.639 ms | 23.454 ms | 38 |
| A repeat | RAG scoring | 84.242 ms | 99.335 ms | 89.875 ms | 38 |
| A repeat | RAG score sort | 0.123 ms | 0.207 ms | 0.145 ms | 38 |
| A repeat | RAG rerank | 150.278 ms | 164.082 ms | 154.890 ms | 38 |
| A repeat | RAG total | 255.125 ms | 359.063 ms | 268.364 ms | 38 |
| A repeat | Prompt build | 0.014 ms | 0.029 ms | 0.019 ms | 38 |
| A repeat | LLM request | 0.129 ms | 0.332 ms | 0.248 ms | 38 |
| A repeat | Cache store | 10.704 ms | 22.125 ms | 16.096 ms | 38 |

#### Scale 1 (100 rows) B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 20.650 ms | 360.007 ms | 282.055 ms | 50 |
| B first | Total (RAG+LLM) | 721.150 ms | 1060.507 ms | 982.555 ms | 50 |
| B first | Embedding | 4.753 ms | 15.415 ms | 8.462 ms | 50 |
| B first | Cache lookup DB | 0.253 ms | 3.422 ms | 1.369 ms | 50 |
| B first | Cache lookup scoring | 0.000 ms | 6.850 ms | 3.248 ms | 50 |
| B first | Cache lookup total | 0.253 ms | 10.272 ms | 4.617 ms | 50 |
| B first | Validation | 2.981 ms | 4.275 ms | 3.802 ms | 3 |
| B first | Full retrieval DB | 17.570 ms | 34.603 ms | 20.924 ms | 47 |
| B first | Full retrieval scoring | 84.359 ms | 136.844 ms | 90.284 ms | 47 |
| B first | Full retrieval score sort | 0.123 ms | 0.160 ms | 0.141 ms | 47 |
| B first | Full retrieval rerank | 150.145 ms | 164.184 ms | 155.119 ms | 47 |
| B first | Full retrieval total | 254.441 ms | 326.568 ms | 266.467 ms | 47 |
| B first | Prompt build | 0.007 ms | 0.035 ms | 0.021 ms | 50 |
| B first | LLM request | 0.130 ms | 0.405 ms | 0.255 ms | 50 |
| B first | Cache store | 10.600 ms | 37.744 ms | 16.954 ms | 47 |
| B repeat | Total | 15.881 ms | 31.957 ms | 21.025 ms | 50 |
| B repeat | Total (RAG+LLM) | 716.381 ms | 732.457 ms | 721.525 ms | 50 |
| B repeat | Embedding | 4.907 ms | 15.646 ms | 8.356 ms | 50 |
| B repeat | Cache lookup DB | 2.024 ms | 3.529 ms | 2.345 ms | 50 |
| B repeat | Cache lookup scoring | 4.766 ms | 7.087 ms | 5.634 ms | 50 |
| B repeat | Cache lookup total | 6.837 ms | 10.476 ms | 7.979 ms | 50 |
| B repeat | Validation | 2.413 ms | 4.241 ms | 3.081 ms | 50 |
| B repeat | Prompt build | 0.006 ms | 0.012 ms | 0.008 ms | 50 |
| B repeat | LLM request | 0.106 ms | 0.669 ms | 0.282 ms | 50 |

#### Scale 2 (200 rows) NoCache timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| NoCache | Total | 342.054 ms | 380.958 ms | 356.386 ms | 50 |
| NoCache | Total (RAG+LLM) | 1042.554 ms | 1081.458 ms | 1056.886 ms | 50 |
| NoCache | Embedding | 4.917 ms | 27.337 ms | 8.317 ms | 50 |
| NoCache | Full retrieval DB | 21.992 ms | 34.918 ms | 24.185 ms | 50 |
| NoCache | Full retrieval scoring | 158.682 ms | 182.035 ms | 167.429 ms | 50 |
| NoCache | Full retrieval score sort | 0.223 ms | 0.546 ms | 0.271 ms | 50 |
| NoCache | Full retrieval rerank | 149.466 ms | 164.793 ms | 154.693 ms | 50 |
| NoCache | Full retrieval total | 333.205 ms | 369.644 ms | 346.579 ms | 50 |
| NoCache | Prompt build | 0.016 ms | 0.031 ms | 0.019 ms | 50 |
| NoCache | LLM request | 0.121 ms | 0.329 ms | 0.245 ms | 50 |

#### Scale 2 (200 rows) A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 9.613 ms | 405.087 ms | 371.834 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 1058.324 ms | 50 |
| A first | Embedding | 4.994 ms | 15.106 ms | 8.023 ms | 50 |
| A first | Route | 1.035 ms | 1.683 ms | 1.278 ms | 50 |
| A first | Cache lookup | 0.530 ms | 2.027 ms | 0.900 ms | 14 |
| A first | Validation | 0.000 ms | 2.720 ms | 0.194 ms | 14 |
| A first | RAG DB | 21.250 ms | 37.602 ms | 25.737 ms | 49 |
| A first | RAG scoring | 161.952 ms | 179.379 ms | 169.758 ms | 49 |
| A first | RAG score sort | 0.237 ms | 0.564 ms | 0.291 ms | 49 |
| A first | RAG rerank | 149.530 ms | 165.402 ms | 155.398 ms | 49 |
| A first | RAG total | 335.419 ms | 378.664 ms | 351.183 ms | 49 |
| A first | Prompt build | 0.017 ms | 0.031 ms | 0.020 ms | 49 |
| A first | LLM request | 0.121 ms | 0.334 ms | 0.249 ms | 49 |
| A first | Cache store | 10.445 ms | 26.690 ms | 17.020 ms | 49 |
| A repeat | Total | 9.746 ms | 403.299 ms | 274.023 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 778.383 ms | 50 |
| A repeat | Embedding | 4.947 ms | 14.373 ms | 7.987 ms | 50 |
| A repeat | Route | 1.029 ms | 1.732 ms | 1.183 ms | 50 |
| A repeat | Cache lookup | 0.709 ms | 2.315 ms | 1.349 ms | 14 |
| A repeat | Validation | 2.633 ms | 4.135 ms | 2.982 ms | 14 |
| A repeat | RAG DB | 21.599 ms | 33.928 ms | 23.295 ms | 36 |
| A repeat | RAG scoring | 159.728 ms | 177.141 ms | 168.040 ms | 36 |
| A repeat | RAG score sort | 0.236 ms | 0.597 ms | 0.293 ms | 36 |
| A repeat | RAG rerank | 148.877 ms | 165.619 ms | 155.136 ms | 36 |
| A repeat | RAG total | 333.836 ms | 369.108 ms | 346.764 ms | 36 |
| A repeat | Prompt build | 0.016 ms | 0.033 ms | 0.020 ms | 36 |
| A repeat | LLM request | 0.138 ms | 0.341 ms | 0.253 ms | 36 |
| A repeat | Cache store | 10.261 ms | 25.122 ms | 17.921 ms | 36 |

#### Scale 2 (200 rows) B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 15.056 ms | 416.312 ms | 342.641 ms | 50 |
| B first | Total (RAG+LLM) | 715.556 ms | 1116.812 ms | 1043.141 ms | 50 |
| B first | Embedding | 4.925 ms | 16.273 ms | 7.826 ms | 50 |
| B first | Cache lookup DB | 0.310 ms | 3.026 ms | 1.259 ms | 50 |
| B first | Cache lookup scoring | 0.000 ms | 6.135 ms | 2.988 ms | 50 |
| B first | Cache lookup total | 0.469 ms | 9.161 ms | 4.246 ms | 50 |
| B first | Validation | 2.550 ms | 4.408 ms | 3.139 ms | 5 |
| B first | Full retrieval DB | 20.349 ms | 34.053 ms | 23.152 ms | 45 |
| B first | Full retrieval scoring | 158.731 ms | 185.603 ms | 168.869 ms | 45 |
| B first | Full retrieval score sort | 0.238 ms | 0.435 ms | 0.288 ms | 45 |
| B first | Full retrieval rerank | 150.333 ms | 164.655 ms | 154.991 ms | 45 |
| B first | Full retrieval total | 335.804 ms | 373.854 ms | 347.301 ms | 45 |
| B first | Prompt build | 0.007 ms | 0.033 ms | 0.021 ms | 50 |
| B first | LLM request | 0.130 ms | 0.414 ms | 0.253 ms | 50 |
| B first | Cache store | 10.767 ms | 34.390 ms | 16.832 ms | 45 |
| B repeat | Total | 15.510 ms | 27.218 ms | 19.850 ms | 50 |
| B repeat | Total (RAG+LLM) | 716.010 ms | 727.718 ms | 720.350 ms | 50 |
| B repeat | Embedding | 4.933 ms | 15.770 ms | 8.053 ms | 50 |
| B repeat | Cache lookup DB | 1.955 ms | 3.423 ms | 2.229 ms | 50 |
| B repeat | Cache lookup scoring | 4.542 ms | 6.747 ms | 5.227 ms | 50 |
| B repeat | Cache lookup total | 6.572 ms | 9.924 ms | 7.455 ms | 50 |
| B repeat | Validation | 2.466 ms | 4.042 ms | 2.754 ms | 50 |
| B repeat | Prompt build | 0.005 ms | 0.016 ms | 0.007 ms | 50 |
| B repeat | LLM request | 0.142 ms | 0.444 ms | 0.276 ms | 50 |

#### Scale 3 (300 rows) NoCache timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| NoCache | Total | 438.115 ms | 472.748 ms | 450.950 ms | 50 |
| NoCache | Total (RAG+LLM) | 1138.615 ms | 1173.248 ms | 1151.450 ms | 50 |
| NoCache | Embedding | 4.598 ms | 15.148 ms | 7.427 ms | 50 |
| NoCache | Full retrieval DB | 25.426 ms | 45.850 ms | 27.936 ms | 50 |
| NoCache | Full retrieval scoring | 246.900 ms | 269.355 ms | 257.841 ms | 50 |
| NoCache | Full retrieval score sort | 0.382 ms | 0.506 ms | 0.433 ms | 50 |
| NoCache | Full retrieval rerank | 150.012 ms | 165.779 ms | 155.366 ms | 50 |
| NoCache | Full retrieval total | 427.613 ms | 461.454 ms | 441.577 ms | 50 |
| NoCache | Prompt build | 0.017 ms | 0.042 ms | 0.021 ms | 50 |
| NoCache | LLM request | 0.140 ms | 0.318 ms | 0.251 ms | 50 |

#### Scale 3 (300 rows) A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A first | Total | 11.186 ms | 568.583 ms | 445.401 ms | 50 |
| A first | Total (RAG+LLM) | - | - | 1103.871 ms | 50 |
| A first | Embedding | 4.727 ms | 14.826 ms | 7.500 ms | 50 |
| A first | Route | 1.331 ms | 2.240 ms | 1.603 ms | 50 |
| A first | Cache lookup | 0.550 ms | 1.016 ms | 0.784 ms | 17 |
| A first | Validation | 0.000 ms | 3.198 ms | 0.557 ms | 17 |
| A first | RAG DB | 25.144 ms | 44.560 ms | 28.405 ms | 47 |
| A first | RAG scoring | 247.281 ms | 355.136 ms | 260.329 ms | 47 |
| A first | RAG score sort | 0.366 ms | 0.518 ms | 0.453 ms | 47 |
| A first | RAG rerank | 150.083 ms | 167.046 ms | 155.807 ms | 47 |
| A first | RAG total | 425.553 ms | 541.104 ms | 444.993 ms | 47 |
| A first | Prompt build | 0.017 ms | 0.026 ms | 0.021 ms | 47 |
| A first | LLM request | 0.145 ms | 0.335 ms | 0.250 ms | 47 |
| A first | Cache store | 10.479 ms | 22.927 ms | 16.849 ms | 47 |
| A repeat | Total | 10.968 ms | 503.215 ms | 314.680 ms | 50 |
| A repeat | Total (RAG+LLM) | - | - | 777.010 ms | 50 |
| A repeat | Embedding | 5.093 ms | 13.975 ms | 7.652 ms | 50 |
| A repeat | Route | 1.332 ms | 2.281 ms | 1.652 ms | 50 |
| A repeat | Cache lookup | 0.852 ms | 1.738 ms | 1.114 ms | 17 |
| A repeat | Validation | 2.624 ms | 4.270 ms | 3.176 ms | 17 |
| A repeat | RAG DB | 24.929 ms | 47.925 ms | 28.726 ms | 33 |
| A repeat | RAG scoring | 244.490 ms | 279.037 ms | 257.751 ms | 33 |
| A repeat | RAG score sort | 0.416 ms | 0.521 ms | 0.461 ms | 33 |
| A repeat | RAG rerank | 150.644 ms | 164.617 ms | 155.105 ms | 33 |
| A repeat | RAG total | 421.798 ms | 470.819 ms | 442.043 ms | 33 |
| A repeat | Prompt build | 0.018 ms | 0.028 ms | 0.022 ms | 33 |
| A repeat | LLM request | 0.200 ms | 0.380 ms | 0.262 ms | 33 |
| A repeat | Cache store | 11.173 ms | 22.119 ms | 16.543 ms | 33 |

#### Scale 3 (300 rows) B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B first | Total | 14.712 ms | 501.089 ms | 428.013 ms | 50 |
| B first | Total (RAG+LLM) | 715.212 ms | 1201.589 ms | 1128.513 ms | 50 |
| B first | Embedding | 4.983 ms | 15.643 ms | 7.593 ms | 50 |
| B first | Cache lookup DB | 0.297 ms | 2.946 ms | 1.273 ms | 50 |
| B first | Cache lookup scoring | 0.000 ms | 6.458 ms | 3.041 ms | 50 |
| B first | Cache lookup total | 0.405 ms | 8.848 ms | 4.314 ms | 50 |
| B first | Validation | 2.526 ms | 2.705 ms | 2.624 ms | 5 |
| B first | Full retrieval DB | 24.594 ms | 45.790 ms | 28.506 ms | 45 |
| B first | Full retrieval scoring | 247.408 ms | 267.950 ms | 258.205 ms | 45 |
| B first | Full retrieval score sort | 0.421 ms | 0.522 ms | 0.475 ms | 45 |
| B first | Full retrieval rerank | 150.284 ms | 163.289 ms | 154.895 ms | 45 |
| B first | Full retrieval total | 429.130 ms | 469.809 ms | 442.081 ms | 45 |
| B first | Prompt build | 0.007 ms | 0.038 ms | 0.023 ms | 50 |
| B first | LLM request | 0.174 ms | 0.351 ms | 0.255 ms | 50 |
| B first | Cache store | 10.629 ms | 26.813 ms | 16.643 ms | 45 |
| B repeat | Total | 15.547 ms | 26.914 ms | 18.738 ms | 50 |
| B repeat | Total (RAG+LLM) | 716.047 ms | 727.414 ms | 719.238 ms | 50 |
| B repeat | Embedding | 4.841 ms | 15.186 ms | 7.302 ms | 50 |
| B repeat | Cache lookup DB | 1.932 ms | 3.365 ms | 2.107 ms | 50 |
| B repeat | Cache lookup scoring | 4.600 ms | 6.797 ms | 5.084 ms | 50 |
| B repeat | Cache lookup total | 6.571 ms | 9.239 ms | 7.190 ms | 50 |
| B repeat | Validation | 2.419 ms | 4.565 ms | 2.719 ms | 50 |
| B repeat | Prompt build | 0.004 ms | 0.009 ms | 0.007 ms | 50 |
| B repeat | LLM request | 0.175 ms | 0.394 ms | 0.269 ms | 50 |

## 4. TC3 Similar Query Pair Quality

TC3은 `techqa` 유사질문 pair에서 A안 Answer Cache 재사용과 B안 Context Cache 재사용의 응답/품질 비교용 raw data를 기록한다.

### 공통 세팅

| 항목 | 값 |
|---|---|
| Test Case | TC3 Similar Query Pair Quality |
| Dataset | RAGBench `techqa` |
| Split | `test` |
| Source ID | `dp3_ragbench_techqa_test_314` |
| RAG corpus row 수 | 314 |
| Base EU rows | 5,986 |
| Versioned EU rows | 12,971 |
| Pair 수 | 10 |
| Query 수 | 20 |
| Requested version | V1 |
| User scope | A |
| LLM provider | groq |
| LLM model | llama-3.1-8b-instant |
| Route threshold | 0.70 |
| Cache hit threshold | 0.86 |
| LLM latency basis | `timings_ms.llm_ms` 실제 HTTP 요청 왕복시간 포함 |
| Reranker | On |
| Reranker requested device | cuda |
| Reranker resolved device | cuda |
| Rerank model | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Rerank candidates | 30 |
| Route pool mode | sampled |
| Route pool sample rate | 10% |
| Route pool min per dataset | 5 |
| Route pool seed | 42 |
| Query seed | 7 |
| Route pool | `ragbench:techqa:test` 31개 / 314 |
| Route pool indexes | `3, 12, 13, 15, 16, 44, 47, 52, 57, 71, 79, 81, 101, 110, 111, 112, 114, 119, 125, 140, 142, 174, 214, 216, 229, 258, 279, 287, 301, 302, 308` |
| RAGAS input | `/home/seungho/Shared/Archi/2026_architect_9B_team/src/data/ragbench/techqa/test_tc4_ragas_input.jsonl` |
| Run log file | `20260629_214358_similar_pair_quality_techqa_rerank-on_cuda_e791ca78.json` |
| Job ID | `e791ca78-a6de-4ad8-9e62-f2306af6a298` |
| Saved at | 20260629_214358 |
| Post-hoc total wait correction | true |

`Total (RAG+LLM) avg = Total avg` (실제 LLM 실행은 `total_ms`에 이미 포함)

### 전체 요약

| Mode | Total | Route passed | Cache hit | Validation passed | RAG fallback | LLM calls | Total avg | Total (RAG+LLM) avg |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A left_seed | 10 | 10 | 4 | 4 | 6 | 6 | 973.103 ms | 973.103 ms |
| A right_probe | 10 | 10 | 8 | 8 | 2 | 2 | 270.358 ms | 270.358 ms |
| B left_seed | 10 | 0 | 0 | 0 | 10 | 10 | 1184.845 ms | 1184.845 ms |
| B right_probe | 10 | 0 | 10 | 10 | 0 | 10 | 719.092 ms | 719.092 ms |

### Decision reasons

| Mode | Reason | Count |
|---|---|---:|
| A left_seed | `cache_candidate_not_found_fallback_to_roi_rag` | 6 |
| A left_seed | `answer_cache_hit_valid` | 4 |
| A right_probe | `answer_cache_hit_valid` | 8 |
| A right_probe | `cache_candidate_not_found_fallback_to_roi_rag` | 2 |
| B left_seed | `context_cache_candidate_not_found_full_fallback` | 10 |
| B right_probe | `context_cache_hit_all_valid` | 10 |

### A안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| A left_seed | Total | 13.942 ms | 1111.612 ms | 973.103 ms | 10 |
| A left_seed | Total (RAG+LLM) | 13.942 ms | 1111.612 ms | 973.103 ms | 10 |
| A left_seed | Embedding | 6.169 ms | 11.384 ms | 7.930 ms | 10 |
| A left_seed | Route | 1.357 ms | 4.251 ms | 1.904 ms | 10 |
| A left_seed | Cache lookup | 0.565 ms | 2.041 ms | 0.932 ms | 10 |
| A left_seed | Validation | 0.000 ms | 3.900 ms | 1.388 ms | 10 |
| A left_seed | RAG DB | 25.492 ms | 38.844 ms | 29.412 ms | 6 |
| A left_seed | RAG scoring | 264.032 ms | 280.880 ms | 272.145 ms | 6 |
| A left_seed | RAG score sort | 0.326 ms | 0.448 ms | 0.397 ms | 6 |
| A left_seed | RAG rerank | 156.392 ms | 2742.752 ms | 605.788 ms | 6 |
| A left_seed | RAG total | 471.374 ms | 3051.141 ms | 907.741 ms | 6 |
| A left_seed | Prompt build | 0.022 ms | 0.027 ms | 0.024 ms | 6 |
| A left_seed | LLM request | 590.732 ms | 817.210 ms | 679.989 ms | 6 |
| A left_seed | LLM wall | 664.457 ms | 55228.131 ms | 20061.880 ms | 6 |
| A left_seed | LLM throttle wait | 0.000 ms | 54614.129 ms | 19381.405 ms | 6 |
| A left_seed | LLM retry wait | 0.000 ms | 0.000 ms | 0.000 ms | 6 |
| A left_seed | LLM API reported queue | 9.915 ms | 228.112 ms | 107.160 ms | 6 |
| A left_seed | LLM API reported total | 244.182 ms | 394.384 ms | 314.779 ms | 6 |
| A left_seed | Cache store | 9.216 ms | 14.079 ms | 12.228 ms | 6 |
| A right_probe | Total | 12.973 ms | 1336.029 ms | 270.358 ms | 10 |
| A right_probe | Total (RAG+LLM) | 12.973 ms | 1336.029 ms | 270.358 ms | 10 |
| A right_probe | Embedding | 7.015 ms | 10.228 ms | 8.730 ms | 10 |
| A right_probe | Route | 1.350 ms | 2.037 ms | 1.686 ms | 10 |
| A right_probe | Cache lookup | 0.728 ms | 1.434 ms | 0.969 ms | 10 |
| A right_probe | Validation | 0.000 ms | 4.527 ms | 2.584 ms | 10 |
| A right_probe | RAG DB | 29.077 ms | 30.415 ms | 29.746 ms | 2 |
| A right_probe | RAG scoring | 269.208 ms | 284.510 ms | 276.859 ms | 2 |
| A right_probe | RAG score sort | 0.418 ms | 0.420 ms | 0.419 ms | 2 |
| A right_probe | RAG rerank | 174.104 ms | 187.018 ms | 180.561 ms | 2 |
| A right_probe | RAG total | 474.147 ms | 501.023 ms | 487.585 ms | 2 |
| A right_probe | Prompt build | 0.024 ms | 0.024 ms | 0.024 ms | 2 |
| A right_probe | LLM request | 710.789 ms | 853.116 ms | 781.952 ms | 2 |
| A right_probe | LLM wall | 1796.373 ms | 1923.751 ms | 1860.062 ms | 2 |
| A right_probe | LLM throttle wait | 1070.158 ms | 1085.163 ms | 1077.660 ms | 2 |
| A right_probe | LLM retry wait | 0.000 ms | 0.000 ms | 0.000 ms | 2 |
| A right_probe | LLM API reported queue | 61.873 ms | 64.471 ms | 63.172 ms | 2 |
| A right_probe | LLM API reported total | 402.137 ms | 517.262 ms | 459.699 ms | 2 |
| A right_probe | Cache store | 8.893 ms | 12.241 ms | 10.567 ms | 2 |

### B안 timing

| Pass | Metric | Min | Max | Avg | N |
|---|---|---:|---:|---:|---:|
| B left_seed | Total | 1068.737 ms | 1370.788 ms | 1184.845 ms | 10 |
| B left_seed | Total (RAG+LLM) | 1068.737 ms | 1370.788 ms | 1184.845 ms | 10 |
| B left_seed | Embedding | 6.260 ms | 10.140 ms | 8.020 ms | 10 |
| B left_seed | Cache lookup DB | 0.557 ms | 0.827 ms | 0.667 ms | 10 |
| B left_seed | Cache lookup scoring | 0.000 ms | 0.001 ms | 0.001 ms | 10 |
| B left_seed | Cache lookup total | 0.557 ms | 0.828 ms | 0.667 ms | 10 |
| B left_seed | Full retrieval DB | 25.659 ms | 41.070 ms | 29.393 ms | 10 |
| B left_seed | Full retrieval scoring | 258.980 ms | 353.421 ms | 277.730 ms | 10 |
| B left_seed | Full retrieval score sort | 0.362 ms | 0.488 ms | 0.428 ms | 10 |
| B left_seed | Full retrieval rerank | 149.403 ms | 197.107 ms | 166.622 ms | 10 |
| B left_seed | Full retrieval total | 445.897 ms | 533.318 ms | 474.172 ms | 10 |
| B left_seed | Prompt build | 0.025 ms | 0.033 ms | 0.029 ms | 10 |
| B left_seed | LLM request | 435.283 ms | 990.585 ms | 686.206 ms | 10 |
| B left_seed | LLM wall | 1448.451 ms | 53006.631 ms | 7200.580 ms | 10 |
| B left_seed | LLM throttle wait | 850.803 ms | 52119.290 ms | 6513.882 ms | 10 |
| B left_seed | LLM retry wait | 0.000 ms | 0.000 ms | 0.000 ms | 10 |
| B left_seed | LLM API reported queue | 60.652 ms | 192.290 ms | 107.407 ms | 10 |
| B left_seed | LLM API reported total | 174.468 ms | 565.257 ms | 359.940 ms | 10 |
| B left_seed | Cache store | 11.317 ms | 16.127 ms | 12.810 ms | 10 |
| B right_probe | Total | 535.243 ms | 964.237 ms | 719.092 ms | 10 |
| B right_probe | Total (RAG+LLM) | 535.243 ms | 964.237 ms | 719.092 ms | 10 |
| B right_probe | Embedding | 6.915 ms | 10.266 ms | 8.199 ms | 10 |
| B right_probe | Cache lookup DB | 0.585 ms | 0.887 ms | 0.702 ms | 10 |
| B right_probe | Cache lookup scoring | 0.159 ms | 0.274 ms | 0.198 ms | 10 |
| B right_probe | Cache lookup total | 0.758 ms | 1.161 ms | 0.900 ms | 10 |
| B right_probe | Validation | 2.985 ms | 3.849 ms | 3.502 ms | 10 |
| B right_probe | Prompt build | 0.008 ms | 0.013 ms | 0.010 ms | 10 |
| B right_probe | LLM request | 522.418 ms | 1046.702 ms | 705.180 ms | 10 |
| B right_probe | LLM wall | 1895.189 ms | 60076.337 ms | 39599.999 ms | 10 |
| B right_probe | LLM throttle wait | 1372.190 ms | 55181.989 ms | 37838.068 ms | 10 |
| B right_probe | LLM retry wait | 0.000 ms | 3000.000 ms | 500.000 ms | 10 |
| B right_probe | LLM API reported queue | 64.048 ms | 192.282 ms | 118.266 ms | 10 |
| B right_probe | LLM API reported total | 171.254 ms | 713.686 ms | 353.064 ms | 10 |

### Pair별 raw

| Pair ID | Similarity | Answer Jaccard | A right hit | B right hit | A answers equal | B answers equal | A right decision | B right decision | A right LLM request | B right LLM request | B right throttle wait |
|---|---:|---:|---|---|---|---|---|---|---:|---:|---:|
| `tc4:techqa_DEV_Q072:techqa_DEV_Q182` | 0.9569 | 0.3509 | true | true | true | false | `answer_cache_hit_valid` | `context_cache_hit_all_valid` | - | 578.874 ms | 1645.565 ms |
| `tc4:techqa_DEV_Q098:techqa_DEV_Q059` | 0.8685 | 0.2812 | true | true | true | false | `answer_cache_hit_valid` | `context_cache_hit_all_valid` | - | 522.418 ms | 1372.190 ms |
| `tc4:techqa_DEV_Q180:techqa_DEV_Q220` | 0.8633 | 0.1803 | true | true | false | false | `answer_cache_hit_valid` | `context_cache_hit_all_valid` | - | 1046.702 ms | 54628.182 ms |
| `tc4:techqa_DEV_Q072:techqa_DEV_Q019` | 0.9167 | 0.3492 | true | true | true | false | `answer_cache_hit_valid` | `context_cache_hit_all_valid` | - | 593.543 ms | 55181.989 ms |
| `tc4:techqa_DEV_Q044:techqa_DEV_Q115` | 0.8748 | 0.2078 | false | true | false | false | `cache_candidate_not_found_fallback_to_roi_rag` | `context_cache_hit_all_valid` | 853.116 ms | 550.491 ms | 55020.565 ms |
| `tc4:techqa_DEV_Q037:techqa_DEV_Q066` | 0.9412 | 0.3750 | false | true | false | false | `cache_candidate_not_found_fallback_to_roi_rag` | `context_cache_hit_all_valid` | 710.789 ms | 533.494 ms | 50150.911 ms |
| `tc4:techqa_DEV_Q182:techqa_DEV_Q236` | 0.8661 | 0.3387 | true | true | true | false | `answer_cache_hit_valid` | `context_cache_hit_all_valid` | - | 771.036 ms | 52750.593 ms |
| `tc4:techqa_DEV_Q098:techqa_DEV_Q182` | 0.8920 | 0.3077 | true | true | true | false | `answer_cache_hit_valid` | `context_cache_hit_all_valid` | - | 726.091 ms | 52745.850 ms |
| `tc4:techqa_DEV_Q037:techqa_DEV_Q105` | 0.8940 | 0.3261 | true | true | true | false | `answer_cache_hit_valid` | `context_cache_hit_all_valid` | - | 781.594 ms | 1510.273 ms |
| `tc4:techqa_DEV_Q058:techqa_DEV_Q105` | 0.9935 | 0.3704 | true | true | true | false | `answer_cache_hit_valid` | `context_cache_hit_all_valid` | - | 947.559 ms | 53374.563 ms |
