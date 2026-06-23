# DP3 PoC TODO

완료된 작업은 `DP3_PoC_Done.md`로 이동했다. 이 문서는 앞으로 남은 작업과 다음 실험 계획만 관리한다.

## 1. DP3 평가용 Test Case 재구성

현재 HTML PoC에는 기본 테스트 흐름이 들어가 있지만, 최종 DP3 평가에서는 다음 4개 Test Case를 명확히 분리해서 실행할 수 있어야 한다.

### TC1. Cache Benefit

Cache가 될 때와 안 될 때의 평균 시간 비용 차이를 확인한다.

#### 목적

- Answer Cache / Context Cache가 실제로 시간 비용을 줄이는지 확인한다.
- cache hit 평균 시간과 cache miss 평균 시간을 분리해서 비교한다.
- `Cache off baseline`을 포함해 cache 자체의 효과를 더 명확히 보여준다.

#### 구성

- A안, B안을 각각 실행한다.
- V1 기준 동일 또는 유사 질문을 1회 실행한 뒤, 같은 workload를 반복 실행한다.
- 비교군에 `Cache off`를 포함한다.
- LLM은 기본적으로 mock을 사용한다.

#### 주요 지표

- cache hit 비율
- validator 통과 비율
- cache off 평균 시간
- cache miss 평균 시간
- cache hit 평균 시간
- embedding / route / cache / RAG / LLM 구간별 min, max, avg

### TC2. Mixed Workload Performance

권한, 버전, 질문 유형이 섞인 현실형 workload에서 평균 수행시간을 확인한다.

#### 목적

- 실제 사용에 가까운 혼합 workload에서 A안/B안의 평균 수행시간을 비교한다.
- 권한/버전/fingerprint validation이 포함된 상태에서도 cache 전략이 안정적으로 동작하는지 확인한다.

#### 구성

- RAGBench 계열 데이터셋으로 전환하는 것을 목표로 한다.
- 유사 질문이 충분히 포함된 query set을 사용한다.
- scope A/B, version V1/V2/V3 요청을 섞는다.
- LLM은 HTML에서 선택 가능하게 둔다.

#### 주요 지표

- A안: route 통과 비율, cache hit 비율, validation 통과 비율
- B안: cache hit 비율, full validation 통과 비율, partial validation 통과 비율
- 평균 total latency
- embedding / route / cache / full RAG / delta RAG / LLM 구간별 min, max, avg

### TC3. Scale Cost

RAG 대상 규모가 커질수록 RAG 비용이 증가하는지 확인한다.

#### 목적

- 데이터 규모가 커질수록 no-cache RAG 비용이 증가한다는 DP3 필요성 근거를 만든다.
- cache 전략이 데이터 규모 증가에 대해 어느 정도 비용을 완화하는지 확인한다.

#### 구성

- 데이터셋 크기를 1배, 2배, 3배, 4배, 5배로 늘린다.
- `Cache off`, A안, B안을 비교한다.
- LLM은 기본적으로 mock을 사용한다.
- 필요하면 LLM 선택 옵션은 유지하되, scale 실험의 기본 해석에서는 RAG 비용 중심으로 본다.

#### 주요 지표

- 데이터 규모별 RAG 평균 시간
- 데이터 규모별 retrieval / reranking / validation / delta retrieval 시간
- cache off 대비 A/B안 평균 시간 절감률

### TC4. Similar Query Pair Quality

B안이 유사질문에 강하다는 점을 품질과 함께 검증한다.

#### 목적

- Context Cache 기반 B안이 Answer Cache 기반 A안보다 유사질문 재사용에 더 유리한지 확인한다.
- cache 재사용이 답변 품질을 과도하게 떨어뜨리지 않는지 RAGAS로 확인한다.

#### 구성

- cache에 걸릴 가능성이 높은 유사질문 pair를 여러 쌍 준비한다.
- A안, B안을 모두 LLM 포함 상태로 실행한다.
- 생성 답변, 사용 context, reference answer를 저장한다.
- RAGAS로 품질을 비교한다.

#### 주요 지표

- 유사질문 pair별 cache hit 여부
- A안/B안 cache hit 비율
- A안/B안 평균 수행시간
- faithfulness
- answer relevancy
- context precision
- context recall

### Optional. Validation Safety

권한 mismatch, version stale, fingerprint mismatch 상황에서 cache가 안전하게 차단되는지 별도 통계로 확인한다. 단독 Test Case로 만들 수도 있지만, 우선은 TC2 결과에 reject reason별 count를 추가하는 방식으로 둔다.

#### 후보 지표

- scope mismatch reject count
- version stale reject count
- fingerprint mismatch reject count
- full reject / partial reject 비율
- reject 후 fallback 또는 delta retrieval 정상 수행 여부

## 2. RAGBench 계열 데이터셋 추가

현재 DP3 PoC는 LongBench를 기본 데이터셋으로 사용한다. LongBench는 long-context understanding 평가에는 적합하지만, RAGAS 기반 RAG 품질 평가에는 최적의 데이터셋은 아니다. RAGAS 평가까지 고려하면 RAGBench 또는 Open RAG Bench류 데이터셋을 추가하는 편이 좋다.

### TODO

- RAGBench 또는 Open RAG Bench 후보를 조사한다.
- DP3 전용 loader를 추가한다.
- dataset abstraction을 정리한다.
  - `LongBench`
  - `RAGBench`
  - `Open RAG Bench` 또는 기타 후보
- RAGBench row를 DP3 `context_units` / `versioned_evidence_units` 구조로 변환한다.
- 기존 A/B안 cache runner가 dataset 종류와 무관하게 동작하도록 정리한다.
- LongBench는 개발/초기 PoC용, RAGBench는 RAG 품질 평가용으로 역할을 분리한다.

### 확인 기준

- RAGBench 계열 데이터셋으로 A/B안 cache test를 실행할 수 있어야 한다.
- RAGBench 계열 데이터셋으로 RAGAS 입력 형식을 만들 수 있어야 한다.
- DP1/DP2 기존 LongBench loader와 실행 방식은 영향을 받지 않아야 한다.

## 3. Cache-Friendly 유사질문 셋 생성

랜덤 LongBench 질문만 사용하면 semantic embedding 기준으로 첫 실행 cache hit 비율이 낮게 나오는 것이 자연스럽다. cache 전략 자체의 동작을 검증하려면 의도적으로 유사 질문을 포함한 query set이 필요하다.

### TODO

- 테스트 질문 셋을 미리 생성한다.
- 전체 질문 중 약 50%는 유사 질문으로 구성한다.
- 나머지 50%는 일반 random 질문으로 유지한다.
- 유사 질문은 다음 유형으로 분리한다.
  - `same`: 동일 질문 반복
  - `paraphrase`: 의미는 같지만 표현이 다른 질문
  - `near_miss`: 표면적으로 비슷하지만 답이나 근거가 달라야 하는 질문
  - `random`: 일반 LongBench/RAGBench 랜덤 질문
- HTML에서 query mix 모드를 선택할 수 있게 한다.
  - `random`
  - `cache_friendly_50`
  - `custom`
- query set은 매 실행마다 새로 만들지 않고 로컬 산출물로 저장 후 재사용한다.

### 권장 기본 비율

```text
same        20%
paraphrase  20%
near_miss   10%
random      50%
```

### 확인 기준

- TC1 첫 실행에서도 cache hit이 일정 비율 이상 발생해야 한다.
- TC1 반복 실행에서는 route 통과 질문의 cache hit이 높게 나와야 한다.
- `near_miss`는 cache threshold/validator에 의해 과도하게 재사용되지 않아야 한다.
- random workload 결과와 cache-friendly workload 결과를 분리해서 해석할 수 있어야 한다.

## 4. RAGAS 평가 추가

cache 전략은 속도만 개선하면 안 되고, answer quality 또는 groundedness를 과도하게 해치지 않아야 한다. RAGAS를 사용해 cache 적용 전후 품질을 비교한다.

### TODO

- RAGAS 실행 스크립트를 추가한다.
- A안, B안, no-cache baseline 결과를 RAGAS 입력 형식으로 저장한다.
- 우선 검토할 metric을 정한다.
  - faithfulness
  - answer relevancy
  - context precision
  - context recall
- LongBench 기반 RAGAS 가능성을 제한적으로 확인한다.
- RAGBench 계열 데이터셋으로 본 평가를 수행한다.

### 확인 기준

- cache hit 결과와 no-cache 결과의 품질 차이를 비교할 수 있어야 한다.
- A안/B안이 latency를 줄이면서 품질 저하를 얼마나 유발하는지 확인할 수 있어야 한다.
- RAGAS 비용이 크면 subset 평가를 지원해야 한다.

## 5. BM25 및 Hybrid Retrieval 추가

현재 retrieval과 route filtering은 vector similarity 중심이다. BM25 또는 hybrid retrieval을 추가하면 실제 RAG 시스템에 더 가까운 조건에서 cache 전략을 비교할 수 있다.

### TODO

- DP3 전용 EU 기준 BM25 index를 생성한다.
- vector-only, BM25-only, hybrid retrieval 결과를 비교한다.
- hybrid score 정책을 정한다.
  - vector score
  - BM25 score
  - weighted hybrid score
- A안 route filtering에 BM25/hybrid를 적용할지 검토한다.
- B안 context retrieval에 BM25/hybrid를 적용할지 검토한다.
- BM25/hybrid timing을 별도 저장한다.
  - `bm25_ms`
  - `hybrid_merge_ms`
  - `hybrid_total_ms`

### 확인 기준

- vector-only, BM25-only, hybrid를 같은 query set에서 비교할 수 있어야 한다.
- cache 전략 비교와 retrieval 방식 비교가 섞여 해석되지 않도록 실험 조건을 분리해야 한다.
- BM25 추가가 DP1/DP2 실행 방식에 영향을 주지 않아야 한다.

## 6. CPU/GPU 환경 비교

현재 노트북 환경은 CPU-only PyTorch를 사용한다. Cross-Encoder reranking은 CPU에서 비용이 크게 나타날 수 있으므로, GPU 환경에서 같은 테스트를 반복해 사양 의존성을 분리한다.

### TODO

- GPU 환경에서 CUDA PyTorch를 설치한다.
- `torch.cuda.is_available()` 값을 결과에 기록한다.
- reranker device 정보를 로그에 남긴다.
- 같은 테스트를 다음 조건으로 비교한다.
  - CPU, reranker OFF
  - CPU, reranker ON
  - GPU, reranker ON
- HTML 또는 결과 summary에 실행 환경 정보를 표시한다.

### 확인 기준

- 낮은 사양 노트북 때문에 latency가 과도하게 보였다는 리뷰에 대응할 수 있어야 한다.
- CPU/GPU 차이를 분리해 cache 전략 자체의 상대 효과를 설명할 수 있어야 한다.

## 7. 테스트 결과 저장 및 비교 리포트

현재 HTML에서 결과를 볼 수 있지만, 여러 조건의 결과를 장기 비교하기에는 부족하다.

### TODO

- test suite 실행 결과를 JSON 파일로 저장한다.
- 실행 조건 metadata를 함께 저장한다.
  - dataset
  - query count
  - query mix
  - route threshold
  - cache threshold
  - reranker on/off
  - rerank candidate count
  - LLM provider/model
  - CPU/GPU 환경
- 이전 실행 결과와 현재 실행 결과를 비교하는 간단한 report를 만든다.
- 발표용 표/그래프에 쓸 수 있는 CSV export를 추가한다.

### 확인 기준

- 같은 조건을 반복 실행했을 때 결과를 비교할 수 있어야 한다.
- threshold 또는 rerank 후보 수 변경에 따른 cache hit/latency 변화를 추적할 수 있어야 한다.
