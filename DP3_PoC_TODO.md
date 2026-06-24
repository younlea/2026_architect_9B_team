# DP3 PoC TODO

완료된 작업은 `DP3_PoC_Done.md`로 이동했다. 이 문서는 앞으로 남은 작업과 다음 실험 계획만 관리한다.

## 1. DP3 평가용 Test Case 재구성

최종 DP3 평가는 RAGBench 기반으로 다음 4개 Test Case를 분리해 실행한다. 모든 TC는 원칙적으로 "사용 중인 RAG corpus row 안에 포함된 질문"만 사용한다. 즉, EU를 만들지 않은 row의 질문을 테스트 query로 던지지 않는다.

공통 데이터 구성 원칙:

- TC1/TC2는 RAGBench `techqa`를 기본 dataset으로 둔다.
- TC3/TC4는 RAGBench `emanual`을 기본 dataset으로 둔다.
- TC1/TC2용 `techqa` corpus는 row scale `100`, `200`, `300`을 기본 단위로 만든다.
- TC1은 사용자가 선택한 row scale을 사용하고, TC2는 `100 -> 200 -> 300` 세 scale을 고정 비교한다.
- TC1/TC2의 질문은 해당 scale에 포함된 row의 질문에서 seed 기반 random sampling으로 뽑는다.
- TC1/TC2의 A안 route pool도 같은 row 범위 안에서 seed 기반 random sampling으로 뽑되, 테스트 질문과 exact overlap은 피한다.
- TC3/TC4는 `emanual` 전체 row를 EU corpus로 사용한다.
- 권한/버전 테스트 안정성을 위해, 추후 metadata 생성 방식은 base EU를 scope A/B 양쪽으로 복제한 뒤 각 scope 안에서 V1/V2/V3를 생성하는 구조로 전환하는 것을 검토한다.
  - V1: 전체 EU
  - V2: V1 중 2/3
  - V3: V2 중 1/2
  - 이 구조는 같은 질문을 A/B 권한에서 모두 테스트하기 쉽게 만든다.

### TC1. Cache Benefit

Cache가 될 때와 안 될 때의 평균 시간 비용 차이를 확인한다.

#### 목적

- Answer Cache / Context Cache가 실제로 시간 비용을 줄이는지 확인한다.
- NoCache 상태의 RAG + LLM 비용과, A/B cache hit 시의 비용 차이를 비교한다.
- mock LLM 실행 시에도 `GROQ llama-3.1-8b-instant`, 약 5.6K token 기준 LLM 700ms를 더한 추정 total을 함께 본다.

#### 구성

- 기본 dataset은 RAGBench `techqa`이다.
- row scale은 사용자가 선택한다.
  - 기본 후보: `100`, `200`, `300`
  - TC2에서 만든 동일 source를 재사용한다.
- 선택된 row scale 안에서 EU를 만든다.
- 질문은 선택된 row 범위 안의 질문에서 seed 기반 random sampling으로 뽑는다.
- A안 route pool도 같은 row 범위 안에서 seed 기반 random sampling으로 만들되, 테스트 질문과 exact overlap은 제외한다.
- 실행 순서:
  - NoCache 1회
  - A안 V1 첫 실행
  - A안 V1 반복 실행
  - B안 V1 첫 실행
  - B안 V1 반복 실행
- LLM은 기본적으로 mock을 사용한다.

#### 주요 지표

- NoCache total / full RAG / estimated total with LLM
- cache hit 비율
- validator 통과 비율
- A/B 반복 실행 total latency
- A/B 반복 실행 estimated total with LLM
- cache hit 요청과 fallback 요청의 평균 시간 차이
- embedding / route / cache / RAG / LLM 구간별 min, max, avg

### TC2. Scale Cost

RAG 대상 규모가 커질수록 RAG 비용이 증가하는지 확인한다.

#### 목적

- 데이터 규모가 커질수록 no-cache RAG 비용이 증가한다는 DP3 필요성 근거를 만든다.
- A/B cache 적용 결과는 참고용으로 함께 보되, 핵심 해석은 NoCache RAG 비용 증가에 둔다.

#### 구성

- 기본 dataset은 RAGBench `techqa`이다.
- row scale은 `100`, `200`, `300`으로 고정한다.
- 각 scale은 별도 source로 한 번 EU를 만들고 이후 재사용한다.
  - `dp3_ragbench_techqa_test_100`
  - `dp3_ragbench_techqa_test_200`
  - `dp3_ragbench_techqa_test_300`
- 질문은 각 scale에 포함된 row 범위 안에서 seed 기반 random sampling으로 뽑는다.
- NoCache는 필수로 실행한다.
- A안/B안은 첫 실행과 반복 실행을 모두 수행한다.
- TC2에서는 반복 cache benefit보다 corpus 규모 증가에 따른 retrieval/scoring 비용 증가를 본다.
- LLM은 기본적으로 mock을 사용한다.

#### 주요 지표

- scale별 실제 base EU 수
- scale별 NoCache full RAG 평균 시간
- scale별 retrieval DB / scoring / sort / reranking 시간
- A/B 첫 실행과 반복 실행의 cache hit, validation, fallback/delta retrieval 양상
- 전체 corpus 증가에 따라 scoring 비용이 증가하는지

### TC3. Mixed Workload Performance

유사질문 set이 포함된 혼합 workload에서 A/B안의 평균 수행시간과 cache 동작을 확인한다.

#### 목적

- 실제 사용에 가까운 혼합 workload에서 A안/B안의 평균 수행시간을 비교한다.
- 유사질문이 포함된 workload에서 A안 route/cache, B안 context cache/delta retrieval이 어떻게 동작하는지 확인한다.

#### 구성

- 기본 후보는 RAGBench `emanual`로 둔다.
- `emanual` 전체 row를 EU corpus로 사용한다.
- 질문은 유사질문 set asset을 사용한다.
- 유사질문 set 생성 방식:
  - `emanual` 전체 질문에 대해 brute force similarity를 계산한다.
  - cache hit threshold `0.86` 이상인 유사 질문이 하나라도 있는 질문을 set 후보로 둔다.
  - 필요하면 `same`, `similar`, `near_miss`, `random` 역할을 같이 기록한다.
  - 생성된 set은 매 실행마다 새로 만들지 않고 local asset으로 저장해 재사용한다.
- 같은 질문 set을 NoCache, A안, B안에 던진다.
- LLM은 HTML에서 선택 가능하게 둔다.

#### 주요 지표

- A안: route 통과 비율, cache hit 비율, validation 통과 비율
- B안: cache hit 비율, full validation 통과 비율, partial validation 통과 비율
- B안: full RAG / delta RAG 호출 비율
- 평균 total latency
- estimated total with LLM
- embedding / route / cache / full RAG / delta RAG / LLM 구간별 min, max, avg

### TC4. Similar Query Pair Quality

B안이 유사질문에 강하다는 점을 품질과 함께 검증한다.

#### 목적

- Context Cache 기반 B안이 Answer Cache 기반 A안보다 유사질문 재사용에 더 유리한지 확인한다.
- cache 재사용이 답변 품질을 과도하게 떨어뜨리지 않는지 RAGAS로 확인한다.

#### 구성

- 기본 후보는 RAGBench `emanual`로 둔다.
- `emanual` 전체 row를 EU corpus로 사용한다.
- 유사질문 pair 생성 방식:
  - `emanual` 전체 질문에 대해 brute force similarity를 계산한다.
  - cache hit threshold `0.86` 이상인 질문 pair를 최대한 많이 만든다.
  - 가능하면 reference answer가 다르거나 source row/id가 다른 pair를 우선한다.
  - 생성된 pair는 local asset으로 저장해 재사용한다.
- pair 단위로 실행한다.
  - A안: pair의 첫 질문으로 answer cache를 seed하고, 두 번째 질문이 cache hit되는지 확인한다.
  - B안: A안에 사용한 동일 pair를 그대로 사용한다.
  - 총 질문 수는 `N pair * 2`이다.
- A안의 핵심 관찰점은 answer-level cache reuse로 두 번째 질문에 동일 답변을 반환할 위험이다.
- B안의 핵심 관찰점은 context-level cache reuse 후 질문별로 다른 답변을 생성할 수 있다는 점이다.
- A안, B안을 모두 LLM 포함 상태로 실행한다.
- 생성 답변, 사용 context, reference answer를 저장한다.
- RAGAS로 품질을 비교한다.

#### 주요 지표

- 유사질문 pair별 cache hit 여부
- A안/B안 cache hit 비율
- A안/B안 두 번째 질문 답변이 첫 번째 답변과 동일한지
- A안/B안 평균 수행시간
- faithfulness
- answer relevancy
- context precision
- context recall

### Optional. Validation Safety

권한 mismatch, version stale, fingerprint mismatch 상황에서 cache가 안전하게 차단되는지 별도 통계로 확인한다. 단독 Test Case로 만들 수도 있지만, 우선은 TC3 결과에 reject reason별 count를 추가하는 방식으로 둔다.

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

## 8. Open Question - Cache Reuse vs New Cache

version 또는 scope 조건 때문에 기존 cache가 현재 요청에서 invalid로 판정될 때, 해당 cache가 실제로 잘못된 cache인지 아니면 현재 요청 조건과만 맞지 않는 cache인지 구분할 필요가 있다.

### TODO

- partial invalid 또는 version mismatch 발생 시 기존 cache를 수정할지, 새 cache item을 생성할지 정책을 검토한다.
- 새 cache 생성 방식은 이력 추적과 실험 해석이 쉽지만 cache 중복이 늘 수 있다.
- 기존 cache 수정 방식은 저장 공간을 줄일 수 있지만, 어떤 query/version/scope 조합에서 만들어진 cache였는지 추적이 흐려질 수 있다.
- 권한 검증 구조를 별도로 조정한 뒤 version mismatch cache 처리 정책을 다시 결정한다.

