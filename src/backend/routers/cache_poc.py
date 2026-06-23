from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from backend.cache.answer_cache import (
    run_answer_cache_query,
    setup_answer_cache_poc,
)
from backend.cache.context_cache import clear_context_cache_for_source, run_context_cache_query
from load_longbench_dp3 import prepare_dp3_longbench
from run_dp3_answer_cache_longbench_batch import _run_pass, _sample_queries
from seed_dp3_question_pool import DATA_DIR, seed_question_pool

router = APIRouter(prefix="/api/dp3", tags=["dp3-cache-poc"])


class SetupRequest(BaseModel):
    thread_id: str
    reset: bool = False


class AnswerCacheRunRequest(BaseModel):
    thread_id: str
    query: str
    user_scope: str = "A"
    requested_version: Optional[str] = None
    model: Optional[str] = None
    llm_provider: Optional[str] = None
    route_threshold: float = 0.70
    cache_threshold: float = 0.86
    use_reranker: bool = False
    rerank_candidates: int = 30
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class LongBenchPrepareRequest(BaseModel):
    dataset_name: str = "multifieldqa_en"
    num_examples: int = 5
    auto_download: bool = True
    reset: bool = False
    reset_metadata: bool = False


class QuestionPoolSeedRequest(BaseModel):
    dataset: Optional[str] = None
    sample_rate: float = 0.10
    min_per_dataset: int = 5
    seed: int = 42
    reset: bool = True
    include_smoke: bool = False


class AnswerCacheBatchRequest(BaseModel):
    source_id: str = "dp3_longbench_multifieldqa_en_5"
    dataset: Optional[str] = None
    count: int = 100
    seed: int = 7
    user_scope: str = "A"
    route_threshold: float = 0.70
    cache_threshold: float = 0.86
    llm_provider: Optional[str] = None
    model: Optional[str] = None
    use_reranker: bool = False
    rerank_candidates: int = 30
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    include_smoke: bool = False
    reset_cache: bool = True


class ContextCacheBatchRequest(BaseModel):
    source_id: str = "dp3_longbench_multifieldqa_en_5"
    dataset: Optional[str] = None
    count: int = 100
    seed: int = 7
    user_scope: str = "A"
    cache_threshold: float = 0.86
    llm_provider: Optional[str] = None
    model: Optional[str] = None
    use_reranker: bool = False
    rerank_candidates: int = 30
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    include_smoke: bool = False
    reset_cache: bool = True


@router.post("/answer-cache/setup")
def setup_answer_cache(body: SetupRequest):
    return setup_answer_cache_poc(body.thread_id, reset=body.reset)


@router.post("/answer-cache/run")
def run_answer_cache(body: AnswerCacheRunRequest):
    return run_answer_cache_query(
        thread_id=body.thread_id,
        query=body.query,
        user_scope=body.user_scope,
        requested_version=body.requested_version,
        model=body.model,
        llm_provider=body.llm_provider,
        route_threshold=body.route_threshold,
        cache_threshold=body.cache_threshold,
        use_reranker=body.use_reranker,
        rerank_candidates=body.rerank_candidates,
        rerank_model=body.rerank_model,
    )


@router.get("/longbench/datasets")
def list_longbench_datasets():
    datasets = []
    for path in sorted(DATA_DIR.glob("*.jsonl")):
        if path.name.startswith("dp3_smoke"):
            continue
        count = 0
        with path.open(encoding="utf-8-sig") as f:
            for line in f:
                if line.strip():
                    count += 1
        datasets.append({
            "dataset": path.stem,
            "rows": count,
            "size_kb": round(path.stat().st_size / 1024, 1),
        })
    return {"datasets": datasets, "total_rows": sum(d["rows"] for d in datasets)}


@router.post("/longbench/prepare")
def prepare_longbench(body: LongBenchPrepareRequest):
    return prepare_dp3_longbench(
        dataset_name=body.dataset_name,
        num_examples=body.num_examples,
        auto_download=body.auto_download,
        reset=body.reset,
        reset_metadata=body.reset_metadata,
    )


@router.get("/question-pool/stats")
def question_pool_stats():
    from backend.db.database import get_conn, init_db
    from backend.cache.answer_cache import init_dp3_cache_schema

    init_db()
    init_dp3_cache_schema()
    with get_conn() as conn:
        total = conn.execute(
            "SELECT COUNT(*) AS cnt FROM dp3_answerable_question_pool"
        ).fetchone()["cnt"]
        rows = conn.execute(
            """SELECT route_type, COUNT(*) AS cnt
               FROM dp3_answerable_question_pool
               GROUP BY route_type
               ORDER BY route_type"""
        ).fetchall()
    return {
        "total": total,
        "by_route_type": [dict(row) for row in rows],
    }


@router.post("/question-pool/seed")
def seed_pool(body: QuestionPoolSeedRequest):
    return seed_question_pool(
        dataset=body.dataset,
        sample_rate=body.sample_rate,
        min_per_dataset=body.min_per_dataset,
        seed=body.seed,
        reset=body.reset,
        include_smoke=body.include_smoke,
    )


@router.post("/answer-cache/batch")
def run_answer_cache_batch(body: AnswerCacheBatchRequest):
    from backend.cache.answer_cache import clear_answer_cache_for_source
    from backend.db.database import init_db
    from collections import Counter

    init_db()
    if body.reset_cache:
        clear_answer_cache_for_source(body.source_id)

    queries = _sample_queries(body.dataset, body.count, body.seed, body.include_smoke)
    passes = [
        _run_pass(
            "v1_first",
            body.source_id,
            queries,
            body.user_scope,
            "V1",
            body.route_threshold,
            body.cache_threshold,
            body.llm_provider,
            body.model,
            body.use_reranker,
            body.rerank_candidates,
            body.rerank_model,
        ),
        _run_pass(
            "v1_repeat",
            body.source_id,
            queries,
            body.user_scope,
            "V1",
            body.route_threshold,
            body.cache_threshold,
            body.llm_provider,
            body.model,
            body.use_reranker,
            body.rerank_candidates,
            body.rerank_model,
        ),
        _run_pass(
            "v2_validation",
            body.source_id,
            queries,
            body.user_scope,
            "V2",
            body.route_threshold,
            body.cache_threshold,
            body.llm_provider,
            body.model,
            body.use_reranker,
            body.rerank_candidates,
            body.rerank_model,
        ),
    ]
    return {
        "source_id": body.source_id,
        "sampled_query_count": len(queries),
        "sampled_datasets": dict(Counter(q["dataset"] for q in queries)),
        "route_threshold": body.route_threshold,
        "cache_threshold": body.cache_threshold,
        "passes": [{"pass": p["pass"], "summary": p["summary"]} for p in passes],
    }


def _summarize_context_results(results: list[dict]) -> dict:
    from collections import Counter

    def avg(values: list[float]) -> float | None:
        if not values:
            return None
        return round(sum(values) / len(values), 3)

    def timing_value(row: dict, key: str) -> float | None:
        value = (row.get("timings_ms") or {}).get(key)
        if value is None:
            return None
        return float(value)

    def timing_summary() -> dict:
        keys = [
            "embedding_ms",
            "cache_lookup_db_ms",
            "cache_lookup_scoring_ms",
            "cache_lookup_ms",
            "validation_ms",
            "valid_current_lookup_ms",
            "delta_retrieval_db_ms",
            "delta_retrieval_scoring_ms",
            "delta_retrieval_score_sort_ms",
            "delta_retrieval_rerank_ms",
            "delta_retrieval_filter_ms",
            "delta_retrieval_total_ms",
            "full_retrieval_db_ms",
            "full_retrieval_scoring_ms",
            "full_retrieval_score_sort_ms",
            "full_retrieval_rerank_ms",
            "full_retrieval_total_ms",
            "prompt_build_ms",
            "llm_ms",
            "cache_store_ms",
            "total_ms",
        ]
        summary = {}
        for key in keys:
            values = [
                value for value in (timing_value(row, key) for row in results)
                if value is not None
            ]
            if values:
                summary[key] = avg(values)

        hit_rows = [row for row in results if row.get("cache_hit")]
        retrieval_rows = [row for row in results if row.get("retrieval_called")]
        full_rows = [row for row in results if row.get("full_retrieval")]
        delta_rows = [row for row in results if int(row.get("delta_retrieval_count", 0)) > 0]
        summary["hit_total_ms"] = avg([
            value for value in (timing_value(row, "total_ms") for row in hit_rows)
            if value is not None
        ])
        summary["retrieval_total_request_ms"] = avg([
            value for value in (timing_value(row, "total_ms") for row in retrieval_rows)
            if value is not None
        ])
        summary["full_retrieval_total_request_ms"] = avg([
            value for value in (timing_value(row, "total_ms") for row in full_rows)
            if value is not None
        ])
        summary["delta_retrieval_total_request_ms"] = avg([
            value for value in (timing_value(row, "total_ms") for row in delta_rows)
            if value is not None
        ])
        return summary

    by_reason = Counter(r.get("decision_reason", "unknown") for r in results)
    by_dataset = {}
    for row in results:
        item = by_dataset.setdefault(row["dataset"], {
            "total": 0,
            "context_cache_hit": 0,
            "validation_passed": 0,
            "delta_retrievals": 0,
            "full_retrievals": 0,
            "llm_calls": 0,
        })
        item["total"] += 1
        item["context_cache_hit"] += int(bool(row.get("cache_hit")))
        item["validation_passed"] += int(bool(row.get("validation_passed")))
        item["delta_retrievals"] += int(row.get("delta_retrieval_count", 0) > 0)
        item["full_retrievals"] += int(bool(row.get("full_retrieval")))
        item["llm_calls"] += int(row.get("llm_call_count", 0))

    return {
        "total": len(results),
        "context_cache_hits": sum(1 for r in results if r.get("cache_hit")),
        "validation_passed": sum(1 for r in results if r.get("validation_passed")),
        "delta_retrievals": sum(1 for r in results if int(r.get("delta_retrieval_count", 0)) > 0),
        "full_retrievals": sum(1 for r in results if r.get("full_retrieval")),
        "fallbacks": sum(1 for r in results if r.get("full_retrieval")),
        "llm_calls": sum(int(r.get("llm_call_count", 0)) for r in results),
        "timing_avg_ms": timing_summary(),
        "decision_reasons": dict(by_reason),
        "by_dataset": by_dataset,
    }


def _run_context_pass(
    pass_name: str,
    source_id: str,
    queries: list[dict],
    user_scope: str,
    requested_version: str,
    cache_threshold: float,
    llm_provider: str | None,
    model: str | None,
    use_reranker: bool,
    rerank_candidates: int,
    rerank_model: str,
) -> dict:
    results = []
    for item in queries:
        result = run_context_cache_query(
            source_id=source_id,
            query=item["query"],
            user_scope=user_scope,
            requested_version=requested_version,
            model=model,
            llm_provider=llm_provider,
            cache_threshold=cache_threshold,
            use_reranker=use_reranker,
            rerank_candidates=rerank_candidates,
            rerank_model=rerank_model,
        )
        result["query_id"] = item["query_id"]
        result["dataset"] = item["dataset"]
        result["index"] = item["index"]
        results.append(result)
    return {"pass": pass_name, "summary": _summarize_context_results(results), "results": results}


@router.post("/context-cache/batch")
def run_context_cache_batch(body: ContextCacheBatchRequest):
    from backend.db.database import init_db
    from collections import Counter

    init_db()
    if body.reset_cache:
        clear_context_cache_for_source(body.source_id)

    queries = _sample_queries(body.dataset, body.count, body.seed, body.include_smoke)
    passes = [
        _run_context_pass(
            "v1_first",
            body.source_id,
            queries,
            body.user_scope,
            "V1",
            body.cache_threshold,
            body.llm_provider,
            body.model,
            body.use_reranker,
            body.rerank_candidates,
            body.rerank_model,
        ),
        _run_context_pass(
            "v1_repeat",
            body.source_id,
            queries,
            body.user_scope,
            "V1",
            body.cache_threshold,
            body.llm_provider,
            body.model,
            body.use_reranker,
            body.rerank_candidates,
            body.rerank_model,
        ),
        _run_context_pass(
            "v2_validation",
            body.source_id,
            queries,
            body.user_scope,
            "V2",
            body.cache_threshold,
            body.llm_provider,
            body.model,
            body.use_reranker,
            body.rerank_candidates,
            body.rerank_model,
        ),
    ]
    return {
        "source_id": body.source_id,
        "sampled_query_count": len(queries),
        "sampled_datasets": dict(Counter(q["dataset"] for q in queries)),
        "cache_threshold": body.cache_threshold,
        "passes": [{"pass": p["pass"], "summary": p["summary"]} for p in passes],
    }
