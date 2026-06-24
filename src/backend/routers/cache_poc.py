import json
import random
import threading
import time
import uuid
from collections import Counter
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

import backend.cache.answer_cache as answer_cache_module
from backend.cache.answer_cache import (
    TOP_K_SOURCES,
    _build_prompt,
    _embed,
    _elapsed_ms,
    _retrieve_context_units,
    _set_timing,
    _set_total_ms,
    _timer,
    clear_answer_cache_for_source,
    run_answer_cache_query,
    setup_answer_cache_poc,
)
from backend.cache.cache_llm import get_dp3_answer, get_dp3_llm_provider, is_mock_llm
from backend.cache.context_cache import clear_context_cache_for_source, run_context_cache_query
from backend.db.database import init_db
from build_dp3_ragbench_query_assets import build_query_assets
from load_longbench_dp3 import prepare_dp3_longbench
from load_ragbench_dp3 import (
    RAGBENCH_DATA_DIR,
    RAGBENCH_SUBSETS,
    iter_ragbench_queries,
    list_local_ragbench_datasets,
    prepare_dp3_ragbench,
    seed_ragbench_question_pool_sampled,
)
from run_dp3_answer_cache_longbench_batch import _run_pass, _sample_queries
from seed_dp3_question_pool import DATA_DIR, seed_question_pool

router = APIRouter(prefix="/api/dp3", tags=["dp3-cache-poc"])

_SUITE_JOBS: dict[str, dict] = {}
_SUITE_JOB_LOCK = threading.Lock()


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


class RAGBenchPrepareRequest(BaseModel):
    dataset_name: str = "techqa"
    dataset_split: str = "test"
    num_examples: int = 20
    auto_download: bool = True
    reset: bool = False
    reset_metadata: bool = False


class QuestionPoolSeedRequest(BaseModel):
    dataset_family: str = "longbench"
    dataset: Optional[str] = None
    dataset_split: str = "test"
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


class TestSuiteRunRequest(BaseModel):
    test_case: str = "cache"
    dataset_family: str = "longbench"
    dataset_name: str = "multifieldqa_en"
    dataset_split: str = "test"
    num_examples: int = 5
    query_count: int = 100
    seed: int = 7
    warmup_count: int = 3
    user_scope: str = "A"
    route_threshold: float = 0.70
    cache_threshold: float = 0.86
    sample_rate: float = 0.10
    min_per_dataset: int = 5
    pool_seed: int = 42
    llm_provider: Optional[str] = None
    model: Optional[str] = None
    use_reranker: bool = False
    rerank_candidates: int = 30
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reset_metadata: bool = False
    max_scale: int = 5
    include_smoke: bool = False


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


@router.get("/ragbench/datasets")
def list_ragbench_datasets():
    local = list_local_ragbench_datasets()
    return {
        "available_subsets": list(RAGBENCH_SUBSETS),
        **local,
    }


@router.post("/ragbench/prepare")
def prepare_ragbench(body: RAGBenchPrepareRequest):
    return prepare_dp3_ragbench(
        subset=body.dataset_name,
        split=body.dataset_split,
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
    if body.dataset_family.strip().lower() == "ragbench":
        return seed_ragbench_question_pool_sampled(
            subset=body.dataset or "techqa",
            split=body.dataset_split,
            reset=body.reset,
            sample_rate=body.sample_rate,
            min_count=body.min_per_dataset,
            seed=body.seed,
        )
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


A_TIMING_KEYS = [
    "embedding_ms",
    "route_ms",
    "cache_lookup_ms",
    "validation_ms",
    "rag_db_ms",
    "rag_scoring_ms",
    "rag_score_sort_ms",
    "rag_rerank_ms",
    "rag_total_ms",
    "prompt_build_ms",
    "llm_ms",
    "cache_store_ms",
    "total_ms",
]

B_TIMING_KEYS = [
    "embedding_ms",
    "cache_lookup_db_ms",
    "cache_lookup_scoring_ms",
    "cache_lookup_ms",
    "validation_ms",
    "valid_current_lookup_ms",
    "full_retrieval_db_ms",
    "full_retrieval_scoring_ms",
    "full_retrieval_score_sort_ms",
    "full_retrieval_rerank_ms",
    "full_retrieval_total_ms",
    "delta_retrieval_db_ms",
    "delta_retrieval_scoring_ms",
    "delta_retrieval_score_sort_ms",
    "delta_retrieval_rerank_ms",
    "delta_retrieval_filter_ms",
    "delta_retrieval_total_ms",
    "prompt_build_ms",
    "llm_ms",
    "cache_store_ms",
    "total_ms",
]

NO_CACHE_TIMING_KEYS = [
    "embedding_ms",
    "full_retrieval_db_ms",
    "full_retrieval_scoring_ms",
    "full_retrieval_score_sort_ms",
    "full_retrieval_rerank_ms",
    "full_retrieval_total_ms",
    "prompt_build_ms",
    "llm_ms",
    "total_ms",
]

MOCK_LLM_ESTIMATE_MS = 700.0


def _update_suite_job(job_id: str | None, **fields) -> None:
    if not job_id:
        return
    with _SUITE_JOB_LOCK:
        job = _SUITE_JOBS.get(job_id)
        if not job:
            return
        job.update(fields)
        if job.get("total"):
            job["progress_percent"] = round((job.get("completed", 0) / job["total"]) * 100, 1)


class _SuiteProgress:
    def __init__(self, job_id: str | None):
        self.job_id = job_id

    def reset(self, total: int, step: str) -> None:
        _update_suite_job(
            self.job_id,
            current_step=step,
            completed=0,
            total=max(1, total),
            progress_percent=0.0,
        )

    def step(self, step: str) -> None:
        _update_suite_job(self.job_id, current_step=step)

    def advance(self, step: str, count: int = 1) -> None:
        if not self.job_id:
            return
        with _SUITE_JOB_LOCK:
            job = _SUITE_JOBS.get(self.job_id)
            if not job:
                return
            job["current_step"] = step
            job["completed"] = min(job.get("total", 1), job.get("completed", 0) + count)
            job["progress_percent"] = round((job["completed"] / max(1, job.get("total", 1))) * 100, 1)


def _ratio(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count / total, 4)


def _timing_stats(results: list[dict], keys: list[str]) -> dict:
    stats = {}
    for key in keys:
        values = []
        for row in results:
            value = (row.get("timings_ms") or {}).get(key)
            if value is not None:
                values.append(float(value))
        if values:
            stats[key] = {
                "avg": round(sum(values) / len(values), 3),
                "min": round(min(values), 3),
                "max": round(max(values), 3),
                "count": len(values),
            }
    return stats


def _estimated_total_with_llm_stats(results: list[dict]) -> dict:
    values = []
    for row in results:
        total = (row.get("timings_ms") or {}).get("total_ms")
        if total is None:
            total = row.get("total_ms")
        if total is None:
            continue
        llm_calls = int(row.get("llm_call_count", 0))
        values.append(float(total) + llm_calls * MOCK_LLM_ESTIMATE_MS)
    if not values:
        return {}
    return {
        "avg": round(sum(values) / len(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "count": len(values),
        "mock_llm_estimate_ms": MOCK_LLM_ESTIMATE_MS,
        "basis": "total_ms + llm_call_count * 700ms",
    }


def _query_dataset_counts(queries: list[dict]) -> dict:
    return dict(Counter(item["dataset"] for item in queries))


def _dataset_family(body: TestSuiteRunRequest) -> str:
    family = (body.dataset_family or "longbench").strip().lower()
    if family not in {"longbench", "ragbench"}:
        raise ValueError(f"Unknown dataset_family: {body.dataset_family}")
    return family


def _summarize_a_detailed(results: list[dict]) -> dict:
    total = len(results)
    route_passed = sum(1 for row in results if row.get("routing_passed"))
    cache_hits = sum(1 for row in results if row.get("cache_hit"))
    validation_passed = sum(1 for row in results if row.get("validation_passed"))
    fallbacks = sum(1 for row in results if row.get("roi_rag_called"))
    return {
        "total": total,
        "route_passed": route_passed,
        "route_pass_ratio": _ratio(route_passed, total),
        "cache_hits": cache_hits,
        "cache_hit_ratio": _ratio(cache_hits, total),
        "validation_passed": validation_passed,
        "validation_pass_ratio": _ratio(validation_passed, total),
        "fallbacks": fallbacks,
        "fallback_ratio": _ratio(fallbacks, total),
        "llm_calls": sum(int(row.get("llm_call_count", 0)) for row in results),
        "decision_reasons": dict(Counter(row.get("decision_reason", "unknown") for row in results)),
        "timing_stats_ms": _timing_stats(results, A_TIMING_KEYS),
        "estimated_total_with_llm_ms": _estimated_total_with_llm_stats(results),
    }


def _summarize_b_detailed(results: list[dict]) -> dict:
    total = len(results)
    cache_hits = sum(1 for row in results if row.get("cache_hit"))
    full_valid = sum(1 for row in results if row.get("validation_passed"))
    partial_valid = sum(
        1
        for row in results
        if row.get("decision_reason") == "context_cache_partial_invalid_delta_rebuilt"
    )
    full_retrievals = sum(1 for row in results if row.get("full_retrieval"))
    delta_retrievals = sum(1 for row in results if int(row.get("delta_retrieval_count", 0)) > 0)
    return {
        "total": total,
        "context_cache_hits": cache_hits,
        "cache_hit_ratio": _ratio(cache_hits, total),
        "validation_full_passed": full_valid,
        "validation_full_pass_ratio": _ratio(full_valid, total),
        "validation_partial_passed": partial_valid,
        "validation_partial_pass_ratio": _ratio(partial_valid, total),
        "full_retrievals": full_retrievals,
        "full_retrieval_ratio": _ratio(full_retrievals, total),
        "delta_retrievals": delta_retrievals,
        "delta_retrieval_ratio": _ratio(delta_retrievals, total),
        "llm_calls": sum(int(row.get("llm_call_count", 0)) for row in results),
        "decision_reasons": dict(Counter(row.get("decision_reason", "unknown") for row in results)),
        "timing_stats_ms": _timing_stats(results, B_TIMING_KEYS),
        "estimated_total_with_llm_ms": _estimated_total_with_llm_stats(results),
    }


def _summarize_no_cache(results: list[dict]) -> dict:
    total = len(results)
    return {
        "total": total,
        "full_retrievals": sum(1 for row in results if row.get("full_retrieval")),
        "llm_calls": sum(int(row.get("llm_call_count", 0)) for row in results),
        "decision_reasons": dict(Counter(row.get("decision_reason", "unknown") for row in results)),
        "timing_stats_ms": _timing_stats(results, NO_CACHE_TIMING_KEYS),
        "estimated_total_with_llm_ms": _estimated_total_with_llm_stats(results),
    }


def _prepare_suite_source(body: TestSuiteRunRequest, num_examples: int | None = None) -> dict:
    if _dataset_family(body) == "ragbench":
        return prepare_dp3_ragbench(
            subset=body.dataset_name,
            split=body.dataset_split,
            num_examples=num_examples or body.num_examples,
            auto_download=True,
            reset=False,
            reset_metadata=body.reset_metadata,
            seed_route_pool=False,
        )
    return prepare_dp3_longbench(
        dataset_name=body.dataset_name,
        num_examples=num_examples or body.num_examples,
        auto_download=True,
        reset=False,
        reset_metadata=body.reset_metadata,
    )


def _seed_suite_route_pool(body: TestSuiteRunRequest, exclude_indexes: set[int] | None = None) -> dict:
    sample_rate = max(0.0, min(1.0, body.sample_rate))
    min_count = max(1, body.min_per_dataset)
    if _dataset_family(body) == "ragbench":
        return seed_ragbench_question_pool_sampled(
            subset=body.dataset_name,
            split=body.dataset_split,
            reset=True,
            sample_rate=sample_rate,
            min_count=min_count,
            seed=body.pool_seed,
            exclude_indexes=exclude_indexes,
        )
    return seed_question_pool(
        dataset=body.dataset_name,
        sample_rate=sample_rate,
        min_per_dataset=min_count,
        seed=body.pool_seed,
        reset=True,
        include_smoke=body.include_smoke,
    )


def _sample_ragbench_queries(
    dataset_name: str,
    split: str,
    count: int,
    seed: int,
    exclude_indexes: set[int] | None = None,
) -> list[dict]:
    exclude_indexes = exclude_indexes or set()
    queries = [
        item
        for item in iter_ragbench_queries(dataset_name, split)
        if int(item["index"]) not in exclude_indexes
    ]
    rng = random.Random(seed)
    if count >= len(queries):
        return queries
    return sorted(rng.sample(queries, count), key=lambda item: (item["dataset"], item["index"]))


def _ragbench_asset_path(dataset_name: str, split: str, filename: str):
    return RAGBENCH_DATA_DIR / dataset_name.strip().lower() / f"{split.strip().lower()}_{filename}.jsonl"


def _ensure_emanual_assets(body: TestSuiteRunRequest) -> dict:
    dataset = body.dataset_name.strip().lower()
    split = body.dataset_split.strip().lower()
    tc2_path = _ragbench_asset_path(dataset, split, "tc2_query_sets")
    tc4_path = _ragbench_asset_path(dataset, split, "tc4_query_pairs")
    if tc2_path.exists() and tc4_path.exists():
        return {
            "subset": dataset,
            "split": split,
            "tc2_path": str(tc2_path),
            "tc4_path": str(tc4_path),
            "reused": True,
        }
    result = build_query_assets(
        subset=dataset,
        split=split,
        seed=body.seed,
        cache_threshold=body.cache_threshold,
    )
    result["reused"] = False
    return result


def _read_jsonl(path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _tc2_asset_queries(body: TestSuiteRunRequest) -> list[dict]:
    assets = _ensure_emanual_assets(body)
    rows = _read_jsonl(assets["tc2_path"])
    rng = random.Random(body.seed)
    grouped = {}
    for row in rows:
        grouped.setdefault(row["group_id"], []).append(row)
    groups = list(grouped.values())
    rng.shuffle(groups)
    flattened = [row for group in groups for row in sorted(group, key=lambda r: r["role"])]
    count = min(body.query_count, len(flattened)) if body.query_count > 0 else len(flattened)
    return flattened[:count]


def _tc4_asset_pairs(body: TestSuiteRunRequest) -> list[dict]:
    assets = _ensure_emanual_assets(body)
    rows = _read_jsonl(assets["tc4_path"])
    rng = random.Random(body.seed)
    rng.shuffle(rows)
    max_pairs = body.query_count // 2 if body.query_count > 0 else len(rows)
    return rows[: max(1, min(max_pairs, len(rows)))]


def _suite_queries(
    body: TestSuiteRunRequest,
    count: int | None = None,
    route_pool_indexes: set[int] | None = None,
) -> list[dict]:
    if _dataset_family(body) == "ragbench":
        queries = _sample_ragbench_queries(
            body.dataset_name,
            body.dataset_split,
            count or body.query_count,
            body.seed,
            exclude_indexes=route_pool_indexes,
        )
        if not queries:
            raise RuntimeError("RAGBench 질문을 찾지 못했습니다. dataset 준비 상태를 확인하세요.")
        return queries

    queries = _sample_queries(
        body.dataset_name,
        count or body.query_count,
        body.seed,
        body.include_smoke,
    )
    if not queries:
        raise RuntimeError("LongBench 질문을 찾지 못했습니다. 데이터셋 준비 상태를 확인하세요.")
    return queries


def _mixed_queries(queries: list[dict], seed: int) -> list[dict]:
    rng = random.Random(seed)
    versions = ["V1", "V2", "V3"]
    scopes = ["A", "B"]
    mixed = []
    for item in queries:
        mixed.append({
            **item,
            "user_scope": rng.choice(scopes),
            "requested_version": rng.choice(versions),
        })
    return mixed


def _run_answer_items(
    source_id: str,
    queries: list[dict],
    user_scope: str,
    requested_version: str,
    route_threshold: float,
    cache_threshold: float,
    llm_provider: str | None,
    model: str | None,
    use_reranker: bool,
    rerank_candidates: int,
    rerank_model: str,
    progress: _SuiteProgress | None = None,
    progress_label: str = "A 실행",
) -> list[dict]:
    results = []
    for item in queries:
        scope = item.get("user_scope", user_scope)
        version = item.get("requested_version", requested_version)
        result = run_answer_cache_query(
            thread_id=source_id,
            query=item["query"],
            user_scope=scope,
            requested_version=version,
            model=model,
            llm_provider=llm_provider,
            route_threshold=route_threshold,
            cache_threshold=cache_threshold,
            use_reranker=use_reranker,
            rerank_candidates=rerank_candidates,
            rerank_model=rerank_model,
        )
        result["query_id"] = item["query_id"]
        result["dataset"] = item["dataset"]
        result["index"] = item["index"]
        results.append(result)
        if progress:
            progress.advance(progress_label)
    return results


def _run_context_items(
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
    progress: _SuiteProgress | None = None,
    progress_label: str = "B 실행",
) -> list[dict]:
    results = []
    for item in queries:
        scope = item.get("user_scope", user_scope)
        version = item.get("requested_version", requested_version)
        result = run_context_cache_query(
            source_id=source_id,
            query=item["query"],
            user_scope=scope,
            requested_version=version,
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
        if progress:
            progress.advance(progress_label)
    return results


def _run_no_cache_query(
    source_id: str,
    query: str,
    user_scope: str,
    requested_version: str,
    model: str | None,
    llm_provider: str | None,
    use_reranker: bool,
    rerank_candidates: int,
    rerank_model: str,
) -> dict:
    start = _timer()
    embedding_start = _timer()
    query_embedding = _embed(query)
    log = {
        "mode": "no_cache_rag",
        "thread_id": source_id,
        "query": query,
        "user_scope": user_scope,
        "requested_version": requested_version,
        "cache_hit": False,
        "validation_passed": False,
        "full_retrieval": True,
        "llm_provider": get_dp3_llm_provider(llm_provider),
        "llm_model": model,
        "llm_mocked": is_mock_llm(llm_provider),
        "llm_call_count": 0,
        "decision_reason": "no_cache_full_rag",
    }
    _set_timing(log, "embedding_ms", _elapsed_ms(embedding_start))

    retrieval_timing = {}
    sources = _retrieve_context_units(
        source_id,
        query,
        query_embedding,
        user_scope,
        requested_version,
        top_k=TOP_K_SOURCES,
        timing=retrieval_timing,
        use_reranker=use_reranker,
        rerank_candidates=rerank_candidates,
        rerank_model=rerank_model,
    )
    for key, value in retrieval_timing.items():
        if key.endswith("_ms"):
            _set_timing(log, f"full_retrieval_{key}", value)

    prompt_start = _timer()
    prompt = _build_prompt(query, sources)
    _set_timing(log, "prompt_build_ms", _elapsed_ms(prompt_start))

    llm_start = _timer()
    log["answer"] = get_dp3_answer(prompt, model, llm_provider)
    _set_timing(log, "llm_ms", _elapsed_ms(llm_start))
    log["llm_call_count"] = 1
    log["fallback_source_count"] = len(sources)
    _set_total_ms(log, start)
    return log


def _run_no_cache_items(
    source_id: str,
    queries: list[dict],
    user_scope: str,
    requested_version: str,
    llm_provider: str | None,
    model: str | None,
    use_reranker: bool,
    rerank_candidates: int,
    rerank_model: str,
    progress: _SuiteProgress | None = None,
    progress_label: str = "No-cache 실행",
) -> list[dict]:
    results = []
    for item in queries:
        result = _run_no_cache_query(
            source_id,
            item["query"],
            item.get("user_scope", user_scope),
            item.get("requested_version", requested_version),
            model,
            llm_provider,
            use_reranker,
            rerank_candidates,
            rerank_model,
        )
        result["query_id"] = item["query_id"]
        result["dataset"] = item["dataset"]
        result["index"] = item["index"]
        results.append(result)
        if progress:
            progress.advance(progress_label)
    return results


def _warm_up_suite(
    source_id: str,
    queries: list[dict],
    body: TestSuiteRunRequest,
    include_no_cache: bool = False,
    llm_provider: str | None = None,
    model: str | None = None,
    progress: _SuiteProgress | None = None,
) -> None:
    warmup = queries[: max(0, body.warmup_count)]
    if not warmup:
        return
    warmup_llm_provider = body.llm_provider if llm_provider is None else llm_provider
    warmup_model = body.model if model is None else model
    if include_no_cache:
        _run_no_cache_items(
            source_id,
            warmup,
            body.user_scope,
            "V1",
            warmup_llm_provider,
            warmup_model,
            body.use_reranker,
            body.rerank_candidates,
            body.rerank_model,
            progress,
            "Warm-up No-cache",
        )
    _run_answer_items(
        source_id,
        warmup,
        body.user_scope,
        "V1",
        body.route_threshold,
        body.cache_threshold,
        warmup_llm_provider,
        warmup_model,
        body.use_reranker,
        body.rerank_candidates,
        body.rerank_model,
        progress,
        "Warm-up A안",
    )
    _run_context_items(
        source_id,
        warmup,
        body.user_scope,
        "V1",
        body.cache_threshold,
        warmup_llm_provider,
        warmup_model,
        body.use_reranker,
        body.rerank_candidates,
        body.rerank_model,
        progress,
        "Warm-up B안",
    )
    clear_answer_cache_for_source(source_id)
    clear_context_cache_for_source(source_id)


def _pass_result(pass_name: str, mode: str, results: list[dict]) -> dict:
    if mode == "A":
        summary = _summarize_a_detailed(results)
    elif mode == "B":
        summary = _summarize_b_detailed(results)
    else:
        summary = _summarize_no_cache(results)
    return {"pass": pass_name, "mode": mode, "summary": summary}


def _run_cache_test(body: TestSuiteRunRequest, progress: _SuiteProgress | None = None) -> dict:
    if progress:
        warm_count = min(body.warmup_count, body.query_count)
        progress.reset(body.query_count * 5 + warm_count * 2 + 1, "Dataset/EU 준비")
        progress.step("Dataset/EU 준비")
    prepared = _prepare_suite_source(body)
    if progress:
        progress.advance("Dataset/EU 준비 완료")
    source_id = prepared["source_id"]
    route_pool = _seed_suite_route_pool(body)
    prepared["route_pool"] = route_pool
    queries = _suite_queries(body, route_pool_indexes=set(route_pool.get("seeded_indexes", [])))
    llm_provider = "mock"
    model = None

    _warm_up_suite(source_id, queries, body, llm_provider=llm_provider, model=model, progress=progress)
    no_cache = _run_no_cache_items(
        source_id,
        queries,
        body.user_scope,
        "V1",
        llm_provider,
        model,
        body.use_reranker,
        body.rerank_candidates,
        body.rerank_model,
        progress,
        "No-cache V1 실행",
    )
    clear_answer_cache_for_source(source_id)
    a_first = _run_answer_items(
        source_id,
        queries,
        body.user_scope,
        "V1",
        body.route_threshold,
        body.cache_threshold,
        llm_provider,
        model,
        body.use_reranker,
        body.rerank_candidates,
        body.rerank_model,
        progress,
        "A안 V1 첫 실행",
    )
    a_repeat = _run_answer_items(
        source_id,
        queries,
        body.user_scope,
        "V1",
        body.route_threshold,
        body.cache_threshold,
        llm_provider,
        model,
        body.use_reranker,
        body.rerank_candidates,
        body.rerank_model,
        progress,
        "A안 V1 반복",
    )

    clear_context_cache_for_source(source_id)
    b_first = _run_context_items(
        source_id,
        queries,
        body.user_scope,
        "V1",
        body.cache_threshold,
        llm_provider,
        model,
        body.use_reranker,
        body.rerank_candidates,
        body.rerank_model,
        progress,
        "B안 V1 첫 실행",
    )
    b_repeat = _run_context_items(
        source_id,
        queries,
        body.user_scope,
        "V1",
        body.cache_threshold,
        llm_provider,
        model,
        body.use_reranker,
        body.rerank_candidates,
        body.rerank_model,
        progress,
        "B안 V1 반복",
    )

    return {
        "test_case": "cache",
        "prepared": prepared,
        "source_id": source_id,
        "query_count": len(queries),
        "sampled_datasets": _query_dataset_counts(queries),
        "llm_provider": llm_provider,
        "no_cache": {"passes": [_pass_result("v1", "no_cache", no_cache)]},
        "a": {
            "passes": [
                _pass_result("v1_first", "A", a_first),
                _pass_result("v1_repeat", "A", a_repeat),
            ]
        },
        "b": {
            "passes": [
                _pass_result("v1_first", "B", b_first),
                _pass_result("v1_repeat", "B", b_repeat),
            ]
        },
    }


def _run_mixed_timing_test(body: TestSuiteRunRequest, progress: _SuiteProgress | None = None) -> dict:
    use_tc2_assets = _dataset_family(body) == "ragbench" and body.dataset_name.strip().lower() == "emanual"
    if use_tc2_assets:
        base_queries = _tc2_asset_queries(body)
        prepare_count = max(body.num_examples, max((int(q["index"]) for q in base_queries), default=0) + 1)
    else:
        base_queries = _suite_queries(body)
        prepare_count = body.num_examples

    if progress:
        warm_count = min(body.warmup_count, len(base_queries))
        progress.reset(len(base_queries) * 2 + warm_count * 2 + 1, "Dataset/EU 준비")
        progress.step("Dataset/EU 준비")
    prepared = _prepare_suite_source(body, prepare_count)
    if progress:
        progress.advance("Dataset/EU 준비 완료")
    source_id = prepared["source_id"]
    query_indexes = {int(q["index"]) for q in base_queries if "index" in q}
    route_pool = _seed_suite_route_pool(body, exclude_indexes=None if use_tc2_assets else query_indexes)
    prepared["route_pool"] = route_pool
    queries = _mixed_queries(base_queries, body.seed + 101)

    _warm_up_suite(source_id, queries, body, progress=progress)
    clear_answer_cache_for_source(source_id)
    a_results = _run_answer_items(
        source_id,
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
        progress,
        "A안 혼합 실행",
    )

    clear_context_cache_for_source(source_id)
    b_results = _run_context_items(
        source_id,
        queries,
        body.user_scope,
        "V1",
        body.cache_threshold,
        body.llm_provider,
        body.model,
        body.use_reranker,
        body.rerank_candidates,
        body.rerank_model,
        progress,
        "B안 혼합 실행",
    )

    return {
        "test_case": "mixed_timing",
        "prepared": prepared,
        "source_id": source_id,
        "query_count": len(queries),
        "sampled_datasets": _query_dataset_counts(queries),
        "query_asset": "tc2_query_sets" if use_tc2_assets else "random",
        "llm_provider": get_dp3_llm_provider(body.llm_provider),
        "llm_model": body.model,
        "a": {"passes": [_pass_result("mixed", "A", a_results)]},
        "b": {"passes": [_pass_result("mixed", "B", b_results)]},
    }


def _run_scalability_test(body: TestSuiteRunRequest, progress: _SuiteProgress | None = None) -> dict:
    scale_results = []
    max_scale = max(1, min(body.max_scale, 5))
    llm_provider = body.llm_provider or "mock"
    model = body.model if llm_provider != "mock" else None
    if progress:
        per_scale = body.query_count * 3 + min(body.warmup_count, body.query_count) * 3 + 1
        progress.reset(per_scale * max_scale, "TC3 준비")

    for scale in range(1, max_scale + 1):
        if progress:
            progress.step(f"Scale {scale}x LongBench/EU 준비")
        prepared = _prepare_suite_source(body, body.num_examples * scale)
        if progress:
            progress.advance(f"Scale {scale}x LongBench/EU 준비 완료")
        source_id = prepared["source_id"]
        route_pool = _seed_suite_route_pool(body)
        prepared["route_pool"] = route_pool
        queries = _suite_queries(body, route_pool_indexes=set(route_pool.get("seeded_indexes", [])))
        _warm_up_suite(
            source_id,
            queries,
            body,
            include_no_cache=True,
            llm_provider=llm_provider,
            model=model,
            progress=progress,
        )

        no_cache = _run_no_cache_items(
            source_id,
            queries,
            body.user_scope,
            "V1",
            llm_provider,
            model,
            body.use_reranker,
            body.rerank_candidates,
            body.rerank_model,
            progress,
            f"Scale {scale}x No-cache",
        )

        clear_answer_cache_for_source(source_id)
        a_results = _run_answer_items(
            source_id,
            queries,
            body.user_scope,
            "V1",
            body.route_threshold,
            body.cache_threshold,
            llm_provider,
            model,
            body.use_reranker,
            body.rerank_candidates,
            body.rerank_model,
            progress,
            f"Scale {scale}x A안",
        )

        clear_context_cache_for_source(source_id)
        b_results = _run_context_items(
            source_id,
            queries,
            body.user_scope,
            "V1",
            body.cache_threshold,
            llm_provider,
            model,
            body.use_reranker,
            body.rerank_candidates,
            body.rerank_model,
            progress,
            f"Scale {scale}x B안",
        )

        scale_results.append({
            "scale": scale,
            "num_examples": prepared["num_examples"],
            "source_id": source_id,
            "prepared": prepared,
            "no_cache": _pass_result("scale", "no_cache", no_cache),
            "a": _pass_result("scale", "A", a_results),
            "b": _pass_result("scale", "B", b_results),
        })

    return {
        "test_case": "scalability",
        "dataset": body.dataset_name,
        "dataset_family": _dataset_family(body),
        "base_num_examples": body.num_examples,
        "query_count": body.query_count,
        "max_scale": max_scale,
        "llm_provider": get_dp3_llm_provider(llm_provider),
        "llm_model": model,
        "scales": scale_results,
    }


def _tc4_pair_query(pair: dict, side: str) -> dict:
    item = pair[side]
    return {
        "query_id": item["query_id"],
        "dataset": pair["dataset"],
        "index": item["index"],
        "query": item["query"],
        "reference_answer": item.get("reference_answer", ""),
        "pair_id": pair["pair_id"],
        "pair_side": side,
    }


def _answers_equal(left: dict, right: dict) -> bool:
    return " ".join(str(left.get("answer", "")).split()) == " ".join(str(right.get("answer", "")).split())


def _run_similar_pair_quality_test(body: TestSuiteRunRequest, progress: _SuiteProgress | None = None) -> dict:
    if _dataset_family(body) != "ragbench" or body.dataset_name.strip().lower() != "emanual":
        raise ValueError("TC4 requires dataset_family=ragbench and dataset_name=emanual.")

    pairs = _tc4_asset_pairs(body)
    if not pairs:
        raise RuntimeError("TC4 pair asset is empty. Run the RAGBench query asset builder first.")
    prepare_count = max(body.num_examples, max(max(p["left"]["index"], p["right"]["index"]) for p in pairs) + 1)

    if progress:
        progress.reset(len(pairs) * 4 + 1, "TC4 Dataset/EU 준비")
        progress.step("TC4 Dataset/EU 준비")
    prepared = _prepare_suite_source(body, prepare_count)
    if progress:
        progress.advance("TC4 Dataset/EU 준비 완료")
    source_id = prepared["source_id"]
    answer_cache_module._READY_SOURCE_IDS.add(source_id)

    a_left_results = []
    a_right_results = []
    b_left_results = []
    b_right_results = []
    pair_results = []
    for pair in pairs:
        left = _tc4_pair_query(pair, "left")
        right = _tc4_pair_query(pair, "right")

        clear_answer_cache_for_source(source_id)
        previous_route_cache = answer_cache_module._ROUTE_CACHE
        answer_cache_module._ROUTE_CACHE = [{
            "route_id": f"tc4_shared:{pair['pair_id']}",
            "route_type": "tc4_shared_pair",
            "route_question": left["query"],
            "embedding": _embed(left["query"]),
        }]
        try:
            a_left = _run_answer_items(
                source_id,
                [left],
                body.user_scope,
                "V1",
                body.route_threshold,
                body.cache_threshold,
                body.llm_provider,
                body.model,
                body.use_reranker,
                body.rerank_candidates,
                body.rerank_model,
                progress,
                "TC4 A left",
            )[0]
            a_right = _run_answer_items(
                source_id,
                [right],
                body.user_scope,
                "V1",
                body.route_threshold,
                body.cache_threshold,
                body.llm_provider,
                body.model,
                body.use_reranker,
                body.rerank_candidates,
                body.rerank_model,
                progress,
                "TC4 A right",
            )[0]
        finally:
            answer_cache_module._ROUTE_CACHE = previous_route_cache

        clear_context_cache_for_source(source_id)
        b_left = _run_context_items(
            source_id,
            [left],
            body.user_scope,
            "V1",
            body.cache_threshold,
            body.llm_provider,
            body.model,
            body.use_reranker,
            body.rerank_candidates,
            body.rerank_model,
            progress,
            "TC4 B left",
        )[0]
        b_right = _run_context_items(
            source_id,
            [right],
            body.user_scope,
            "V1",
            body.cache_threshold,
            body.llm_provider,
            body.model,
            body.use_reranker,
            body.rerank_candidates,
            body.rerank_model,
            progress,
            "TC4 B right",
        )[0]

        a_left_results.append(a_left)
        a_right_results.append(a_right)
        b_left_results.append(b_left)
        b_right_results.append(b_right)
        pair_results.append({
            "pair_id": pair["pair_id"],
            "similarity": pair["similarity"],
            "answer_jaccard": pair["answer_jaccard"],
            "left_query": left["query"],
            "right_query": right["query"],
            "a_right_cache_hit": bool(a_right.get("cache_hit")),
            "b_right_cache_hit": bool(b_right.get("cache_hit")),
            "a_answers_equal": _answers_equal(a_left, a_right),
            "b_answers_equal": _answers_equal(b_left, b_right),
            "a_right_decision": a_right.get("decision_reason"),
            "b_right_decision": b_right.get("decision_reason"),
            "left_reference_answer": left.get("reference_answer", ""),
            "right_reference_answer": right.get("reference_answer", ""),
            "a_left_answer": a_left.get("answer", ""),
            "a_right_answer": a_right.get("answer", ""),
            "b_left_answer": b_left.get("answer", ""),
            "b_right_answer": b_right.get("answer", ""),
        })

    return {
        "test_case": "similar_pair_quality",
        "prepared": prepared,
        "source_id": source_id,
        "query_asset": "tc4_query_pairs",
        "pair_count": len(pairs),
        "query_count": len(pairs) * 2,
        "sampled_datasets": _query_dataset_counts([_tc4_pair_query(p, "left") for p in pairs]),
        "llm_provider": get_dp3_llm_provider(body.llm_provider),
        "llm_model": body.model,
        "a": {
            "passes": [
                _pass_result("left_seed", "A", a_left_results),
                _pass_result("right_probe", "A", a_right_results),
            ]
        },
        "b": {
            "passes": [
                _pass_result("left_seed", "B", b_left_results),
                _pass_result("right_probe", "B", b_right_results),
            ]
        },
        "pairs": pair_results,
    }


def _run_test_suite_internal(body: TestSuiteRunRequest, progress: _SuiteProgress | None = None):
    init_db()
    normalized = body.test_case.strip().lower()
    if normalized == "cache":
        return _run_cache_test(body, progress)
    if normalized in {"mixed", "mixed_timing", "timing"}:
        return _run_mixed_timing_test(body, progress)
    if normalized in {"scalability", "scale"}:
        return _run_scalability_test(body, progress)
    if normalized in {"similar_pair_quality", "tc4", "pair_quality"}:
        return _run_similar_pair_quality_test(body, progress)
    raise ValueError(f"Unknown DP3 test_case: {body.test_case}")


@router.post("/test-suite/run")
def run_test_suite(body: TestSuiteRunRequest):
    return _run_test_suite_internal(body)


def _run_suite_job(job_id: str, body: TestSuiteRunRequest) -> None:
    _update_suite_job(job_id, status="running", current_step="테스트 시작", started_at=time.time())
    try:
        result = _run_test_suite_internal(body, _SuiteProgress(job_id))
        _update_suite_job(
            job_id,
            status="completed",
            current_step="완료",
            completed=_SUITE_JOBS.get(job_id, {}).get("total", 1),
            result=result,
            finished_at=time.time(),
        )
    except Exception as exc:
        _update_suite_job(
            job_id,
            status="failed",
            current_step="오류",
            error=f"{type(exc).__name__}: {exc}",
            finished_at=time.time(),
        )


@router.post("/test-suite/start")
def start_test_suite(body: TestSuiteRunRequest):
    job_id = str(uuid.uuid4())
    with _SUITE_JOB_LOCK:
        _SUITE_JOBS[job_id] = {
            "job_id": job_id,
            "status": "queued",
            "current_step": "대기 중",
            "completed": 0,
            "total": 1,
            "progress_percent": 0.0,
            "created_at": time.time(),
        }
    thread = threading.Thread(target=_run_suite_job, args=(job_id, body), daemon=True)
    thread.start()
    return _SUITE_JOBS[job_id]


@router.get("/test-suite/jobs/{job_id}")
def get_test_suite_job(job_id: str):
    with _SUITE_JOB_LOCK:
        job = _SUITE_JOBS.get(job_id)
        if not job:
            raise ValueError(f"Unknown DP3 test suite job: {job_id}")
        return dict(job)
