import json
import random
import re
import threading
import time
import uuid
from collections import Counter
from pathlib import Path
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:
    class BaseCallbackHandler:
        pass

import backend.cache.answer_cache as answer_cache_module
from backend.cache.answer_cache import (
    TOP_K_SOURCES,
    _build_prompt,
    _embed,
    _elapsed_ms,
    _embedding_to_json,
    _retrieve_context_units,
    _set_llm_timings,
    _set_timing,
    _set_total_ms,
    _store_log,
    _timer,
    clear_answer_cache_for_source,
    run_answer_cache_query,
    setup_answer_cache_poc,
)
from backend.cache.cache_llm import get_dp3_answer_with_metadata, get_dp3_llm_provider, is_mock_llm
from backend.cache.context_cache import clear_context_cache_for_source, run_context_cache_query
from backend.db.database import init_db
from build_dp3_ragbench_query_assets import ensure_query_assets
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
DP3_RUN_LOG_DIR = Path(__file__).resolve().parents[2] / "data" / "dp3_run_logs"
DP3_RAGAS_INPUT_DIR = Path(__file__).resolve().parents[2] / "data" / "dp3_ragas_inputs"


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
    row_limit: Optional[int] = None
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
    route_pool_mode: str = "sampled"
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


class RagasRunRequest(BaseModel):
    input_path: str
    run_official: bool = False
    max_rows: int = 2
    model: Optional[str] = None


def _safe_filename_part(value: object, fallback: str = "run") -> str:
    text = str(value or fallback).strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "-", text)
    return text.strip("-") or fallback


def _save_test_suite_run(body: TestSuiteRunRequest, result: dict, job_id: str | None = None) -> dict:
    DP3_RUN_LOG_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    test_case = _safe_filename_part(result.get("test_case") or body.test_case)
    dataset = _safe_filename_part(result.get("dataset") or body.dataset_name)
    reranker = "rerank-on" if body.use_reranker else "rerank-off"
    device = "na"
    if body.use_reranker:
        marker = "||device="
        device = body.rerank_model.split(marker, 1)[1] if marker in body.rerank_model else "auto"
        device = _safe_filename_part(device, "auto")
    name_parts = [timestamp, test_case, dataset, reranker, device]
    if job_id:
        name_parts.append(job_id[:8])
    _persist_run_specific_ragas_input(result, name_parts)
    path = DP3_RUN_LOG_DIR / ("_".join(name_parts) + ".json")
    payload = {
        "saved_at": timestamp,
        "job_id": job_id,
        "request": body.model_dump(),
        "result": result,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"path": str(path), "file": path.name}


def _persist_run_specific_ragas_input(result: dict, name_parts: list[str]) -> None:
    source_value = result.get("ragas_input_path")
    if not source_value:
        return
    source = Path(source_value)
    if not source.exists():
        return

    DP3_RAGAS_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    destination = DP3_RAGAS_INPUT_DIR / ("_".join([*name_parts, "ragas-input"]) + ".jsonl")
    content = source.read_text(encoding="utf-8")
    destination.write_text(content, encoding="utf-8")
    row_count = sum(1 for line in content.splitlines() if line.strip())
    result["ragas_input_legacy_path"] = str(source)
    result["ragas_input_path"] = str(destination)
    result["ragas_input_export"] = {
        "path": str(destination),
        "file": destination.name,
        "row_count": row_count,
        "legacy_path": str(source),
    }


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
            max_index=body.row_limit,
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
            "llm_wall_ms",
            "llm_throttle_wait_ms",
            "llm_retry_wait_ms",
            "llm_api_reported_queue_ms",
            "llm_api_reported_total_ms",
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
    "llm_wall_ms",
    "llm_throttle_wait_ms",
    "llm_retry_wait_ms",
    "llm_api_reported_queue_ms",
    "llm_api_reported_total_ms",
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
    "llm_wall_ms",
    "llm_throttle_wait_ms",
    "llm_retry_wait_ms",
    "llm_api_reported_queue_ms",
    "llm_api_reported_total_ms",
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
    "llm_wall_ms",
    "llm_throttle_wait_ms",
    "llm_retry_wait_ms",
    "llm_api_reported_queue_ms",
    "llm_api_reported_total_ms",
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


def _reranker_device_stats(results: list[dict]) -> dict:
    enabled = [row for row in results if row.get("reranker_enabled")]
    requested = Counter(
        row.get("reranker_requested_device") or row.get("reranker_device") or "unknown"
        for row in enabled
    )
    resolved = Counter(row.get("reranker_resolved_device") or "unknown" for row in enabled)
    models = Counter(row.get("reranker_model") or "unknown" for row in enabled)
    return {
        "enabled_count": len(enabled),
        "requested_devices": dict(requested),
        "resolved_devices": dict(resolved),
        "models": dict(models),
    }


def _estimated_total_with_llm_stats(results: list[dict]) -> dict:
    values = []
    mock_rows = 0
    actual_rows = 0
    for row in results:
        total = (row.get("timings_ms") or {}).get("total_ms")
        if total is None:
            total = row.get("total_ms")
        if total is None:
            continue
        llm_calls = int(row.get("llm_call_count", 0))
        if row.get("llm_mocked"):
            mock_rows += 1
            values.append(float(total) + llm_calls * MOCK_LLM_ESTIMATE_MS)
        else:
            actual_rows += 1
            values.append(float(total))
    if not values:
        return {}
    return {
        "avg": round(sum(values) / len(values), 3),
        "min": round(min(values), 3),
        "max": round(max(values), 3),
        "count": len(values),
        "mock_llm_estimate_ms": MOCK_LLM_ESTIMATE_MS,
        "mock_rows": mock_rows,
        "actual_llm_rows": actual_rows,
        "basis": "mock rows: total_ms + llm_call_count * 700ms; actual LLM rows: total_ms",
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
        "reranker_devices": _reranker_device_stats(results),
        "timing_stats_ms": _timing_stats(results, A_TIMING_KEYS),
        "estimated_total_with_llm_ms": _estimated_total_with_llm_stats(results),
    }


def _summarize_b_detailed(results: list[dict]) -> dict:
    total = len(results)
    full_valid = sum(
        1
        for row in results
        if row.get("decision_reason") == "context_cache_hit_all_valid"
    )
    partial_valid = sum(
        1
        for row in results
        if row.get("decision_reason") == "context_cache_partial_invalid_delta_rebuilt"
    )
    validation_failed = sum(
        1
        for row in results
        if row.get("decision_reason") in {
            "context_cache_invalid_ratio_full_fallback",
            "context_cache_delta_insufficient_full_fallback",
        }
    )
    cache_hits = full_valid + partial_valid + validation_failed
    context_reused = full_valid + partial_valid
    full_retrievals = sum(1 for row in results if row.get("full_retrieval"))
    delta_retrievals = sum(1 for row in results if int(row.get("delta_retrieval_count", 0)) > 0)
    return {
        "total": total,
        "context_cache_hits": cache_hits,
        "cache_hit_ratio": _ratio(cache_hits, total),
        "context_cache_reused": context_reused,
        "context_cache_reuse_ratio": _ratio(context_reused, total),
        "validation_full_passed": full_valid,
        "validation_full_pass_ratio": _ratio(full_valid, total),
        "validation_partial_passed": partial_valid,
        "validation_partial_pass_ratio": _ratio(partial_valid, total),
        "validation_failed": validation_failed,
        "validation_failed_ratio": _ratio(validation_failed, total),
        "full_retrievals": full_retrievals,
        "full_retrieval_ratio": _ratio(full_retrievals, total),
        "delta_retrievals": delta_retrievals,
        "delta_retrieval_ratio": _ratio(delta_retrievals, total),
        "llm_calls": sum(int(row.get("llm_call_count", 0)) for row in results),
        "decision_reasons": dict(Counter(row.get("decision_reason", "unknown") for row in results)),
        "reranker_devices": _reranker_device_stats(results),
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
        "reranker_devices": _reranker_device_stats(results),
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


def _seed_suite_route_pool(
    body: TestSuiteRunRequest,
    exclude_indexes: set[int] | None = None,
    row_limit: int | None = None,
) -> dict:
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
            max_index=row_limit,
        )
    return seed_question_pool(
        dataset=body.dataset_name,
        sample_rate=sample_rate,
        min_per_dataset=min_count,
        seed=body.pool_seed,
        reset=True,
        include_smoke=body.include_smoke,
    )


def _route_pool_mode(body: TestSuiteRunRequest, default: str = "sampled") -> str:
    value = (body.route_pool_mode or default).strip().lower()
    aliases = {
        "same": "sampled",
        "normal": "sampled",
        "default": "sampled",
        "include": "include_similar",
        "include_set": "include_similar",
        "similar": "similar_only",
        "set_only": "similar_only",
        "pair_only": "similar_only",
    }
    return aliases.get(value, value)


def _seed_route_pool_from_queries(
    queries: list[dict],
    reset: bool,
    route_type: str,
) -> dict:
    from backend.cache.answer_cache import _clear_runtime_caches, init_dp3_cache_schema
    from backend.db.database import get_conn

    init_db()
    init_dp3_cache_schema()
    inserted = 0
    indexes = []
    seen = set()
    with get_conn() as conn:
        if reset:
            conn.execute("DELETE FROM dp3_answerable_question_pool")
        for item in queries:
            query = item.get("query")
            if not query:
                continue
            route_id = item.get("query_id") or f"{route_type}:{inserted}"
            if route_id in seen:
                continue
            seen.add(route_id)
            conn.execute(
                """INSERT OR REPLACE INTO dp3_answerable_question_pool
                   (route_id, question_text, route_type, embedding_json)
                   VALUES (?, ?, ?, ?)""",
                (
                    route_id,
                    query,
                    route_type,
                    _embedding_to_json(_embed(query)),
                ),
            )
            inserted += 1
            if "index" in item:
                indexes.append(int(item["index"]))
    _clear_runtime_caches()
    return {
        "route_pool_mode": route_type,
        "seeded_questions": inserted,
        "seeded_indexes": indexes,
    }


def _sample_ragbench_queries(
    dataset_name: str,
    split: str,
    count: int,
    seed: int,
    exclude_indexes: set[int] | None = None,
    row_limit: int | None = None,
) -> list[dict]:
    exclude_indexes = exclude_indexes or set()
    queries = [
        item
        for item in iter_ragbench_queries(dataset_name, split)
        if int(item["index"]) not in exclude_indexes
        and (row_limit is None or int(item["index"]) < row_limit)
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
    return ensure_query_assets(
        subset=dataset,
        split=split,
        seed=body.seed,
        cache_threshold=body.cache_threshold,
        tc4_min_similarity=body.cache_threshold,
    )


def _seed_tc3_route_pool(
    body: TestSuiteRunRequest,
    base_queries: list[dict],
) -> dict:
    row_limit = max(body.num_examples, max((int(q["index"]) for q in base_queries), default=-1) + 1)
    result = _seed_suite_route_pool(body, row_limit=row_limit)
    result["route_pool_mode"] = "sampled"
    return result


def _read_jsonl(path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _tc3_asset_queries(body: TestSuiteRunRequest) -> list[dict]:
    assets = _ensure_emanual_assets(body)
    rows = _read_jsonl(assets.get("tc3_path") or assets["tc2_path"])
    rng = random.Random(body.seed)
    role_order = {"same": 0, "similar": 1, "paraphrase": 1, "near_miss": 2, "random": 3}
    grouped = {}
    for row in rows:
        grouped.setdefault(row["group_id"], []).append(row)
    groups = list(grouped.values())
    rng.shuffle(groups)
    flattened = [
        row
        for group in groups
        for row in sorted(group, key=lambda r: (role_order.get(r["role"], 99), r["query_id"]))
    ]
    count = min(body.query_count, len(flattened)) if body.query_count > 0 else len(flattened)
    return flattened[:count]


def _tc4_route_queries(pairs: list[dict]) -> list[dict]:
    queries = []
    for pair in pairs:
        left = _tc4_pair_query(pair, "left")
        left["query_id"] = f"tc4-route:{pair['pair_id']}"
        queries.append(left)
    return queries


def _seed_tc4_route_pool(body: TestSuiteRunRequest, pairs: list[dict]) -> dict:
    query_indexes = [
        int(item["index"])
        for pair in pairs
        for item in (pair["left"], pair["right"])
        if "index" in item
    ]
    row_limit = max(body.num_examples, max(query_indexes, default=-1) + 1)
    result = _seed_suite_route_pool(body, row_limit=row_limit)
    result["route_pool_mode"] = "sampled"
    return result


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
    row_limit: int | None = None,
) -> list[dict]:
    query_count = count or body.query_count
    if _dataset_family(body) == "ragbench":
        queries = _sample_ragbench_queries(
            body.dataset_name,
            body.dataset_split,
            query_count,
            body.seed,
            row_limit=row_limit,
        )
        if not queries:
            raise RuntimeError("RAGBench 질문을 찾지 못했습니다. dataset 준비 상태를 확인하세요.")
        return queries[:query_count]

    queries = _sample_queries(
        body.dataset_name,
        query_count,
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
    if retrieval_timing.get("reranker_enabled"):
        log["reranker_model"] = retrieval_timing.get("reranker_model")
        log["reranker_requested_device"] = retrieval_timing.get("reranker_requested_device")
        log["reranker_resolved_device"] = retrieval_timing.get("reranker_resolved_device")
        log["full_retrieval_reranker_requested_device"] = retrieval_timing.get("reranker_requested_device")
        log["full_retrieval_reranker_resolved_device"] = retrieval_timing.get("reranker_resolved_device")

    prompt_start = _timer()
    prompt = _build_prompt(query, sources)
    _set_timing(log, "prompt_build_ms", _elapsed_ms(prompt_start))

    llm_start = _timer()
    llm_result = get_dp3_answer_with_metadata(prompt, model, llm_provider)
    log["answer"] = llm_result["answer"]
    log["llm_usage"] = llm_result.get("usage", {})
    log["llm_prompt_fit"] = llm_result.get("prompt_fit", {})
    log["llm_estimated_tokens"] = llm_result.get("estimated_tokens")
    _set_llm_timings(log, llm_result, _elapsed_ms(llm_start))
    log["llm_call_count"] = 1
    log["fallback_source_count"] = len(sources)
    _set_total_ms(log, start)
    _store_log(source_id, query, log)
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
    row_limit = prepared.get("num_examples", body.num_examples)
    route_pool = _seed_suite_route_pool(body, row_limit=row_limit)
    prepared["route_pool"] = route_pool
    queries = _suite_queries(
        body,
        row_limit=row_limit,
    )
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
    clear_answer_cache_for_source(source_id, clear_logs=False)
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
    use_tc3_assets = _dataset_family(body) == "ragbench" and body.dataset_name.strip().lower() == "emanual"
    if use_tc3_assets:
        base_queries = _tc3_asset_queries(body)
        prepare_count = max(body.num_examples, max((int(q["index"]) for q in base_queries), default=0) + 1)
    else:
        base_queries = _suite_queries(body)
        prepare_count = body.num_examples

    if progress:
        warm_count = min(body.warmup_count, len(base_queries))
        progress.reset(len(base_queries) * 5 + warm_count * 2 + 1, "Dataset/EU prepare")
        progress.step("Dataset/EU prepare")
    prepared = _prepare_suite_source(body, prepare_count)
    if progress:
        progress.advance("Dataset/EU prepared")
    source_id = prepared["source_id"]
    route_pool = (
        _seed_tc3_route_pool(body, base_queries)
        if use_tc3_assets
        else _seed_suite_route_pool(body)
    )
    prepared["route_pool"] = route_pool
    queries = _mixed_queries(base_queries, body.seed + 101)
    answer_cache_module._READY_SOURCE_IDS.add(source_id)

    _warm_up_suite(source_id, queries, body, progress=progress)
    no_cache = _run_no_cache_items(
        source_id,
        queries,
        body.user_scope,
        "V1",
        body.llm_provider,
        body.model,
        body.use_reranker,
        body.rerank_candidates,
        body.rerank_model,
        progress,
        "No-cache mixed run",
    )
    clear_answer_cache_for_source(source_id, clear_logs=False)
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
        "A mixed first run",
    )
    a_repeat = _run_answer_items(
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
        "A mixed repeat run",
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
        "B mixed first run",
    )
    b_repeat = _run_context_items(
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
        "B mixed repeat run",
    )

    return {
        "test_case": "mixed_timing",
        "prepared": prepared,
        "source_id": source_id,
        "query_count": len(queries),
        "sampled_datasets": _query_dataset_counts(queries),
        "query_asset": "tc3_query_sets" if use_tc3_assets else "random",
        "llm_provider": get_dp3_llm_provider(body.llm_provider),
        "llm_model": body.model,
        "no_cache": {"passes": [_pass_result("mixed", "no_cache", no_cache)]},
        "a": {
            "passes": [
                _pass_result("mixed", "A", a_results),
                _pass_result("mixed_repeat", "A", a_repeat),
            ]
        },
        "b": {
            "passes": [
                _pass_result("mixed", "B", b_results),
                _pass_result("mixed_repeat", "B", b_repeat),
            ]
        },
    }

def _run_scalability_test(body: TestSuiteRunRequest, progress: _SuiteProgress | None = None) -> dict:
    scale_results = []
    scale_rows = [100, 200, 300]
    llm_provider = body.llm_provider or "mock"
    model = body.model if llm_provider != "mock" else None
    if progress:
        per_scale = body.query_count * 5 + min(body.warmup_count, body.query_count) * 3 + 1
        progress.reset(per_scale * len(scale_rows), "TC2 준비")

    for scale, row_count in enumerate(scale_rows, start=1):
        if progress:
            progress.step(f"Scale {scale} ({row_count} rows) Dataset/EU 준비")
        prepared = _prepare_suite_source(body, row_count)
        if progress:
            progress.advance(f"Scale {scale} ({row_count} rows) Dataset/EU 준비 완료")
        source_id = prepared["source_id"]
        route_pool = _seed_suite_route_pool(body, row_limit=row_count)
        prepared["route_pool"] = route_pool
        queries = _suite_queries(
            body,
            row_limit=row_count,
        )
        _warm_up_suite(
            source_id,
            queries,
            body,
            include_no_cache=True,
            llm_provider=llm_provider,
            model=model,
            progress=progress,
        )
        clear_answer_cache_for_source(clear_logs=False)
        clear_context_cache_for_source()

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

        clear_answer_cache_for_source(clear_logs=False)
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
            f"Scale {scale}x A안 첫 실행",
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
            f"Scale {scale}x A안 반복",
        )

        clear_context_cache_for_source()
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
            f"Scale {scale}x B안 첫 실행",
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
            f"Scale {scale}x B안 반복",
        )

        scale_results.append({
            "scale": scale,
            "row_count": row_count,
            "num_examples": prepared["num_examples"],
            "source_id": source_id,
            "prepared": prepared,
            "no_cache": _pass_result("scale", "no_cache", no_cache),
            "a_first": _pass_result("scale_first", "A", a_first),
            "a_repeat": _pass_result("scale_repeat", "A", a_repeat),
            "b_first": _pass_result("scale_first", "B", b_first),
            "b_repeat": _pass_result("scale_repeat", "B", b_repeat),
        })

    return {
        "test_case": "scalability",
        "dataset": body.dataset_name,
        "dataset_family": _dataset_family(body),
        "base_num_examples": scale_rows[0],
        "query_count": body.query_count,
        "max_scale": len(scale_rows),
        "scale_rows": scale_rows,
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


def _llm_usage_summary(result: dict) -> dict:
    usage = result.get("llm_usage") or {}
    prompt_fit = result.get("llm_prompt_fit") or {}
    timings = result.get("timings_ms") or {}
    return {
        "llm_calls": int(result.get("llm_call_count", 0)),
        "prompt_tokens": usage.get("prompt_tokens") or usage.get("estimated_prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens") or usage.get("estimated_total_tokens") or result.get("llm_estimated_tokens"),
        "estimated_tokens": result.get("llm_estimated_tokens"),
        "request_ms": timings.get("llm_ms"),
        "wall_ms": timings.get("llm_wall_ms"),
        "throttle_wait_ms": timings.get("llm_throttle_wait_ms"),
        "api_reported_queue_ms": timings.get("llm_api_reported_queue_ms"),
        "api_reported_total_ms": timings.get("llm_api_reported_total_ms"),
        "reported_queue_time_s": usage.get("queue_time"),
        "reported_total_time_s": usage.get("total_time"),
        "prompt_trimmed": bool(prompt_fit.get("trimmed", False)),
        "prompt_fit": prompt_fit,
    }


def _contexts_for_result(source_id: str, result: dict) -> list[str]:
    from backend.db.database import get_conn

    context_cache_id = result.get("context_cache_id") or result.get("context_cache_candidate_id")
    if context_cache_id:
        with get_conn() as conn:
            rows = conn.execute(
                """SELECT s.versioned_eu_id, v.text
                   FROM dp3_context_cache_sources s
                   JOIN dp3_versioned_evidence_units v
                     ON v.versioned_eu_id = s.versioned_eu_id
                    AND v.source_id = ?
                   WHERE s.context_cache_id=?
                   ORDER BY s.source_order""",
                (source_id, context_cache_id),
            ).fetchall()
            if rows:
                contexts = []
                seen = set()
                for row in rows:
                    text = row["text"]
                    if not text:
                        continue
                    key = row["versioned_eu_id"] or text
                    if key in seen:
                        continue
                    seen.add(key)
                    contexts.append(text)
                if contexts:
                    return contexts
            row = conn.execute(
                """SELECT context_pack_text
                   FROM dp3_context_cache_entries
                   WHERE context_cache_id=?""",
                (context_cache_id,),
            ).fetchone()
        if row and row["context_pack_text"]:
            return [row["context_pack_text"]]

    versioned_ids = []
    cache_id = result.get("cache_candidate_id") or result.get("cache_id")
    if cache_id:
        with get_conn() as conn:
            rows = conn.execute(
                """SELECT versioned_eu_id
                   FROM dp3_answer_cache_sources
                   WHERE cache_id=?
                   ORDER BY source_order""",
                (cache_id,),
            ).fetchall()
        versioned_ids.extend([row["versioned_eu_id"] for row in rows if row["versioned_eu_id"]])

    for source in result.get("fallback_sources", []) or []:
        if source.get("versioned_eu_id"):
            versioned_ids.append(source["versioned_eu_id"])

    if not versioned_ids:
        return []
    placeholders = ",".join("?" for _ in versioned_ids)
    with get_conn() as conn:
        rows = conn.execute(
            f"""SELECT versioned_eu_id, text
                FROM dp3_versioned_evidence_units
                WHERE source_id=? AND versioned_eu_id IN ({placeholders})""",
            [source_id, *versioned_ids],
        ).fetchall()
    by_id = {row["versioned_eu_id"]: row["text"] for row in rows}
    return [by_id[vid] for vid in versioned_ids if vid in by_id]


def _write_tc4_ragas_input(
    dataset_name: str,
    split: str,
    source_id: str,
    pair_results: list[dict],
) -> str:
    out_dir = RAGBENCH_DATA_DIR / dataset_name.strip().lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split.strip().lower()}_tc4_ragas_input.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for pair in pair_results:
            for mode in ("A", "B"):
                result = pair[f"{mode.lower()}_right_result"]
                row = {
                    "pair_id": pair["pair_id"],
                    "mode": mode,
                    "question": pair["right_query"],
                    "answer": result.get("answer", ""),
                    "contexts": pair.get(f"{mode.lower()}_right_contexts")
                    or _contexts_for_result(source_id, result),
                    "ground_truth": pair.get("right_reference_answer", ""),
                    "cache_hit": bool(result.get("cache_hit")),
                    "similarity": pair.get("similarity"),
                    "answer_jaccard": pair.get("answer_jaccard"),
                    "decision_reason": result.get("decision_reason"),
                    "route_id": result.get("embedding_route_id"),
                    "route_score": result.get("embedding_score"),
                    "cache_lookup_strategy": result.get("cache_lookup_strategy"),
                    "cache_candidate_id": result.get("cache_candidate_id"),
                    "cache_candidate_route_id": result.get("cache_candidate_route_id"),
                    "cache_similarity_score": result.get("cache_similarity_score"),
                    "llm_usage": pair.get(f"{mode.lower()}_right_usage"),
                }
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return str(path)


def _token_set(value: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9가-힣]+", str(value).lower()))


def _jaccard_text(left: str, right: str) -> float:
    a = _token_set(left)
    b = _token_set(right)
    if not a or not b:
        return 0.0
    return round(len(a & b) / max(1, len(a | b)), 4)


def _ragas_proxy_from_input(path: str) -> dict:
    rows = _read_jsonl(path)
    by_mode = {}
    for row in rows:
        mode = row.get("mode", "unknown")
        contexts = "\n\n".join(row.get("contexts") or [])
        item = by_mode.setdefault(mode, {
            "count": 0,
            "answer_ground_truth_jaccard_sum": 0.0,
            "answer_context_jaccard_sum": 0.0,
            "question_context_jaccard_sum": 0.0,
        })
        item["count"] += 1
        item["answer_ground_truth_jaccard_sum"] += _jaccard_text(row.get("answer", ""), row.get("ground_truth", ""))
        item["answer_context_jaccard_sum"] += _jaccard_text(row.get("answer", ""), contexts)
        item["question_context_jaccard_sum"] += _jaccard_text(row.get("question", ""), contexts)

    for item in by_mode.values():
        count = max(1, item["count"])
        item["answer_ground_truth_jaccard_avg"] = round(item.pop("answer_ground_truth_jaccard_sum") / count, 4)
        item["answer_context_jaccard_avg"] = round(item.pop("answer_context_jaccard_sum") / count, 4)
        item["question_context_jaccard_avg"] = round(item.pop("question_context_jaccard_sum") / count, 4)
    return {
        "type": "lightweight_ragas_proxy",
        "input_path": path,
        "by_mode": by_mode,
    }


def _resolve_ragas_input_path(path_value: str) -> str:
    requested = Path(path_value)
    if not requested.is_absolute():
        requested = (Path.cwd() / requested).resolve()
    else:
        requested = requested.resolve()
    data_root = (Path.cwd() / "data").resolve()
    src_data_root = (Path(__file__).resolve().parents[2] / "data").resolve()
    allowed_roots = [data_root, src_data_root]
    if not any(requested == root or root in requested.parents for root in allowed_roots):
        raise ValueError("RAGAS input_path must be under the project data directory.")
    if not requested.exists():
        raise FileNotFoundError(f"RAGAS input file not found: {requested}")
    return str(requested)


def _import_ragas_metrics():
    try:
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError:
        from ragas.metrics._answer_relevance import answer_relevancy
        from ragas.metrics._context_precision import context_precision
        from ragas.metrics._context_recall import context_recall
        from ragas.metrics._faithfulness import faithfulness
    return [faithfulness, answer_relevancy, context_precision, context_recall]


def _official_ragas_metrics():
    metrics = _import_ragas_metrics()
    for metric in metrics:
        if getattr(metric, "name", "") == "answer_relevancy" and hasattr(metric, "strictness"):
            metric.strictness = 1
    return metrics


RAGAS_GROQ_MODEL_FALLBACKS = (
    "meta-llama/llama-4-scout-17b-16e-instruct",
    "llama-3.3-70b-versatile",
    "qwen/qwen3-32b",
    "qwen/qwen3.6-27b",
)
RAGAS_EVALUATOR_INTERVAL_SECONDS = 65
RAGAS_EVALUATOR_TIMEOUT_SECONDS = 300
RAGAS_EVALUATOR_MAX_WORKERS = 1
RAGAS_EVALUATOR_BATCH_SIZE = 1
RAGAS_EVALUATOR_MAX_RETRIES = 2


def _ragas_model_candidates(model: str | None) -> list[str]:
    if model:
        return [model]
    return list(RAGAS_GROQ_MODEL_FALLBACKS)


def _is_token_per_day_limit(exc: Exception) -> bool:
    text = str(exc).lower()
    daily_markers = [
        "tokens per day",
        "token per day",
        "tpd",
        "daily token",
        "tokens per day (tpd)",
    ]
    return (
        ("rate limit" in text or "rate_limit" in text or "429" in text)
        and any(marker in text for marker in daily_markers)
    )


def _estimate_text_tokens_for_ragas(text: str) -> int:
    words = re.findall(r"\S+", str(text))
    by_words = int(len(words) * 1.35) if words else 0
    by_chars = int(len(str(text)) / 6.0)
    return max(1, by_words, by_chars)


def _ragas_message_text(message) -> str:
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return " ".join(
            str(part.get("text", part)) if isinstance(part, dict) else str(part)
            for part in content
        )
    return str(content)


class _RagasUsageCallback(BaseCallbackHandler):
    def __init__(self):
        self.llm_calls = 0
        self.prompt_count = 0
        self.estimated_prompt_tokens = 0
        self.reported_prompt_tokens = 0
        self.reported_completion_tokens = 0
        self.reported_total_tokens = 0

    def on_llm_start(self, serialized, prompts, **kwargs):
        prompt_list = prompts or []
        self.llm_calls += len(prompt_list)
        self.prompt_count += len(prompt_list)
        self.estimated_prompt_tokens += sum(_estimate_text_tokens_for_ragas(prompt) for prompt in prompt_list)

    def on_chat_model_start(self, serialized, messages, **kwargs):
        message_batches = messages or []
        self.llm_calls += len(message_batches)
        self.prompt_count += len(message_batches)
        for batch in message_batches:
            text = "\n".join(_ragas_message_text(message) for message in (batch or []))
            self.estimated_prompt_tokens += _estimate_text_tokens_for_ragas(text)

    def on_llm_end(self, response, **kwargs):
        usage = {}
        llm_output = getattr(response, "llm_output", None) or {}
        if isinstance(llm_output, dict):
            usage = llm_output.get("token_usage") or llm_output.get("usage") or {}
        if not usage:
            for generation_list in getattr(response, "generations", []) or []:
                for generation in generation_list or []:
                    message = getattr(generation, "message", None)
                    usage = getattr(message, "usage_metadata", None) or getattr(message, "response_metadata", {}).get("token_usage", {})
                    if usage:
                        break
                if usage:
                    break
        self.reported_prompt_tokens += int(
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or 0
        )
        self.reported_completion_tokens += int(
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or 0
        )
        self.reported_total_tokens += int(
            usage.get("total_tokens")
            or usage.get("total_token_count")
            or 0
        )

    def summary(self) -> dict:
        return {
            "llm_calls": self.llm_calls,
            "prompt_count": self.prompt_count,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "reported_prompt_tokens": self.reported_prompt_tokens,
            "reported_completion_tokens": self.reported_completion_tokens,
            "reported_total_tokens": self.reported_total_tokens,
        }


def _official_ragas_from_input(path: str, max_rows: int = 18, model: str | None = None) -> dict:
    started = time.perf_counter()
    rows = _read_jsonl(path)
    if max_rows and max_rows > 0:
        rows = rows[:max_rows]
    rows = [
        row for row in rows
        if row.get("question") and row.get("answer") and row.get("contexts") and row.get("ground_truth")
    ]
    if not rows:
        return {
            "type": "official_ragas",
            "status": "skipped",
            "reason": "No complete RAGAS rows with question, answer, contexts, and ground_truth.",
            "input_path": path,
            "row_count": 0,
        }

    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from ragas.llms import LangchainLLMWrapper
        from ragas.run_config import RunConfig
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_core.rate_limiters import InMemoryRateLimiter
        from langchain_openai import ChatOpenAI
        from backend.config import EMBEDDING_MODEL, GROQ_API_KEY, GROQ_BASE_URL, GROQ_MODEL, OPENAI_API_KEY, OPENAI_MODEL
    except Exception as exc:
        return {
            "type": "official_ragas",
            "status": "unavailable",
            "reason": f"{type(exc).__name__}: {exc}",
            "input_path": path,
            "row_count": len(rows),
        }

    if not GROQ_API_KEY and not OPENAI_API_KEY:
        return {
            "type": "official_ragas",
            "status": "unavailable",
            "reason": "GROQ_API_KEY or OPENAI_API_KEY is required for the evaluator LLM.",
            "input_path": path,
            "row_count": len(rows),
        }

    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        dataset = Dataset.from_list([
            {
                "user_input": row["question"],
                "response": row["answer"],
                "retrieved_contexts": row["contexts"],
                "reference": row["ground_truth"],
                "pair_id": row.get("pair_id"),
                "mode": row.get("mode"),
            }
            for row in rows
        ])

        attempts = []
        if GROQ_API_KEY:
            provider = "groq"
            model_candidates = _ragas_model_candidates(model)
        else:
            provider = "openai"
            model_candidates = [model or OPENAI_MODEL]

        for evaluator_model in model_candidates:
            usage_callback = _RagasUsageCallback()
            rate_limiter = InMemoryRateLimiter(
                requests_per_second=1 / RAGAS_EVALUATOR_INTERVAL_SECONDS,
                check_every_n_seconds=1,
                max_bucket_size=1,
            )
            run_config = RunConfig(
                timeout=RAGAS_EVALUATOR_TIMEOUT_SECONDS,
                max_retries=RAGAS_EVALUATOR_MAX_RETRIES,
                max_wait=RAGAS_EVALUATOR_INTERVAL_SECONDS,
                max_workers=RAGAS_EVALUATOR_MAX_WORKERS,
            )
            try:
                if provider == "groq":
                    llm = ChatOpenAI(
                        model=evaluator_model,
                        api_key=GROQ_API_KEY,
                        base_url=GROQ_BASE_URL,
                        temperature=0,
                        max_tokens=1024,
                        timeout=RAGAS_EVALUATOR_TIMEOUT_SECONDS,
                        max_retries=RAGAS_EVALUATOR_MAX_RETRIES,
                        rate_limiter=rate_limiter,
                    )
                else:
                    llm = ChatOpenAI(
                        model=evaluator_model,
                        api_key=OPENAI_API_KEY,
                        temperature=0,
                        max_tokens=1024,
                        timeout=RAGAS_EVALUATOR_TIMEOUT_SECONDS,
                        max_retries=RAGAS_EVALUATOR_MAX_RETRIES,
                        rate_limiter=rate_limiter,
                    )
                result = evaluate(
                    dataset,
                    metrics=_official_ragas_metrics(),
                    llm=LangchainLLMWrapper(llm),
                    embeddings=LangchainEmbeddingsWrapper(embeddings),
                    callbacks=[usage_callback],
                    raise_exceptions=False,
                    show_progress=False,
                    run_config=run_config,
                    batch_size=RAGAS_EVALUATOR_BATCH_SIZE,
                )
                records = result.to_pandas().to_dict(orient="records")
                for index, record in enumerate(records):
                    if index < len(rows):
                        record["pair_id"] = rows[index].get("pair_id")
                        record["mode"] = rows[index].get("mode")
                score_path = str(Path(path).with_suffix(".official_ragas_scores.json"))
                Path(score_path).write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
                return {
                    "type": "official_ragas",
                    "status": "completed",
                    "input_path": path,
                    "score_path": score_path,
                    "row_count": len(records),
                    "evaluator_provider": provider,
                    "evaluator_model": evaluator_model,
                    "evaluator_attempts": attempts,
                    "evaluator_rate_limit": {
                        "interval_seconds": RAGAS_EVALUATOR_INTERVAL_SECONDS,
                        "max_workers": RAGAS_EVALUATOR_MAX_WORKERS,
                        "batch_size": RAGAS_EVALUATOR_BATCH_SIZE,
                        "timeout_seconds": RAGAS_EVALUATOR_TIMEOUT_SECONDS,
                        "max_retries": RAGAS_EVALUATOR_MAX_RETRIES,
                    },
                    "evaluator_usage": usage_callback.summary(),
                    "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
                    "by_mode": _summarize_official_ragas(records),
                }
            except Exception as model_exc:
                attempts.append({
                    "model": evaluator_model,
                    "error": f"{type(model_exc).__name__}: {model_exc}",
                    "fallback_reason": "tokens_per_day" if _is_token_per_day_limit(model_exc) else "none",
                    "usage": usage_callback.summary(),
                })
                if provider == "groq" and _is_token_per_day_limit(model_exc):
                    continue
                raise

        return {
            "type": "official_ragas",
            "status": "failed",
            "reason": "All RAGAS evaluator model candidates failed.",
            "input_path": path,
            "row_count": len(rows),
            "evaluator_attempts": attempts,
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }
    except Exception as exc:
        return {
            "type": "official_ragas",
            "status": "failed",
            "reason": f"{type(exc).__name__}: {exc}",
            "input_path": path,
            "row_count": len(rows),
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 1),
        }


def _summarize_official_ragas(records: list[dict]) -> dict:
    metric_names = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    by_mode: dict[str, dict] = {}
    for row in records:
        mode = str(row.get("mode") or "unknown")
        item = by_mode.setdefault(
            mode,
            {
                "count": 0,
                "valid_counts": {name: 0 for name in metric_names},
                "nan_counts": {name: 0 for name in metric_names},
                **{name: 0.0 for name in metric_names},
            },
        )
        item["count"] += 1
        for name in metric_names:
            value = row.get(name)
            if isinstance(value, (int, float)) and value == value:
                item[name] += float(value)
                item["valid_counts"][name] += 1
            else:
                item["nan_counts"][name] += 1
    for item in by_mode.values():
        for name in metric_names:
            valid_count = item["valid_counts"][name]
            item[name] = round(item[name] / valid_count, 4) if valid_count else None
    return by_mode


def _run_similar_pair_quality_test(body: TestSuiteRunRequest, progress: _SuiteProgress | None = None) -> dict:
    dataset_name = body.dataset_name.strip().lower()
    if _dataset_family(body) != "ragbench" or dataset_name not in {"emanual", "techqa"}:
        raise ValueError("TC3 requires dataset_family=ragbench and dataset_name=techqa or emanual.")

    pairs = _tc4_asset_pairs(body)
    if not pairs:
        raise RuntimeError("TC3 pair asset is empty. Run the RAGBench query asset builder first.")
    prepare_count = max(body.num_examples, max(max(p["left"]["index"], p["right"]["index"]) for p in pairs) + 1)

    if progress:
        progress.reset(len(pairs) * 4 + 1, "TC3 Dataset/EU 준비")
        progress.step("TC3 Dataset/EU 준비")
    prepared = _prepare_suite_source(body, prepare_count)
    if progress:
        progress.advance("TC3 Dataset/EU 준비 완료")
    source_id = prepared["source_id"]
    route_pool = _seed_tc4_route_pool(body, pairs)
    prepared["route_pool"] = route_pool
    route_mode = route_pool["route_pool_mode"]
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
        if route_mode == "similar_only":
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
                    "TC3 A left",
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
                    "TC3 A right",
                )[0]
        finally:
            if route_mode == "similar_only":
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
            "TC3 B left",
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
            "TC3 B right",
        )[0]

        a_left_results.append(a_left)
        a_right_results.append(a_right)
        b_left_results.append(b_left)
        b_right_results.append(b_right)
        a_right_contexts = _contexts_for_result(source_id, a_right)
        b_right_contexts = _contexts_for_result(source_id, b_right)
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
            "a_right_route_id": a_right.get("embedding_route_id"),
            "a_right_route_score": a_right.get("embedding_score"),
            "a_right_cache_lookup_strategy": a_right.get("cache_lookup_strategy"),
            "a_right_cache_candidate_id": a_right.get("cache_candidate_id"),
            "a_right_cache_candidate_route_id": a_right.get("cache_candidate_route_id"),
            "a_right_cache_similarity_score": a_right.get("cache_similarity_score"),
            "left_reference_answer": left.get("reference_answer", ""),
            "right_reference_answer": right.get("reference_answer", ""),
            "a_left_answer": a_left.get("answer", ""),
            "a_right_answer": a_right.get("answer", ""),
            "b_left_answer": b_left.get("answer", ""),
            "b_right_answer": b_right.get("answer", ""),
            "a_left_usage": _llm_usage_summary(a_left),
            "a_right_usage": _llm_usage_summary(a_right),
            "b_left_usage": _llm_usage_summary(b_left),
            "b_right_usage": _llm_usage_summary(b_right),
            "a_right_contexts": a_right_contexts,
            "b_right_contexts": b_right_contexts,
            "a_right_result": a_right,
            "b_right_result": b_right,
        })

    ragas_input_path = _write_tc4_ragas_input(
        body.dataset_name,
        body.dataset_split,
        source_id,
        pair_results,
    )
    public_pair_results = [
        {k: v for k, v in pair.items() if k not in {"a_right_result", "b_right_result"}}
        for pair in pair_results
    ]
    return {
        "test_case": "similar_pair_quality",
        "prepared": prepared,
        "source_id": source_id,
        "query_asset": "tc4_query_pairs",
        "route_pool_mode": route_mode,
        "pair_count": len(pairs),
        "query_count": len(pairs) * 2,
        "ragas_input_path": ragas_input_path,
        "ragas_proxy": None,
        "official_ragas": None,
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
        "pairs": public_pair_results,
    }


def _run_test_suite_internal(body: TestSuiteRunRequest, progress: _SuiteProgress | None = None):
    init_db()
    normalized = body.test_case.strip().lower()
    if normalized == "cache":
        return _run_cache_test(body, progress)
    if normalized in {"mixed", "mixed_timing", "timing", "tc_add"}:
        return _run_mixed_timing_test(body, progress)
    if normalized in {"scalability", "scale"}:
        return _run_scalability_test(body, progress)
    if normalized in {"similar_pair_quality", "tc3", "tc4", "pair_quality"}:
        return _run_similar_pair_quality_test(body, progress)
    raise ValueError(f"Unknown DP3 test_case: {body.test_case}")


@router.post("/test-suite/run")
def run_test_suite(body: TestSuiteRunRequest):
    result = _run_test_suite_internal(body)
    result["saved_run_log"] = _save_test_suite_run(body, result)
    return result


@router.post("/ragas/run")
def run_ragas(body: RagasRunRequest):
    input_path = _resolve_ragas_input_path(body.input_path)
    result = {
        "input_path": input_path,
        "ragas_proxy": _ragas_proxy_from_input(input_path),
        "official_ragas": None,
    }
    if body.run_official:
        result["official_ragas"] = _official_ragas_from_input(
            input_path,
            max_rows=body.max_rows,
            model=body.model,
        )
    return result


def _run_suite_job(job_id: str, body: TestSuiteRunRequest) -> None:
    _update_suite_job(job_id, status="running", current_step="테스트 시작", started_at=time.time())
    try:
        result = _run_test_suite_internal(body, _SuiteProgress(job_id))
        result["saved_run_log"] = _save_test_suite_run(body, result, job_id=job_id)
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
