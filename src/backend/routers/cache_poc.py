from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from backend.cache.answer_cache import (
    run_answer_cache_query,
    setup_answer_cache_poc,
)
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
    route_threshold: float = 0.70
    cache_threshold: float = 0.86


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
        route_threshold=body.route_threshold,
        cache_threshold=body.cache_threshold,
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
        ),
        _run_pass(
            "v1_repeat",
            body.source_id,
            queries,
            body.user_scope,
            "V1",
            body.route_threshold,
            body.cache_threshold,
        ),
        _run_pass(
            "v2_validation",
            body.source_id,
            queries,
            body.user_scope,
            "V2",
            body.route_threshold,
            body.cache_threshold,
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
