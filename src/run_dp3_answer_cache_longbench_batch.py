"""
Run DP3 Answer Cache A-option against sampled LongBench source questions.

This script samples test queries from the raw LongBench JSONL files, not from
the answerable_question_pool. The answerable_question_pool is used only as the
front route/filter gate.

Usage:
    python run_dp3_answer_cache_longbench_batch.py --source-id dp3_longbench_multifieldqa_en_5 --count 100
"""
import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("SQLITE_DB_PATH", str(BASE_DIR / "data" / "poc.db"))
os.environ.setdefault("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma"))

sys.path.insert(0, str(BASE_DIR))

from backend.cache.answer_cache import clear_answer_cache_for_source, run_answer_cache_query
from backend.db.database import init_db
from seed_dp3_question_pool import _question_text

DATA_DIR = BASE_DIR / "data" / "longbench"


def _iter_longbench_questions(dataset: str | None, include_smoke: bool):
    files = [DATA_DIR / f"{dataset}.jsonl"] if dataset else sorted(DATA_DIR.glob("*.jsonl"))
    for path in files:
        if not path.exists() or (not include_smoke and path.name.startswith("dp3_smoke")):
            continue
        with path.open(encoding="utf-8-sig") as f:
            for index, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                query = _question_text(path.stem, row)
                if query:
                    yield {
                        "query_id": f"longbench:{path.stem}:{index}",
                        "dataset": path.stem,
                        "index": index,
                        "query": query,
                    }


def _sample_queries(dataset: str | None, count: int, seed: int, include_smoke: bool):
    queries = list(_iter_longbench_questions(dataset, include_smoke))
    rng = random.Random(seed)
    if count >= len(queries):
        return queries
    return sorted(rng.sample(queries, count), key=lambda item: (item["dataset"], item["index"]))


def _avg(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 3)


def _timing_value(row: dict, key: str) -> float | None:
    timings = row.get("timings_ms") or {}
    value = timings.get(key)
    if value is None:
        return None
    return float(value)


def _timing_summary(results: list[dict]) -> dict:
    keys = [
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
    summary = {}
    for key in keys:
        values = [
            value for value in (_timing_value(row, key) for row in results)
            if value is not None
        ]
        if values:
            summary[key] = _avg(values)

    hit_rows = [row for row in results if row.get("cache_hit")]
    rag_rows = [row for row in results if row.get("roi_rag_called")]
    summary["hit_total_ms"] = _avg([
        value for value in (_timing_value(row, "total_ms") for row in hit_rows)
        if value is not None
    ])
    summary["rag_total_request_ms"] = _avg([
        value for value in (_timing_value(row, "total_ms") for row in rag_rows)
        if value is not None
    ])
    return summary


def _summarize(results: list[dict]) -> dict:
    by_reason = Counter(r.get("decision_reason", "unknown") for r in results)
    by_dataset = {}
    for row in results:
        item = by_dataset.setdefault(row["dataset"], {
            "total": 0,
            "route_passed": 0,
            "cache_hit": 0,
            "validation_passed": 0,
            "fallbacks": 0,
            "llm_calls": 0,
        })
        item["total"] += 1
        item["route_passed"] += int(bool(row.get("routing_passed")))
        item["cache_hit"] += int(bool(row.get("cache_hit")))
        item["validation_passed"] += int(bool(row.get("validation_passed")))
        item["fallbacks"] += int(bool(row.get("roi_rag_called")))
        item["llm_calls"] += int(row.get("llm_call_count", 0))

    total = len(results)
    return {
        "total": total,
        "route_passed": sum(1 for r in results if r.get("routing_passed")),
        "cache_hits": sum(1 for r in results if r.get("cache_hit")),
        "validation_passed": sum(1 for r in results if r.get("validation_passed")),
        "fallbacks": sum(1 for r in results if r.get("roi_rag_called")),
        "llm_calls": sum(int(r.get("llm_call_count", 0)) for r in results),
        "timing_avg_ms": _timing_summary(results),
        "decision_reasons": dict(by_reason),
        "by_dataset": by_dataset,
    }


def _run_pass(
    pass_name: str,
    source_id: str,
    queries: list[dict],
    user_scope: str,
    requested_version: str,
    route_threshold: float,
    cache_threshold: float,
    llm_provider: str | None = None,
    model: str | None = None,
    use_reranker: bool = False,
    rerank_candidates: int = 30,
    rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
) -> dict:
    results = []
    for item in queries:
        result = run_answer_cache_query(
            thread_id=source_id,
            query=item["query"],
            user_scope=user_scope,
            requested_version=requested_version,
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
    return {"pass": pass_name, "summary": _summarize(results), "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-id", default="dp3_longbench_multifieldqa_en_5")
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--scope", default="A")
    parser.add_argument("--route-threshold", type=float, default=0.70)
    parser.add_argument("--cache-threshold", type=float, default=0.86)
    parser.add_argument("--llm-provider", default=None, choices=[None, "mock", "groq", "default"])
    parser.add_argument("--model", default=None)
    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--rerank-candidates", type=int, default=30)
    parser.add_argument("--rerank-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()

    init_db()
    clear_answer_cache_for_source(args.source_id)
    queries = _sample_queries(args.dataset, args.count, args.seed, args.include_smoke)
    if not queries:
        raise RuntimeError("No LongBench queries found.")

    passes = [
        _run_pass(
            "v1_first",
            args.source_id,
            queries,
            args.scope,
            "V1",
            args.route_threshold,
            args.cache_threshold,
            args.llm_provider,
            args.model,
            args.use_reranker,
            args.rerank_candidates,
            args.rerank_model,
        ),
        _run_pass(
            "v1_repeat",
            args.source_id,
            queries,
            args.scope,
            "V1",
            args.route_threshold,
            args.cache_threshold,
            args.llm_provider,
            args.model,
            args.use_reranker,
            args.rerank_candidates,
            args.rerank_model,
        ),
        _run_pass(
            "v2_validation",
            args.source_id,
            queries,
            args.scope,
            "V2",
            args.route_threshold,
            args.cache_threshold,
            args.llm_provider,
            args.model,
            args.use_reranker,
            args.rerank_candidates,
            args.rerank_model,
        ),
    ]

    output = {
        "source_id": args.source_id,
        "sampled_query_count": len(queries),
        "sampled_datasets": dict(Counter(q["dataset"] for q in queries)),
        "route_threshold": args.route_threshold,
        "cache_threshold": args.cache_threshold,
        "passes": [{"pass": p["pass"], "summary": p["summary"]} for p in passes],
    }
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
