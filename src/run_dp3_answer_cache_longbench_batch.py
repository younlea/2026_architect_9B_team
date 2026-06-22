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
) -> dict:
    results = []
    for item in queries:
        result = run_answer_cache_query(
            thread_id=source_id,
            query=item["query"],
            user_scope=user_scope,
            requested_version=requested_version,
            route_threshold=route_threshold,
            cache_threshold=cache_threshold,
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
        ),
        _run_pass(
            "v1_repeat",
            args.source_id,
            queries,
            args.scope,
            "V1",
            args.route_threshold,
            args.cache_threshold,
        ),
        _run_pass(
            "v2_validation",
            args.source_id,
            queries,
            args.scope,
            "V2",
            args.route_threshold,
            args.cache_threshold,
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
