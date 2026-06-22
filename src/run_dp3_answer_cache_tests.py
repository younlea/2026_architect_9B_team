"""
DP3 A안 Verified Answer Cache mock 테스트 runner.

Usage:
    python run_dp3_answer_cache_tests.py <source_id>

source_id는 load_longbench_dp3.py 출력값을 사용한다.
"""
import json
import os
import sys
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("SQLITE_DB_PATH", str(BASE_DIR / "data" / "poc.db"))
os.environ.setdefault("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma"))

sys.path.insert(0, str(BASE_DIR))

from backend.cache.answer_cache import clear_answer_cache_for_source, run_answer_cache_query
from backend.db.database import get_conn, init_db


def _load_queries(source_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT query_id, query_type, query_text, user_scope, requested_version, expected_behavior
               FROM dp3_query_sets
               WHERE source_id=?
               ORDER BY query_id""",
            (source_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def _summarize(results: list[dict]) -> dict:
    total = len(results)
    by_reason = Counter(r.get("decision_reason", "unknown") for r in results)
    by_type = {}
    for r in results:
        qtype = r.get("query_type", "unknown")
        item = by_type.setdefault(qtype, {
            "total": 0,
            "cache_hit": 0,
            "validation_fail": 0,
            "llm_calls": 0,
            "fallbacks": 0,
        })
        item["total"] += 1
        item["cache_hit"] += int(bool(r.get("cache_hit")))
        item["validation_fail"] += int(bool(r.get("cache_candidate_id")) and not bool(r.get("validation_passed")))
        item["llm_calls"] += int(r.get("llm_call_count", 0))
        item["fallbacks"] += int(bool(r.get("roi_rag_called")))
    return {
        "total": total,
        "cache_hits": sum(1 for r in results if r.get("cache_hit")),
        "fallbacks": sum(1 for r in results if r.get("roi_rag_called")),
        "llm_calls": sum(int(r.get("llm_call_count", 0)) for r in results),
        "decision_reasons": dict(by_reason),
        "by_query_type": by_type,
    }


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_dp3_answer_cache_tests.py <source_id>")
        sys.exit(1)

    source_id = sys.argv[1]
    init_db()
    clear_answer_cache_for_source(source_id)
    queries = _load_queries(source_id)
    if not queries:
        print(f"No dp3_query_sets for source_id={source_id}")
        sys.exit(1)

    results = []
    for q in queries:
        result = run_answer_cache_query(
            thread_id=source_id,
            query=q["query_text"],
            user_scope=q["user_scope"],
            requested_version=q["requested_version"],
        )
        result["query_id"] = q["query_id"]
        result["query_type"] = q["query_type"]
        result["expected_behavior"] = q["expected_behavior"]
        results.append(result)
        print(json.dumps({
            "query_id": q["query_id"],
            "query_type": q["query_type"],
            "scope": q["user_scope"],
            "version": q["requested_version"],
            "cache_hit": result.get("cache_hit"),
            "validation_passed": result.get("validation_passed"),
            "decision_reason": result.get("decision_reason"),
            "llm_call_count": result.get("llm_call_count"),
            "roi_rag_called": result.get("roi_rag_called"),
            "cache_candidate_id": result.get("cache_candidate_id"),
            "source_validation": result.get("source_validation"),
        }, ensure_ascii=False))

    print("[summary]")
    print(json.dumps(_summarize(results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
