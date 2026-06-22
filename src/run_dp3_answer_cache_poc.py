"""
DP3 Verified Answer Cache PoC runner.

Usage:
    python run_dp3_answer_cache_poc.py <thread_id> [query] [scope] [version]

The runner uses DP3_MOCK_LLM=true by default, so it can verify cache routing,
metadata validation, and hit/miss behavior without an external LLM.
"""
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("SQLITE_DB_PATH", str(BASE_DIR / "data" / "poc.db"))
os.environ.setdefault("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma"))

sys.path.insert(0, str(BASE_DIR))

from backend.cache.answer_cache import run_answer_cache_query, setup_answer_cache_poc
from backend.db.database import init_db


def main():
    if len(sys.argv) < 2:
        print("Usage: python run_dp3_answer_cache_poc.py <thread_id> [query] [scope] [version]")
        sys.exit(1)

    thread_id = sys.argv[1]
    query = sys.argv[2] if len(sys.argv) > 2 else "V1에서 이 문서의 핵심 내용을 요약해줘."
    scope = sys.argv[3] if len(sys.argv) > 3 else "A"
    version = sys.argv[4] if len(sys.argv) > 4 else None

    init_db()
    setup = setup_answer_cache_poc(thread_id, reset=False)
    print("[setup]", setup)

    result = run_answer_cache_query(
        thread_id=thread_id,
        query=query,
        user_scope=scope,
        requested_version=version,
    )
    print("[result]")
    import json
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
