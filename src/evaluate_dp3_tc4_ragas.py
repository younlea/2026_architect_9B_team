"""
Build optional RAGAS input/evaluation data for DP3 TC4.

Default behavior runs TC4 with mock LLM and writes a RAGAS-compatible JSONL.
Use --evaluate only in an environment where ragas and its evaluator LLM are
configured.

Usage:
    python evaluate_dp3_tc4_ragas.py
    python evaluate_dp3_tc4_ragas.py --llm-provider groq --model llama-3.1-8b-instant
    python evaluate_dp3_tc4_ragas.py --input src/data/dp3_tc4_result.json --evaluate
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("SQLITE_DB_PATH", str(BASE_DIR / "data" / "poc.db"))
os.environ.setdefault("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma"))

sys.path.insert(0, str(BASE_DIR))

from backend.cache.answer_cache import TOP_K_SOURCES, _embed, _retrieve_context_units  # noqa: E402
from backend.routers.cache_poc import TestSuiteRunRequest, _run_test_suite_internal  # noqa: E402


DEFAULT_OUTPUT = BASE_DIR / "data" / "ragbench" / "emanual" / "test_tc4_ragas_input.jsonl"


def main() -> int:
    args = parse_args()
    result = load_or_run_tc4(args)
    rows = build_ragas_rows(result, top_k=args.top_k)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output, rows)
    print(json.dumps({
        "output": str(output),
        "rows": len(rows),
        "pairs": result.get("pair_count"),
        "llm_provider": result.get("llm_provider"),
        "llm_model": result.get("llm_model"),
    }, ensure_ascii=False, indent=2))

    if args.evaluate:
        scores = run_ragas_evaluate(rows)
        score_path = output.with_suffix(".scores.json")
        score_path.write_text(json.dumps(scores, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"scores": str(score_path)}, ensure_ascii=False, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Existing TC4 result JSON file.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--llm-provider", default="mock")
    parser.add_argument("--model", default=None)
    parser.add_argument("--query-count", type=int, default=18)
    parser.add_argument("--cache-threshold", type=float, default=0.78)
    parser.add_argument("--route-threshold", type=float, default=0.70)
    parser.add_argument("--top-k", type=int, default=TOP_K_SOURCES)
    parser.add_argument("--evaluate", action="store_true")
    return parser.parse_args()


def load_or_run_tc4(args: argparse.Namespace) -> dict:
    if args.input:
        return json.loads(Path(args.input).read_text(encoding="utf-8"))
    body = TestSuiteRunRequest(
        test_case="similar_pair_quality",
        dataset_family="ragbench",
        dataset_name="emanual",
        dataset_split="test",
        num_examples=132,
        query_count=args.query_count,
        seed=7,
        warmup_count=0,
        user_scope="A",
        route_threshold=args.route_threshold,
        cache_threshold=args.cache_threshold,
        llm_provider=args.llm_provider,
        model=args.model,
        use_reranker=False,
    )
    return _run_test_suite_internal(body)


def build_ragas_rows(result: dict, top_k: int) -> list[dict]:
    source_id = result["source_id"]
    rows = []
    for pair in result.get("pairs", []):
        contexts = retrieve_contexts(
            source_id=source_id,
            question=pair["right_query"],
            user_scope="A",
            requested_version="V1",
            top_k=top_k,
        )
        for mode in ("a", "b"):
            rows.append({
                "pair_id": pair["pair_id"],
                "mode": mode.upper(),
                "question": pair["right_query"],
                "answer": pair[f"{mode}_right_answer"],
                "contexts": contexts,
                "ground_truth": pair["right_reference_answer"],
                "cache_hit": pair[f"{mode}_right_cache_hit"],
                "answers_equal_to_left": pair[f"{mode}_answers_equal"],
                "similarity": pair["similarity"],
                "answer_jaccard": pair["answer_jaccard"],
            })
    return rows


def retrieve_contexts(
    source_id: str,
    question: str,
    user_scope: str,
    requested_version: str,
    top_k: int,
) -> list[str]:
    sources = _retrieve_context_units(
        thread_id=source_id,
        query=question,
        query_embedding=_embed(question),
        user_scope=user_scope,
        requested_version=requested_version,
        top_k=top_k,
        timing={},
        use_reranker=False,
    )
    return [str(source.get("text", "")) for source in sources if str(source.get("text", "")).strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_ragas_evaluate(rows: list[dict]) -> dict:
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "ragas is not installed. Install ragas and configure evaluator LLM credentials, "
            "or run without --evaluate to create the JSONL input only."
        ) from exc

    dataset = Dataset.from_list([
        {
            "question": row["question"],
            "answer": row["answer"],
            "contexts": row["contexts"],
            "ground_truth": row["ground_truth"],
        }
        for row in rows
    ])
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    )
    try:
        return result.to_pandas().to_dict(orient="records")
    except Exception:
        return dict(result)


if __name__ == "__main__":
    raise SystemExit(main())
