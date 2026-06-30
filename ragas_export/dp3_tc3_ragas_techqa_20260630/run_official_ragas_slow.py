#!/usr/bin/env python3
"""Run official RAGAS for the exported DP3 TC3 JSONL input.

This script is intentionally standalone. It does not import project backend
code, so the export folder can be copied to another machine and executed there.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path(__file__).with_name("dp3_tc3_ragas_input_techqa_20rows.jsonl")
DEFAULT_OUTPUT_DIR = Path(__file__).with_name("official_ragas_output")
DEFAULT_GROQ_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--provider", choices=["groq", "openai"], default=os.getenv("RAGAS_PROVIDER", "groq"))
    parser.add_argument("--model", default=os.getenv("RAGAS_MODEL"))
    parser.add_argument("--embedding-model", default=os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    parser.add_argument("--base-url", default=os.getenv("GROQ_BASE_URL", DEFAULT_GROQ_BASE_URL))
    parser.add_argument("--max-rows", type=int, default=0, help="0 means all rows.")
    parser.add_argument("--start-row", type=int, default=0, help="0-based inclusive start row.")
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--seconds-per-request", type=float, default=65.0)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--metrics",
        default="faithfulness,answer_relevancy,context_precision,context_recall",
        help="Comma-separated metrics. Use without context_precision if that metric keeps timing out.",
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def import_metrics(names: list[str]) -> list[Any]:
    try:
        from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
    except ImportError:
        from ragas.metrics._answer_relevance import answer_relevancy
        from ragas.metrics._context_precision import context_precision
        from ragas.metrics._context_recall import context_recall
        from ragas.metrics._faithfulness import faithfulness

    registry = {
        "faithfulness": faithfulness,
        "answer_relevancy": answer_relevancy,
        "context_precision": context_precision,
        "context_recall": context_recall,
    }
    metrics = [registry[name] for name in names]
    for metric in metrics:
        if getattr(metric, "name", "") == "answer_relevancy" and hasattr(metric, "strictness"):
            metric.strictness = 1
    return metrics


def build_dataset(rows: list[dict[str, Any]]) -> Any:
    from datasets import Dataset

    return Dataset.from_list(
        [
            {
                "user_input": row["question"],
                "response": row["answer"],
                "retrieved_contexts": row["contexts"],
                "reference": row["ground_truth"],
            }
            for row in rows
        ]
    )


def build_llm(args: argparse.Namespace) -> Any:
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_openai import ChatOpenAI

    rate_limiter = InMemoryRateLimiter(
        requests_per_second=1 / args.seconds_per_request,
        check_every_n_seconds=1,
        max_bucket_size=1,
    )

    if args.provider == "groq":
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY is required when --provider groq.")
        return ChatOpenAI(
            model=args.model or DEFAULT_GROQ_MODEL,
            api_key=api_key,
            base_url=args.base_url,
            temperature=0,
            max_tokens=1024,
            timeout=args.timeout,
            max_retries=args.max_retries,
            rate_limiter=rate_limiter,
        )

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required when --provider openai.")
    return ChatOpenAI(
        model=args.model or DEFAULT_OPENAI_MODEL,
        api_key=api_key,
        temperature=0,
        max_tokens=1024,
        timeout=args.timeout,
        max_retries=args.max_retries,
        rate_limiter=rate_limiter,
    )


def clean_for_json(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {key: clean_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_for_json(item) for item in value]
    return value


def evaluate_chunk(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.run_config import RunConfig

    dataset = build_dataset(rows)
    llm = build_llm(args)
    embeddings = HuggingFaceEmbeddings(model_name=args.embedding_model)
    run_config = RunConfig(
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_wait=int(args.seconds_per_request),
        max_workers=1,
    )
    metric_names = [name.strip() for name in args.metrics.split(",") if name.strip()]
    result = evaluate(
        dataset,
        metrics=import_metrics(metric_names),
        llm=LangchainLLMWrapper(llm),
        embeddings=LangchainEmbeddingsWrapper(embeddings),
        raise_exceptions=False,
        show_progress=True,
        run_config=run_config,
        batch_size=1,
    )
    records = result.to_pandas().to_dict(orient="records")
    merged = []
    for index, record in enumerate(records):
        source = rows[index]
        for key, value in source.items():
            if key not in record:
                record[key] = value
        merged.append(clean_for_json(record))
    return merged


def chunk_ranges(total: int, chunk_size: int) -> list[tuple[int, int]]:
    return [(start, min(start + chunk_size, total)) for start in range(0, total, chunk_size)]


def combine_chunks(output_dir: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted((output_dir / "chunks").glob("chunk_*.json")):
        records.extend(json.loads(path.read_text(encoding="utf-8")))
    return records


def write_summary(output_dir: Path, args: argparse.Namespace, rows: list[dict[str, Any]], records: list[dict[str, Any]]) -> None:
    metrics = [name.strip() for name in args.metrics.split(",") if name.strip()]
    by_mode: dict[str, dict[str, Any]] = {}
    for mode in sorted({str(row.get("mode")) for row in records}):
        items = [row for row in records if str(row.get("mode")) == mode]
        summary: dict[str, Any] = {"n": len(items)}
        for metric in metrics:
            values = [row.get(metric) for row in items]
            valid = [float(value) for value in values if isinstance(value, (int, float)) and not math.isnan(float(value))]
            summary[metric] = {
                "avg": round(sum(valid) / len(valid), 6) if valid else None,
                "valid": len(valid),
                "missing_or_nan": len(values) - len(valid),
            }
        by_mode[mode] = summary

    payload = {
        "input": str(Path(args.input).resolve()),
        "rows_requested": len(rows),
        "rows_scored": len(records),
        "provider": args.provider,
        "model": args.model or (DEFAULT_GROQ_MODEL if args.provider == "groq" else DEFAULT_OPENAI_MODEL),
        "embedding_model": args.embedding_model,
        "rate_limit": {
            "seconds_per_request": args.seconds_per_request,
            "chunk_size": args.chunk_size,
            "max_workers": 1,
            "batch_size": 1,
            "timeout": args.timeout,
            "max_retries": args.max_retries,
        },
        "metrics": metrics,
        "by_mode": by_mode,
    }
    (output_dir / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    chunk_dir = output_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    if args.start_row:
        rows = rows[args.start_row :]
    if args.max_rows and args.max_rows > 0:
        rows = rows[: args.max_rows]

    started = time.time()
    for start, end in chunk_ranges(len(rows), args.chunk_size):
        chunk_path = chunk_dir / f"chunk_{start:03d}_{end - 1:03d}.json"
        if args.skip_existing and chunk_path.exists():
            print(f"[skip] {chunk_path}")
            continue
        print(f"[run] rows {start}-{end - 1} -> {chunk_path}")
        records = evaluate_chunk(rows[start:end], args)
        chunk_path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
        combined = combine_chunks(output_dir)
        (output_dir / "official_ragas_scores_combined.json").write_text(
            json.dumps(combined, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        write_summary(output_dir, args, rows, combined)
        print(f"[done] chunk rows={len(records)} combined={len(combined)} elapsed_s={time.time() - started:.1f}")

    combined = combine_chunks(output_dir)
    (output_dir / "official_ragas_scores_combined.json").write_text(
        json.dumps(combined, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    write_summary(output_dir, args, rows, combined)
    print(json.dumps({"rows_requested": len(rows), "rows_scored": len(combined), "summary": str(output_dir / "summary.json")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
