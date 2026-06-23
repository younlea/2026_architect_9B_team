"""
DP3-only RAGBench preparation script.

This does not touch the DP1/DP2 LongBench loader or benchmark tables.

Usage:
    python load_ragbench_dp3.py techqa 20
    python load_ragbench_dp3.py emanual 50 --split test
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

from backend.cache.answer_cache import (  # noqa: E402
    _embed,
    _embedding_to_json,
    init_dp3_cache_schema,
    seed_answerable_question_pool,
)
from backend.db.database import get_conn, init_db  # noqa: E402
from load_longbench_dp3 import (  # noqa: E402
    _clear_source,
    _clear_versioned_source,
    _insert_context_units,
    _insert_versioned_units_from_base,
)

RAGBENCH_REPO = "rungalileo/ragbench"
RAGBENCH_SUBSETS = ("techqa", "emanual")
RAGBENCH_DATA_DIR = BASE_DIR / "data" / "ragbench"


def _raw_path(subset: str, split: str) -> Path:
    return RAGBENCH_DATA_DIR / subset / f"{split}.jsonl"


def _normalize_row(row: dict, subset: str, split: str, index: int) -> dict:
    documents = row.get("documents") or []
    if isinstance(documents, str):
        documents = [documents]
    documents = [str(doc).strip() for doc in documents if str(doc).strip()]
    question = str(row.get("question", "")).strip()
    response = str(row.get("response", "")).strip()
    row_id = str(row.get("id") or f"{subset}_{split}_{index}")
    return {
        "id": row_id,
        "input": question,
        "context": "\n\n".join(documents),
        "answers": [response] if response else [],
        "response": response,
        "documents": documents,
        "dataset_name": f"ragbench_{subset}_{split}",
        "source_index": index,
    }


def _download_ragbench(subset: str, split: str) -> Path:
    try:
        from datasets import load_dataset
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The 'datasets' package is required for RAGBench. "
            "Install dependencies from src/requirements.txt."
        ) from exc

    target = _raw_path(subset, split)
    target.parent.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(RAGBENCH_REPO, subset, split=split)
    with target.open("w", encoding="utf-8") as f:
        for index, row in enumerate(ds):
            normalized = _normalize_row(dict(row), subset, split, index)
            f.write(json.dumps(normalized, ensure_ascii=False) + "\n")
    return target


def ensure_ragbench_dataset(subset: str, split: str = "test", auto_download: bool = True) -> Path:
    subset = subset.strip().lower()
    split = split.strip().lower()
    if subset not in RAGBENCH_SUBSETS:
        raise ValueError(f"Unsupported RAGBench subset: {subset}. Use one of {RAGBENCH_SUBSETS}.")

    path = _raw_path(subset, split)
    if path.exists():
        return path
    if auto_download:
        return _download_ragbench(subset, split)
    raise FileNotFoundError(f"{path} does not exist. Enable auto_download or prepare it manually.")


def load_ragbench_examples(subset: str, split: str, num_examples: int, auto_download: bool = True) -> list[dict]:
    path = ensure_ragbench_dataset(subset, split, auto_download)
    examples = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            examples.append(json.loads(line))
            if len(examples) >= num_examples:
                break
    return examples


def iter_ragbench_queries(subset: str, split: str = "test"):
    path = ensure_ragbench_dataset(subset, split, auto_download=True)
    with path.open(encoding="utf-8") as f:
        for index, line in enumerate(f):
            if not line.strip():
                continue
            row = json.loads(line)
            question = str(row.get("input", "")).strip()
            if not question:
                continue
            yield {
                "query_id": f"ragbench:{subset}:{split}:{index}",
                "dataset": f"ragbench_{subset}",
                "index": index,
                "query": question,
                "reference_answer": row.get("response") or "",
                "source_row_id": row.get("id") or f"{subset}_{split}_{index}",
            }


def seed_ragbench_question_pool(subset: str, split: str, reset: bool = False, sample_limit: int | None = None) -> dict:
    init_db()
    init_dp3_cache_schema()
    inserted = 0
    with get_conn() as conn:
        if reset:
            conn.execute("DELETE FROM dp3_answerable_question_pool")
        for item in iter_ragbench_queries(subset, split):
            if sample_limit is not None and inserted >= sample_limit:
                break
            conn.execute(
                """INSERT OR REPLACE INTO dp3_answerable_question_pool
                   (route_id, question_text, route_type, embedding_json)
                   VALUES (?, ?, ?, ?)""",
                (
                    item["query_id"],
                    item["query"],
                    f"ragbench:{subset}:{split}",
                    _embedding_to_json(_embed(item["query"])),
                ),
            )
            inserted += 1
    return {"subset": subset, "split": split, "seeded_questions": inserted}


def list_local_ragbench_datasets() -> dict:
    datasets = []
    for subset in RAGBENCH_SUBSETS:
        for path in sorted((RAGBENCH_DATA_DIR / subset).glob("*.jsonl")):
            rows = 0
            with path.open(encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        rows += 1
            datasets.append({
                "family": "ragbench",
                "dataset": subset,
                "split": path.stem,
                "rows": rows,
                "size_kb": round(path.stat().st_size / 1024, 1),
            })
    return {"datasets": datasets, "total_rows": sum(item["rows"] for item in datasets)}


def prepare_dp3_ragbench(
    subset: str,
    num_examples: int,
    split: str = "test",
    auto_download: bool = True,
    reset: bool = False,
    reset_metadata: bool = False,
) -> dict:
    subset = subset.strip().lower()
    split = split.strip().lower()
    examples = load_ragbench_examples(subset, split, num_examples, auto_download=auto_download)
    if not examples:
        raise RuntimeError(f"No RAGBench examples found for subset={subset}, split={split}.")

    init_db()
    init_dp3_cache_schema()
    seed_answerable_question_pool(reset=False)
    seed_result = seed_ragbench_question_pool(subset, split, reset=False)

    source_id = f"dp3_ragbench_{subset}_{split}_{len(examples)}"
    if reset:
        _clear_source(source_id)
    elif reset_metadata:
        _clear_versioned_source(source_id)

    with get_conn() as conn:
        existing_base = conn.execute(
            "SELECT COUNT(*) AS cnt FROM dp3_evidence_units WHERE source_id=?",
            (source_id,),
        ).fetchone()["cnt"]
        existing_versioned = conn.execute(
            "SELECT COUNT(*) AS cnt FROM dp3_versioned_evidence_units WHERE source_id=?",
            (source_id,),
        ).fetchone()["cnt"]

    if existing_versioned:
        context_result = {
            "base_eu_count": existing_base,
            "context_unit_count": "reused",
            "version_rows": existing_versioned,
        }
    elif existing_base:
        context_result = _insert_versioned_units_from_base(source_id, len(examples))
    else:
        context_result = _insert_context_units(source_id, examples)

    return {
        "source_id": source_id,
        "family": "ragbench",
        "dataset": subset,
        "split": split,
        "num_examples": len(examples),
        "jsonl_path": str(_raw_path(subset, split)),
        **context_result,
        "route_pool": seed_result,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("subset", nargs="?", default="techqa", choices=RAGBENCH_SUBSETS)
    parser.add_argument("num_examples", nargs="?", type=int, default=20)
    parser.add_argument("--split", default="test")
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--reset-metadata", action="store_true")
    args = parser.parse_args()

    result = prepare_dp3_ragbench(
        subset=args.subset,
        num_examples=args.num_examples,
        split=args.split,
        auto_download=not args.no_download,
        reset=args.reset,
        reset_metadata=args.reset_metadata,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
