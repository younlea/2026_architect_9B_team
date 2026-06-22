"""
Seed DP3 answerable_question_pool from LongBench input questions.

Usage:
    python seed_dp3_question_pool.py --sample-rate 0.1 --reset
    python seed_dp3_question_pool.py --dataset multifieldqa_en --sample-rate 0.1
"""
import argparse
import json
import os
import random
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("SQLITE_DB_PATH", str(BASE_DIR / "data" / "poc.db"))
os.environ.setdefault("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma"))

sys.path.insert(0, str(BASE_DIR))

from backend.cache.answer_cache import _embed, _embedding_to_json, init_dp3_cache_schema
from backend.db.database import get_conn, init_db

DATA_DIR = BASE_DIR / "data" / "longbench"

DATASET_FALLBACK_QUESTIONS = {
    "gov_report": "Summarize the provided government report.",
    "gov_report_e": "Summarize the provided government report.",
    "multi_news": "Summarize the provided news articles.",
    "multi_news_e": "Summarize the provided news articles.",
    "vcsum": "请总结给定的会议或对话内容。",
    "lcc": "Complete the next line of code based on the provided code context.",
    "lcc_e": "Complete the next line of code based on the provided code context.",
    "passage_count": "Count how many passages in the provided context satisfy the task.",
    "passage_count_e": "Count how many passages in the provided context satisfy the task.",
}


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8-sig") as f:
        for index, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            yield index, json.loads(line)


def _dataset_files(dataset: str | None, include_smoke: bool) -> list[Path]:
    if dataset:
        path = DATA_DIR / f"{dataset}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"{path} 파일이 없습니다.")
        return [path]

    files = sorted(DATA_DIR.glob("*.jsonl"))
    if not include_smoke:
        files = [path for path in files if not path.name.startswith("dp3_smoke")]
    return files


def _sample_items(items: list[tuple[int, dict]], sample_rate: float, min_count: int, seed: int):
    if sample_rate >= 1.0:
        return items
    sample_count = max(min_count, int(round(len(items) * sample_rate)))
    sample_count = min(sample_count, len(items))
    rng = random.Random(seed)
    return sorted(rng.sample(items, sample_count), key=lambda item: item[0])


def _question_text(dataset_name: str, row: dict) -> str:
    question = str(row.get("input", "")).strip()
    if question:
        return question
    return DATASET_FALLBACK_QUESTIONS.get(
        dataset_name,
        "Answer the task using the provided context.",
    )


def seed_question_pool(
    dataset: str | None,
    sample_rate: float,
    min_per_dataset: int,
    seed: int,
    reset: bool,
    include_smoke: bool,
) -> dict:
    init_db()
    init_dp3_cache_schema()
    files = _dataset_files(dataset, include_smoke)
    inserted = 0
    total_questions = 0
    by_dataset = {}

    with get_conn() as conn:
        if reset:
            conn.execute("DELETE FROM dp3_answerable_question_pool")

        for path in files:
            dataset_name = path.stem
            items = [
                (index, row)
                for index, row in _iter_jsonl(path)
                if _question_text(dataset_name, row)
            ]
            total_questions += len(items)
            sampled = _sample_items(items, sample_rate, min_per_dataset, seed)
            dataset_inserted = 0

            for index, row in sampled:
                question = _question_text(dataset_name, row)
                route_id = f"longbench:{dataset_name}:{index}"
                conn.execute(
                    """INSERT OR REPLACE INTO dp3_answerable_question_pool
                       (route_id, question_text, route_type, embedding_json)
                       VALUES (?, ?, ?, ?)""",
                    (
                        route_id,
                        question,
                        f"longbench:{dataset_name}",
                        _embedding_to_json(_embed(question)),
                    ),
                )
                inserted += 1
                dataset_inserted += 1

            by_dataset[dataset_name] = {
                "total_questions": len(items),
                "seeded_questions": dataset_inserted,
            }

    return {
        "dataset": dataset or "all",
        "sample_rate": sample_rate,
        "min_per_dataset": min_per_dataset,
        "source_files": len(files),
        "total_questions": total_questions,
        "seeded_questions": inserted,
        "by_dataset": by_dataset,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--sample-rate", type=float, default=0.1)
    parser.add_argument("--min-per-dataset", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--include-smoke", action="store_true")
    args = parser.parse_args()

    if args.sample_rate <= 0:
        raise ValueError("--sample-rate must be positive")

    result = seed_question_pool(
        dataset=args.dataset,
        sample_rate=args.sample_rate,
        min_per_dataset=args.min_per_dataset,
        seed=args.seed,
        reset=args.reset,
        include_smoke=args.include_smoke,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
