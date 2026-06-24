"""
Build DP3 TC3/TC4 query assets from RAGBench eManual.

Outputs are local generated artifacts under src/data/ragbench/emanual/ and are
git-ignored. The scripts are committed so the workload can be regenerated.

Usage:
    python build_dp3_ragbench_query_assets.py --subset emanual --split test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("SQLITE_DB_PATH", str(BASE_DIR / "data" / "poc.db"))
os.environ.setdefault("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma"))

sys.path.insert(0, str(BASE_DIR))

from backend.cache.answer_cache import _cosine, _embed  # noqa: E402
from load_ragbench_dp3 import RAGBENCH_DATA_DIR, iter_ragbench_queries  # noqa: E402


def build_query_assets(
    subset: str = "emanual",
    split: str = "test",
    seed: int = 42,
    cache_threshold: float = 0.86,
    near_miss_min: float = 0.55,
    tc4_min_similarity: float | None = None,
    max_sets: int = 50,
    max_pairs: int = 50,
) -> dict:
    tc4_min_similarity = cache_threshold if tc4_min_similarity is None else tc4_min_similarity
    rng = random.Random(seed)
    items = list(iter_ragbench_queries(subset, split))
    if len(items) < 4:
        raise RuntimeError(f"Not enough queries for {subset}/{split}: {len(items)}")

    embeddings = [_embed(item["query"]) for item in items]
    pair_scores = _pair_scores(items, embeddings)
    tc3_rows = _build_tc3_rows(
        items,
        pair_scores,
        rng,
        near_miss_min=near_miss_min,
        cache_threshold=cache_threshold,
        max_sets=max_sets,
    )
    tc4_rows = _build_tc4_pairs(
        items,
        pair_scores,
        min_similarity=tc4_min_similarity,
        max_pairs=max_pairs,
    )

    out_dir = RAGBENCH_DATA_DIR / subset
    out_dir.mkdir(parents=True, exist_ok=True)
    tc3_path = out_dir / f"{split}_tc3_query_sets.jsonl"
    legacy_tc2_path = out_dir / f"{split}_tc2_query_sets.jsonl"
    tc4_path = out_dir / f"{split}_tc4_query_pairs.jsonl"
    meta_path = out_dir / f"{split}_query_assets_meta.json"
    _write_jsonl(tc3_path, tc3_rows)
    _write_jsonl(legacy_tc2_path, tc3_rows)
    _write_jsonl(tc4_path, tc4_rows)

    result = {
        "subset": subset,
        "split": split,
        "source_queries": len(items),
        "cache_threshold": cache_threshold,
        "near_miss_min": near_miss_min,
        "tc4_min_similarity": tc4_min_similarity,
        "max_sets": max_sets,
        "max_pairs": max_pairs,
        "tc3_path": str(tc3_path),
        "tc2_path": str(legacy_tc2_path),
        "tc3_rows": len(tc3_rows),
        "tc3_groups": len({row["group_id"] for row in tc3_rows}),
        "tc3_by_role": _count_by(tc3_rows, "role"),
        "tc4_path": str(tc4_path),
        "tc4_pairs": len(tc4_rows),
        "tc4_similarity_min": round(min((row["similarity"] for row in tc4_rows), default=0.0), 4),
        "tc4_similarity_max": round(max((row["similarity"] for row in tc4_rows), default=0.0), 4),
        "meta_path": str(meta_path),
    }
    meta_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def ensure_query_assets(
    subset: str = "emanual",
    split: str = "test",
    seed: int = 42,
    cache_threshold: float = 0.86,
    near_miss_min: float = 0.55,
    tc4_min_similarity: float | None = None,
    max_sets: int = 50,
    max_pairs: int = 50,
    force: bool = False,
) -> dict:
    tc4_min_similarity = cache_threshold if tc4_min_similarity is None else tc4_min_similarity
    out_dir = RAGBENCH_DATA_DIR / subset
    tc3_path = out_dir / f"{split}_tc3_query_sets.jsonl"
    legacy_tc2_path = out_dir / f"{split}_tc2_query_sets.jsonl"
    tc4_path = out_dir / f"{split}_tc4_query_pairs.jsonl"
    meta_path = out_dir / f"{split}_query_assets_meta.json"
    expected = {
        "subset": subset,
        "split": split,
        "seed": seed,
        "cache_threshold": cache_threshold,
        "near_miss_min": near_miss_min,
        "tc4_min_similarity": tc4_min_similarity,
        "max_sets": max_sets,
        "max_pairs": max_pairs,
    }
    query_set_path = tc3_path if tc3_path.exists() else legacy_tc2_path
    if not force and query_set_path.exists() and tc4_path.exists() and meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
        if all(meta.get(key) == value for key, value in expected.items()):
            meta.setdefault("tc3_path", str(query_set_path))
            meta.setdefault("tc3_rows", meta.get("tc2_rows", 0))
            meta.setdefault("tc3_groups", meta.get("tc2_groups", 0))
            meta.setdefault("tc3_by_role", meta.get("tc2_by_role", {}))
            meta["reused"] = True
            return meta

    result = build_query_assets(
        subset=subset,
        split=split,
        seed=seed,
        cache_threshold=cache_threshold,
        near_miss_min=near_miss_min,
        tc4_min_similarity=tc4_min_similarity,
        max_sets=max_sets,
        max_pairs=max_pairs,
    )
    result.update(expected)
    result["reused"] = False
    Path(result["meta_path"]).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _pair_scores(items: list[dict], embeddings: list[list[float]]) -> list[dict]:
    rows = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            rows.append({
                "i": i,
                "j": j,
                "similarity": _cosine(embeddings[i], embeddings[j]),
                "answer_jaccard": _token_jaccard(
                    items[i].get("reference_answer", ""),
                    items[j].get("reference_answer", ""),
                ),
            })
    rows.sort(key=lambda row: row["similarity"], reverse=True)
    return rows


def _build_tc3_rows(
    items: list[dict],
    pair_scores: list[dict],
    rng: random.Random,
    near_miss_min: float,
    cache_threshold: float,
    max_sets: int,
) -> list[dict]:
    by_seed: dict[int, list[dict]] = {}
    for row in pair_scores:
        by_seed.setdefault(row["i"], []).append(row)
        by_seed.setdefault(row["j"], []).append({"i": row["j"], "j": row["i"], **{k: v for k, v in row.items() if k not in {"i", "j"}}})

    used_groups = 0
    rows = []
    seed_order = list(range(len(items)))
    rng.shuffle(seed_order)
    for seed_idx in seed_order:
        candidates = by_seed.get(seed_idx, [])
        seed_query_key = _normalize_query(items[seed_idx]["query"])
        similar = next(
            (
                row for row in sorted(candidates, key=lambda r: r["similarity"], reverse=True)
                if row["similarity"] >= cache_threshold
                and _normalize_query(items[row["j"]]["query"]) != seed_query_key
            ),
            None,
        )
        near_miss = next(
            (
                row for row in sorted(candidates, key=lambda r: abs(r["similarity"] - cache_threshold))
                if near_miss_min <= row["similarity"] < cache_threshold
                and _normalize_query(items[row["j"]]["query"]) != seed_query_key
            ),
            None,
        )
        random_pool = [idx for idx in range(len(items)) if idx != seed_idx]
        random_idx = rng.choice(random_pool)
        if similar is None or near_miss is None:
            continue

        group_id = f"tc3:{used_groups + 1:03d}:{items[seed_idx]['source_row_id']}"
        rows.append(_tc3_row(group_id, "same", items[seed_idx], seed_idx, 1.0))
        rows.append(_tc3_row(group_id, "similar", items[similar["j"]], similar["j"], similar["similarity"]))
        rows.append(_tc3_row(group_id, "near_miss", items[near_miss["j"]], near_miss["j"], near_miss["similarity"]))
        rows.append(_tc3_row(group_id, "random", items[random_idx], random_idx, None))
        used_groups += 1
        if used_groups >= max_sets:
            break
    return rows


def _tc3_row(group_id: str, role: str, item: dict, index: int, similarity: float | None) -> dict:
    return {
        "query_id": f"{group_id}:{role}:{index}",
        "group_id": group_id,
        "role": role,
        "dataset": item["dataset"],
        "index": item["index"],
        "query": item["query"],
        "reference_answer": item.get("reference_answer", ""),
        "source_row_id": item.get("source_row_id", ""),
        "similarity_to_seed": None if similarity is None else round(float(similarity), 4),
    }


def _build_tc4_pairs(
    items: list[dict],
    pair_scores: list[dict],
    min_similarity: float,
    max_pairs: int,
) -> list[dict]:
    rows = []
    used: set[tuple[str, str]] = set()
    for row in pair_scores:
        if row["similarity"] < min_similarity:
            break
        left = items[row["i"]]
        right = items[row["j"]]
        if _normalize_query(left["query"]) == _normalize_query(right["query"]):
            continue
        if left["source_row_id"] == right["source_row_id"]:
            continue
        if row["answer_jaccard"] >= 0.72:
            continue
        key = tuple(sorted([left["source_row_id"], right["source_row_id"]]))
        if key in used:
            continue
        used.add(key)
        rows.append({
            "pair_id": f"tc4:{left['source_row_id']}:{right['source_row_id']}",
            "dataset": left["dataset"],
            "left": _pair_item(left),
            "right": _pair_item(right),
            "similarity": round(float(row["similarity"]), 4),
            "answer_jaccard": round(float(row["answer_jaccard"]), 4),
            "expected_behavior": "A may reuse identical answer; B should regenerate from cached context.",
        })
        if len(rows) >= max_pairs:
            break
    return rows


def _pair_item(item: dict) -> dict:
    return {
        "query_id": item["query_id"],
        "index": item["index"],
        "query": item["query"],
        "reference_answer": item.get("reference_answer", ""),
        "source_row_id": item.get("source_row_id", ""),
    }


def _token_jaccard(left: str, right: str) -> float:
    a = set(re.findall(r"[A-Za-z0-9가-힣]+", left.lower()))
    b = set(re.findall(r"[A-Za-z0-9가-힣]+", right.lower()))
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _normalize_query(value: str) -> str:
    return " ".join(re.findall(r"[A-Za-z0-9가-힣]+", value.lower()))


def _count_by(rows: list[dict], key: str) -> dict:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row.get(key, "unknown"))
        counts[value] = counts.get(value, 0) + 1
    return counts


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset", default="emanual")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cache-threshold", type=float, default=0.86)
    parser.add_argument("--near-miss-min", type=float, default=0.55)
    parser.add_argument("--tc4-min-similarity", type=float, default=None)
    parser.add_argument("--max-sets", type=int, default=50)
    parser.add_argument("--max-pairs", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    result = ensure_query_assets(
        subset=args.subset,
        split=args.split,
        seed=args.seed,
        cache_threshold=args.cache_threshold,
        near_miss_min=args.near_miss_min,
        tc4_min_similarity=args.tc4_min_similarity,
        max_sets=args.max_sets,
        max_pairs=args.max_pairs,
        force=args.force,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
