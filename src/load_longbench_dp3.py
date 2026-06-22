"""
DP3 전용 LongBench 준비 스크립트.

기존 load_longbench.py, DP1/DP2 benchmark/session/thread 테이블은 건드리지 않는다.

Usage:
    python load_longbench_dp3.py multifieldqa_en 10
    python load_longbench_dp3.py multifieldqa_en 10 --no-download
    python load_longbench_dp3.py multifieldqa_en 10 --reset
    python load_longbench_dp3.py multifieldqa_en 10 --reset-metadata
"""
import argparse
import json
import os
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
os.environ.setdefault("SQLITE_DB_PATH", str(BASE_DIR / "data" / "poc.db"))
os.environ.setdefault("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chroma"))

sys.path.insert(0, str(BASE_DIR))

from backend.cache.answer_cache import (
    _chunk_text,
    _embed,
    _embedding_to_json,
    _mutation_type_for_version,
    _version_rows_for_unit,
    clear_answer_cache_for_source,
    init_dp3_cache_schema,
    seed_answerable_question_pool,
)
from backend.db.database import get_conn, init_db

LONGBENCH_URL = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
DATA_DIR = BASE_DIR / "data" / "longbench"
ZIP_PATH = BASE_DIR / "data" / "longbench.zip"


def _download_longbench() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_PATH.exists():
        print(f"Downloading LongBench data.zip -> {ZIP_PATH}")
        urllib.request.urlretrieve(LONGBENCH_URL, ZIP_PATH)
    print(f"Extracting {ZIP_PATH} -> {DATA_DIR}")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)

    # zip 내부가 data/*.jsonl 형태인 경우 root로 복사해 기존 loader 관례와 맞춘다.
    for path in DATA_DIR.rglob("*.jsonl"):
        target = DATA_DIR / path.name
        if path != target and not target.exists():
            shutil.copy2(path, target)


def _ensure_dataset(dataset_name: str, auto_download: bool) -> Path:
    jsonl_path = DATA_DIR / f"{dataset_name}.jsonl"
    if jsonl_path.exists():
        return jsonl_path
    if auto_download:
        _download_longbench()
    if jsonl_path.exists():
        return jsonl_path
    raise FileNotFoundError(
        f"{jsonl_path} 파일이 없습니다. 네트워크가 막혀 있다면 직접 src/data/longbench에 넣어 주세요."
    )


def _load_examples(jsonl_path: Path, num_examples: int) -> list[dict]:
    examples = []
    with jsonl_path.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if len(examples) >= num_examples:
                break
    return examples


def _scope_for_example(example_index: int, total_examples: int) -> str:
    return "A" if example_index < max(1, total_examples // 2) else "B"


def _clear_source(source_id: str) -> None:
    clear_answer_cache_for_source(source_id)
    with get_conn() as conn:
        conn.execute("DELETE FROM dp3_versioned_evidence_units WHERE source_id=?", (source_id,))
        conn.execute("DELETE FROM dp3_evidence_units WHERE source_id=?", (source_id,))
        conn.execute("DELETE FROM dp3_context_units WHERE source_example_id=?", (source_id,))
        conn.execute("DELETE FROM dp3_query_sets WHERE source_id=?", (source_id,))


def _clear_versioned_source(source_id: str) -> None:
    clear_answer_cache_for_source(source_id)
    with get_conn() as conn:
        conn.execute("DELETE FROM dp3_versioned_evidence_units WHERE source_id=?", (source_id,))
        conn.execute("DELETE FROM dp3_query_sets WHERE source_id=?", (source_id,))


def _chunk_fallback_eus(source_id: str, ex_idx: int, context: str) -> dict:
    chunks = _chunk_text(context)
    return {
        "builder": "fallback_chunk",
        "regime": "N/A",
        "segment_count": len(chunks),
        "evidence_units": [
            {
                "roi_eu_id": f"{source_id}_ex_{ex_idx}_chunk_{chunk_idx}",
                "text": chunk,
                "embedding": _embed(chunk),
                "segments": [chunk],
                "segment_indices": [chunk_idx],
                "segment_count": 1,
                "re": 0.0,
                "de": 1.0,
                "regime": "N/A",
            }
            for chunk_idx, chunk in enumerate(chunks)
        ],
    }


def _build_dp3_eus(source_id: str, ex_idx: int, context: str) -> dict:
    try:
        from backend.rag.roi_rag import build_evidence_units_for_text

        result = build_evidence_units_for_text(
            context,
            id_prefix=f"{source_id}_ex_{ex_idx}",
            use_summary=False,
        )
        result["builder"] = "roi_rag"
        return result
    except Exception as exc:
        print(
            "[WARN] ROI-RAG EU build failed; using chunk fallback for "
            f"{source_id}:ex:{ex_idx}. reason={type(exc).__name__}: {exc}"
        )
        return _chunk_fallback_eus(source_id, ex_idx, context)


def _insert_context_units(source_id: str, examples: list[dict]) -> dict:
    total_units = 0
    total_rows = 0
    builder_counts = Counter()
    regime_counts = Counter()
    with get_conn() as conn:
        global_unit_index = 0
        for ex_idx, ex in enumerate(examples):
            context = ex.get("context", "")
            eu_result = _build_dp3_eus(source_id, ex_idx, context)
            evidence_units = eu_result.get("evidence_units", [])
            builder_counts[eu_result.get("builder", "unknown")] += 1
            regime_counts[eu_result.get("regime", "unknown")] += 1
            scope = _scope_for_example(ex_idx, len(examples))
            for eu_idx, eu in enumerate(evidence_units):
                eu_text = eu.get("text", "")
                if not eu_text.strip():
                    continue
                base_eu_id = f"{source_id}:ex:{ex_idx}:base_eu:{eu_idx}"
                logical_eu_id = f"{source_id}:ex:{ex_idx}:logical_eu:{eu_idx}"
                embedding = eu.get("embedding") or _embed(eu_text)
                embedding_json = _embedding_to_json([float(x) for x in embedding])
                roi_metadata = {
                    "builder": eu_result.get("builder", "unknown"),
                    "roi_eu_id": eu.get("roi_eu_id"),
                    "segment_indices": eu.get("segment_indices", []),
                    "segment_count": eu.get("segment_count", 0),
                    "re": eu.get("re"),
                    "de": eu.get("de"),
                    "regime": eu.get("regime", eu_result.get("regime")),
                }
                conn.execute(
                    """INSERT OR REPLACE INTO dp3_evidence_units
                       (base_eu_id, source_id, source_example_id, eu_index, text, embedding_json, roi_metadata_json)
                       VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (
                        base_eu_id,
                        source_id,
                        f"{source_id}:ex:{ex_idx}",
                        eu_idx,
                        eu_text,
                        embedding_json,
                        json.dumps(roi_metadata, ensure_ascii=False),
                    ),
                )
                for version, fingerprint in _version_rows_for_unit(global_unit_index, eu_text):
                    versioned_eu_id = f"{logical_eu_id}:{version}"
                    conn.execute(
                        """INSERT OR REPLACE INTO dp3_versioned_evidence_units
                           (versioned_eu_id, base_eu_id, logical_eu_id, source_id, source_example_id,
                            version, scope, text, fingerprint, embedding_json, mutation_type, is_available)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                        (
                            versioned_eu_id,
                            base_eu_id,
                            logical_eu_id,
                            source_id,
                            f"{source_id}:ex:{ex_idx}",
                            version,
                            scope,
                            eu_text,
                            fingerprint,
                            embedding_json,
                            _mutation_type_for_version(global_unit_index, version, eu_text, fingerprint),
                        ),
                    )
                    total_rows += 1
                total_units += 1
                global_unit_index += 1
    return {
        "context_unit_count": total_units,
        "version_rows": total_rows,
        "eu_builders": dict(builder_counts),
        "roi_regimes": dict(regime_counts),
    }


def _example_index_from_source_example_id(value: str) -> int:
    try:
        return int(value.rsplit(":ex:", 1)[1])
    except (IndexError, ValueError):
        return 0


def _insert_versioned_units_from_base(source_id: str, total_examples: int) -> dict:
    total_rows = 0
    with get_conn() as conn:
        base_rows = conn.execute(
            """SELECT base_eu_id, source_example_id, eu_index, text, embedding_json
               FROM dp3_evidence_units
               WHERE source_id=?
               ORDER BY source_example_id, eu_index""",
            (source_id,),
        ).fetchall()
        for global_unit_index, row in enumerate(base_rows):
            ex_idx = _example_index_from_source_example_id(row["source_example_id"])
            scope = _scope_for_example(ex_idx, total_examples)
            logical_eu_id = f"{source_id}:ex:{ex_idx}:logical_eu:{row['eu_index']}"
            for version, fingerprint in _version_rows_for_unit(global_unit_index, row["text"]):
                versioned_eu_id = f"{logical_eu_id}:{version}"
                conn.execute(
                    """INSERT OR REPLACE INTO dp3_versioned_evidence_units
                       (versioned_eu_id, base_eu_id, logical_eu_id, source_id, source_example_id,
                        version, scope, text, fingerprint, embedding_json, mutation_type, is_available)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)""",
                    (
                        versioned_eu_id,
                        row["base_eu_id"],
                        logical_eu_id,
                        source_id,
                        row["source_example_id"],
                        version,
                        scope,
                        row["text"],
                        fingerprint,
                        row["embedding_json"],
                        _mutation_type_for_version(global_unit_index, version, row["text"], fingerprint),
                    ),
                )
                total_rows += 1
    return {"context_unit_count": len(base_rows), "version_rows": total_rows, "metadata_rebuilt": True}


def _insert_query_sets(source_id: str, examples: list[dict]) -> dict:
    rows = []
    for scope in ("A", "B"):
        rows.extend([
            (
                f"{source_id}:{scope}:same:v1",
                "same",
                "V1에서 이 문서의 핵심 내용을 요약해줘.",
                scope,
                "V1",
                "first_run_miss_then_hit",
            ),
            (
                f"{source_id}:{scope}:same:v1-repeat",
                "same",
                "V1에서 이 문서의 핵심 내용을 요약해줘.",
                scope,
                "V1",
                "cache_hit_expected_after_seed",
            ),
            (
                f"{source_id}:{scope}:paraphrase:v1",
                "paraphrase",
                "V1에서 주요 내용을 간단히 정리해줘.",
                scope,
                "V1",
                "semantic_cache_reuse_possible",
            ),
            (
                f"{source_id}:{scope}:near_miss:v1",
                "near_miss",
                "V1에서 이 문서의 예외나 한계를 알려줘.",
                scope,
                "V1",
                "over_hit_risk_check",
            ),
            (
                f"{source_id}:{scope}:version:v2",
                "version_mismatch",
                "V2에서 이 문서의 핵심 내용을 요약해줘.",
                scope,
                "V2",
                "fingerprint_or_missing_version_fallback_possible",
            ),
        ])

    with get_conn() as conn:
        for query_id, query_type, query_text, user_scope, requested_version, expected in rows:
            conn.execute(
                """INSERT OR REPLACE INTO dp3_query_sets
                   (query_id, source_id, query_type, query_text, user_scope, requested_version, expected_behavior)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (query_id, source_id, query_type, query_text, user_scope, requested_version, expected),
            )
    return {"query_count": len(rows)}


def prepare_dp3_longbench(
    dataset_name: str,
    num_examples: int,
    auto_download: bool,
    reset: bool,
    reset_metadata: bool = False,
) -> dict:
    jsonl_path = _ensure_dataset(dataset_name, auto_download)
    init_db()
    init_dp3_cache_schema()
    seed_answerable_question_pool(reset=False)
    examples = _load_examples(jsonl_path, num_examples)
    if not examples:
        raise RuntimeError(f"{jsonl_path}에서 예제를 읽지 못했습니다.")

    source_id = f"dp3_longbench_{dataset_name}_{len(examples)}"
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
    query_result = _insert_query_sets(source_id, examples)

    return {
        "source_id": source_id,
        "dataset": dataset_name,
        "num_examples": len(examples),
        "jsonl_path": str(jsonl_path),
        **context_result,
        **query_result,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", nargs="?", default="multifieldqa_en")
    parser.add_argument("num_examples", nargs="?", type=int, default=5)
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="LongBench 파일이 없을 때 자동 다운로드하지 않는다.",
    )
    parser.add_argument("--reset", action="store_true")
    parser.add_argument(
        "--reset-metadata",
        action="store_true",
        help="원본 EU는 유지하고 scope/version/fingerprint 실험 metadata만 다시 만든다.",
    )
    args = parser.parse_args()

    result = prepare_dp3_longbench(
        args.dataset_name,
        args.num_examples,
        auto_download=not args.no_download,
        reset=args.reset,
        reset_metadata=args.reset_metadata,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
