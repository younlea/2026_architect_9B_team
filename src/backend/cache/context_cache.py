import json
import time
import uuid

from backend.cache.answer_cache import (
    DEFAULT_CACHE_THRESHOLD,
    TOP_K_SOURCES,
    _build_prompt,
    _cosine,
    _embed,
    _embedding_from_json,
    _embedding_to_json,
    _elapsed_ms,
    _retrieve_context_units,
    _set_timing,
    _set_total_ms,
    _timer,
    init_dp3_cache_schema,
    validate_eu,
)
from backend.cache.cache_llm import get_dp3_answer, get_dp3_llm_provider, is_mock_llm
from backend.db.database import get_conn


CONTEXT_CACHE_SCHEMA = """
CREATE TABLE IF NOT EXISTS dp3_context_cache_entries (
    context_cache_id TEXT PRIMARY KEY,
    anchor_query_text TEXT,
    anchor_query_embedding_json TEXT,
    context_pack_text TEXT,
    scope TEXT,
    created_version TEXT,
    source_count INTEGER DEFAULT 0,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    ttl_seconds INTEGER DEFAULT 86400
);

CREATE TABLE IF NOT EXISTS dp3_context_cache_sources (
    context_cache_id TEXT,
    logical_eu_id TEXT,
    versioned_eu_id TEXT,
    eu_version TEXT,
    fingerprint TEXT,
    source_order INTEGER DEFAULT 0,
    PRIMARY KEY (context_cache_id, logical_eu_id),
    FOREIGN KEY (context_cache_id) REFERENCES dp3_context_cache_entries(context_cache_id)
);
"""


def init_context_cache_schema() -> None:
    init_dp3_cache_schema()
    with get_conn() as conn:
        conn.executescript(CONTEXT_CACHE_SCHEMA)


def clear_context_cache_for_source(source_id: str | None = None) -> dict:
    init_context_cache_schema()
    with get_conn() as conn:
        if source_id:
            rows = conn.execute(
                """SELECT DISTINCT cce.context_cache_id
                   FROM dp3_context_cache_entries cce
                   JOIN dp3_context_cache_sources ccs
                     ON ccs.context_cache_id = cce.context_cache_id
                   WHERE ccs.logical_eu_id LIKE ?""",
                (f"{source_id}:%",),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT context_cache_id FROM dp3_context_cache_entries"
            ).fetchall()

        cache_ids = [r["context_cache_id"] for r in rows]
        if cache_ids:
            placeholders = ",".join("?" for _ in cache_ids)
            conn.execute(
                f"DELETE FROM dp3_context_cache_sources WHERE context_cache_id IN ({placeholders})",
                cache_ids,
            )
            conn.execute(
                f"DELETE FROM dp3_context_cache_entries WHERE context_cache_id IN ({placeholders})",
                cache_ids,
            )
    return {"cleared_context_cache_entries": len(cache_ids)}


def _context_pack_text(sources: list[dict]) -> str:
    return "\n\n".join(f"[{s['logical_eu_id']}]\n{s['text']}" for s in sources)


def _store_context_cache(
    query: str,
    query_embedding: list[float],
    context_pack_text: str,
    user_scope: str,
    created_version: str,
    sources: list[dict],
) -> str:
    init_context_cache_schema()
    context_cache_id = f"ctx-{uuid.uuid4()}"
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO dp3_context_cache_entries
               (context_cache_id, anchor_query_text, anchor_query_embedding_json,
                context_pack_text, scope, created_version, source_count)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                context_cache_id,
                query,
                _embedding_to_json(query_embedding),
                context_pack_text,
                user_scope,
                created_version,
                len(sources),
            ),
        )
        for idx, source in enumerate(sources):
            conn.execute(
                """INSERT INTO dp3_context_cache_sources
                   (context_cache_id, logical_eu_id, versioned_eu_id,
                    eu_version, fingerprint, source_order)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    context_cache_id,
                    source["logical_eu_id"],
                    source.get("versioned_eu_id"),
                    source["version"],
                    source["fingerprint"],
                    idx,
                ),
            )
    return context_cache_id


def _find_context_cache_candidate(
    user_scope: str,
    query_embedding: list[float],
    timing: dict | None = None,
) -> dict | None:
    init_context_cache_schema()
    db_start = _timer()
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT context_cache_id, anchor_query_text, anchor_query_embedding_json,
                      context_pack_text, scope, created_version, source_count
               FROM dp3_context_cache_entries
               WHERE scope=?""",
            (user_scope,),
        ).fetchall()
    if timing is not None:
        timing["db_ms"] = _elapsed_ms(db_start)
        timing["candidate_count"] = len(rows)

    scoring_start = _timer()
    best = None
    for row in rows:
        score = _cosine(query_embedding, _embedding_from_json(row["anchor_query_embedding_json"]))
        if best is None or score > best["cache_similarity_score"]:
            best = dict(row)
            best["cache_similarity_score"] = round(score, 4)
    if timing is not None:
        timing["scoring_ms"] = _elapsed_ms(scoring_start)
        timing["total_ms"] = round(timing.get("db_ms", 0.0) + timing["scoring_ms"], 3)
    return best


def _cache_sources(context_cache_id: str) -> list[dict]:
    init_context_cache_schema()
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT logical_eu_id, versioned_eu_id, eu_version, fingerprint, source_order
               FROM dp3_context_cache_sources
               WHERE context_cache_id=?
               ORDER BY source_order""",
            (context_cache_id,),
        ).fetchall()
    return [dict(row) for row in rows]


def _validate_context_sources(
    context_cache_id: str,
    user_scope: str,
    requested_version: str,
) -> dict:
    sources = _cache_sources(context_cache_id)
    valid = []
    invalid = []
    for source in sources:
        result = validate_eu(
            source["logical_eu_id"],
            source["fingerprint"],
            user_scope,
            requested_version,
        )
        item = {**source, "validation_reason": result["reason"]}
        if result["valid"]:
            valid.append(item)
        else:
            invalid.append(item)
    return {
        "valid": len(invalid) == 0 and len(sources) > 0,
        "source_count": len(sources),
        "valid_sources": valid,
        "invalid_sources": invalid,
        "invalid_count": len(invalid),
    }


def _current_units_for_sources(
    sources: list[dict],
    user_scope: str,
    requested_version: str,
) -> list[dict]:
    if not sources:
        return []
    logical_ids = [s["logical_eu_id"] for s in sources]
    placeholders = ",".join("?" for _ in logical_ids)
    with get_conn() as conn:
        rows = conn.execute(
            f"""SELECT logical_eu_id, versioned_eu_id, version, fingerprint,
                       scope, text, embedding_json
                FROM dp3_versioned_evidence_units
                WHERE logical_eu_id IN ({placeholders})
                  AND version=?
                  AND scope=?
                  AND is_available=1""",
            [*logical_ids, requested_version, user_scope],
        ).fetchall()
    by_id = {row["logical_eu_id"]: dict(row) for row in rows}
    ordered = []
    for source in sources:
        row = by_id.get(source["logical_eu_id"])
        if row:
            ordered.append(row)
    return ordered


def _delta_retrieve(
    source_id: str,
    query_embedding: list[float],
    user_scope: str,
    requested_version: str,
    valid_sources: list[dict],
    invalid_sources: list[dict],
) -> dict:
    needed = len(invalid_sources)
    candidate_count = max(needed * 2, needed)
    valid_logical_ids = {s["logical_eu_id"] for s in valid_sources}
    invalid_logical_ids = {s["logical_eu_id"] for s in invalid_sources}
    excluded = valid_logical_ids | invalid_logical_ids

    retrieval_timing = {}
    candidates = _retrieve_context_units(
        source_id,
        query_embedding,
        user_scope,
        requested_version,
        top_k=candidate_count,
        timing=retrieval_timing,
    )
    filter_start = _timer()
    replacements = []
    seen = set()
    for candidate in candidates:
        logical_id = candidate["logical_eu_id"]
        if logical_id in excluded or logical_id in seen:
            continue
        replacements.append(candidate)
        seen.add(logical_id)
        if len(replacements) >= needed:
            break
    filter_ms = _elapsed_ms(filter_start)

    return {
        "needed": needed,
        "candidate_count": candidate_count,
        "replacement_count": len(replacements),
        "replacements": replacements,
        "timing": {
            "db_ms": retrieval_timing.get("db_ms", 0.0),
            "scoring_ms": retrieval_timing.get("scoring_ms", 0.0),
            "rerank_ms": retrieval_timing.get("rerank_ms", 0.0),
            "filter_ms": filter_ms,
            "total_ms": round(retrieval_timing.get("total_ms", 0.0) + filter_ms, 3),
        },
    }


def _store_context_log(thread_id: str, query: str, log: dict) -> None:
    init_context_cache_schema()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO dp3_answer_cache_logs
               (mode, thread_id, query_text, requested_version, user_scope, decision_reason,
                cache_hit, validation_passed, llm_mocked, llm_call_count,
                roi_rag_called, total_ms, log_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "incremental_context_cache",
                thread_id,
                query,
                log.get("requested_version"),
                log.get("user_scope"),
                log.get("decision_reason"),
                int(bool(log.get("cache_hit"))),
                int(bool(log.get("validation_passed"))),
                int(bool(log.get("llm_mocked"))),
                int(log.get("llm_call_count", 0)),
                int(bool(log.get("retrieval_called"))),
                int(log.get("total_ms", 0)),
                json.dumps(log, ensure_ascii=False),
            ),
        )


def _generate_and_store(
    source_id: str,
    query: str,
    query_embedding: list[float],
    sources: list[dict],
    user_scope: str,
    requested_version: str,
    model: str | None,
    llm_provider: str | None,
    log: dict,
    start: float,
) -> dict:
    prompt_start = _timer()
    context_pack = _context_pack_text(sources)
    prompt = _build_prompt(query, sources)
    _set_timing(log, "prompt_build_ms", _elapsed_ms(prompt_start))

    llm_start = _timer()
    answer = get_dp3_answer(prompt, model, llm_provider)
    _set_timing(log, "llm_ms", _elapsed_ms(llm_start))

    store_start = _timer()
    context_cache_id = _store_context_cache(
        query,
        query_embedding,
        context_pack,
        user_scope,
        requested_version,
        sources,
    )
    _set_timing(log, "cache_store_ms", _elapsed_ms(store_start))
    log.update({
        "context_cache_id": context_cache_id,
        "context_source_count": len(sources),
        "answer": answer,
        "llm_call_count": 1,
    })
    _set_total_ms(log, start)
    _store_context_log(source_id, query, log)
    return log


def _full_retrieval_and_store(
    source_id: str,
    query: str,
    query_embedding: list[float],
    user_scope: str,
    requested_version: str,
    model: str | None,
    llm_provider: str | None,
    log: dict,
    start: float,
) -> dict:
    retrieval_timing = {}
    sources = _retrieve_context_units(
        source_id,
        query_embedding,
        user_scope,
        requested_version,
        top_k=TOP_K_SOURCES,
        timing=retrieval_timing,
    )
    for key, value in retrieval_timing.items():
        if key.endswith("_ms"):
            _set_timing(log, f"full_retrieval_{key}", value)
    log["full_retrieval_candidate_count"] = retrieval_timing.get("candidate_count", 0)
    log["full_retrieval_top_k"] = retrieval_timing.get("top_k", len(sources))
    log.update({
        "cache_hit": False,
        "validation_passed": False,
        "retrieval_called": True,
        "full_retrieval": True,
        "fallback_source_count": len(sources),
    })
    return _generate_and_store(
        source_id,
        query,
        query_embedding,
        sources,
        user_scope,
        requested_version,
        model,
        llm_provider,
        log,
        start,
    )


def run_context_cache_query(
    source_id: str,
    query: str,
    user_scope: str = "A",
    requested_version: str = "V1",
    model: str | None = None,
    llm_provider: str | None = None,
    cache_threshold: float = DEFAULT_CACHE_THRESHOLD,
) -> dict:
    start = _timer()
    init_context_cache_schema()
    embedding_start = _timer()
    query_embedding = _embed(query)
    embedding_ms = _elapsed_ms(embedding_start)

    log = {
        "mode": "incremental_context_cache",
        "thread_id": source_id,
        "query": query,
        "user_scope": user_scope,
        "requested_version": requested_version,
        "cache_threshold": cache_threshold,
        "cache_hit": False,
        "validation_passed": False,
        "llm_provider": get_dp3_llm_provider(llm_provider),
        "llm_model": model,
        "llm_mocked": is_mock_llm(llm_provider),
        "llm_call_count": 0,
        "retrieval_called": False,
        "delta_retrieval_count": 0,
        "full_retrieval": False,
    }
    _set_timing(log, "embedding_ms", embedding_ms)

    cache_lookup_timing = {}
    candidate = _find_context_cache_candidate(
        user_scope,
        query_embedding,
        timing=cache_lookup_timing,
    )
    _set_timing(log, "cache_lookup_db_ms", cache_lookup_timing.get("db_ms", 0))
    _set_timing(log, "cache_lookup_scoring_ms", cache_lookup_timing.get("scoring_ms", 0))
    _set_timing(log, "cache_lookup_ms", cache_lookup_timing.get("total_ms", 0))
    log["cache_lookup_candidate_count"] = cache_lookup_timing.get("candidate_count", 0)
    if not candidate:
        log["decision_reason"] = "context_cache_candidate_not_found_full_fallback"
        return _full_retrieval_and_store(
            source_id,
            query,
            query_embedding,
            user_scope,
            requested_version,
            model,
            llm_provider,
            log,
            start,
        )

    log.update({
        "context_cache_candidate_id": candidate["context_cache_id"],
        "cache_similarity_score": candidate["cache_similarity_score"],
    })

    if candidate["cache_similarity_score"] < cache_threshold:
        log["decision_reason"] = "context_cache_similarity_below_threshold"
        return _full_retrieval_and_store(
            source_id,
            query,
            query_embedding,
            user_scope,
            requested_version,
            model,
            llm_provider,
            log,
            start,
        )

    validation_start = _timer()
    validation = _validate_context_sources(
        candidate["context_cache_id"],
        user_scope,
        requested_version,
    )
    _set_timing(log, "validation_ms", _elapsed_ms(validation_start))
    log["source_validation"] = {
        "source_count": validation["source_count"],
        "invalid_count": validation["invalid_count"],
        "invalid_sources": [
            {
                "logical_eu_id": s["logical_eu_id"],
                "reason": s["validation_reason"],
            }
            for s in validation["invalid_sources"]
        ],
    }

    if validation["valid"]:
        prompt_start = _timer()
        prompt = f"{candidate['context_pack_text']}\n\n[Question]\n{query}\n\n[Answer]"
        _set_timing(log, "prompt_build_ms", _elapsed_ms(prompt_start))
        llm_start = _timer()
        answer = get_dp3_answer(
            prompt,
            model,
            llm_provider,
        )
        _set_timing(log, "llm_ms", _elapsed_ms(llm_start))
        log.update({
            "cache_hit": True,
            "validation_passed": True,
            "decision_reason": "context_cache_hit_all_valid",
            "answer": answer,
            "llm_call_count": 1,
        })
        _set_total_ms(log, start)
        _store_context_log(source_id, query, log)
        return log

    total_count = validation["source_count"]
    invalid_count = validation["invalid_count"]
    if invalid_count * 2 >= total_count:
        log["decision_reason"] = "context_cache_invalid_ratio_full_fallback"
        return _full_retrieval_and_store(
            source_id,
            query,
            query_embedding,
            user_scope,
            requested_version,
            model,
            llm_provider,
            log,
            start,
        )

    current_units_start = _timer()
    valid_current = _current_units_for_sources(
        validation["valid_sources"],
        user_scope,
        requested_version,
    )
    _set_timing(log, "valid_current_lookup_ms", _elapsed_ms(current_units_start))
    delta = _delta_retrieve(
        source_id,
        query_embedding,
        user_scope,
        requested_version,
        validation["valid_sources"],
        validation["invalid_sources"],
    )
    log["delta_retrieval"] = {
        "needed": delta["needed"],
        "candidate_count": delta["candidate_count"],
        "replacement_count": delta["replacement_count"],
    }
    for key, value in delta.get("timing", {}).items():
        _set_timing(log, f"delta_retrieval_{key}", value)

    if len(valid_current) != len(validation["valid_sources"]) or delta["replacement_count"] < delta["needed"]:
        log["decision_reason"] = "context_cache_delta_insufficient_full_fallback"
        return _full_retrieval_and_store(
            source_id,
            query,
            query_embedding,
            user_scope,
            requested_version,
            model,
            llm_provider,
            log,
            start,
        )

    rebuilt_sources = valid_current + delta["replacements"]
    log.update({
        "cache_hit": True,
        "validation_passed": False,
        "retrieval_called": True,
        "delta_retrieval_count": delta["replacement_count"],
        "full_retrieval": False,
        "decision_reason": "context_cache_partial_invalid_delta_rebuilt",
    })
    return _generate_and_store(
        source_id,
        query,
        query_embedding,
        rebuilt_sources,
        user_scope,
        requested_version,
        model,
        llm_provider,
        log,
        start,
    )
