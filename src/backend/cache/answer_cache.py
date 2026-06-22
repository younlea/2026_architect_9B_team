import hashlib
import json
import math
import time
import uuid
from typing import Optional

from backend.cache.cache_llm import get_dp3_answer, is_mock_llm
from backend.db.database import get_conn, get_thread_text

DEFAULT_ROUTES = [
    ("summarize_document", "이 문서의 핵심 내용을 요약해줘.", "summarize"),
    ("key_points", "이 문서의 주요 포인트를 정리해줘.", "summary"),
    ("definition", "이 문서의 핵심 개념을 설명해줘.", "definition"),
    ("fact_check", "이 문서의 주요 사실을 확인해줘.", "fact_check"),
]

VERSIONS = ("V1", "V2", "V3")
DEFAULT_ROUTE_THRESHOLD = 0.70
DEFAULT_CACHE_THRESHOLD = 0.86
TOP_K_SOURCES = 5


DP3_SCHEMA = """
CREATE TABLE IF NOT EXISTS dp3_context_units (
    logical_eu_id TEXT,
    version TEXT,
    fingerprint TEXT,
    scope TEXT,
    text TEXT,
    source_example_id TEXT,
    embedding_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (logical_eu_id, version)
);

CREATE TABLE IF NOT EXISTS dp3_evidence_units (
    base_eu_id TEXT PRIMARY KEY,
    source_id TEXT,
    source_example_id TEXT,
    eu_index INTEGER,
    text TEXT,
    embedding_json TEXT,
    roi_metadata_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dp3_versioned_evidence_units (
    versioned_eu_id TEXT PRIMARY KEY,
    base_eu_id TEXT,
    logical_eu_id TEXT,
    source_id TEXT,
    source_example_id TEXT,
    version TEXT,
    scope TEXT,
    text TEXT,
    fingerprint TEXT,
    embedding_json TEXT,
    mutation_type TEXT DEFAULT 'original',
    is_available INTEGER DEFAULT 1,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(logical_eu_id, version),
    FOREIGN KEY (base_eu_id) REFERENCES dp3_evidence_units(base_eu_id)
);

CREATE TABLE IF NOT EXISTS dp3_answerable_question_pool (
    route_id TEXT PRIMARY KEY,
    question_text TEXT,
    route_type TEXT,
    embedding_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dp3_answer_cache_entries (
    cache_id TEXT PRIMARY KEY,
    route_id TEXT,
    query_text TEXT,
    query_embedding_json TEXT,
    answer_text TEXT,
    scope TEXT,
    cache_version TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    ttl_seconds INTEGER DEFAULT 86400
);

CREATE TABLE IF NOT EXISTS dp3_answer_cache_sources (
    cache_id TEXT,
    logical_eu_id TEXT,
    versioned_eu_id TEXT,
    eu_version TEXT,
    fingerprint TEXT,
    source_order INTEGER DEFAULT 0,
    PRIMARY KEY (cache_id, logical_eu_id),
    FOREIGN KEY (cache_id) REFERENCES dp3_answer_cache_entries(cache_id)
);

CREATE TABLE IF NOT EXISTS dp3_answer_cache_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    mode TEXT,
    thread_id TEXT,
    query_text TEXT,
    requested_version TEXT,
    user_scope TEXT,
    decision_reason TEXT,
    cache_hit INTEGER DEFAULT 0,
    validation_passed INTEGER DEFAULT 0,
    llm_mocked INTEGER DEFAULT 1,
    llm_call_count INTEGER DEFAULT 0,
    roi_rag_called INTEGER DEFAULT 0,
    total_ms INTEGER DEFAULT 0,
    log_json TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS dp3_query_sets (
    query_id TEXT PRIMARY KEY,
    source_id TEXT,
    query_type TEXT,
    query_text TEXT,
    user_scope TEXT,
    requested_version TEXT,
    expected_behavior TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);
"""


def init_dp3_cache_schema() -> None:
    with get_conn() as conn:
        conn.executescript(DP3_SCHEMA)
        _ensure_dp3_schema_migrations(conn)


def _ensure_dp3_schema_migrations(conn) -> None:
    columns = {
        row["name"]
        for row in conn.execute("PRAGMA table_info(dp3_answer_cache_sources)").fetchall()
    }
    if "versioned_eu_id" not in columns:
        conn.execute("ALTER TABLE dp3_answer_cache_sources ADD COLUMN versioned_eu_id TEXT")


def _chunk_text(text: str, chunk_size: int = 512, overlap: int = 80) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks


def _ef():
    from backend.rag import _ef as _shared_ef
    return _shared_ef.get()


def _hash_embed(text: str, dim: int = 384) -> list[float]:
    vector = [0.0] * dim
    tokens = text.lower().split()
    if not tokens:
        tokens = [text.lower()]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i, byte in enumerate(digest):
            idx = (byte + i * 17) % dim
            vector[idx] += 1.0
    norm = math.sqrt(sum(x * x for x in vector)) + 1e-10
    return [float(x / norm) for x in vector]


def _embed(text: str) -> list[float]:
    try:
        return [float(x) for x in _ef()([text])[0]]
    except ModuleNotFoundError:
        return _hash_embed(text)


def _embedding_to_json(embedding: list[float]) -> str:
    return json.dumps(embedding, separators=(",", ":"))


def _embedding_from_json(value: str) -> list[float]:
    return [float(x) for x in json.loads(value)]


def _cosine(a: list[float], b: list[float]) -> float:
    denom = (math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(x * x for x in b))) + 1e-10
    return float(sum(x * y for x, y in zip(a, b)) / denom)


def _fingerprint(text: str, salt: str = "") -> str:
    return hashlib.sha256(f"{salt}\n{text}".encode("utf-8")).hexdigest()


def _scope_for_index(index: int, total: int) -> str:
    return "A" if index < max(1, total // 2) else "B"


def _version_rows_for_unit(index: int, text: str) -> list[tuple[str, str]]:
    rows = [("V1", _fingerprint(text))]
    if index % 3 != 2:
        salt = "v2-changed" if index % 5 == 0 else ""
        rows.append(("V2", _fingerprint(text, salt)))
    if index % 6 in {0, 1, 3}:
        salt = "v3-changed" if index % 7 == 0 else ("v2-changed" if index % 5 == 0 else "")
        rows.append(("V3", _fingerprint(text, salt)))
    return rows


def _mutation_type_for_version(index: int, version: str, text: str, fingerprint: str) -> str:
    if version == "V1":
        return "original"
    base_fingerprint = _fingerprint(text)
    if fingerprint != base_fingerprint:
        return "modified"
    return "copied"


def setup_context_units_from_thread(thread_id: str, reset: bool = False) -> dict:
    """Create shared DP3 context_units from an existing thread.

    The synthetic scope/version/fingerprint rules are deterministic so A/B tests
    can reuse the exact same source EU set.
    """
    init_dp3_cache_schema()
    with get_conn() as conn:
        if not reset:
            existing = conn.execute(
                """SELECT COUNT(DISTINCT logical_eu_id) AS unit_count,
                          COUNT(*) AS row_count
                   FROM dp3_versioned_evidence_units
                   WHERE source_id=?""",
                (thread_id,),
            ).fetchone()
            if existing and existing["row_count"] > 0:
                return {
                    "thread_id": thread_id,
                    "context_unit_count": existing["unit_count"],
                    "version_rows": existing["row_count"],
                    "reused": True,
                }
        if reset:
            conn.execute("DELETE FROM dp3_versioned_evidence_units WHERE source_id=?", (thread_id,))
            conn.execute("DELETE FROM dp3_evidence_units WHERE source_id=?", (thread_id,))
            conn.execute("DELETE FROM dp3_context_units WHERE source_example_id=?", (thread_id,))

    text = get_thread_text(thread_id)
    chunks = _chunk_text(text)
    if not chunks:
        return {"thread_id": thread_id, "context_unit_count": 0, "version_rows": 0}

    embeddings = [_embed(chunk) for chunk in chunks]
    with get_conn() as conn:
        inserted = 0
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            base_eu_id = f"{thread_id}:base_eu:{i}"
            logical_eu_id = f"{thread_id}:logical_eu:{i}"
            scope = _scope_for_index(i, len(chunks))
            embedding_json = _embedding_to_json(embedding)
            conn.execute(
                """INSERT OR REPLACE INTO dp3_evidence_units
                   (base_eu_id, source_id, source_example_id, eu_index, text, embedding_json, roi_metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    base_eu_id,
                    thread_id,
                    thread_id,
                    i,
                    chunk,
                    embedding_json,
                    json.dumps({"builder": "thread_chunk"}, ensure_ascii=False),
                ),
            )
            for version, fingerprint in _version_rows_for_unit(i, chunk):
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
                        thread_id,
                        thread_id,
                        version,
                        scope,
                        chunk,
                        fingerprint,
                        embedding_json,
                        _mutation_type_for_version(i, version, chunk, fingerprint),
                    ),
                )
                inserted += 1
    return {
        "thread_id": thread_id,
        "context_unit_count": len(chunks),
        "version_rows": inserted,
    }


def seed_answerable_question_pool(reset: bool = False) -> dict:
    init_dp3_cache_schema()
    with get_conn() as conn:
        if reset:
            conn.execute("DELETE FROM dp3_answerable_question_pool")
        count = 0
        for route_id, question_text, route_type in DEFAULT_ROUTES:
            conn.execute(
                """INSERT OR REPLACE INTO dp3_answerable_question_pool
                   (route_id, question_text, route_type, embedding_json)
                   VALUES (?, ?, ?, ?)""",
                (route_id, question_text, route_type, _embedding_to_json(_embed(question_text))),
            )
            count += 1
    return {"route_count": count}


def clear_answer_cache_for_source(source_id: str | None = None) -> dict:
    init_dp3_cache_schema()
    with get_conn() as conn:
        if source_id:
            cache_rows = conn.execute(
                """SELECT DISTINCT ace.cache_id
                   FROM dp3_answer_cache_entries ace
                   JOIN dp3_answer_cache_sources acs ON acs.cache_id = ace.cache_id
                   WHERE acs.logical_eu_id LIKE ?""",
                (f"{source_id}:%",),
            ).fetchall()
            cache_ids = [r["cache_id"] for r in cache_rows]
            conn.execute("DELETE FROM dp3_answer_cache_logs WHERE thread_id=?", (source_id,))
        else:
            cache_ids = [
                r["cache_id"]
                for r in conn.execute("SELECT cache_id FROM dp3_answer_cache_entries").fetchall()
            ]
            conn.execute("DELETE FROM dp3_answer_cache_logs")

        if cache_ids:
            placeholders = ",".join("?" for _ in cache_ids)
            conn.execute(
                f"DELETE FROM dp3_answer_cache_sources WHERE cache_id IN ({placeholders})",
                cache_ids,
            )
            conn.execute(
                f"DELETE FROM dp3_answer_cache_entries WHERE cache_id IN ({placeholders})",
                cache_ids,
            )
    return {"cleared_cache_entries": len(cache_ids)}


def setup_answer_cache_poc(thread_id: str, reset: bool = False) -> dict:
    init_dp3_cache_schema()
    if reset:
        clear_answer_cache_for_source(thread_id)
    context_result = setup_context_units_from_thread(thread_id, reset=reset)
    route_result = seed_answerable_question_pool(reset=reset)
    return {**context_result, **route_result}


def _parse_requested_version(query: str, default: str = "V1") -> str:
    normalized = query.upper()
    for version in VERSIONS:
        if version in normalized:
            return version
    return default


def _find_route(query_embedding: list[float]) -> dict:
    init_dp3_cache_schema()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT route_id, question_text, route_type, embedding_json FROM dp3_answerable_question_pool"
        ).fetchall()
    if not rows:
        seed_answerable_question_pool()
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT route_id, question_text, route_type, embedding_json FROM dp3_answerable_question_pool"
            ).fetchall()

    best = None
    for row in rows:
        score = _cosine(query_embedding, _embedding_from_json(row["embedding_json"]))
        if best is None or score > best["embedding_score"]:
            best = {
                "route_id": row["route_id"],
                "route_type": row["route_type"],
                "route_question": row["question_text"],
                "embedding_score": round(score, 4),
            }
    return best or {
        "route_id": None,
        "route_type": None,
        "route_question": None,
        "embedding_score": 0.0,
    }


def validate_eu(
    logical_eu_id: str,
    cached_fingerprint: str,
    user_scope: str,
    requested_version: str,
) -> dict:
    init_dp3_cache_schema()
    with get_conn() as conn:
        row = conn.execute(
            """SELECT logical_eu_id, versioned_eu_id, version, fingerprint, scope
               FROM dp3_versioned_evidence_units
               WHERE logical_eu_id=? AND version=? AND is_available=1""",
            (logical_eu_id, requested_version),
        ).fetchone()
        if not row:
            row = conn.execute(
                """SELECT logical_eu_id, version, fingerprint, scope
                   FROM dp3_context_units
                   WHERE logical_eu_id=? AND version=?""",
                (logical_eu_id, requested_version),
            ).fetchone()
    if not row:
        return {"valid": False, "reason": "missing_requested_version"}
    if row["scope"] != user_scope:
        return {"valid": False, "reason": "scope_mismatch"}
    if row["fingerprint"] != cached_fingerprint:
        return {"valid": False, "reason": "fingerprint_mismatch"}
    return {"valid": True, "reason": "valid"}


def _find_cache_candidate(
    route_id: str,
    user_scope: str,
    query_embedding: list[float],
) -> Optional[dict]:
    init_dp3_cache_schema()
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT cache_id, query_text, query_embedding_json, answer_text, scope, cache_version
               FROM dp3_answer_cache_entries
               WHERE route_id=? AND scope=?""",
            (route_id, user_scope),
        ).fetchall()

    best = None
    for row in rows:
        score = _cosine(query_embedding, _embedding_from_json(row["query_embedding_json"]))
        if best is None or score > best["cache_similarity_score"]:
            best = dict(row)
            best["cache_similarity_score"] = round(score, 4)
    return best


def _validate_cache_sources(cache_id: str, user_scope: str, requested_version: str) -> dict:
    init_dp3_cache_schema()
    with get_conn() as conn:
        sources = conn.execute(
            """SELECT logical_eu_id, versioned_eu_id, eu_version, fingerprint
               FROM dp3_answer_cache_sources
               WHERE cache_id=?
               ORDER BY source_order""",
            (cache_id,),
        ).fetchall()

    invalid = []
    for source in sources:
        result = validate_eu(
            source["logical_eu_id"],
            source["fingerprint"],
            user_scope,
            requested_version,
        )
        if not result["valid"]:
            invalid.append({
                "logical_eu_id": source["logical_eu_id"],
                "reason": result["reason"],
            })
    return {
        "valid": len(invalid) == 0 and len(sources) > 0,
        "source_count": len(sources),
        "invalid_sources": invalid,
    }


def _retrieve_context_units(
    thread_id: str,
    query_embedding: list[float],
    user_scope: str,
    requested_version: str,
    top_k: int = TOP_K_SOURCES,
) -> list[dict]:
    init_dp3_cache_schema()
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT logical_eu_id, versioned_eu_id, version, fingerprint, scope, text, embedding_json
               FROM dp3_versioned_evidence_units
               WHERE source_id=? AND version=? AND scope=? AND is_available=1""",
            (thread_id, requested_version, user_scope),
        ).fetchall()
        if not rows:
            rows = conn.execute(
                """SELECT logical_eu_id, NULL AS versioned_eu_id, version, fingerprint, scope, text, embedding_json
                   FROM dp3_context_units
                   WHERE source_example_id=? AND version=? AND scope=?""",
                (thread_id, requested_version, user_scope),
            ).fetchall()
    scored = []
    for row in rows:
        score = _cosine(query_embedding, _embedding_from_json(row["embedding_json"]))
        item = dict(row)
        item["score"] = round(score, 4)
        scored.append(item)
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def _build_prompt(query: str, sources: list[dict]) -> str:
    context = "\n\n".join(f"[{s['logical_eu_id']}]\n{s['text']}" for s in sources)
    return f"""아래 Context를 참고하여 질문에 답변해 주세요.

[Context]
{context}

[Question]
{query}

[Answer]"""


def _store_answer_cache(
    route_id: str,
    query: str,
    query_embedding: list[float],
    answer: str,
    user_scope: str,
    cache_version: str,
    sources: list[dict],
) -> str:
    init_dp3_cache_schema()
    cache_id = f"ans-{uuid.uuid4()}"
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO dp3_answer_cache_entries
               (cache_id, route_id, query_text, query_embedding_json, answer_text, scope, cache_version)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                cache_id,
                route_id,
                query,
                _embedding_to_json(query_embedding),
                answer,
                user_scope,
                cache_version,
            ),
        )
        for idx, source in enumerate(sources):
            conn.execute(
                """INSERT INTO dp3_answer_cache_sources
                   (cache_id, logical_eu_id, versioned_eu_id, eu_version, fingerprint, source_order)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    cache_id,
                    source["logical_eu_id"],
                    source.get("versioned_eu_id"),
                    source["version"],
                    source["fingerprint"],
                    idx,
                ),
            )
    return cache_id


def _store_log(thread_id: str, query: str, log: dict) -> None:
    init_dp3_cache_schema()
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO dp3_answer_cache_logs
               (mode, thread_id, query_text, requested_version, user_scope, decision_reason,
                cache_hit, validation_passed, llm_mocked, llm_call_count,
                roi_rag_called, total_ms, log_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "verified_answer_cache",
                thread_id,
                query,
                log.get("requested_version"),
                log.get("user_scope"),
                log.get("decision_reason"),
                int(bool(log.get("cache_hit"))),
                int(bool(log.get("validation_passed"))),
                int(bool(log.get("llm_mocked"))),
                int(log.get("llm_call_count", 0)),
                int(bool(log.get("roi_rag_called"))),
                int(log.get("total_ms", 0)),
                json.dumps(log, ensure_ascii=False),
            ),
        )


def run_answer_cache_query(
    thread_id: str,
    query: str,
    user_scope: str = "A",
    requested_version: Optional[str] = None,
    model: str = None,
    route_threshold: float = DEFAULT_ROUTE_THRESHOLD,
    cache_threshold: float = DEFAULT_CACHE_THRESHOLD,
) -> dict:
    start = time.time()
    setup_context_units_from_thread(thread_id, reset=False)
    seed_answerable_question_pool(reset=False)

    requested_version = requested_version or _parse_requested_version(query)
    query_embedding_list = _embed(query)
    query_embedding = query_embedding_list

    route = _find_route(query_embedding)
    routing_passed = bool(route["route_id"]) and route["embedding_score"] >= route_threshold

    log = {
        "mode": "verified_answer_cache",
        "thread_id": thread_id,
        "query": query,
        "user_scope": user_scope,
        "requested_version": requested_version,
        "routing_passed": routing_passed,
        "embedding_route_id": route["route_id"],
        "embedding_score": route["embedding_score"],
        "route_threshold": route_threshold,
        "cache_threshold": cache_threshold,
        "cache_hit": False,
        "validation_passed": False,
        "llm_mocked": is_mock_llm(),
        "llm_call_count": 0,
        "roi_rag_called": False,
    }

    if not routing_passed:
        log["decision_reason"] = "embedding_score_below_threshold"
        result = _fallback_and_store(
            thread_id,
            query,
            query_embedding_list,
            query_embedding,
            user_scope,
            requested_version,
            route["route_id"] or "unrouted",
            model,
            log,
        )
        return result

    candidate = _find_cache_candidate(route["route_id"], user_scope, query_embedding)
    if candidate and candidate["cache_similarity_score"] >= cache_threshold:
        validation = _validate_cache_sources(candidate["cache_id"], user_scope, requested_version)
        log.update({
            "cache_candidate_id": candidate["cache_id"],
            "cache_similarity_score": candidate["cache_similarity_score"],
            "source_validation": validation,
        })
        if validation["valid"]:
            log.update({
                "cache_hit": True,
                "validation_passed": True,
                "decision_reason": "answer_cache_hit_valid",
                "answer": candidate["answer_text"],
                "total_ms": int((time.time() - start) * 1000),
            })
            _store_log(thread_id, query, log)
            return log
        log["decision_reason"] = "cache_candidate_invalid_fallback_to_roi_rag"
    else:
        log["cache_candidate_id"] = candidate["cache_id"] if candidate else None
        log["cache_similarity_score"] = candidate["cache_similarity_score"] if candidate else None
        log["decision_reason"] = "cache_candidate_not_found_fallback_to_roi_rag"

    return _fallback_and_store(
        thread_id,
        query,
        query_embedding_list,
        query_embedding,
        user_scope,
        requested_version,
        route["route_id"],
        model,
        log,
        start=start,
    )


def _fallback_and_store(
    thread_id: str,
    query: str,
    query_embedding_list: list[float],
    query_embedding: list[float],
    user_scope: str,
    requested_version: str,
    route_id: str,
    model: str,
    log: dict,
    start: float = None,
) -> dict:
    start = start or time.time()
    sources = _retrieve_context_units(thread_id, query_embedding, user_scope, requested_version)
    prompt = _build_prompt(query, sources)
    answer = get_dp3_answer(prompt, model)
    cache_id = _store_answer_cache(
        route_id=route_id,
        query=query,
        query_embedding=query_embedding_list,
        answer=answer,
        user_scope=user_scope,
        cache_version=requested_version,
        sources=sources,
    )
    log.update({
        "cache_hit": False,
        "validation_passed": False,
        "decision_reason": log.get("decision_reason", "fallback_to_roi_rag"),
        "cache_id": cache_id,
        "answer": answer,
        "fallback_source_count": len(sources),
        "fallback_sources": [
            {
                "logical_eu_id": s["logical_eu_id"],
                "versioned_eu_id": s.get("versioned_eu_id"),
                "version": s["version"],
                "fingerprint": s["fingerprint"],
                "score": s["score"],
            }
            for s in sources
        ],
        "llm_call_count": 1,
        "roi_rag_called": True,
        "total_ms": int((time.time() - start) * 1000),
    })
    _store_log(thread_id, query, log)
    return log
