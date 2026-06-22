"""
ROI-RAG: Redundancy-Optimized Indexing for RAG
KDD 2026 — "When to Optimize Offline: A Regime-Based Framework for
Redundancy in Knowledge-Grounded Generation"

오프라인: 엔트로피 기반 Evidence Unit(EU) 구성 + 적응형 요약
온라인:   단일 ANN 조회 (순회 없음, 예측 가능한 레이턴시)
"""
import json
import hashlib
import time
import numpy as np
try:
    import chromadb
except ModuleNotFoundError:
    chromadb = None
from backend.db.database import get_conn, get_thread_text
from backend.config import CHROMA_PERSIST_DIR
from backend.rag.llm_client import get_llm_answer

# ── 하이퍼파라미터 (논문 Appendix A 기준) ──────────────────────────────────
CHUNK_SIZE = 300           # 기존 코드베이스 기준 (논문: ~200 tokens)
CHUNK_OVERLAP = 50
KNN_K = 10                 # 후보 이웃 크기
MAX_SEGMENTS_PER_EU = 6    # EU당 최대 세그먼트 수 (논문 Appendix A)
TOP_K = 5                  # 검색할 EU 수

# Regime 임계값 (논문 §4.1: 전역 RE 분포에 Otsu thresholding으로 1회 유도한
# 고정 기준값. dataset별 튜닝 아님). RE = mean redundancy entropy.
#   RE ≥ 1e-2  → HIGH,  3e-3 ≤ RE < 1e-2 → MID,  RE < 3e-3 → LOW
HIGH_RE_THRESHOLD = 0.01
MID_RE_THRESHOLD = 0.003

REDUNDANCY_TAU = 0.6       # R(C) 계산용 유사도 임계값 (논문 Equation 1)


# ── ChromaDB 헬퍼 ──────────────────────────────────────────────────────────

def _get_client():
    if chromadb is None:
        raise ModuleNotFoundError("chromadb")
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def _get_ef():
    from backend.rag import _ef as _shared_ef
    return _shared_ef.get()


def _get_collection(col_name: str):
    # 쿼리 시 query_texts 임베딩에 ef 사용; upsert 시 embeddings 직접 주입
    return _get_client().get_or_create_collection(
        name=col_name.replace("-", "_"),
        embedding_function=_get_ef(),
    )


# ── 텍스트 전처리 ──────────────────────────────────────────────────────────

def _chunk_text(text: str) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunks.append(text[start:end])
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in chunks if c.strip()]


def _embed_texts(texts: list[str]) -> np.ndarray:
    try:
        ef = _get_ef()
        embeddings = np.array(ef(texts), dtype=float)
    except Exception:
        embeddings = np.array([_hash_embed_text(text) for text in texts], dtype=float)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10
    return embeddings / norms


def _hash_embed_text(text: str, dim: int = 384) -> list[float]:
    vector = np.zeros(dim, dtype=float)
    tokens = text.lower().split() or [text.lower()]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i, byte in enumerate(digest):
            vector[(byte + i * 17) % dim] += 1.0
    norm = np.linalg.norm(vector) + 1e-10
    return (vector / norm).tolist()


# ── 엔트로피 계산 (논문 Algorithm 1) ────────────────────────────────────────

def _compute_entropy_indices(embeddings: np.ndarray) -> tuple[float, float]:
    """
    RE(C), DE(C) 계산.

    D_ij = 1 - cos(z_i, z_j)
    p_ij = D_ij / Σ D_uv
    H    = -Σ p_ij · log(p_ij)
    DE   = H / log(n²)
    RE   = 1 - DE
    """
    n = len(embeddings)
    if n < 2:
        return 0.0, 1.0  # singleton: RE=0, DE=1

    # embeddings는 normalize_embeddings=True로 이미 정규화됨 → dot = cos
    sim_matrix = embeddings @ embeddings.T
    dist_matrix = 1.0 - sim_matrix
    np.fill_diagonal(dist_matrix, 0.0)

    total = dist_matrix.sum() + 1e-10
    p = dist_matrix / total

    H = float(-np.sum(p * np.log(p + 1e-10)))

    log_n2 = np.log(float(n) ** 2)
    DE = float(np.clip(H / log_n2, 0.0, 1.0)) if log_n2 > 1e-10 else 0.0
    RE = 1.0 - DE

    return RE, DE


# ── kNN 이웃 구성 ──────────────────────────────────────────────────────────

def _build_knn_neighborhoods(
    embeddings: np.ndarray, k: int = KNN_K
) -> list[list[int]]:
    """각 세그먼트의 top-k 의미적 이웃 인덱스를 반환합니다."""
    n = len(embeddings)
    k_actual = min(k, n - 1)
    if k_actual <= 0:
        return [[] for _ in range(n)]

    try:
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(
            n_neighbors=k_actual + 1, metric="cosine", algorithm="brute"
        )
        nbrs.fit(embeddings)
        _, indices = nbrs.kneighbors(embeddings)

        neighborhoods = []
        for i in range(n):
            neighbors = [int(idx) for idx in indices[i] if int(idx) != i][:k_actual]
            neighborhoods.append(neighbors)
        return neighborhoods
    except Exception:
        sim = embeddings @ embeddings.T
        neighborhoods = []
        for i in range(n):
            order = np.argsort(-sim[i])
            neighbors = [int(idx) for idx in order if int(idx) != i][:k_actual]
            neighborhoods.append(neighbors)
        return neighborhoods


# ── Greedy EU 구성 (논문 Section 3.4) ─────────────────────────────────────

def _greedy_eu_construction(
    segments: list[str],
    embeddings: np.ndarray,
    neighborhoods: list[list[int]],
) -> list[dict]:
    """
    Non-overlap Greedy EU 구성.

    1. 각 이웃 neighborhood의 RE 계산
    2. RE 높은 순 (중복 많은 순)으로 seed 선택
    3. 미할당 이웃 중 다양성 최대 세그먼트를 반복 추가 (최대 MAX_SEGMENTS_PER_EU)
    4. 모든 EU에 걸쳐 non-overlap 보장
    """
    n = len(segments)
    assigned: set[int] = set()
    evidence_units: list[dict] = []

    # 이웃 neighborhood RE로 seed 우선순위 결정
    neighborhood_re = []
    for i in range(n):
        nbr_idx = [i] + neighborhoods[i]
        re, _ = _compute_entropy_indices(embeddings[nbr_idx])
        neighborhood_re.append(re)

    seed_order = sorted(range(n), key=lambda i: neighborhood_re[i], reverse=True)

    for seed in seed_order:
        if seed in assigned:
            continue

        eu_indices = [seed]
        assigned.add(seed)
        candidates = [j for j in neighborhoods[seed] if j not in assigned]

        while len(eu_indices) < MAX_SEGMENTS_PER_EU and candidates:
            eu_mean = embeddings[eu_indices].mean(axis=0)
            eu_norm = eu_mean / (np.linalg.norm(eu_mean) + 1e-10)

            # 다양성 극대화: EU와 유사도가 가장 낮은 후보 선택
            best = max(
                candidates,
                key=lambda j: 1.0 - float(embeddings[j] @ eu_norm),
            )
            eu_indices.append(best)
            assigned.add(best)
            candidates = [j for j in candidates if j not in assigned]

        eu_segs = [segments[i] for i in eu_indices]
        eu_embeds = embeddings[eu_indices]
        eu_re, eu_de = _compute_entropy_indices(eu_embeds)

        evidence_units.append({
            "indices": eu_indices,
            "segments": eu_segs,
            "re": eu_re,
            "de": eu_de,
            "embedding": eu_embeds.mean(axis=0),
        })

    # 미할당 세그먼트 → singleton EU
    for i in range(n):
        if i not in assigned:
            evidence_units.append({
                "indices": [i],
                "segments": [segments[i]],
                "re": 0.0,
                "de": 1.0,
                "embedding": embeddings[i],
            })

    return evidence_units


# ── Regime 분류 ────────────────────────────────────────────────────────────

def _classify_regime(eu_re_values: list[float]) -> str:
    """Corpus-level regime 분류 (논문 Section 4.1)."""
    if not eu_re_values:
        return "LOW"
    mean_re = float(np.mean(eu_re_values))
    if mean_re >= HIGH_RE_THRESHOLD:
        return "HIGH"
    if mean_re >= MID_RE_THRESHOLD:
        return "MID"
    return "LOW"


# ── 적응형 요약 (논문 Section 3.5) ────────────────────────────────────────

def _adaptive_summarize(
    eu: dict, regime: str, model: str = None
) -> str:
    """
    논문 §3.5 적응형 요약 정책.

    요약 강도는 corpus regime(전역 Otsu로 유도한 절대 RE 임계값)과
    EU 자체의 절대 RE를 따른다. corpus-relative min-max 정규화(이전 버그)는
    중복이 적은 코퍼스에서도 강제로 공격적 요약을 유발해 추출형 정답을 손상시켰다.

    - LOW regime           : 요약 생략 (논문 "when to optimize offline" — 오프라인 최적화 보류)
    - EU RE < MID(3e-3)    : 요약 생략
    - MID ≤ EU RE < HIGH   : 부분 요약 (~75%)
    - EU RE ≥ HIGH(1e-2)   : 공격적 요약 (~50%, 중복 제거)
    """
    segs = eu["segments"]
    if len(segs) == 1:
        return segs[0]

    joined = "\n".join(segs)

    # LOW regime → 오프라인 요약 비용 대비 이득이 작아 원문 유지
    if regime == "LOW":
        return joined

    eu_re = float(eu["re"])
    if eu_re < MID_RE_THRESHOLD:
        return joined

    if eu_re < HIGH_RE_THRESHOLD:
        prompt = (
            "다음 내용들의 중요한 정보를 유지하면서 간결하게 요약해 주세요"
            " (원문의 약 75% 분량 목표).\n\n[내용]\n" + joined + "\n\n[요약]"
        )
    else:
        prompt = (
            "다음 내용들의 핵심만 추려 간결하게 요약해 주세요"
            " (원문의 약 50% 분량 목표). 중복된 정보는 제거하세요.\n\n[내용]\n"
            + joined + "\n\n[요약]"
        )

    return get_llm_answer(prompt, model)


# ── R(C) 메트릭 (논문 Equation 1) ─────────────────────────────────────────

def _compute_r_c(embeddings: np.ndarray, tau: float = REDUNDANCY_TAU) -> float:
    """검색된 EU 간 중복도 측정."""
    n = len(embeddings)
    if n < 2:
        return 0.0
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    safe = embeddings / (norms + 1e-10)
    sim = safe @ safe.T
    near_dup = [
        float(sim[i, j])
        for i in range(n)
        for j in range(i + 1, n)
        if sim[i, j] > tau
    ]
    return float(np.mean(near_dup)) if near_dup else 0.0


# ── 오프라인 인덱싱 ────────────────────────────────────────────────────────

def _build_index(
    col_name: str, text: str, id_prefix: str, model: str = None
) -> dict:
    """EU 구성 + ChromaDB 인덱싱 전체 파이프라인."""
    segments = _chunk_text(text)
    if not segments:
        return {"eu_count": 0, "regime": "LOW", "segment_count": 0}

    embeddings = _embed_texts(segments)
    neighborhoods = _build_knn_neighborhoods(embeddings)
    evidence_units = _greedy_eu_construction(segments, embeddings, neighborhoods)

    eu_re_values = [eu["re"] for eu in evidence_units]
    regime = _classify_regime(eu_re_values)

    col = _get_collection(col_name)
    docs, embeds, ids, metas = [], [], [], []

    for i, eu in enumerate(evidence_units):
        # 요약 정책은 regime(절대 RE 임계값)을 따른다 — 논문 §3.5
        summary = _adaptive_summarize(eu, regime, model)

        docs.append(summary)
        embeds.append(eu["embedding"].tolist())
        ids.append(f"{id_prefix}_roi_eu_{i}")
        # 논문 §3.4: 프롬프트는 'EU 요약 + 근거 원문 세그먼트'를 결합하므로
        # 근거 세그먼트를 함께 저장한다 (크기 제한 위해 최대 3개).
        metas.append({
            "eu_id": i,
            "segment_count": len(eu["segments"]),
            "re": float(eu["re"]),
            "regime": regime,
            "segments_json": json.dumps(
                eu["segments"][:3], ensure_ascii=False
            ),
        })

    col.upsert(documents=docs, embeddings=embeds, ids=ids, metadatas=metas)

    return {
        "eu_count": len(evidence_units),
        "regime": regime,
        "segment_count": len(segments),
    }


def build_evidence_units_for_text(
    text: str,
    id_prefix: str = "dp3",
    use_summary: bool = False,
    model: str = None,
) -> dict:
    """Build ROI-RAG Evidence Units without writing a Chroma index.

    DP3 uses this helper to create reusable EU-level metadata from LongBench
    while keeping the existing DP1/DP2 index/query paths unchanged. Summary
    generation is disabled by default so preprocessing does not require an LLM.
    """
    segments = _chunk_text(text)
    if not segments:
        return {
            "evidence_units": [],
            "eu_count": 0,
            "regime": "LOW",
            "segment_count": 0,
        }

    embeddings = _embed_texts(segments)
    neighborhoods = _build_knn_neighborhoods(embeddings)
    evidence_units = _greedy_eu_construction(segments, embeddings, neighborhoods)

    eu_re_values = [eu["re"] for eu in evidence_units]
    regime = _classify_regime(eu_re_values)

    rows = []
    for i, eu in enumerate(evidence_units):
        eu_text = (
            _adaptive_summarize(eu, regime, model)
            if use_summary
            else "\n".join(eu["segments"])
        )
        rows.append({
            "roi_eu_id": f"{id_prefix}_roi_eu_{i}",
            "text": eu_text,
            "embedding": eu["embedding"].tolist(),
            "segments": eu["segments"],
            "segment_indices": [int(idx) for idx in eu["indices"]],
            "segment_count": len(eu["segments"]),
            "re": float(eu["re"]),
            "de": float(eu["de"]),
            "regime": regime,
        })

    return {
        "evidence_units": rows,
        "eu_count": len(rows),
        "regime": regime,
        "segment_count": len(segments),
    }


def index_thread(thread_id: str, model: str = None) -> dict:
    """스레드의 ROI-RAG EU 인덱스를 구성합니다."""
    full_text = get_thread_text(thread_id)
    result = _build_index(f"roi_t_{thread_id}", full_text, thread_id, model)
    with get_conn() as conn:
        conn.execute(
            "UPDATE threads SET roi_indexed=1, roi_eu_count=?, roi_regime=? WHERE id=?",
            (result["eu_count"], result["regime"], thread_id),
        )
    return result


# ── 온라인 검색 ────────────────────────────────────────────────────────────

def query_thread(thread_id: str, question: str, model: str = None) -> dict:
    return _query_col(f"roi_t_{thread_id}", question, model)


def _query_col(col_name: str, question: str, model: str = None) -> dict:
    col = _get_collection(col_name)
    count = col.count()

    if count == 0:
        return {
            "answer": "ROI-RAG 인덱스가 비어 있습니다. 먼저 인덱싱을 실행하세요.",
            "references": [],
            "latency_ms": 0,
            "retrieval_ms": 0,
            "generation_ms": 0,
            "model": model or "default",
            "r_c": 0.0,
            "regime": "",
            "eu_count": 0,
        }

    # ── 검색(단일 ANN 조회) 단계 — 논문상 Single-stage ANN ──
    retr_start = time.time()
    results = col.query(
        query_texts=[question],
        n_results=min(TOP_K, count),
        include=["documents", "metadatas", "embeddings"],
    )

    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []
    raw_embeds = results.get("embeddings")
    eu_embeds = raw_embeds[0] if raw_embeds else None

    # R(C) 계산
    r_c = 0.0
    if eu_embeds is not None and len(eu_embeds) >= 2:
        r_c = _compute_r_c(np.array(eu_embeds, dtype=float))

    regime = metas[0].get("regime", "") if metas else ""

    # EU 컨텍스트 조합 — 논문 §3.4: 'EU 요약 + 근거 원문 세그먼트' 결합
    references: list[str] = []
    context_parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        seg_n = meta.get("segment_count", 1)
        r = meta.get("regime", "")
        segs = json.loads(meta.get("segments_json", "[]"))
        references.extend(segs)
        part = f"[EU {i + 1} | {seg_n}개 세그먼트 | {r}]\n[요약]\n{doc}"
        # 요약본과 근거 원문이 다를 때만 원문 세그먼트를 함께 제공
        support = [s for s in segs if s and s.strip() and s.strip() != doc.strip()]
        if support:
            part += "\n[근거 원문]\n" + "\n---\n".join(support)
        context_parts.append(part)

    context = "\n\n".join(context_parts)
    retrieval_ms = int((time.time() - retr_start) * 1000)

    prompt = (
        "아래 최적화된 증거 단위(Evidence Unit)를 참고하여 질문에 답변해 주세요.\n"
        "각 EU는 중복이 제거된 요약과 그 근거 원문을 함께 담고 있습니다.\n\n"
        f"[증거 단위]\n{context}\n\n"
        f"[질문]\n{question}\n\n[답변]"
    )

    # ── LLM 생성 단계 ──
    gen_start = time.time()
    answer = get_llm_answer(prompt, model, deterministic=True)
    generation_ms = int((time.time() - gen_start) * 1000)

    references = references[:TOP_K]

    return {
        "answer": answer,
        "references": references,
        "latency_ms": retrieval_ms + generation_ms,
        "retrieval_ms": retrieval_ms,
        "generation_ms": generation_ms,
        "model": model or "default",
        "r_c": round(r_c, 4),
        "regime": regime,
        "eu_count": count,
    }
