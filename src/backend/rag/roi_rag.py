"""
ROI-RAG: Redundancy-Optimized Indexing for RAG
KDD 2026 — "When to Optimize Offline: A Regime-Based Framework for
Redundancy in Knowledge-Grounded Generation"

오프라인: 엔트로피 기반 Evidence Unit(EU) 구성 + 적응형 요약
온라인:   단일 ANN 조회 (순회 없음, 예측 가능한 레이턴시)
"""
import json
import time
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from backend.db.database import get_conn, get_thread_text
from backend.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL
from backend.rag.llm_client import get_llm_answer

# ── 하이퍼파라미터 (논문 Appendix A 기준) ──────────────────────────────────
CHUNK_SIZE = 300           # 기존 코드베이스 기준 (논문: ~200 tokens)
CHUNK_OVERLAP = 50
KNN_K = 10                 # 후보 이웃 크기
MAX_SEGMENTS_PER_EU = 6    # EU당 최대 세그먼트 수 (논문 Appendix A)
TOP_K = 5                  # 검색할 EU 수

# Regime 임계값 (논문 Section 4.1, 원스케일 RE)
HIGH_RE_THRESHOLD = 0.01    # RE ≥ 0.01 → HIGH
MID_RE_THRESHOLD = 0.003    # 0.003 ≤ RE < 0.01 → MID

# Otsu 정규화 임계값 (논문 Appendix B.3, normalized RE [0,1])
OTSU_LOW = 0.3    # normalized RE < 0.3 → 요약 없음
OTSU_HIGH = 0.7   # normalized RE ≥ 0.7 → 공격적 요약

REDUNDANCY_TAU = 0.6       # R(C) 계산용 유사도 임계값 (논문 Equation 1)


# ── ChromaDB 헬퍼 ──────────────────────────────────────────────────────────

def _get_client():
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def _get_ef():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


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
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model.encode(texts, normalize_embeddings=True)


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
    from sklearn.neighbors import NearestNeighbors

    n = len(embeddings)
    k_actual = min(k, n - 1)
    if k_actual <= 0:
        return [[] for _ in range(n)]

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
    eu: dict, re_normalized: float, model: str = None
) -> str:
    """
    정규화된 RE 기반 요약 정책:
      LOW  (< 0.3): 요약 없음
      MID  (0.3~0.7): 부분 요약 (~75%)
      HIGH (≥ 0.7): 공격적 요약 (~50%, 중복 제거)
    """
    segs = eu["segments"]
    if len(segs) == 1:
        return segs[0]

    joined = "\n".join(segs)

    if re_normalized < OTSU_LOW:
        return joined

    if re_normalized < OTSU_HIGH:
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

    # min-max 정규화 → summarization 정책 결정에 사용
    re_arr = np.array(eu_re_values)
    re_min, re_max = float(re_arr.min()), float(re_arr.max())
    re_range = re_max - re_min

    col = _get_collection(col_name)
    docs, embeds, ids, metas = [], [], [], []

    for i, eu in enumerate(evidence_units):
        re_norm = (eu["re"] - re_min) / re_range if re_range > 1e-10 else 0.0
        summary = _adaptive_summarize(eu, float(re_norm), model)

        docs.append(summary)
        embeds.append(eu["embedding"].tolist())
        ids.append(f"{id_prefix}_roi_eu_{i}")
        # metadata에 원문 세그먼트는 최대 3개만 저장 (크기 제한)
        metas.append({
            "eu_id": i,
            "segment_count": len(eu["segments"]),
            "re": float(eu["re"]),
            "re_normalized": float(re_norm),
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
    start = time.time()
    col = _get_collection(col_name)
    count = col.count()

    if count == 0:
        return {
            "answer": "ROI-RAG 인덱스가 비어 있습니다. 먼저 인덱싱을 실행하세요.",
            "references": [],
            "latency_ms": 0,
            "model": model or "default",
            "r_c": 0.0,
            "regime": "",
            "eu_count": 0,
        }

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
    if eu_embeds and len(eu_embeds) >= 2:
        r_c = _compute_r_c(np.array(eu_embeds, dtype=float))

    regime = metas[0].get("regime", "") if metas else ""

    # EU 컨텍스트 조합
    context_parts = []
    for i, (doc, meta) in enumerate(zip(docs, metas)):
        seg_n = meta.get("segment_count", 1)
        r = meta.get("regime", "")
        context_parts.append(f"[EU {i + 1} | {seg_n}개 세그먼트 | {r}]\n{doc}")

    context = "\n\n".join(context_parts)
    prompt = (
        "아래 최적화된 증거 단위(Evidence Unit)를 참고하여 질문에 답변해 주세요.\n"
        "각 EU는 중복이 제거된 핵심 정보를 담고 있습니다.\n\n"
        f"[증거 단위]\n{context}\n\n"
        f"[질문]\n{question}\n\n[답변]"
    )

    answer = get_llm_answer(prompt, model)
    latency = int((time.time() - start) * 1000)

    # 원문 세그먼트 참조용 (최대 TOP_K개)
    references: list[str] = []
    for meta in metas:
        segs = json.loads(meta.get("segments_json", "[]"))
        references.extend(segs)
    references = references[:TOP_K]

    return {
        "answer": answer,
        "references": references,
        "latency_ms": latency,
        "model": model or "default",
        "r_c": round(r_c, 4),
        "regime": regime,
        "eu_count": count,
    }
