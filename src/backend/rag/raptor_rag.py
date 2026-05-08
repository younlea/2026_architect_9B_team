"""
RAPTOR RAG: Recursive Abstractive Processing for Tree-Organized Retrieval
재귀적 요약 트리 구조로 다층 검색을 구현합니다.
"""
import time
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from backend.db.database import get_conn, get_thread_text
from backend.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL
from backend.rag.llm_client import get_llm_answer

CHUNK_SIZE = 512      # 한국어 기준 약 250 어절
CHUNK_OVERLAP = 80    # 약 15% overlap
TOP_K = 5
MAX_LEVELS = 3
MIN_CLUSTER_SIZE = 2
DBSCAN_EPS = 0.25     # 코사인 거리 임계값 (0~2, 작을수록 엄격하게 묶음)
DBSCAN_MIN_SAMPLES = 2  # 클러스터 형성 최소 샘플 수


def _get_client():
    return chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)


def _get_ef():
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def _get_collection(col_name: str):
    return _get_client().get_or_create_collection(
        name=col_name.replace("-", "_"),
        embedding_function=_get_ef(),
    )


def _chunk_text(text: str) -> list[str]:
    """문장 경계 인식 스마트 청킹 (문장이 중간에서 잘리지 않도록)."""
    import re
    sentences = re.split(r'(?<=[.!?\n])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > CHUNK_SIZE and current:
            chunks.append(current.strip())
            overlap_text = current[-CHUNK_OVERLAP:] if len(current) > CHUNK_OVERLAP else current
            current = overlap_text + " " + sent
        else:
            current = (current + " " + sent).strip() if current else sent
    if current.strip():
        chunks.append(current.strip())

    result = []
    for chunk in chunks:
        if len(chunk) <= CHUNK_SIZE * 2:
            result.append(chunk)
        else:
            start = 0
            while start < len(chunk):
                result.append(chunk[start:start + CHUNK_SIZE])
                start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in result if c.strip()]


def _embed_texts(texts: list[str]) -> np.ndarray:
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBEDDING_MODEL)
    return model.encode(texts, normalize_embeddings=True)


def _cluster_texts(embeddings: np.ndarray) -> list[int]:
    """DBSCAN으로 코사인 거리 기반 의미 클러스터링.
    - 유사한 청크라리 자동으로 묶임 (KMeans의 고정 수 문제 해결)
    - 잘엠 노이즈(독립된 내용) 자동 감지
    """
    from sklearn.cluster import DBSCAN
    n = len(embeddings)
    if n <= DBSCAN_MIN_SAMPLES:
        return list(range(n))

    # 코사인 거리로 DBSCAN (코사인 거리 = 1 - 코사인 유사도)
    db = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric='cosine')
    labels = db.fit_predict(embeddings).tolist()

    # 노이즈(-1)는 각자 독립 클러스터로 배정
    max_label = max(labels) if labels else -1
    result = []
    for lbl in labels:
        if lbl == -1:
            max_label += 1
            result.append(max_label)
        else:
            result.append(lbl)
    return result


def _summarize_cluster(texts: list[str], model: str = None) -> str:
    joined = "\n".join(texts)
    prompt = f"""다음 대화 내용들의 핵심을 간결하게 요약해 주세요. 중요한 정보, 결정사항, 감정적 맥락을 유지하세요.

[내용]
{joined}

[요약]"""
    return get_llm_answer(prompt, model)


def _build_tree(chunks: list[str], level: int = 0, model: str = None) -> list[dict]:
    """재귀적으로 요약 트리를 구성하여 모든 노드를 반환합니다."""
    nodes = [{"text": c, "level": level} for c in chunks]

    if len(chunks) < MIN_CLUSTER_SIZE * 2 or level >= MAX_LEVELS:
        return nodes

    embeddings = _embed_texts(chunks)
    labels = _cluster_texts(embeddings)

    clusters: dict[int, list[str]] = {}
    for text, label in zip(chunks, labels):
        clusters.setdefault(label, []).append(text)

    summaries = []
    for cluster_texts in clusters.values():
        if len(cluster_texts) >= MIN_CLUSTER_SIZE:
            summaries.append(_summarize_cluster(cluster_texts, model))

    if summaries:
        upper_nodes = _build_tree(summaries, level + 1, model)
        nodes.extend(upper_nodes)

    return nodes


def _index_text(col_name: str, text: str, id_prefix: str, model: str = None) -> int:
    chunks = _chunk_text(text)
    all_nodes = _build_tree(chunks, model=model)

    col = _get_collection(col_name)
    col.upsert(
        documents=[n["text"] for n in all_nodes],
        ids=[f"{id_prefix}_raptor_{i}" for i in range(len(all_nodes))],
        metadatas=[{"level": n["level"]} for n in all_nodes],
    )
    return len(all_nodes)


def index_session(session_id: str):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT speaker, content FROM messages WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
    full_text = "\n".join(f"[{r['speaker']}] {r['content']}" for r in rows)
    _index_text(f"raptor_s_{session_id}", full_text, session_id)
    with get_conn() as conn:
        conn.execute("UPDATE sessions SET is_indexed = 1 WHERE id = ?", (session_id,))


def index_thread(thread_id: str, model: str = None) -> int:
    """스레드의 모든 세션 메시지를 합쳐 단일 RAPTOR 트리 인덱스로 구성합니다."""
    full_text = get_thread_text(thread_id)
    node_count = _index_text(f"raptor_t_{thread_id}", full_text, thread_id, model)
    with get_conn() as conn:
        conn.execute(
            "UPDATE threads SET raptor_indexed=1, raptor_node_count=? WHERE id=?",
            (node_count, thread_id),
        )
    return node_count


def query(session_id: str, question: str, model: str = None) -> dict:
    return _query_col(f"raptor_s_{session_id}", question, model)


def query_thread(thread_id: str, question: str, model: str = None) -> dict:
    return _query_col(f"raptor_t_{thread_id}", question, model)


def _query_col(col_name: str, question: str, model: str = None) -> dict:
    start = time.time()
    col = _get_collection(col_name)
    count = col.count()
    if count == 0:
        return {"answer": "", "references": [], "latency_ms": 0, "model": model or "default"}
    results = col.query(
        query_texts=[question],
        n_results=min(TOP_K, count),
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []

    level_groups: dict[int, list[str]] = {}
    for doc, meta in zip(docs, metas):
        lvl = meta.get("level", 0)
        level_groups.setdefault(lvl, []).append(doc)

    context_parts = []
    for lvl in sorted(level_groups.keys(), reverse=True):
        label = "전체 요약" if lvl > 0 else "세부 내용"
        context_parts.append(f"[{label} (레벨 {lvl})]")
        context_parts.extend(level_groups[lvl])

    context = "\n\n".join(context_parts)
    prompt = f"""아래 대화 분석 내용을 참고하여 질문에 답변해 주세요.
전체 요약은 맥락 파악에, 세부 내용은 구체적 사실 확인에 활용하세요.
(주의: 반드시 질문과 동일한 언어로 답변을 작성해야 합니다.)

[분석 내용]
{context}

[질문]
{question}

[답변]"""
    answer = get_llm_answer(prompt, model)
    latency = int((time.time() - start) * 1000)
    return {"answer": answer, "references": docs, "latency_ms": latency, "model": model or "default"}
