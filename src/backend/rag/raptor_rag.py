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

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 5
MAX_LEVELS = 3
MIN_CLUSTER_SIZE = 2


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


def _cluster_texts(embeddings: np.ndarray, n_clusters: int) -> list[int]:
    from sklearn.cluster import KMeans
    if len(embeddings) <= n_clusters:
        return list(range(len(embeddings)))
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    return km.fit_predict(embeddings).tolist()


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
    n_clusters = max(2, len(chunks) // 3)
    labels = _cluster_texts(embeddings, n_clusters)

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
    results = col.query(
        query_texts=[question],
        n_results=min(TOP_K, col.count()),
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

[분석 내용]
{context}

[질문]
{question}

[답변]"""
    answer = get_llm_answer(prompt, model)
    latency = int((time.time() - start) * 1000)
    return {"answer": answer, "references": docs, "latency_ms": latency, "model": model or "default"}
