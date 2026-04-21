"""
RAPTOR RAG: Recursive Abstractive Processing for Tree-Organized Retrieval
재귀적 요약 트리 구조로 다층 검색을 구현합니다.
"""
import time
import numpy as np
import chromadb
from chromadb.utils import embedding_functions
from backend.db.database import get_conn
from backend.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL
from backend.rag.llm_client import get_llm_answer

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 5
MAX_LEVELS = 3
MIN_CLUSTER_SIZE = 2


def _get_collection(session_id: str):
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(
        name=f"raptor_{session_id.replace('-', '_')}",
        embedding_function=ef,
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
    """간단한 K-Means 클러스터링 (UMAP 없이도 동작)."""
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


def index_session(session_id: str):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT speaker, content FROM messages WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()

    full_text = "\n".join(f"[{r['speaker']}] {r['content']}" for r in rows)
    chunks = _chunk_text(full_text)

    all_nodes = _build_tree(chunks)

    collection = _get_collection(session_id)
    documents = [n["text"] for n in all_nodes]
    ids = [f"{session_id}_raptor_{i}" for i in range(len(all_nodes))]
    metadatas = [{"level": n["level"]} for n in all_nodes]

    collection.upsert(documents=documents, ids=ids, metadatas=metadatas)

    with get_conn() as conn:
        conn.execute("UPDATE sessions SET is_indexed = 1 WHERE id = ?", (session_id,))


def query(session_id: str, question: str, model: str = None) -> dict:
    start = time.time()

    collection = _get_collection(session_id)
    results = collection.query(
        query_texts=[question],
        n_results=min(TOP_K, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs = results["documents"][0] if results["documents"] else []
    metas = results["metadatas"][0] if results["metadatas"] else []

    # 레벨별로 분류하여 컨텍스트 구성
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
