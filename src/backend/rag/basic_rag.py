import time
import chromadb
from chromadb.utils import embedding_functions
from backend.db.database import get_conn
from backend.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL
from backend.rag.llm_client import get_llm_answer

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 5


def _get_collection(session_id: str):
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)
    return client.get_or_create_collection(
        name=f"basic_{session_id.replace('-', '_')}",
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


def index_session(session_id: str):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT speaker, content FROM messages WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()

    full_text = "\n".join(f"[{r['speaker']}] {r['content']}" for r in rows)
    chunks = _chunk_text(full_text)

    collection = _get_collection(session_id)
    collection.upsert(
        documents=chunks,
        ids=[f"{session_id}_chunk_{i}" for i in range(len(chunks))],
    )

    with get_conn() as conn:
        conn.execute("UPDATE sessions SET is_indexed = 1 WHERE id = ?", (session_id,))


def query(session_id: str, question: str, model: str = None) -> dict:
    start = time.time()

    collection = _get_collection(session_id)
    results = collection.query(query_texts=[question], n_results=min(TOP_K, collection.count()))
    docs = results["documents"][0] if results["documents"] else []

    context = "\n\n".join(docs)
    prompt = f"""아래 대화 내용을 참고하여 질문에 답변해 주세요.

[대화 내용]
{context}

[질문]
{question}

[답변]"""

    answer = get_llm_answer(prompt, model)
    latency = int((time.time() - start) * 1000)

    return {"answer": answer, "references": docs, "latency_ms": latency, "model": model or "default"}
