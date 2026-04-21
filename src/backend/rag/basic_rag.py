import time
import chromadb
from chromadb.utils import embedding_functions
from backend.db.database import get_conn, get_thread_text
from backend.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL
from backend.rag.llm_client import get_llm_answer

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 5


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


def _index_text(col_name: str, text: str, id_prefix: str) -> int:
    chunks = _chunk_text(text)
    col = _get_collection(col_name)
    col.upsert(
        documents=chunks,
        ids=[f"{id_prefix}_chunk_{i}" for i in range(len(chunks))],
    )
    return len(chunks)


def index_session(session_id: str):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT speaker, content FROM messages WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
    full_text = "\n".join(f"[{r['speaker']}] {r['content']}" for r in rows)
    _index_text(f"basic_s_{session_id}", full_text, session_id)
    with get_conn() as conn:
        conn.execute("UPDATE sessions SET is_indexed = 1 WHERE id = ?", (session_id,))


def index_thread(thread_id: str) -> int:
    """스레드의 모든 세션 메시지를 합쳐 단일 BasicRAG 인덱스로 구성합니다."""
    full_text = get_thread_text(thread_id)
    chunk_count = _index_text(f"basic_t_{thread_id}", full_text, thread_id)
    with get_conn() as conn:
        conn.execute(
            "UPDATE threads SET basic_indexed=1, basic_chunk_count=? WHERE id=?",
            (chunk_count, thread_id),
        )
    return chunk_count


def query(session_id: str, question: str, model: str = None) -> dict:
    return _query_col(f"basic_s_{session_id}", question, model)


def query_thread(thread_id: str, question: str, model: str = None) -> dict:
    return _query_col(f"basic_t_{thread_id}", question, model)


def _query_col(col_name: str, question: str, model: str = None) -> dict:
    start = time.time()
    col = _get_collection(col_name)
    results = col.query(query_texts=[question], n_results=min(TOP_K, col.count()))
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
