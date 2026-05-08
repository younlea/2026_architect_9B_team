import time
import chromadb
from chromadb.utils import embedding_functions
from backend.db.database import get_conn, get_thread_text
from backend.config import CHROMA_PERSIST_DIR, EMBEDDING_MODEL
from backend.rag.llm_client import get_llm_answer

CHUNK_SIZE = 512      # 한국어 기준 약 250 어절 (기존 300자에서 확대)
CHUNK_OVERLAP = 80    # 약 15% overlap으로 문맥 연결 강화
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
    """문장 경계를 인식하는 스마트 청킹.
    마침표/줄바꿈 단위로 먼저 나눈 뒤 CHUNK_SIZE를 초과하면 청크를 확정하고,
    이전 청크의 마지막 CHUNK_OVERLAP 글자를 다음 청크 앞에 붙여 문맥을 이어줍니다.
    """
    import re
    # 한국어 문장 구분자: 마침표/느낌표/물음표 + 공백 or 줄바꿈
    sentences = re.split(r'(?<=[.!?\n])\s*', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) + 1 > CHUNK_SIZE and current:
            chunks.append(current.strip())
            # overlap: 이전 청크 끝부분을 다음 청크 앞에 붙임
            overlap_text = current[-CHUNK_OVERLAP:] if len(current) > CHUNK_OVERLAP else current
            current = overlap_text + " " + sent
        else:
            current = (current + " " + sent).strip() if current else sent

    if current.strip():
        chunks.append(current.strip())

    # 매우 긴 단일 문장 처리 (문장 구분자가 없는 경우)
    result = []
    for chunk in chunks:
        if len(chunk) <= CHUNK_SIZE * 2:
            result.append(chunk)
        else:
            # 강제로 CHUNK_SIZE 단위로 분할
            start = 0
            while start < len(chunk):
                result.append(chunk[start:start + CHUNK_SIZE])
                start += CHUNK_SIZE - CHUNK_OVERLAP
    return [c for c in result if c.strip()]


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
    count = col.count()
    if count == 0:
        return {"answer": "", "references": [], "latency_ms": 0, "model": model or "default"}
    results = col.query(query_texts=[question], n_results=min(TOP_K, count))
    docs = results["documents"][0] if results["documents"] else []

    context = "\n\n".join(docs)
    prompt = f"""아래 대화 내용을 참고하여 질문에 답변해 주세요.
(주의: 반드시 질문과 동일한 언어로 답변을 작성해야 합니다.)

[대화 내용]
{context}

[질문]
{question}

[답변]"""
    answer = get_llm_answer(prompt, model)
    latency = int((time.time() - start) * 1000)
    return {"answer": answer, "references": docs, "latency_ms": latency, "model": model or "default"}
