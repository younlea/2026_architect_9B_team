"""세 RAG 모듈이 공유하는 SentenceTransformer EF 싱글톤."""
import threading
from chromadb.utils import embedding_functions
from backend.config import EMBEDDING_MODEL

_instance = None
_lock = threading.Lock()


def get():
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = embedding_functions.SentenceTransformerEmbeddingFunction(
                    model_name=EMBEDDING_MODEL,
                    device="cpu",
                )
    return _instance
