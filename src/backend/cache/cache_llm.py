import os
from backend.config import DP3_MOCK_LLM


def is_mock_llm() -> bool:
    return os.getenv("DP3_MOCK_LLM", DP3_MOCK_LLM).lower() in {"1", "true", "yes", "y"}


def get_dp3_answer(prompt: str, model: str = None) -> str:
    if is_mock_llm():
        preview = " ".join(prompt.split())[:160]
        return f"[MOCK ANSWER] DP3 Answer Cache PoC 임시 답변입니다. prompt_preview={preview}"
    from backend.rag.llm_client import get_llm_answer
    return get_llm_answer(prompt, model, deterministic=True)
