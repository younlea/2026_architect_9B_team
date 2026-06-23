import os

from backend.config import (
    DP3_LLM_PROVIDER,
    DP3_MOCK_LLM,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
)


def _selected_provider(provider: str | None = None) -> str:
    explicit = (provider or os.getenv("DP3_LLM_PROVIDER", DP3_LLM_PROVIDER) or "").strip().lower()
    if explicit:
        return explicit
    legacy_mock = os.getenv("DP3_MOCK_LLM", DP3_MOCK_LLM).lower() in {"1", "true", "yes", "y"}
    return "mock" if legacy_mock else "default"


def is_mock_llm(provider: str | None = None) -> bool:
    return _selected_provider(provider) == "mock"


def get_dp3_llm_provider(provider: str | None = None) -> str:
    return _selected_provider(provider)


def get_dp3_answer(prompt: str, model: str = None, provider: str | None = None) -> str:
    selected = _selected_provider(provider)
    if selected == "mock":
        preview = " ".join(prompt.split())[:160]
        return f"[MOCK ANSWER] DP3 Answer Cache PoC 임시 답변입니다. prompt_preview={preview}"
    if selected == "groq":
        return _groq(prompt, model or GROQ_MODEL)
    from backend.rag.llm_client import get_llm_answer
    return get_llm_answer(prompt, model, deterministic=True)


def _groq(prompt: str, model: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to src/.env or environment variables.")
    import requests

    resp = requests.post(
        f"{GROQ_BASE_URL.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return (data["choices"][0]["message"]["content"] or "").strip()
