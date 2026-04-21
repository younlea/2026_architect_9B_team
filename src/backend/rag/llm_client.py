import requests as _requests
from backend.config import LLM_PROVIDER, OPENAI_API_KEY, OPENAI_MODEL, OLLAMA_BASE_URL, OLLAMA_MODEL


def get_llm_answer(prompt: str, model: str = None) -> str:
    if LLM_PROVIDER == "ollama":
        return _ollama(prompt, model or OLLAMA_MODEL)
    return _openai(prompt, model or OPENAI_MODEL)


def list_ollama_models() -> list[str]:
    try:
        resp = _requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except Exception:
        return [OLLAMA_MODEL]


def _openai(prompt: str, model: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
    )
    return resp.choices[0].message.content.strip()


def _ollama(prompt: str, model: str) -> str:
    resp = _requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=180,
    )
    resp.raise_for_status()
    return resp.json().get("response", "").strip()
