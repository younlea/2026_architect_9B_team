import os
import re
import threading
import time
from collections import deque

from backend.config import (
    DP3_LLM_PROVIDER,
    DP3_MOCK_LLM,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MAX_RETRIES,
    GROQ_MAX_OUTPUT_TOKENS,
    GROQ_MIN_INTERVAL_SECONDS,
    GROQ_MODEL,
    GROQ_RATE_LIMIT_SAFETY,
)

_GROQ_LOCK = threading.Lock()
_GROQ_LAST_CALL_AT = 0.0
_GROQ_CALL_WINDOW = deque()

_GROQ_MODEL_LIMITS = {
    "allam-2-7b": {"rpm": 30, "tpm": 6000},
    "groq/compound": {"rpm": 30, "tpm": 70000},
    "groq/compound-mini": {"rpm": 30, "tpm": 70000},
    "llama-3.1-8b-instant": {"rpm": 30, "tpm": 6000},
    "llama-3.3-70b-versatile": {"rpm": 30, "tpm": 12000},
    "meta-llama/llama-4-scout-17b-16e-instruct": {"rpm": 30, "tpm": 30000},
    "meta-llama/llama-prompt-guard-2-22m": {"rpm": 30, "tpm": 15000},
    "meta-llama/llama-prompt-guard-2-86m": {"rpm": 30, "tpm": 15000},
    "openai/gpt-oss-120b": {"rpm": 30, "tpm": 8000},
    "openai/gpt-oss-20b": {"rpm": 30, "tpm": 8000},
    "openai/gpt-oss-safeguard-20b": {"rpm": 30, "tpm": 8000},
    "qwen/qwen3-32b": {"rpm": 60, "tpm": 6000},
    "qwen/qwen3.6-27b": {"rpm": 30, "tpm": 8000},
}


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
    return get_dp3_answer_with_metadata(prompt, model, provider)["answer"]


def get_dp3_answer_with_metadata(prompt: str, model: str = None, provider: str | None = None) -> dict:
    selected = _selected_provider(provider)
    if selected == "mock":
        preview = " ".join(prompt.split())[:160]
        return {
            "answer": f"[MOCK ANSWER] DP3 Answer Cache PoC ?? ?????. prompt_preview={preview}",
            "provider": "mock",
            "model": model,
            "usage": {
                "estimated_prompt_tokens": _estimate_prompt_tokens(prompt),
                "estimated_total_tokens": _estimate_groq_tokens(prompt),
            },
        }
    if selected == "groq":
        return _groq(prompt, model or GROQ_MODEL)
    from backend.rag.llm_client import get_llm_answer
    return {
        "answer": get_llm_answer(prompt, model, deterministic=True),
        "provider": selected,
        "model": model,
        "usage": {
            "estimated_prompt_tokens": _estimate_prompt_tokens(prompt),
            "estimated_total_tokens": _estimate_groq_tokens(prompt),
        },
    }


def _groq(prompt: str, model: str) -> str:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set. Add it to src/.env or environment variables.")
    import requests

    prompt, prompt_fit = _fit_prompt_to_groq_limit(prompt, model)
    url = f"{GROQ_BASE_URL.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": GROQ_MAX_OUTPUT_TOKENS,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    last_error = None
    estimated_tokens = _estimate_groq_tokens(prompt)
    for attempt in range(GROQ_MAX_RETRIES + 1):
        _throttle_groq(model, estimated_tokens)
        resp = requests.post(url, headers=headers, json=payload, timeout=120)
        if resp.status_code != 429:
            resp.raise_for_status()
            data = resp.json()
            return {
                "answer": (data["choices"][0]["message"]["content"] or "").strip(),
                "provider": "groq",
                "model": model,
                "usage": data.get("usage") or {},
                "estimated_tokens": estimated_tokens,
                "prompt_fit": prompt_fit,
                "rate_headers": _groq_rate_headers(resp),
            }

        last_error = resp
        if attempt >= GROQ_MAX_RETRIES:
            break
        time.sleep(_groq_retry_wait_seconds(resp, attempt))

    detail = ""
    try:
        detail = last_error.text[:500] if last_error is not None else ""
    except Exception:
        detail = ""
    raise RuntimeError(
        "Groq rate limit exceeded after retries. "
        "Use Mock for large batches, reduce query count, or lower prompt/output tokens. "
        f"limit_hint={_groq_limit_hint(model, estimated_tokens, prompt_fit)} "
        f"last_response={detail} headers={_groq_rate_headers(last_error)}"
    )


def _throttle_groq(model: str, estimated_tokens: int) -> None:
    global _GROQ_LAST_CALL_AT
    min_interval = max(0.0, float(GROQ_MIN_INTERVAL_SECONDS))
    with _GROQ_LOCK:
        while True:
            now = time.monotonic()
            _trim_groq_window(now)
            wait_interval = max(0.0, (_GROQ_LAST_CALL_AT + min_interval) - now)
            wait_window = _groq_window_wait_seconds(model, estimated_tokens, now)
            wait = max(wait_interval, wait_window)
            if wait <= 0:
                _GROQ_LAST_CALL_AT = time.monotonic()
                _GROQ_CALL_WINDOW.append((_GROQ_LAST_CALL_AT, estimated_tokens, model))
                return
            time.sleep(wait)


def _groq_retry_wait_seconds(resp, attempt: int) -> float:
    retry_after = resp.headers.get("retry-after")
    if retry_after:
        try:
            return min(60.0, max(1.0, float(retry_after)))
        except ValueError:
            pass
    return min(60.0, max(2.0, GROQ_MIN_INTERVAL_SECONDS * (attempt + 2)))


def _estimate_groq_tokens(prompt: str) -> int:
    # Conservative approximation for English/Korean mixed RAG prompts.
    prompt_tokens = _estimate_prompt_tokens(prompt)
    return prompt_tokens + max(0, GROQ_MAX_OUTPUT_TOKENS)


def _estimate_prompt_tokens(prompt: str) -> int:
    words = re.findall(r"\S+", prompt)
    by_words = int(len(words) * 1.35) if words else 0
    by_chars = int(len(prompt) / 6.0)
    return max(1, by_words, by_chars)


def _fit_prompt_to_groq_limit(prompt: str, model: str) -> tuple[str, dict]:
    """Trim a single prompt so it can fit inside the model TPM budget.

    Groq exposes rate-limit windows but not a local tokenizer. This estimate is
    deliberately approximate and slightly conservative; the retry path still
    respects server-provided retry-after headers if Groq counts differently.
    """
    raw_tpm = _groq_limits(model)["tpm"]
    output_budget = max(0, int(GROQ_MAX_OUTPUT_TOKENS))
    prompt_budget = max(256, raw_tpm - output_budget - 64)
    prompt_tokens = _estimate_prompt_tokens(prompt)
    if prompt_tokens <= prompt_budget:
        return prompt, {
            "trimmed": False,
            "estimated_prompt_tokens": prompt_tokens,
            "prompt_budget": prompt_budget,
        }

    char_budget = max(512, int(prompt_budget * 6.0))
    marker = "\n\n[... prompt truncated to fit Groq TPM budget ...]\n\n"
    while True:
        head_budget = max(256, int((char_budget - len(marker)) * 0.72))
        tail_budget = max(256, char_budget - len(marker) - head_budget)
        fitted = prompt[:head_budget].rstrip() + marker + prompt[-tail_budget:].lstrip()
        if _estimate_prompt_tokens(fitted) <= prompt_budget or char_budget <= 1024:
            break
        char_budget = int(char_budget * 0.94)
    return fitted, {
        "trimmed": True,
        "estimated_prompt_tokens": prompt_tokens,
        "estimated_fitted_prompt_tokens": _estimate_prompt_tokens(fitted),
        "prompt_budget": prompt_budget,
        "original_chars": len(prompt),
        "fitted_chars": len(fitted),
    }


def _groq_limits(model: str) -> dict:
    return _GROQ_MODEL_LIMITS.get(model, {"rpm": 30, "tpm": 6000})


def _groq_effective_limits(model: str) -> tuple[int, int]:
    limits = _groq_limits(model)
    safety = min(1.0, max(0.1, float(GROQ_RATE_LIMIT_SAFETY)))
    return max(1, int(limits["rpm"] * safety)), max(1, int(limits["tpm"] * safety))


def _trim_groq_window(now: float) -> None:
    while _GROQ_CALL_WINDOW and now - _GROQ_CALL_WINDOW[0][0] >= 60:
        _GROQ_CALL_WINDOW.popleft()


def _groq_window_wait_seconds(model: str, estimated_tokens: int, now: float) -> float:
    rpm, tpm = _groq_effective_limits(model)
    current_requests = len(_GROQ_CALL_WINDOW)
    current_tokens = sum(item[1] for item in _GROQ_CALL_WINDOW)
    # If one raw-safe request is larger than the conservative effective TPM,
    # send it only when the local window is empty, then let the next request wait.
    if estimated_tokens > tpm:
        return 0.0 if current_requests == 0 and current_tokens == 0 else 60.0
    if current_requests < rpm and current_tokens + estimated_tokens <= tpm:
        return 0.0

    waits = []
    if current_requests >= rpm and _GROQ_CALL_WINDOW:
        waits.append(60 - (now - _GROQ_CALL_WINDOW[0][0]) + 0.05)
    if current_tokens + estimated_tokens > tpm:
        running = current_tokens
        for timestamp, tokens, _ in _GROQ_CALL_WINDOW:
            running -= tokens
            if running + estimated_tokens <= tpm:
                waits.append(60 - (now - timestamp) + 0.05)
                break
        else:
            waits.append(60.0)
    return max(0.0, min(waits) if waits else 0.0)


def _groq_limit_hint(model: str, estimated_tokens: int, prompt_fit: dict | None = None) -> str:
    rpm, tpm = _groq_effective_limits(model)
    raw = _groq_limits(model)
    return (
        f"model={model}, estimated_tokens_per_request={estimated_tokens}, "
        f"prompt_fit={prompt_fit or {}}, "
        f"effective_rpm={rpm}/{raw['rpm']}, effective_tpm={tpm}/{raw['tpm']}, "
        f"max_output_tokens={GROQ_MAX_OUTPUT_TOKENS}"
    )


def _groq_rate_headers(resp) -> dict:
    if resp is None:
        return {}
    keys = [
        "retry-after",
        "x-ratelimit-limit-requests",
        "x-ratelimit-limit-tokens",
        "x-ratelimit-remaining-requests",
        "x-ratelimit-remaining-tokens",
        "x-ratelimit-reset-requests",
        "x-ratelimit-reset-tokens",
    ]
    return {key: resp.headers.get(key) for key in keys if resp.headers.get(key) is not None}
