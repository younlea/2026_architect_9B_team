"""Measure Groq LLM latency for DP3 PoC prompts.

Examples:
    python scripts/dp3_groq_latency_check.py --mode short --runs 10
    python scripts/dp3_groq_latency_check.py --mode long --runs 10 --interval 62
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import requests


DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_MODEL = "llama-3.1-8b-instant"


def main() -> int:
    args = parse_args()
    api_key = get_env("GROQ_API_KEY")
    if not api_key:
        print("GROQ_API_KEY is not set.", file=sys.stderr)
        return 1

    base_url = get_env("GROQ_BASE_URL") or DEFAULT_BASE_URL
    url = f"{base_url.rstrip('/')}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    prompt = build_prompt(args.mode)

    print(f"model={args.model}")
    print(f"mode={args.mode}")
    print(f"prompt_chars={len(prompt)}")
    print(f"prompt_words={len(prompt.split())}")
    print(f"rough_char4_tokens={len(prompt) // 4}")
    print(f"runs={args.runs}")
    print(f"interval_seconds={args.interval}, excluded from latency")

    successes: list[dict] = []
    errors: list[dict] = []
    for idx in range(1, args.runs + 1):
        if idx > 1 and args.interval > 0:
            time.sleep(args.interval)

        payload = {
            "model": args.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": args.max_tokens,
        }
        start = time.perf_counter()
        resp = requests.post(url, headers=headers, json=payload, timeout=args.timeout)
        elapsed_ms = (time.perf_counter() - start) * 1000
        headers_subset = rate_limit_headers(resp)

        if resp.status_code == 200:
            data = resp.json()
            usage = data.get("usage", {})
            record = {
                "index": idx,
                "http_ms": round(elapsed_ms, 1),
                "usage": usage,
                "headers": headers_subset,
            }
            successes.append(record)
            print(
                f"{idx}/{args.runs} OK "
                f"http_ms={record['http_ms']} "
                f"usage_total={usage.get('total_tokens')} "
                f"prompt={usage.get('prompt_tokens')} "
                f"completion={usage.get('completion_tokens')} "
                f"queue={usage.get('queue_time')} "
                f"model_total={usage.get('total_time')} "
                f"headers={headers_subset}"
            )
        else:
            body = resp.text[:500].replace("\n", " ")
            record = {
                "index": idx,
                "http_ms": round(elapsed_ms, 1),
                "status": resp.status_code,
                "body_head": body,
                "headers": headers_subset,
            }
            errors.append(record)
            print(
                f"{idx}/{args.runs} ERROR "
                f"status={resp.status_code} "
                f"http_ms={record['http_ms']} "
                f"headers={headers_subset} "
                f"body={body}"
            )

    print_summary(successes, errors)
    if args.output:
        write_json(args.output, args, prompt, successes, errors)
    return 0 if not errors else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure Groq latency for DP3 PoC prompts.")
    parser.add_argument("--mode", choices=["short", "long"], default="short")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--interval", type=float, default=0.0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=240.0)
    parser.add_argument("--output", help="Optional JSON output path.")
    return parser.parse_args()


def build_prompt(mode: str) -> str:
    if mode == "short":
        context = (
            "DP3 cache PoC evaluates answer cache reuse, validation metadata, "
            "permission scope, version freshness, and RAG fallback latency."
        )
        return (
            "Read the following synthetic context and answer with exactly one short sentence.\n\n"
            f"[Context]\n{context}\n\n"
            "[Question]\nWhat is this synthetic context mainly about?"
        )

    base_sentence = (
        "This is a synthetic RAG context sentence about cache validation, evidence units, "
        "metadata fingerprints, user permissions, document versions, retrieval latency, and answer grounding. "
    )
    target_words = 4100
    repeat_count = max(1, target_words // len(base_sentence.split()))
    context = (base_sentence * repeat_count).strip()
    return (
        "Read the following synthetic context and answer with exactly one short sentence.\n\n"
        f"[Context]\n{context}\n\n"
        "[Question]\nWhat is this synthetic context mainly about?"
    )


def get_env(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value

    dotenv_value = read_dotenv(name)
    if dotenv_value:
        return dotenv_value

    if os.name == "nt":
        try:
            import winreg

            with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
                value, _ = winreg.QueryValueEx(key, name)
                return value
        except OSError:
            return None
    return None


def read_dotenv(name: str) -> str | None:
    dotenv = Path("src/.env")
    if not dotenv.exists():
        return None
    for line in dotenv.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == name:
            return value.strip().strip('"').strip("'")
    return None


def rate_limit_headers(resp: requests.Response) -> dict[str, str]:
    return {
        key: value
        for key, value in resp.headers.items()
        if "ratelimit" in key.lower() or key.lower() == "retry-after"
    }


def print_summary(successes: list[dict], errors: list[dict]) -> None:
    print("--- summary: HTTP round-trip only, sleeps excluded ---")
    print(f"success {len(successes)} errors {len(errors)}")
    if not successes:
        if errors:
            print(f"first_error {errors[0]}")
        return

    latencies = [item["http_ms"] for item in successes]
    usages = [item["usage"] for item in successes]
    total_tokens = [usage.get("total_tokens") for usage in usages if usage.get("total_tokens") is not None]
    queue_times = [usage.get("queue_time") for usage in usages if usage.get("queue_time") is not None]
    model_times = [usage.get("total_time") for usage in usages if usage.get("total_time") is not None]

    print(f"min_ms={min(latencies):.1f}")
    print(f"max_ms={max(latencies):.1f}")
    print(f"avg_ms={statistics.mean(latencies):.1f}")
    if total_tokens:
        print(
            f"tokens_min={min(total_tokens)} "
            f"tokens_max={max(total_tokens)} "
            f"tokens_avg={statistics.mean(total_tokens):.1f}"
        )
    if queue_times:
        print(f"queue_time_avg_s={statistics.mean(queue_times):.3f} max_s={max(queue_times):.3f}")
    if model_times:
        print(f"groq_model_total_time_avg_s={statistics.mean(model_times):.3f} max_s={max(model_times):.3f}")
    if errors:
        print(f"first_error {errors[0]}")


def write_json(path: str, args: argparse.Namespace, prompt: str, successes: list[dict], errors: list[dict]) -> None:
    output = {
        "model": args.model,
        "mode": args.mode,
        "runs": args.runs,
        "interval_seconds": args.interval,
        "max_tokens": args.max_tokens,
        "prompt_chars": len(prompt),
        "prompt_words": len(prompt.split()),
        "successes": successes,
        "errors": errors,
    }
    Path(path).write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
