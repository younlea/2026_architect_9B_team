#!/usr/bin/env python3
"""Render DP3 PoC run logs into the desktop raw-result Markdown file."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any


LLM_LATENCY_MS = 700.5
LOG_DIR = Path("src/data/dp3_run_logs")
DEFAULT_DOC = Path("DP3_PoC_Result_Raw_Desktop.md")


TIMING_LABELS = {
    "embedding_ms": "Embedding",
    "route_ms": "Route",
    "cache_lookup_ms": "Cache lookup",
    "validation_ms": "Validation",
    "rag_db_ms": "RAG DB",
    "rag_scoring_ms": "RAG scoring",
    "rag_score_sort_ms": "RAG score sort",
    "rag_rerank_ms": "RAG rerank",
    "rag_total_ms": "RAG total",
    "full_retrieval_db_ms": "Full retrieval DB",
    "full_retrieval_scoring_ms": "Full retrieval scoring",
    "full_retrieval_score_sort_ms": "Full retrieval score sort",
    "full_retrieval_rerank_ms": "Full retrieval rerank",
    "full_retrieval_total_ms": "Full retrieval total",
    "prompt_build_ms": "Prompt build",
    "llm_ms": "LLM request",
    "llm_wall_ms": "LLM wall",
    "llm_throttle_wait_ms": "LLM throttle wait",
    "llm_retry_wait_ms": "LLM retry wait",
    "llm_api_reported_queue_ms": "LLM API reported queue",
    "llm_api_reported_total_ms": "LLM API reported total",
    "cache_store_ms": "Cache store",
    "total_ms": "Total",
    "cache_lookup_db_ms": "Cache lookup DB",
    "cache_lookup_scoring_ms": "Cache lookup scoring",
    "valid_current_lookup_ms": "Valid current lookup",
    "delta_retrieval_db_ms": "Delta retrieval DB",
    "delta_retrieval_scoring_ms": "Delta retrieval scoring",
    "delta_retrieval_score_sort_ms": "Delta retrieval score sort",
    "delta_retrieval_rerank_ms": "Delta retrieval rerank",
    "delta_retrieval_filter_ms": "Delta retrieval filter",
    "delta_retrieval_total_ms": "Delta retrieval total",
}

TIMING_ORDERS = {
    "NoCache": [
        "total_ms",
        "TOTAL_LLM",
        "embedding_ms",
        "full_retrieval_db_ms",
        "full_retrieval_scoring_ms",
        "full_retrieval_score_sort_ms",
        "full_retrieval_rerank_ms",
        "full_retrieval_total_ms",
        "prompt_build_ms",
        "llm_ms",
        "llm_wall_ms",
        "llm_throttle_wait_ms",
        "llm_retry_wait_ms",
        "llm_api_reported_queue_ms",
        "llm_api_reported_total_ms",
    ],
    "A first": [
        "total_ms",
        "TOTAL_LLM",
        "embedding_ms",
        "route_ms",
        "cache_lookup_ms",
        "validation_ms",
        "rag_db_ms",
        "rag_scoring_ms",
        "rag_score_sort_ms",
        "rag_rerank_ms",
        "rag_total_ms",
        "prompt_build_ms",
        "llm_ms",
        "llm_wall_ms",
        "llm_throttle_wait_ms",
        "llm_retry_wait_ms",
        "llm_api_reported_queue_ms",
        "llm_api_reported_total_ms",
        "cache_store_ms",
    ],
    "A repeat": [
        "total_ms",
        "TOTAL_LLM",
        "embedding_ms",
        "route_ms",
        "cache_lookup_ms",
        "validation_ms",
        "rag_db_ms",
        "rag_scoring_ms",
        "rag_score_sort_ms",
        "rag_rerank_ms",
        "rag_total_ms",
        "prompt_build_ms",
        "llm_ms",
        "llm_wall_ms",
        "llm_throttle_wait_ms",
        "llm_retry_wait_ms",
        "llm_api_reported_queue_ms",
        "llm_api_reported_total_ms",
        "cache_store_ms",
    ],
    "B first": [
        "total_ms",
        "TOTAL_LLM",
        "embedding_ms",
        "cache_lookup_db_ms",
        "cache_lookup_scoring_ms",
        "cache_lookup_ms",
        "validation_ms",
        "full_retrieval_db_ms",
        "full_retrieval_scoring_ms",
        "full_retrieval_score_sort_ms",
        "full_retrieval_rerank_ms",
        "full_retrieval_total_ms",
        "prompt_build_ms",
        "llm_ms",
        "llm_wall_ms",
        "llm_throttle_wait_ms",
        "llm_retry_wait_ms",
        "llm_api_reported_queue_ms",
        "llm_api_reported_total_ms",
        "cache_store_ms",
    ],
    "B repeat": [
        "total_ms",
        "TOTAL_LLM",
        "embedding_ms",
        "cache_lookup_db_ms",
        "cache_lookup_scoring_ms",
        "cache_lookup_ms",
        "validation_ms",
        "prompt_build_ms",
        "llm_ms",
        "llm_wall_ms",
        "llm_throttle_wait_ms",
        "llm_retry_wait_ms",
        "llm_api_reported_queue_ms",
        "llm_api_reported_total_ms",
    ],
}


def load_log(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    return json.loads(path.read_text())


def device_from_model(value: str | None) -> str:
    if not value or "||device=" not in value:
        return "auto"
    return value.split("||device=", 1)[1].strip().lower() or "auto"


def latest_log(log_dir: Path, predicate: Callable[[dict[str, Any]], bool]) -> Path | None:
    candidates: list[Path] = []
    for path in log_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text())
        except json.JSONDecodeError:
            continue
        if predicate(data):
            candidates.append(path)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def req(data: dict[str, Any], key: str, default: Any = None) -> Any:
    return data.get("request", {}).get(key, default)


def result(data: dict[str, Any]) -> dict[str, Any]:
    return data["result"]


def route_pool(prepared: dict[str, Any]) -> dict[str, Any]:
    return prepared.get("route_pool") or {}


def total_avg(summary: dict[str, Any]) -> float:
    return float(summary["timing_stats_ms"]["total_ms"]["avg"])


def llm_calls(summary: dict[str, Any]) -> int:
    return int(summary.get("llm_calls", summary.get("full_retrievals", 0)))


def llm_provider(data: dict[str, Any]) -> str:
    return str(req(data, "llm_provider") or result(data).get("llm_provider") or "mock").lower()


def is_mock_run(data: dict[str, Any]) -> bool:
    return llm_provider(data) == "mock"


def rag_llm_avg(summary: dict[str, Any], mock_llm: bool) -> float:
    if not mock_llm:
        return total_avg(summary)
    return total_avg(summary) + (llm_calls(summary) / summary["total"]) * LLM_LATENCY_MS


def rag_llm_min_max(total: dict[str, Any], mock_llm: bool, all_rows_call_llm: bool) -> tuple[float, float] | None:
    if not mock_llm:
        return float(total["min"]), float(total["max"])
    if all_rows_call_llm:
        return float(total["min"]) + LLM_LATENCY_MS, float(total["max"]) + LLM_LATENCY_MS
    return None


def route_passed(summary: dict[str, Any]) -> int:
    return int(summary.get("route_passed", 0))


def cache_hits(summary: dict[str, Any]) -> int:
    return int(summary.get("cache_hits", summary.get("context_cache_hits", 0)))


def validation_passed(summary: dict[str, Any]) -> int:
    return int(summary.get("validation_passed", summary.get("validation_full_passed", 0)))


def rag_fallbacks(summary: dict[str, Any]) -> int:
    return int(summary.get("fallbacks", summary.get("full_retrievals", 0)))


def cache_mode_rows(cache_result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        ("NoCache", cache_result["no_cache"]["passes"][0]["summary"]),
        ("A first", cache_result["a"]["passes"][0]["summary"]),
        ("A repeat", cache_result["a"]["passes"][1]["summary"]),
        ("B first", cache_result["b"]["passes"][0]["summary"]),
        ("B repeat", cache_result["b"]["passes"][1]["summary"]),
    ]


def scale_mode_rows(scale: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        ("NoCache", scale["no_cache"]["summary"]),
        ("A first", scale["a_first"]["summary"]),
        ("A repeat", scale["a_repeat"]["summary"]),
        ("B first", scale["b_first"]["summary"]),
        ("B repeat", scale["b_repeat"]["summary"]),
    ]


def similar_pair_mode_rows(pair_result: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    return [
        ("A left_seed", pair_result["a"]["passes"][0]["summary"]),
        ("A right_probe", pair_result["a"]["passes"][1]["summary"]),
        ("B left_seed", pair_result["b"]["passes"][0]["summary"]),
        ("B right_probe", pair_result["b"]["passes"][1]["summary"]),
    ]


def timing_order(mode: str) -> list[str]:
    if mode in TIMING_ORDERS:
        return TIMING_ORDERS[mode]
    if mode == "A left_seed":
        return TIMING_ORDERS["A first"]
    if mode == "A right_probe":
        return TIMING_ORDERS["A repeat"]
    if mode == "B left_seed":
        return TIMING_ORDERS["B first"]
    if mode == "B right_probe":
        return TIMING_ORDERS["B repeat"]
    raise KeyError(mode)


def metric_label(mode: str, key: str) -> str:
    if key == "cache_lookup_ms" and mode.startswith("B"):
        return "Cache lookup total"
    return TIMING_LABELS[key]


def render_setting_rows(rows: list[tuple[str, str]]) -> list[str]:
    out = ["| 항목 | 값 |", "|---|---|"]
    out.extend(f"| {key} | {value} |" for key, value in rows)
    return out


def llm_latency_setting_rows(mock_llm: bool) -> list[tuple[str, str]]:
    if mock_llm:
        return [
            ("LLM latency basis", f"Mock LLM: `{LLM_LATENCY_MS:.1f} ms/call` 가상 보정"),
        ]
    return [
        ("LLM latency basis", "`timings_ms.llm_ms` 실제 HTTP 요청 왕복시간 포함"),
    ]


def llm_formula_line(mock_llm: bool) -> str:
    if mock_llm:
        return f"`Total (RAG+LLM) avg = Total avg + (LLM calls / Total) * {LLM_LATENCY_MS:.1f} ms`"
    return "`Total (RAG+LLM) avg = Total avg` (실제 LLM 실행은 `total_ms`에 이미 포함)"


def render_summary_table(
    rows: list[tuple[str, dict[str, Any]]],
    scale_prefix: str = "",
    mock_llm: bool = True,
) -> list[str]:
    if scale_prefix:
        out = [
            "| Scale | Row count | Mode | Total | Route passed | Cache hit | Validation passed | RAG fallback | LLM calls | Total avg | Total (RAG+LLM) avg |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    else:
        out = [
            "| Mode | Total | Route passed | Cache hit | Validation passed | RAG fallback | LLM calls | Total avg | Total (RAG+LLM) avg |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]

    for name, summary in rows:
        if scale_prefix:
            scale, row_count = scale_prefix.split("|", 1)
            out.append(
                f"| {scale} | {row_count} | {name} | {summary['total']} | "
                f"{route_passed(summary)} | {cache_hits(summary)} | {validation_passed(summary)} | "
                f"{rag_fallbacks(summary)} | {llm_calls(summary)} | {total_avg(summary):.3f} ms | "
                f"{rag_llm_avg(summary, mock_llm):.3f} ms |"
            )
        else:
            out.append(
                f"| {name} | {summary['total']} | {route_passed(summary)} | {cache_hits(summary)} | "
                f"{validation_passed(summary)} | {rag_fallbacks(summary)} | {llm_calls(summary)} | "
                f"{total_avg(summary):.3f} ms | {rag_llm_avg(summary, mock_llm):.3f} ms |"
            )
    return out


def render_decision_reasons(rows: list[tuple[str, dict[str, Any]]], scale_prefix: str = "") -> list[str]:
    if scale_prefix:
        out = ["| Scale | Row count | Mode | Reason | Count |", "|---:|---:|---|---|---:|"]
        scale, row_count = scale_prefix.split("|", 1)
        for mode, summary in rows:
            for reason, count in summary["decision_reasons"].items():
                out.append(f"| {scale} | {row_count} | {mode} | `{reason}` | {count} |")
        return out

    out = ["| Mode | Reason | Count |", "|---|---|---:|"]
    for mode, summary in rows:
        for reason, count in summary["decision_reasons"].items():
            out.append(f"| {mode} | `{reason}` | {count} |")
    return out


def render_timing_table(
    rows: list[tuple[str, dict[str, Any]]],
    modes: list[str],
    mock_llm: bool = True,
) -> list[str]:
    row_map = dict(rows)
    out = ["| Pass | Metric | Min | Max | Avg | N |", "|---|---|---:|---:|---:|---:|"]
    for mode in modes:
        summary = row_map[mode]
        timings = summary["timing_stats_ms"]
        for key in timing_order(mode):
            if key == "TOTAL_LLM":
                total = timings["total_ms"]
                min_max = rag_llm_min_max(
                    total,
                    mock_llm,
                    all_rows_call_llm=mode in {"NoCache", "B first", "B repeat"},
                )
                if min_max is not None:
                    min_value, max_value = min_max
                    out.append(
                        f"| {mode} | Total (RAG+LLM) | {min_value:.3f} ms | "
                        f"{max_value:.3f} ms | {rag_llm_avg(summary, mock_llm):.3f} ms | "
                        f"{total['count']} |"
                    )
                else:
                    out.append(
                        f"| {mode} | Total (RAG+LLM) | - | - | {rag_llm_avg(summary, mock_llm):.3f} ms | "
                        f"{summary['total']} |"
                    )
                continue
            if key not in timings:
                continue
            timing = timings[key]
            out.append(
                f"| {mode} | {metric_label(mode, key)} | {timing['min']:.3f} ms | "
                f"{timing['max']:.3f} ms | {timing['avg']:.3f} ms | {timing['count']} |"
            )
    return out


def render_cache_run_section(
    title: str,
    data: dict[str, Any],
    log_path: Path,
    auto_data: dict[str, Any] | None = None,
    auto_path: Path | None = None,
) -> str:
    cache_result = result(data)
    mock_llm = is_mock_run(data)
    prepared = cache_result["prepared"]
    pool = route_pool(prepared)
    use_reranker = bool(req(data, "use_reranker"))
    posthoc = data.get("posthoc_reranker_device_check") or {}
    requested_device = posthoc.get("requested_device") or device_from_model(req(data, "rerank_model"))
    resolved_device = posthoc.get("resolved_device") or (
        "N/A" if not use_reranker else "TBD"
    )
    reranker_label = "On" if use_reranker else "Off"
    rows = [
        ("Reranker", reranker_label),
        ("Reranker requested device", requested_device if use_reranker else "N/A"),
        ("Reranker resolved device", resolved_device if use_reranker else "N/A"),
        ("Rerank model", "`cross-encoder/ms-marco-MiniLM-L-6-v2`" if use_reranker else "N/A"),
        ("Rerank candidates", str(req(data, "rerank_candidates")) if use_reranker else "N/A"),
        ("Route threshold", str(req(data, "route_threshold"))),
        ("Route pool sample rate", f"{int(float(req(data, 'sample_rate')) * 100)}%"),
        ("Route pool min per dataset", str(req(data, "min_per_dataset"))),
        ("Route pool seed", str(req(data, "pool_seed"))),
        ("Query seed", str(req(data, "seed"))),
        (
            "Route pool",
            f"`ragbench:techqa:test` {pool.get('seeded_questions')}개 / {pool.get('total_questions')}",
        ),
        ("Route pool indexes", "`" + ", ".join(map(str, pool.get("seeded_indexes", []))) + "`"),
        ("Run log file", f"`{log_path.name}`"),
        ("Job ID", f"`{data.get('job_id')}`"),
        ("Saved at", str(data.get("saved_at"))),
    ]

    out = [f"### {title}", "", "#### 세팅", "", *render_setting_rows(rows)]

    if auto_data is not None and auto_path is not None:
        auto_summary = result(auto_data)["no_cache"]["passes"][0]["summary"]
        auto_posthoc = auto_data.get("posthoc_reranker_device_check") or {}
        auto_timing = auto_summary["timing_stats_ms"]["full_retrieval_rerank_ms"]
        out.extend(
            [
                "",
                "#### Auto 실행 참고",
                "",
                *render_setting_rows(
                    [
                        ("Run log file", f"`{auto_path.name}`"),
                        ("Reranker requested device", auto_posthoc.get("requested_device", "auto")),
                        ("Post-hoc resolved device", auto_posthoc.get("resolved_device", "TBD")),
                        ("NoCache full_retrieval_rerank avg", f"{auto_timing['avg']:.3f} ms"),
                        ("판단 근거", "`CrossEncoder._target_device` 및 `predict()` 동작 확인"),
                    ]
                ),
            ]
        )

    rows_by_mode = cache_mode_rows(cache_result)
    out.extend(["", "#### 전체 요약", "", *render_summary_table(rows_by_mode, mock_llm=mock_llm)])
    out.extend(["", "#### Decision reasons", "", *render_decision_reasons(rows_by_mode)])
    out.extend(["", "#### NoCache timing", "", *render_timing_table(rows_by_mode, ["NoCache"], mock_llm=mock_llm)])
    out.extend(["", "#### A안 timing", "", *render_timing_table(rows_by_mode, ["A first", "A repeat"], mock_llm=mock_llm)])
    out.extend(["", "#### B안 timing", "", *render_timing_table(rows_by_mode, ["B first", "B repeat"], mock_llm=mock_llm)])
    return "\n".join(out)


def render_tc1(
    off_path: Path | None,
    cpu_path: Path | None,
    gpu_path: Path | None,
    auto_path: Path | None,
) -> str | None:
    off = load_log(off_path)
    cpu = load_log(cpu_path)
    gpu = load_log(gpu_path)
    auto = load_log(auto_path)
    if off is None and cpu is None and gpu is None:
        return None

    common = off or cpu or gpu
    assert common is not None
    mock_llm = is_mock_run(common)
    common_result = result(common)
    prepared = common_result["prepared"]
    pool = route_pool(prepared)

    out = [
        "## 2. TC1 Cache Benefit",
        "",
        "TC1은 RAGBench `techqa` 100 rows 기준으로 NoCache, A안 Verified Answer Cache, B안 Incremental Context Cache의 cache off / miss / hit 수행시간을 비교한다.",
        "",
        "### 공통 세팅",
        "",
        *render_setting_rows(
            [
                ("Test Case", "TC1 Cache Benefit"),
                ("Dataset", "RAGBench `techqa`"),
                ("Split", f"`{req(common, 'dataset_split')}`"),
                ("Source ID", f"`{prepared.get('source_id')}`"),
                ("RAG corpus row 수", str(prepared.get("num_examples"))),
                ("Base EU rows", f"{prepared.get('base_eu_count'):,}"),
                ("Versioned EU rows", f"{prepared.get('version_rows'):,}"),
                ("Test query 수", str(req(common, "query_count"))),
                ("Warm-up query 수", str(req(common, "warmup_count"))),
                ("Requested version", "V1"),
                ("User scope", str(req(common, "user_scope"))),
                ("LLM", llm_provider(common)),
                ("Route threshold", f"{float(req(common, 'route_threshold')):.2f}"),
                ("Cache hit threshold", str(req(common, "cache_threshold"))),
                *llm_latency_setting_rows(mock_llm),
                ("Route pool mode", str(req(common, "route_pool_mode"))),
                ("Route pool sample rate", f"{int(float(req(common, 'sample_rate')) * 100)}%"),
                ("Route pool min per dataset", str(pool.get("min_count", req(common, "min_per_dataset")))),
                ("Route pool seed", str(req(common, "pool_seed"))),
                ("Query seed", str(req(common, "seed"))),
            ]
        ),
        "",
        llm_formula_line(mock_llm),
    ]

    if off is not None and off_path is not None:
        out.extend(["", render_cache_run_section("2-1. Reranker Off", off, off_path)])
    if cpu is not None and cpu_path is not None:
        out.extend(["", render_cache_run_section("2-2. Reranker On (CPU)", cpu, cpu_path)])
    if gpu is not None and gpu_path is not None:
        out.extend(
            [
                "",
                render_cache_run_section("2-3. Reranker On (GPU)", gpu, gpu_path, auto, auto_path),
            ]
        )
    return "\n".join(out)


def render_scalability_run_section(
    title: str,
    data: dict[str, Any],
    log_path: Path,
    auto_data: dict[str, Any] | None = None,
    auto_path: Path | None = None,
) -> str:
    scale_result = result(data)
    mock_llm = is_mock_run(data)
    use_reranker = bool(req(data, "use_reranker"))
    posthoc = data.get("posthoc_reranker_device_check") or {}
    requested_device = posthoc.get("requested_device") or device_from_model(req(data, "rerank_model"))
    resolved_device = posthoc.get("resolved_device") or ("N/A" if not use_reranker else "TBD")
    out = [
        f"### {title}",
        "",
        "#### 세팅",
        "",
        *render_setting_rows(
            [
                ("Reranker", "On" if use_reranker else "Off"),
                ("Reranker requested device", requested_device if use_reranker else "N/A"),
                ("Reranker resolved device", resolved_device if use_reranker else "N/A"),
                ("Rerank model", "`cross-encoder/ms-marco-MiniLM-L-6-v2`" if use_reranker else "N/A"),
                ("Rerank candidates", str(req(data, "rerank_candidates")) if use_reranker else "N/A"),
                ("Scale rows", "`" + ", ".join(map(str, scale_result.get("scale_rows", []))) + "`"),
                ("Route threshold", str(req(data, "route_threshold"))),
                ("Route pool sample rate", f"{int(float(req(data, 'sample_rate')) * 100)}%"),
                ("Route pool min per dataset", str(req(data, "min_per_dataset"))),
                ("Route pool seed", str(req(data, "pool_seed"))),
                ("Query seed", str(req(data, "seed"))),
                ("Run log file", f"`{log_path.name}`"),
                ("Job ID", f"`{data.get('job_id')}`"),
                ("Saved at", str(data.get("saved_at"))),
            ]
        ),
    ]
    if auto_data is not None and auto_path is not None:
        auto_scale = result(auto_data)["scales"][0]
        auto_summary = auto_scale["no_cache"]["summary"]
        auto_posthoc = auto_data.get("posthoc_reranker_device_check") or {}
        auto_timing = auto_summary["timing_stats_ms"]["full_retrieval_rerank_ms"]
        out.extend(
            [
                "",
                "#### Auto 실행 참고",
                "",
                *render_setting_rows(
                    [
                        ("Run log file", f"`{auto_path.name}`"),
                        ("Reranker requested device", auto_posthoc.get("requested_device", "auto")),
                        ("Post-hoc resolved device", auto_posthoc.get("resolved_device", "TBD")),
                        ("Scale 1 NoCache full_retrieval_rerank avg", f"{auto_timing['avg']:.3f} ms"),
                        ("판단 근거", "`CrossEncoder._target_device` 및 `predict()` 동작 확인"),
                    ]
                ),
            ]
        )
    out.extend(
        [
            "",
            "#### Scale별 route pool",
            "",
            "| Scale | Row count | Source ID | Base EU rows | Versioned EU rows | Route pool | Route pool indexes |",
            "|---:|---:|---|---:|---:|---:|---|",
        ]
    )

    for scale in scale_result["scales"]:
        prepared = scale["prepared"]
        pool = route_pool(prepared)
        out.append(
            f"| {scale['scale']} | {scale['row_count']} | `{scale['source_id']}` | "
            f"{prepared.get('base_eu_count'):,} | {prepared.get('version_rows'):,} | "
            f"{pool.get('seeded_questions')} / {pool.get('total_questions')} | "
            f"`{', '.join(map(str, pool.get('seeded_indexes', [])))}` |"
        )

    out.extend(
        [
            "",
            "#### Scale별 전체 요약",
            "",
            "| Scale | Row count | Mode | Total | Route passed | Cache hit | Validation passed | RAG fallback | LLM calls | Total avg | Total (RAG+LLM) avg |",
            "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for scale in scale_result["scales"]:
        out.extend(
            render_summary_table(
                scale_mode_rows(scale),
                f"{scale['scale']}|{scale['row_count']}",
                mock_llm=mock_llm,
            )[2:]
        )

    out.extend(
        [
            "",
            "#### Scale별 Decision reasons",
            "",
            "| Scale | Row count | Mode | Reason | Count |",
            "|---:|---:|---|---|---:|",
        ]
    )
    for scale in scale_result["scales"]:
        out.extend(render_decision_reasons(scale_mode_rows(scale), f"{scale['scale']}|{scale['row_count']}")[2:])

    for scale in scale_result["scales"]:
        rows = scale_mode_rows(scale)
        out.extend(["", f"#### Scale {scale['scale']} ({scale['row_count']} rows) NoCache timing", ""])
        out.extend(render_timing_table(rows, ["NoCache"], mock_llm=mock_llm))
        out.extend(["", f"#### Scale {scale['scale']} ({scale['row_count']} rows) A안 timing", ""])
        out.extend(render_timing_table(rows, ["A first", "A repeat"], mock_llm=mock_llm))
        out.extend(["", f"#### Scale {scale['scale']} ({scale['row_count']} rows) B안 timing", ""])
        out.extend(render_timing_table(rows, ["B first", "B repeat"], mock_llm=mock_llm))
    return "\n".join(out)


def render_tc2(
    off_path: Path | None,
    cpu_path: Path | None,
    gpu_path: Path | None,
    auto_path: Path | None,
) -> str | None:
    off = load_log(off_path)
    cpu = load_log(cpu_path)
    gpu = load_log(gpu_path)
    auto = load_log(auto_path)
    if off is None and cpu is None and gpu is None:
        return None

    common = off or cpu or gpu
    assert common is not None
    mock_llm = is_mock_run(common)
    scale_result = result(common)
    out = [
        "## 3. TC2 Scale Cost",
        "",
        "TC2는 RAGBench `techqa`의 row 수를 100, 200, 300으로 늘리며 NoCache, A안, B안의 수행시간 변화를 비교한다.",
        "",
        "### 공통 세팅",
        "",
        *render_setting_rows(
            [
                ("Test Case", "TC2 Scale Cost"),
                ("Dataset", "RAGBench `techqa`"),
                ("Split", f"`{req(common, 'dataset_split')}`"),
                ("Scale rows", "`" + ", ".join(map(str, scale_result.get("scale_rows", []))) + "`"),
                ("Test query 수", str(req(common, "query_count"))),
                ("Warm-up query 수", str(req(common, "warmup_count"))),
                ("Requested version", "V1"),
                ("User scope", str(req(common, "user_scope"))),
                ("LLM", llm_provider(common)),
                ("Route threshold", f"{float(req(common, 'route_threshold')):.2f}"),
                ("Cache hit threshold", str(req(common, "cache_threshold"))),
                *llm_latency_setting_rows(mock_llm),
                ("Route pool mode", str(req(common, "route_pool_mode"))),
                ("Route pool sample rate", f"{int(float(req(common, 'sample_rate')) * 100)}%"),
                ("Route pool min per dataset", str(req(common, "min_per_dataset"))),
                ("Route pool seed", str(req(common, "pool_seed"))),
                ("Query seed", str(req(common, "seed"))),
            ]
        ),
        "",
        llm_formula_line(mock_llm),
    ]
    if off is not None and off_path is not None:
        out.extend(["", render_scalability_run_section("3-1. Reranker Off", off, off_path)])
    if cpu is not None and cpu_path is not None:
        out.extend(["", render_scalability_run_section("3-2. Reranker On (CPU)", cpu, cpu_path)])
    if gpu is not None and gpu_path is not None:
        out.extend(
            [
                "",
                render_scalability_run_section("3-3. Reranker On (GPU)", gpu, gpu_path, auto, auto_path),
            ]
        )
    return "\n".join(out)


def fmt_bool(value: Any) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    return ""


def fmt_ms(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f} ms"


def render_pair_rows(pair_result: dict[str, Any]) -> list[str]:
    out = [
        "| Pair ID | Similarity | Answer Jaccard | A right hit | B right hit | A answers equal | B answers equal | A right decision | B right decision | A right LLM request | B right LLM request | B right throttle wait |",
        "|---|---:|---:|---|---|---|---|---|---|---:|---:|---:|",
    ]
    for pair in pair_result.get("pairs", []):
        a_usage = pair.get("a_right_usage") or {}
        b_usage = pair.get("b_right_usage") or {}
        out.append(
            f"| `{pair.get('pair_id')}` | {float(pair.get('similarity', 0)):.4f} | "
            f"{float(pair.get('answer_jaccard', 0)):.4f} | {fmt_bool(pair.get('a_right_cache_hit'))} | "
            f"{fmt_bool(pair.get('b_right_cache_hit'))} | {fmt_bool(pair.get('a_answers_equal'))} | "
            f"{fmt_bool(pair.get('b_answers_equal'))} | `{pair.get('a_right_decision')}` | "
            f"`{pair.get('b_right_decision')}` | {fmt_ms(a_usage.get('request_ms'))} | "
            f"{fmt_ms(b_usage.get('request_ms'))} | {fmt_ms(b_usage.get('throttle_wait_ms'))} |"
        )
    return out


def render_tc3(gpu_path: Path | None) -> str | None:
    data = load_log(gpu_path)
    if data is None or gpu_path is None:
        return None

    pair_result = result(data)
    prepared = pair_result["prepared"]
    pool = route_pool(prepared)
    mock_llm = is_mock_run(data)
    use_reranker = bool(req(data, "use_reranker"))
    posthoc = data.get("posthoc_reranker_device_check") or {}
    requested_device = posthoc.get("requested_device") or device_from_model(req(data, "rerank_model"))
    resolved_device = posthoc.get("resolved_device") or requested_device
    correction = data.get("posthoc_total_wait_correction") or {}

    out = [
        "## 4. TC3 Similar Query Pair Quality",
        "",
        "TC3은 `techqa` 유사질문 pair에서 A안 Answer Cache 재사용과 B안 Context Cache 재사용의 응답/품질 비교용 raw data를 기록한다.",
        "",
        "### 공통 세팅",
        "",
        *render_setting_rows(
            [
                ("Test Case", "TC3 Similar Query Pair Quality"),
                ("Dataset", "RAGBench `techqa`"),
                ("Split", f"`{req(data, 'dataset_split')}`"),
                ("Source ID", f"`{prepared.get('source_id')}`"),
                ("RAG corpus row 수", str(prepared.get("num_examples"))),
                ("Base EU rows", f"{prepared.get('base_eu_count'):,}"),
                ("Versioned EU rows", f"{prepared.get('version_rows'):,}"),
                ("Pair 수", str(pair_result.get("pair_count"))),
                ("Query 수", str(pair_result.get("query_count"))),
                ("Requested version", "V1"),
                ("User scope", str(req(data, "user_scope"))),
                ("LLM provider", llm_provider(data)),
                ("LLM model", str(req(data, "model"))),
                ("Route threshold", f"{float(req(data, 'route_threshold')):.2f}"),
                ("Cache hit threshold", str(req(data, "cache_threshold"))),
                *llm_latency_setting_rows(mock_llm),
                ("Reranker", "On" if use_reranker else "Off"),
                ("Reranker requested device", requested_device if use_reranker else "N/A"),
                ("Reranker resolved device", resolved_device if use_reranker else "N/A"),
                ("Rerank model", "`cross-encoder/ms-marco-MiniLM-L-6-v2`" if use_reranker else "N/A"),
                ("Rerank candidates", str(req(data, "rerank_candidates")) if use_reranker else "N/A"),
                ("Route pool mode", str(req(data, "route_pool_mode"))),
                ("Route pool sample rate", f"{int(float(req(data, 'sample_rate')) * 100)}%"),
                ("Route pool min per dataset", str(pool.get("min_count", req(data, "min_per_dataset")))),
                ("Route pool seed", str(req(data, "pool_seed"))),
                ("Query seed", str(req(data, "seed"))),
                (
                    "Route pool",
                    f"`ragbench:techqa:test` {pool.get('seeded_questions')}개 / {pool.get('total_questions')}",
                ),
                ("Route pool indexes", "`" + ", ".join(map(str, pool.get("seeded_indexes", []))) + "`"),
                ("RAGAS input", f"`{pair_result.get('ragas_input_path')}`"),
                ("Run log file", f"`{gpu_path.name}`"),
                ("Job ID", f"`{data.get('job_id')}`"),
                ("Saved at", str(data.get("saved_at"))),
                ("Post-hoc total wait correction", str(bool(correction.get("applied"))).lower()),
            ]
        ),
        "",
        llm_formula_line(mock_llm),
    ]

    rows_by_mode = similar_pair_mode_rows(pair_result)
    out.extend(["", "### 전체 요약", "", *render_summary_table(rows_by_mode, mock_llm=mock_llm)])
    out.extend(["", "### Decision reasons", "", *render_decision_reasons(rows_by_mode)])
    out.extend(["", "### A안 timing", "", *render_timing_table(rows_by_mode, ["A left_seed", "A right_probe"], mock_llm=mock_llm)])
    out.extend(["", "### B안 timing", "", *render_timing_table(rows_by_mode, ["B left_seed", "B right_probe"], mock_llm=mock_llm)])
    out.extend(["", "### Pair별 raw", "", *render_pair_rows(pair_result)])
    return "\n".join(out)


def replace_between(text: str, start_heading: str, end_heading: str, replacement: str) -> str:
    start = text.index(start_heading)
    end = text.index(end_heading, start)
    return text[:start] + replacement.rstrip() + "\n\n" + text[end:]


def replace_between_any(
    text: str,
    start_heading: str,
    end_headings: list[str],
    replacement: str,
) -> str:
    start = text.index(start_heading)
    end_candidates = [text.index(heading, start) for heading in end_headings if heading in text[start:]]
    if not end_candidates:
        raise ValueError(f"Could not find any end heading after {start_heading!r}: {end_headings!r}")
    end = min(end_candidates)
    return text[:start] + replacement.rstrip() + "\n\n" + text[end:]


def replace_from(text: str, start_heading: str, replacement: str) -> str:
    start = text.index(start_heading)
    return text[:start] + replacement.rstrip() + "\n"


def resolve_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    def explicit(value: str | None) -> Path | None:
        return Path(value) if value else None

    paths = {
        "tc1_off": explicit(args.tc1_off),
        "tc1_cpu": explicit(args.tc1_cpu),
        "tc1_gpu": explicit(args.tc1_gpu),
        "tc1_auto": explicit(args.tc1_auto),
        "tc2_off": explicit(args.tc2_off),
        "tc2_cpu": explicit(args.tc2_cpu),
        "tc2_gpu": explicit(args.tc2_gpu),
        "tc2_auto": explicit(args.tc2_auto),
        "tc3_gpu": explicit(args.tc3_gpu),
    }
    if not args.latest:
        return paths

    log_dir = Path(args.log_dir)
    paths["tc1_off"] = paths["tc1_off"] or latest_log(
        log_dir,
        lambda data: req(data, "test_case") == "cache" and not bool(req(data, "use_reranker")),
    )
    paths["tc1_cpu"] = paths["tc1_cpu"] or latest_log(
        log_dir,
        lambda data: req(data, "test_case") == "cache"
        and bool(req(data, "use_reranker"))
        and device_from_model(req(data, "rerank_model")) == "cpu",
    )
    paths["tc1_gpu"] = paths["tc1_gpu"] or latest_log(
        log_dir,
        lambda data: req(data, "test_case") == "cache"
        and bool(req(data, "use_reranker"))
        and device_from_model(req(data, "rerank_model")) in {"cuda", "gpu", "cuda:0"},
    )
    paths["tc1_auto"] = paths["tc1_auto"] or latest_log(
        log_dir,
        lambda data: req(data, "test_case") == "cache"
        and bool(req(data, "use_reranker"))
        and device_from_model(req(data, "rerank_model")) == "auto",
    )
    paths["tc2_off"] = paths["tc2_off"] or latest_log(
        log_dir,
        lambda data: req(data, "test_case") == "scalability" and not bool(req(data, "use_reranker")),
    )
    paths["tc2_cpu"] = paths["tc2_cpu"] or latest_log(
        log_dir,
        lambda data: req(data, "test_case") == "scalability"
        and bool(req(data, "use_reranker"))
        and device_from_model(req(data, "rerank_model")) == "cpu",
    )
    paths["tc2_gpu"] = paths["tc2_gpu"] or latest_log(
        log_dir,
        lambda data: req(data, "test_case") == "scalability"
        and bool(req(data, "use_reranker"))
        and device_from_model(req(data, "rerank_model")) in {"cuda", "gpu", "cuda:0"},
    )
    paths["tc2_auto"] = paths["tc2_auto"] or latest_log(
        log_dir,
        lambda data: req(data, "test_case") == "scalability"
        and bool(req(data, "use_reranker"))
        and device_from_model(req(data, "rerank_model")) == "auto",
    )
    paths["tc3_gpu"] = paths["tc3_gpu"] or latest_log(
        log_dir,
        lambda data: req(data, "test_case") == "similar_pair_quality"
        and bool(req(data, "use_reranker"))
        and device_from_model(req(data, "rerank_model")) in {"cuda", "gpu", "cuda:0"},
    )
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--doc", default=str(DEFAULT_DOC))
    parser.add_argument("--log-dir", default=str(LOG_DIR))
    parser.add_argument("--latest", action="store_true", help="Auto-select latest matching logs.")
    parser.add_argument("--write", action="store_true", help="Update the Markdown document in-place.")
    parser.add_argument("--tc1-off")
    parser.add_argument("--tc1-cpu")
    parser.add_argument("--tc1-gpu")
    parser.add_argument("--tc1-auto")
    parser.add_argument("--tc2-off")
    parser.add_argument("--tc2-cpu")
    parser.add_argument("--tc2-gpu")
    parser.add_argument("--tc2-auto")
    parser.add_argument("--tc3-gpu")
    args = parser.parse_args()

    paths = resolve_paths(args)
    tc1 = render_tc1(paths["tc1_off"], paths["tc1_cpu"], paths["tc1_gpu"], paths["tc1_auto"])
    tc2 = render_tc2(paths["tc2_off"], paths["tc2_cpu"], paths["tc2_gpu"], paths["tc2_auto"])
    tc3 = render_tc3(paths["tc3_gpu"])

    if not args.write:
        if tc1:
            print(tc1)
            print()
        if tc2:
            print(tc2)
            print()
        if tc3:
            print(tc3)
        return

    doc_path = Path(args.doc)
    text = doc_path.read_text()
    if tc1:
        text = replace_between(text, "## 2. TC1 Cache Benefit", "## 3. TC2 Scale Cost", tc1)
    if tc2:
        text = replace_between_any(
            text,
            "## 3. TC2 Scale Cost",
            ["## 4. TC3 Mixed Workload Performance", "## 4. TC3 Similar Query Pair Quality"],
            tc2,
        )
    if tc3:
        start_heading = (
            "## 4. TC3 Mixed Workload Performance"
            if "## 4. TC3 Mixed Workload Performance" in text
            else "## 4. TC3 Similar Query Pair Quality"
        )
        text = replace_from(text, start_heading, tc3)
    doc_path.write_text(text)


if __name__ == "__main__":
    main()
