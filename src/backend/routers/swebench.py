"""
SWE-bench PoC 라우터

GET  /api/swebench/status              — 인덱싱 현황 (RAG 방식별)
POST /api/swebench/index               — 데이터 다운로드 & 인덱싱
GET  /api/swebench/issues              — 이슈 목록
GET  /api/swebench/issues/{id}         — 이슈 상세
POST /api/swebench/retrieve            — RAG 방식 × 검색 전략 조합 실행
POST /api/swebench/evaluate            — RAGAS 스타일 메트릭 평가 (SSE)

RAG 인덱싱 방식 (DB 구축):
  BasicRAG  — 고정 청킹 + 코사인 Top-K
  RaptorRAG — DBSCAN 클러스터링 + 계층 노드 (LLM 요약 없음)
  ROIRAG    — kNN 기반 EU 구성 + 단일 ANN   (LLM 요약 없음)

검색 전략:
  Flat       — 통합 DB에서 코사인 Top-K 직접 추출
  PostFilter — 통합 DB Top-N 추출 → repo/version 필터 → Top-K
  Routed     — repo+version으로 파티션 DB 직접 선택 → Top-K

조합: 3 × 3 = 최대 9가지 비교
"""
import json
import asyncio
import chromadb
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from backend.db.database import get_conn
from backend.config import CHROMA_PERSIST_DIR
from backend.rag.swebench_rag_engines import (
    ALL_ENGINES, ALL_STRATEGIES, TOP_K,
    index_issue, retrieve as engine_retrieve, engine_status,
)
from backend.scripts.load_swebench import get_index_status

router = APIRouter(prefix="/api/swebench", tags=["swebench"])

DATASET_IDS = {
    "lite": "princeton-nlp/SWE-bench_Lite",
    "full": "princeton-nlp/SWE-bench",
}


class RetrieveRequest(BaseModel):
    instance_id: str
    rag_methods: list[str] = ALL_ENGINES       # BasicRAG, RaptorRAG, ROIRAG
    strategies:  list[str] = ALL_STRATEGIES    # Flat, PostFilter, Routed


def _sse(data: dict) -> str:
    return f"data: {json.dumps(data, ensure_ascii=False)}\n\n"


# ── RAGAS 메트릭 ──────────────────────────────────────────────────────────────

def _get_fps(chunk: dict) -> list[str]:
    fps = chunk.get("file_paths")
    if fps:
        if isinstance(fps, list):
            return fps
        return [f.strip() for f in str(fps).split(",") if f.strip()]
    fp = chunk.get("file_path", "")
    return [fp] if fp else []


def _hit_metric(chunks: list, answer_files: list) -> bool:
    answer_set = set(answer_files)
    return any(any(fp in answer_set for fp in _get_fps(c)) for c in chunks)


def _context_precision(chunks: list, answer_files: list, k: int) -> float:
    if not chunks or not answer_files:
        return 0.0
    answer_set = set(answer_files)
    relevant = sum(1 for c in chunks[:k] if any(fp in answer_set for fp in _get_fps(c)))
    return relevant / min(k, len(chunks))


def _context_recall(chunks: list, answer_files: list, k: int) -> float:
    if not answer_files:
        return 1.0
    retrieved = {fp for c in chunks[:k] for fp in _get_fps(c)}
    return sum(1 for f in answer_files if f in retrieved) / len(answer_files)


def _mrr(chunks: list, answer_files: list) -> float:
    answer_set = set(answer_files)
    for i, c in enumerate(chunks):
        if any(fp in answer_set for fp in _get_fps(c)):
            return 1.0 / (i + 1)
    return 0.0


# ── 엔드포인트 ────────────────────────────────────────────────────────────────

@router.post("/clear")
def clear_data(rag_engines: str = ""):
    """인덱스 데이터 전체 삭제 (ChromaDB 컬렉션 + SQLite 이슈).

    rag_engines: 콤마 구분 (예: "BasicRAG,RaptorRAG"). 빈 값이면 전체 삭제.
    """
    selected = (
        [e.strip() for e in rag_engines.split(",") if e.strip() in ALL_ENGINES]
        if rag_engines else list(ALL_ENGINES)
    ) or list(ALL_ENGINES)

    from backend.rag.swebench_rag_engines import _flat_col, _FLAT_COL_MAP
    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

    # 모든 컬렉션 조회
    all_cols = [c.name for c in client.list_collections()]

    # 삭제할 컬렉션 prefix 결정
    prefixes = []
    for eng in selected:
        prefixes.append(_flat_col(eng))  # flat 컬렉션 이름 자체

    deleted_cols = []
    for col_name in all_cols:
        for prefix in prefixes:
            if col_name == prefix or col_name.startswith(prefix + "_"):
                try:
                    client.delete_collection(col_name)
                    deleted_cols.append(col_name)
                except Exception:
                    pass
                break

    # SQLite 이슈 삭제 (전체 엔진 삭제 시에만)
    deleted_issues = 0
    if set(selected) == set(ALL_ENGINES):
        with get_conn() as conn:
            deleted_issues = conn.execute(
                "SELECT COUNT(*) as cnt FROM swebench_issues"
            ).fetchone()["cnt"]
            conn.execute("DELETE FROM swebench_issues")

    return {
        "deleted_collections": deleted_cols,
        "deleted_collection_count": len(deleted_cols),
        "deleted_issues": deleted_issues,
        "engines_cleared": selected,
    }


@router.get("/status")
def get_status():
    base = get_index_status()
    base["engines"] = engine_status()
    return base


@router.get("/issues")
def list_issues():
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT instance_id, repo, version,
                      substr(problem_statement, 1, 120) as preview
               FROM swebench_issues ORDER BY repo, version, instance_id"""
        ).fetchall()
    return [dict(r) for r in rows]


@router.get("/issues/{instance_id}")
def get_issue(instance_id: str):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM swebench_issues WHERE instance_id=?", (instance_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Issue not found")
    d = dict(row)
    d["answer_files"] = json.loads(d.get("answer_files") or "[]")
    return d


@router.post("/retrieve")
async def retrieve(body: RetrieveRequest):
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM swebench_issues WHERE instance_id=?", (body.instance_id,)
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Issue not found. Run /index first.")

    issue = dict(row)
    query = issue["problem_statement"]
    repo, version = issue["repo"], issue["version"]
    answer_files = json.loads(issue.get("answer_files") or "[]")
    instance_id = body.instance_id

    valid_methods = [m for m in body.rag_methods if m in ALL_ENGINES]
    valid_strategies = [s for s in body.strategies if s in ALL_STRATEGIES]

    if not valid_methods or not valid_strategies:
        raise HTTPException(status_code=400,
                            detail="최소 1개의 RAG 방식과 검색 전략을 선택하세요.")

    client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    loop = asyncio.get_event_loop()

    results = []
    for rag in valid_methods:
        for strategy in valid_strategies:
            res = await loop.run_in_executor(
                None,
                lambda r=rag, s=strategy: engine_retrieve(
                    client, r, s, query, repo, version, answer_files
                ),
            )
            results.append(res)

    return {
        "instance_id": instance_id,
        "repo": repo,
        "version": version,
        "answer_files": answer_files,
        "results": results,
    }


@router.post("/evaluate")
async def evaluate(request: Request, max_eval: int = 50, rag_methods: str = "", strategies: str = ""):
    """RAGAS 스타일 메트릭 평가. SSE 스트리밍.

    rag_methods: 콤마 구분 (예: "BasicRAG,RaptorRAG"), 빈 값이면 전체
    strategies:  콤마 구분 (예: "Flat,PostFilter"), 빈 값이면 전체
    """
    selected_methods = (
        [m.strip() for m in rag_methods.split(",") if m.strip() in ALL_ENGINES]
        if rag_methods else list(ALL_ENGINES)
    ) or list(ALL_ENGINES)

    selected_strategies = (
        [s.strip() for s in strategies.split(",") if s.strip() in ALL_STRATEGIES]
        if strategies else list(ALL_STRATEGIES)
    ) or list(ALL_STRATEGIES)

    # 평가할 조합 리스트
    combos = [
        (rag, strat)
        for rag in selected_methods
        for strat in selected_strategies
    ]
    combo_names = [f"{r}+{s}" for r, s in combos]

    async def event_stream():
        with get_conn() as conn:
            rows = conn.execute(
                "SELECT instance_id, repo, version, problem_statement, answer_files "
                "FROM swebench_issues LIMIT ?",
                (max_eval,),
            ).fetchall()

        if not rows:
            yield _sse({"type": "error", "message": "인덱싱된 이슈가 없습니다."})
            return

        total = len(rows)
        yield _sse({"type": "log",
                    "message": f"{total}개 이슈 평가 시작 ({', '.join(combo_names)})..."})

        acc = {name: {"hits": 0, "precision": 0.0, "recall": 0.0, "mrr": 0.0, "n": 0}
               for name in combo_names}

        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        loop = asyncio.get_event_loop()
        for idx, row in enumerate(rows):
            if await request.is_disconnected():
                yield _sse({"type": "cancelled", "message": "사용자가 평가를 중지했습니다."})
                return

            issue = dict(row)
            answer_files = json.loads(issue.get("answer_files") or "[]")
            query = issue["problem_statement"]
            repo, version = issue["repo"], issue["version"]
            instance_id = issue["instance_id"]

            for rag, strat in combos:
                name = f"{rag}+{strat}"
                res = await loop.run_in_executor(
                    None,
                    lambda r=rag, s=strat, q=query, repo_=repo, ver_=version, af=answer_files:
                        engine_retrieve(client, r, s, q, repo_, ver_, af),
                )
                if res.get("error") or not res.get("chunks"):
                    continue
                chunks = res["chunks"]
                acc[name]["hits"] += 1 if _hit_metric(chunks, answer_files) else 0
                acc[name]["precision"] += _context_precision(chunks, answer_files, TOP_K)
                acc[name]["recall"] += _context_recall(chunks, answer_files, TOP_K)
                acc[name]["mrr"] += _mrr(chunks, answer_files)
                acc[name]["n"] += 1

            yield _sse({"type": "progress", "current": idx + 1,
                        "total": total, "instance_id": instance_id})
            await asyncio.sleep(0)

        final = {}
        for name in combo_names:
            n = acc[name]["n"] or 1
            final[name] = {
                "hit_at_k":          round(acc[name]["hits"]      / n, 4),
                "context_precision": round(acc[name]["precision"] / n, 4),
                "context_recall":    round(acc[name]["recall"]    / n, 4),
                "mrr":               round(acc[name]["mrr"]       / n, 4),
                "evaluated":         acc[name]["n"],
            }

        yield _sse({"type": "done", "metrics": final, "total": total})

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )


@router.post("/index")
async def index_data(request: Request, dataset: str = "lite", max_issues: int = 300, rag_engines: str = ""):
    """SWE-bench 데이터 다운로드 & 인덱싱. SSE 진행률 스트리밍.

    dataset    : "lite" | "full"
    max_issues : 최대 이슈 수
    rag_engines: 인덱싱할 RAG 방식 (콤마 구분, 예: "BasicRAG,RaptorRAG,ROIRAG")
                 빈 값이면 전체 3가지 인덱싱
    """
    dataset_id = DATASET_IDS.get(dataset, DATASET_IDS["lite"])
    selected_engines = (
        [e.strip() for e in rag_engines.split(",") if e.strip() in ALL_ENGINES]
        if rag_engines else list(ALL_ENGINES)
    ) or list(ALL_ENGINES)

    async def event_stream():
        from datasets import load_dataset
        from backend.db.database import init_db
        from backend.scripts.load_swebench import _parse_patch_files, _save_issue

        init_db()
        label = "Lite (~300건)" if dataset == "lite" else "Full (~2294건)"
        engines_label = ", ".join(selected_engines)
        yield _sse({"type": "log",
                    "message": f"SWE-bench {label} 로딩 중... (RAG 방식: {engines_label})"})

        try:
            ds = load_dataset(dataset_id, split="test")
        except Exception as e:
            yield _sse({"type": "error", "message": f"데이터셋 로드 실패: {e}"})
            return

        total = min(max_issues, len(ds))
        yield _sse({"type": "log", "message": f"{total}개 이슈 인덱싱 시작..."})

        client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)

        for idx in range(total):
            if await request.is_disconnected():
                yield _sse({"type": "cancelled",
                            "message": f"사용자가 인덱싱을 중지했습니다. ({idx}개 완료)",
                            "indexed_so_far": idx})
                return

            row = ds[idx]
            instance_id = row["instance_id"]
            repo = row["repo"]
            version = str(row["version"])
            patch = row["patch"]

            answer_files = _parse_patch_files(patch)
            _save_issue(row, answer_files)

            chunk_counts = {}
            for engine in selected_engines:
                try:
                    n = index_issue(client, engine, instance_id, patch, repo, version)
                    chunk_counts[engine] = n
                except Exception:
                    chunk_counts[engine] = 0

            yield _sse({
                "type": "progress",
                "current": idx + 1,
                "total": total,
                "instance_id": instance_id,
                "chunk_counts": chunk_counts,
                "engines": selected_engines,
            })
            await asyncio.sleep(0)

        yield _sse({
            "type": "done",
            "message": f"완료: {total}개 이슈 ({', '.join(selected_engines)}) 인덱싱",
            "issue_count": total,
        })

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={"X-Accel-Buffering": "no", "Cache-Control": "no-cache"},
    )
