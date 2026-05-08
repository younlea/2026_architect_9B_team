import asyncio
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.db.database import get_conn
from backend.rag import basic_rag, raptor_rag, roi_rag
from backend.rag.llm_client import list_ollama_models
from backend.config import OLLAMA_MODEL

router = APIRouter(prefix="/api", tags=["rag"])


class CompareRequest(BaseModel):
    session_id: Optional[str] = None
    thread_id: Optional[str] = None
    query: str
    model: Optional[str] = None


class MultiModelCompareRequest(BaseModel):
    session_id: Optional[str] = None
    thread_id: Optional[str] = None
    query: str
    models: list[str] = []


def _get_indexed_status(
    session_id: str = None, thread_id: str = None
) -> dict:
    """인덱싱 상태를 반환합니다. basic/raptor 미인덱싱 시 HTTPException."""
    with get_conn() as conn:
        if thread_id:
            t = conn.execute(
                "SELECT basic_indexed, raptor_indexed, roi_indexed FROM threads WHERE id=?",
                (thread_id,),
            ).fetchone()
            if not t:
                raise HTTPException(status_code=404, detail="Thread not found")
            if not t["basic_indexed"] or not t["raptor_indexed"]:
                raise HTTPException(
                    status_code=400,
                    detail="Thread not indexed yet. Call /threads/{id}/index first.",
                )
            return {
                "basic": True,
                "raptor": True,
                "roi": bool(t["roi_indexed"]) if t["roi_indexed"] is not None else False,
            }
        else:
            sess = conn.execute(
                "SELECT is_indexed FROM sessions WHERE id=?", (session_id,)
            ).fetchone()
            if not sess:
                raise HTTPException(status_code=404, detail="Session not found")
            if not sess["is_indexed"]:
                raise HTTPException(
                    status_code=400,
                    detail="Session not indexed yet. Call /index first.",
                )
            return {"basic": True, "raptor": True, "roi": False}


@router.get("/models")
def get_models():
    """Ollama에서 사용 가능한 모델 목록 반환"""
    models = list_ollama_models()
    return {"models": models, "default": OLLAMA_MODEL}


@router.post("/rag/compare")
async def compare_rag(body: CompareRequest):
    if not body.session_id and not body.thread_id:
        raise HTTPException(status_code=400, detail="session_id or thread_id required")

    status = _get_indexed_status(body.session_id, body.thread_id)
    loop = asyncio.get_event_loop()

    use_thread = bool(body.thread_id)

    if use_thread:
        basic_fn = lambda: basic_rag.query_thread(body.thread_id, body.query, body.model)
        raptor_fn = lambda: raptor_rag.query_thread(body.thread_id, body.query, body.model)
        roi_fn = lambda: roi_rag.query_thread(body.thread_id, body.query, body.model)
    else:
        basic_fn = lambda: basic_rag.query(body.session_id, body.query, body.model)
        raptor_fn = lambda: raptor_rag.query(body.session_id, body.query, body.model)
        roi_fn = None  # 세션 레벨 ROI 미지원

    futures = [
        loop.run_in_executor(None, basic_fn),
        loop.run_in_executor(None, raptor_fn),
    ]
    if status["roi"] and roi_fn:
        futures.append(loop.run_in_executor(None, roi_fn))

    done = await asyncio.gather(*futures)
    basic_result, raptor_result = done[0], done[1]
    roi_result = done[2] if len(done) > 2 else None

    ctx_id = body.thread_id or body.session_id
    with get_conn() as conn:
        if use_thread:
            conn.execute(
                """INSERT INTO thread_rag_results
                   (thread_id, query, basic_rag_answer, basic_rag_latency_ms,
                    raptor_rag_answer, raptor_rag_latency_ms,
                    roi_rag_answer, roi_rag_latency_ms, model_name)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    ctx_id, body.query,
                    basic_result["answer"], basic_result["latency_ms"],
                    raptor_result["answer"], raptor_result["latency_ms"],
                    roi_result["answer"] if roi_result else None,
                    roi_result["latency_ms"] if roi_result else None,
                    body.model or OLLAMA_MODEL,
                ),
            )
        else:
            conn.execute(
                """INSERT INTO rag_results
                   (session_id, query, basic_rag_answer, basic_rag_latency_ms,
                    raptor_rag_answer, raptor_rag_latency_ms)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    ctx_id, body.query,
                    basic_result["answer"], basic_result["latency_ms"],
                    raptor_result["answer"], raptor_result["latency_ms"],
                ),
            )

    return {
        "query": body.query,
        "model": body.model or OLLAMA_MODEL,
        "context_type": "thread" if use_thread else "session",
        "basic_rag": basic_result,
        "raptor_rag": raptor_result,
        "roi_rag": roi_result,
    }


@router.get("/rag/results/thread/{thread_id}")
def get_thread_results(thread_id: str):
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, query, basic_rag_answer, basic_rag_latency_ms,
                      raptor_rag_answer, raptor_rag_latency_ms,
                      roi_rag_answer, roi_rag_latency_ms,
                      model_name, created_at
               FROM thread_rag_results WHERE thread_id=? ORDER BY created_at DESC""",
            (thread_id,),
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("/rag/multimodel")
async def multimodel_compare(body: MultiModelCompareRequest):
    """여러 모델로 Basic + RAPTOR + ROI-RAG를 동시에 실행하여 비교"""
    status = _get_indexed_status(body.session_id, body.thread_id)

    target_models = body.models if body.models else list_ollama_models()
    use_thread = bool(body.thread_id)
    loop = asyncio.get_event_loop()

    tasks = []
    for m in target_models:
        if use_thread:
            tasks.append(("basic", m, loop.run_in_executor(
                None, basic_rag.query_thread, body.thread_id, body.query, m
            )))
            tasks.append(("raptor", m, loop.run_in_executor(
                None, raptor_rag.query_thread, body.thread_id, body.query, m
            )))
            if status["roi"]:
                tasks.append(("roi", m, loop.run_in_executor(
                    None, roi_rag.query_thread, body.thread_id, body.query, m
                )))
        else:
            tasks.append(("basic", m, loop.run_in_executor(
                None, basic_rag.query, body.session_id, body.query, m
            )))
            tasks.append(("raptor", m, loop.run_in_executor(
                None, raptor_rag.query, body.session_id, body.query, m
            )))

    results = []
    futures = [t[2] for t in tasks]
    done = await asyncio.gather(*futures, return_exceptions=True)

    with get_conn() as conn:
        for (rag_type, model_name, _), result in zip(tasks, done):
            if isinstance(result, Exception):
                entry = {
                    "rag_type": rag_type, "model": model_name,
                    "answer": f"오류: {str(result)}", "latency_ms": 0, "references": [],
                }
            else:
                entry = {"rag_type": rag_type, "model": model_name, **result}
                conn.execute(
                    """INSERT INTO model_compare_results
                       (session_id, query, rag_type, model_name, answer, latency_ms)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (
                        body.session_id, body.query, rag_type, model_name,
                        result["answer"], result["latency_ms"],
                    ),
                )
            results.append(entry)

    return {"query": body.query, "results": results}


@router.get("/rag/results/{session_id}")
def get_results(session_id: str):
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, query, basic_rag_answer, basic_rag_latency_ms,
                      raptor_rag_answer, raptor_rag_latency_ms, created_at
               FROM rag_results WHERE session_id = ? ORDER BY created_at DESC""",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


@router.get("/rag/multimodel/results/{session_id}")
def get_multimodel_results(session_id: str):
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, query, rag_type, model_name, answer, latency_ms, created_at
               FROM model_compare_results WHERE session_id = ?
               ORDER BY created_at DESC, model_name, rag_type""",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]
