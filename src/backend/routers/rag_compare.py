import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.db.database import get_conn
from backend.rag import basic_rag, raptor_rag
from backend.rag.llm_client import list_ollama_models
from backend.config import OLLAMA_MODEL

router = APIRouter(prefix="/api", tags=["rag"])


class CompareRequest(BaseModel):
    session_id: str
    query: str
    model: str = None          # 단일 모델 지정 (없으면 기본값)


class MultiModelCompareRequest(BaseModel):
    session_id: str
    query: str
    models: list[str] = []     # 빈 리스트면 전체 Ollama 모델 사용


def _check_indexed(session_id: str):
    with get_conn() as conn:
        sess = conn.execute(
            "SELECT is_indexed FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        if not sess["is_indexed"]:
            raise HTTPException(status_code=400, detail="Session not indexed yet. Call /index first.")


@router.get("/models")
def get_models():
    """Ollama에서 사용 가능한 모델 목록 반환"""
    models = list_ollama_models()
    return {"models": models, "default": OLLAMA_MODEL}


@router.post("/rag/compare")
async def compare_rag(body: CompareRequest):
    _check_indexed(body.session_id)

    loop = asyncio.get_event_loop()
    basic_future = loop.run_in_executor(
        None, basic_rag.query, body.session_id, body.query, body.model
    )
    raptor_future = loop.run_in_executor(
        None, raptor_rag.query, body.session_id, body.query, body.model
    )
    basic_result, raptor_result = await asyncio.gather(basic_future, raptor_future)

    with get_conn() as conn:
        conn.execute(
            """INSERT INTO rag_results
               (session_id, query, basic_rag_answer, basic_rag_latency_ms, raptor_rag_answer, raptor_rag_latency_ms)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                body.session_id, body.query,
                basic_result["answer"], basic_result["latency_ms"],
                raptor_result["answer"], raptor_result["latency_ms"],
            ),
        )

    return {
        "query": body.query,
        "model": body.model or OLLAMA_MODEL,
        "basic_rag": basic_result,
        "raptor_rag": raptor_result,
    }


@router.post("/rag/multimodel")
async def multimodel_compare(body: MultiModelCompareRequest):
    """여러 모델로 Basic + RAPTOR RAG를 동시에 실행하여 비교"""
    _check_indexed(body.session_id)

    target_models = body.models if body.models else list_ollama_models()

    loop = asyncio.get_event_loop()
    tasks = []
    for m in target_models:
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
                entry = {"rag_type": rag_type, "model": model_name,
                         "answer": f"오류: {str(result)}", "latency_ms": 0, "references": []}
            else:
                entry = {"rag_type": rag_type, "model": model_name, **result}
                conn.execute(
                    """INSERT INTO model_compare_results
                       (session_id, query, rag_type, model_name, answer, latency_ms)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (body.session_id, body.query, rag_type, model_name,
                     result["answer"], result["latency_ms"]),
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
