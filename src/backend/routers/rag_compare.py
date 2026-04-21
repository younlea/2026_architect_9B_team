import asyncio
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.db.database import get_conn
from backend.rag import basic_rag, raptor_rag

router = APIRouter(prefix="/api", tags=["rag"])


class CompareRequest(BaseModel):
    session_id: str
    query: str


@router.post("/rag/compare")
async def compare_rag(body: CompareRequest):
    with get_conn() as conn:
        sess = conn.execute(
            "SELECT is_indexed FROM sessions WHERE id = ?", (body.session_id,)
        ).fetchone()
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        if not sess["is_indexed"]:
            raise HTTPException(status_code=400, detail="Session not indexed yet. Call /index first.")

    loop = asyncio.get_event_loop()
    basic_future = loop.run_in_executor(None, basic_rag.query, body.session_id, body.query)
    raptor_future = loop.run_in_executor(None, raptor_rag.query, body.session_id, body.query)

    basic_result, raptor_result = await asyncio.gather(basic_future, raptor_future)

    with get_conn() as conn:
        conn.execute(
            """INSERT INTO rag_results
               (session_id, query, basic_rag_answer, basic_rag_latency_ms, raptor_rag_answer, raptor_rag_latency_ms)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                body.session_id,
                body.query,
                basic_result["answer"],
                basic_result["latency_ms"],
                raptor_result["answer"],
                raptor_result["latency_ms"],
            ),
        )

    return {
        "query": body.query,
        "basic_rag": basic_result,
        "raptor_rag": raptor_result,
    }


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
