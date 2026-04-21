"""벤치마크: LongBench 질문으로 Basic RAG vs RAPTOR RAG 정확도 비교"""
import asyncio
import json
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.db.database import get_conn
from backend.rag import basic_rag, raptor_rag
from backend.config import OLLAMA_MODEL

router = APIRouter(prefix="/api", tags=["benchmark"])


class BenchmarkRunRequest(BaseModel):
    model: Optional[str] = None


def _answer_correct(answer: str, ground_truths: list[str]) -> bool:
    ans_lower = answer.lower()
    return any(gt.lower() in ans_lower for gt in ground_truths if gt)


@router.get("/threads/{thread_id}/benchmark/questions")
def get_benchmark_questions(thread_id: str):
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT id, question, ground_truth_answers, dataset_name, source_id, created_at
               FROM benchmark_questions WHERE thread_id=? ORDER BY id""",
            (thread_id,),
        ).fetchall()
    result = []
    for r in rows:
        item = dict(r)
        item["ground_truth_answers"] = json.loads(r["ground_truth_answers"] or "[]")
        result.append(item)
    return result


@router.post("/threads/{thread_id}/benchmark/run")
async def run_benchmark(thread_id: str, body: BenchmarkRunRequest = BenchmarkRunRequest()):
    with get_conn() as conn:
        t = conn.execute(
            "SELECT basic_indexed, raptor_indexed FROM threads WHERE id=?", (thread_id,)
        ).fetchone()
        if not t:
            raise HTTPException(status_code=404, detail="Thread not found")
        if not t["basic_indexed"] or not t["raptor_indexed"]:
            raise HTTPException(status_code=400, detail="Thread not indexed yet")

        questions = conn.execute(
            "SELECT id, question, ground_truth_answers FROM benchmark_questions WHERE thread_id=?",
            (thread_id,),
        ).fetchall()

    if not questions:
        raise HTTPException(status_code=400, detail="No benchmark questions for this thread")

    loop = asyncio.get_event_loop()
    model = body.model

    results = []
    for q in questions:
        qid = q["id"]
        question = q["question"]
        ground_truths = json.loads(q["ground_truth_answers"] or "[]")

        basic_fn = lambda question=question: basic_rag.query_thread(thread_id, question, model)
        raptor_fn = lambda question=question: raptor_rag.query_thread(thread_id, question, model)
        basic_res, raptor_res = await asyncio.gather(
            loop.run_in_executor(None, basic_fn),
            loop.run_in_executor(None, raptor_fn),
        )

        basic_correct = int(_answer_correct(basic_res["answer"], ground_truths))
        raptor_correct = int(_answer_correct(raptor_res["answer"], ground_truths))

        with get_conn() as conn:
            # 이전 결과 삭제 후 새 결과 저장
            conn.execute("DELETE FROM benchmark_results WHERE question_id=? AND thread_id=?", (qid, thread_id))
            conn.execute(
                """INSERT INTO benchmark_results
                   (question_id, thread_id, basic_rag_answer, basic_rag_latency_ms,
                    raptor_rag_answer, raptor_rag_latency_ms, model_name,
                    basic_correct, raptor_correct)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (qid, thread_id,
                 basic_res["answer"], basic_res["latency_ms"],
                 raptor_res["answer"], raptor_res["latency_ms"],
                 model or OLLAMA_MODEL,
                 basic_correct, raptor_correct),
            )

        results.append({
            "question_id": qid,
            "question": question,
            "ground_truth_answers": ground_truths,
            "basic_rag": {**basic_res, "correct": bool(basic_correct)},
            "raptor_rag": {**raptor_res, "correct": bool(raptor_correct)},
        })

    total = len(results)
    basic_score = sum(1 for r in results if r["basic_rag"]["correct"])
    raptor_score = sum(1 for r in results if r["raptor_rag"]["correct"])

    return {
        "thread_id": thread_id,
        "model": model or OLLAMA_MODEL,
        "total": total,
        "basic_correct": basic_score,
        "raptor_correct": raptor_score,
        "basic_accuracy": round(basic_score / total * 100, 1) if total else 0,
        "raptor_accuracy": round(raptor_score / total * 100, 1) if total else 0,
        "results": results,
    }


@router.get("/threads/{thread_id}/benchmark/results")
def get_benchmark_results(thread_id: str):
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT br.id, br.question_id, bq.question, bq.ground_truth_answers,
                      br.basic_rag_answer, br.basic_rag_latency_ms,
                      br.raptor_rag_answer, br.raptor_rag_latency_ms,
                      br.model_name, br.basic_correct, br.raptor_correct, br.created_at
               FROM benchmark_results br
               JOIN benchmark_questions bq ON br.question_id = bq.id
               WHERE br.thread_id=?
               ORDER BY br.created_at DESC, bq.id""",
            (thread_id,),
        ).fetchall()

    if not rows:
        return {"results": [], "summary": None}

    results = []
    for r in rows:
        item = dict(r)
        item["ground_truth_answers"] = json.loads(r["ground_truth_answers"] or "[]")
        results.append(item)

    total = len(results)
    basic_score = sum(1 for r in results if r["basic_correct"])
    raptor_score = sum(1 for r in results if r["raptor_correct"])

    return {
        "results": results,
        "summary": {
            "total": total,
            "basic_correct": basic_score,
            "raptor_correct": raptor_score,
            "basic_accuracy": round(basic_score / total * 100, 1) if total else 0,
            "raptor_accuracy": round(raptor_score / total * 100, 1) if total else 0,
            "model": results[0]["model_name"] if results else None,
        },
    }
