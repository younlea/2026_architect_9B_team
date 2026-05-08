"""벤치마크: LongBench 질문으로 Basic RAG vs RAPTOR RAG vs ROI-RAG 정확도 비교"""
import asyncio
import json
import os
import re
from collections import Counter
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.db.database import get_conn
from backend.rag import basic_rag, raptor_rag, roi_rag
from backend.config import OLLAMA_MODEL

router = APIRouter(prefix="/api", tags=["benchmark"])

_UUID_RE = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.I)


class BenchmarkRunRequest(BaseModel):
    model: Optional[str] = None

def _answer_correct(answer: str, ground_truths: list[str]) -> bool:
    if not answer or not ground_truths:
        return False
        
    ans_lower = answer.lower()
    # 1. 완전 일치 (부분 문자열) 검사
    if any(gt.lower() in ans_lower for gt in ground_truths if gt):
        return True

    # 2. 토큰/글자 기반 유사도(F1 Score) 검사 (완전 일치 실패 시)
    def get_tokens(text):
        if not text: return []
        # 한자(중국어)나 한글이 포함된 경우 글자 단위로 토큰화
        if any('\u4e00' <= char <= '\u9fff' or '\uac00' <= char <= '\ud7a3' for char in text):
            return [c for c in text.strip() if c.strip()]
        # 영어 등은 단어 단위로 토큰화
        return re.findall(r'\w+', text.lower())

    pred_tokens = get_tokens(answer)
    if not pred_tokens:
        return False

    for gt in ground_truths:
        if not gt: continue
        truth_tokens = get_tokens(gt)
        if not truth_tokens: continue
        
        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            continue
            
        precision = 1.0 * num_same / len(pred_tokens)
        recall = 1.0 * num_same / len(truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        
        # 모델 답변이 길어지면 F1이 낮아지므로 Recall(재현율)도 확인
        # 모델 답변이 너무 짧게 요약되면 Recall이 낮아지므로 Precision(정밀도)도 확인
        if f1 >= 0.25 or recall >= 0.5 or precision >= 0.4:
            return True
            
    return False


@router.get("/benchmark/datasets")
def get_datasets():
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "longbench")
    downloaded = []
    if os.path.exists(data_dir):
        for f in os.listdir(data_dir):
            if f.endswith(".jsonl"):
                dataset_name = f[:-6]
                size_mb = os.path.getsize(os.path.join(data_dir, f)) / (1024*1024)
                downloaded.append({"name": dataset_name, "size_mb": round(size_mb, 1)})
    
    loaded = []
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, basic_indexed, raptor_indexed, roi_indexed FROM threads WHERE title LIKE '[LongBench] %'"
        ).fetchall()
        for r in rows:
            name = r["title"].replace("[LongBench] ", "")
            loaded.append({
                "name": name,
                "thread_id": r["id"],
                "basic_indexed": bool(r["basic_indexed"]),
                "raptor_indexed": bool(r["raptor_indexed"]),
                "roi_indexed": bool(r["roi_indexed"]),
            })

    result = []
    loaded_dict = {item["name"]: item for item in loaded}

    for d in downloaded:
        info = loaded_dict.get(d["name"])
        d["loaded"] = info is not None
        if info:
            d["thread_id"] = info["thread_id"]
            d["basic_indexed"] = info["basic_indexed"]
            d["raptor_indexed"] = info["raptor_indexed"]
            d["roi_indexed"] = info["roi_indexed"]
        result.append(d)

    downloaded_names = {d["name"] for d in downloaded}
    for l in loaded:
        if l["name"] not in downloaded_names:
            result.append({
                "name": l["name"], "size_mb": 0, "loaded": True, "missing_file": True,
                "thread_id": l["thread_id"],
                "basic_indexed": l["basic_indexed"],
                "raptor_indexed": l["raptor_indexed"],
                "roi_indexed": l["roi_indexed"],
            })
            
    result.sort(key=lambda x: x["name"])
    return {"datasets": result}


class LoadDatasetRequest(BaseModel):
    dataset_name: str
    num_examples: int = 5


@router.get("/benchmark/datasets/{dataset_name}/view")
def view_dataset(dataset_name: str, limit: int = 5):
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "longbench")
    jsonl_path = os.path.join(data_dir, f"{dataset_name}.jsonl")
    
    if not os.path.exists(jsonl_path):
        raise HTTPException(status_code=404, detail=f"Dataset {dataset_name} not found")
        
    examples = []
    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    ex = json.loads(line)
                    examples.append({
                        "input": ex.get("input", ""),
                        "context": ex.get("context", ""),
                        "answers": ex.get("answers", [])
                    })
                    if len(examples) >= limit:
                        break
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    return {"dataset": dataset_name, "examples": examples}


@router.post("/benchmark/load")
async def load_dataset(body: LoadDatasetRequest):
    script_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "load_longbench.py")
    import sys
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable, script_path, body.dataset_name, str(body.num_examples),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Load failed: {stderr.decode('utf-8', errors='ignore')}")
            
        return {"status": "success", "message": "데이터 로드 완료", "log": stdout.decode('utf-8', errors='ignore')}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
            "SELECT basic_indexed, raptor_indexed, roi_indexed FROM threads WHERE id=?",
            (thread_id,),
        ).fetchone()
        if not t:
            raise HTTPException(status_code=404, detail="Thread not found")
        if not t["basic_indexed"] or not t["raptor_indexed"]:
            raise HTTPException(status_code=400, detail="Thread not indexed yet")

        roi_ready = bool(t["roi_indexed"]) if t["roi_indexed"] is not None else False

        questions = conn.execute(
            "SELECT id, question, ground_truth_answers, source_id FROM benchmark_questions WHERE thread_id=?",
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
        # 요약 태스크(multi_news 등)는 input이 비어있음 → GT 언어에 맞는 요약 프롬프트로 대체
        if not question or not question.strip():
            gt_sample = " ".join(ground_truths[:1])
            has_cjk = any('一' <= c <= '鿿' or '가' <= c <= '힣' for c in gt_sample)
            question = "이 문서의 핵심 내용을 간결하게 요약해 주세요." if has_cjk else \
                       "Summarize the key content of the document concisely in English."

        source_id = dict(q).get("source_id") or ""
        # source_id가 UUID 형식이면 세션 단위 검색, 아니면 스레드 단위로 fallback
        if _UUID_RE.match(source_id):
            basic_fn = lambda question=question, sid=source_id: basic_rag.query(sid, question, model)
            raptor_fn = lambda question=question, sid=source_id: raptor_rag.query(sid, question, model)
        else:
            basic_fn = lambda question=question: basic_rag.query_thread(thread_id, question, model)
            raptor_fn = lambda question=question: raptor_rag.query_thread(thread_id, question, model)

        fns = [basic_fn, raptor_fn]
        if roi_ready:
            roi_fn = lambda question=question: roi_rag.query_thread(thread_id, question, model)
            fns.append(roi_fn)

        done = await asyncio.gather(*[loop.run_in_executor(None, fn) for fn in fns])
        basic_res, raptor_res = done[0], done[1]
        roi_res = done[2] if roi_ready else None

        basic_correct = int(_answer_correct(basic_res["answer"], ground_truths))
        raptor_correct = int(_answer_correct(raptor_res["answer"], ground_truths))
        roi_correct = int(_answer_correct(roi_res["answer"], ground_truths)) if roi_res else 0

        with get_conn() as conn:
            conn.execute(
                "DELETE FROM benchmark_results WHERE question_id=? AND thread_id=?",
                (qid, thread_id),
            )
            conn.execute(
                """INSERT INTO benchmark_results
                   (question_id, thread_id,
                    basic_rag_answer, basic_rag_latency_ms,
                    raptor_rag_answer, raptor_rag_latency_ms,
                    roi_rag_answer, roi_rag_latency_ms,
                    model_name, basic_correct, raptor_correct, roi_correct)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    qid, thread_id,
                    basic_res["answer"], basic_res["latency_ms"],
                    raptor_res["answer"], raptor_res["latency_ms"],
                    roi_res["answer"] if roi_res else None,
                    roi_res["latency_ms"] if roi_res else None,
                    model or OLLAMA_MODEL,
                    basic_correct, raptor_correct, roi_correct,
                ),
            )

        results.append({
            "question_id": qid,
            "question": question,
            "ground_truth_answers": ground_truths,
            "basic_rag": {**basic_res, "correct": bool(basic_correct)},
            "raptor_rag": {**raptor_res, "correct": bool(raptor_correct)},
            "roi_rag": {**roi_res, "correct": bool(roi_correct)} if roi_res else None,
        })

    total = len(results)
    basic_score = sum(1 for r in results if r["basic_rag"]["correct"])
    raptor_score = sum(1 for r in results if r["raptor_rag"]["correct"])
    roi_score = sum(1 for r in results if r.get("roi_rag") and r["roi_rag"]["correct"])

    return {
        "thread_id": thread_id,
        "model": model or OLLAMA_MODEL,
        "roi_ready": roi_ready,
        "total": total,
        "basic_correct": basic_score,
        "raptor_correct": raptor_score,
        "roi_correct": roi_score,
        "basic_accuracy": round(basic_score / total * 100, 1) if total else 0,
        "raptor_accuracy": round(raptor_score / total * 100, 1) if total else 0,
        "roi_accuracy": round(roi_score / total * 100, 1) if total and roi_ready else None,
        "results": results,
    }


@router.get("/threads/{thread_id}/benchmark/results")
def get_benchmark_results(thread_id: str):
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT br.id, br.question_id, bq.question, bq.ground_truth_answers,
                      br.basic_rag_answer, br.basic_rag_latency_ms,
                      br.raptor_rag_answer, br.raptor_rag_latency_ms,
                      br.roi_rag_answer, br.roi_rag_latency_ms,
                      br.model_name, br.basic_correct, br.raptor_correct, br.roi_correct,
                      br.created_at
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
    roi_score = sum(1 for r in results if r.get("roi_correct"))
    roi_ready = any(r.get("roi_rag_answer") for r in results)

    return {
        "results": results,
        "summary": {
            "total": total,
            "basic_correct": basic_score,
            "raptor_correct": raptor_score,
            "roi_correct": roi_score,
            "basic_accuracy": round(basic_score / total * 100, 1) if total else 0,
            "raptor_accuracy": round(raptor_score / total * 100, 1) if total else 0,
            "roi_accuracy": round(roi_score / total * 100, 1) if total and roi_ready else None,
            "roi_ready": roi_ready,
            "model_name": results[0]["model_name"] if results else None,
        },
    }
