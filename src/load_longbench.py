"""
LongBench 벤치마크 데이터를 로드하여 RAG 비교용 스레드를 생성합니다.

Usage:
    python load_longbench.py [dataset_name] [num_examples] [data_dir]

    dataset_name: multifieldqa_en (기본) | hotpotqa | qasper | narrativeqa
    num_examples: 5 (기본)
    data_dir: LongBench JSONL 파일 경로 (기본: /tmp/longbench/extracted/data)

Example:
    python load_longbench.py multifieldqa_en 5
"""
import sys
import uuid
import json
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(__file__))

from backend.db.database import get_conn, init_db
from backend.rag import basic_rag, raptor_rag

DATASET_NAME = sys.argv[1] if len(sys.argv) > 1 else "multifieldqa_en"
NUM_EXAMPLES = int(sys.argv[2]) if len(sys.argv) > 2 else 5
DATA_DIR = sys.argv[3] if len(sys.argv) > 3 else "/tmp/longbench/extracted/data"
THREAD_TITLE = f"[LongBench] {DATASET_NAME}"


def split_context(context: str, target_chunk: int = 400) -> list[str]:
    """Context를 단락 기준으로 분할하고 너무 작으면 합칩니다."""
    paragraphs = [p.strip() for p in context.split("\n\n") if p.strip()]
    if len(paragraphs) < 3:
        # 단락이 적으면 줄바꿈으로 재시도
        paragraphs = [p.strip() for p in context.split("\n") if p.strip()]

    chunks, current = [], ""
    for p in paragraphs:
        if len(current) + len(p) + 2 <= target_chunk:
            current = (current + "\n\n" + p).strip()
        else:
            if current:
                chunks.append(current)
            if len(p) > target_chunk * 2:
                # 너무 긴 단락은 강제 분할
                for start in range(0, len(p), target_chunk - 50):
                    chunks.append(p[start:start + target_chunk])
                current = ""
            else:
                current = p
    if current:
        chunks.append(current)
    return [c for c in chunks if c.strip()]


def _contains_answer(answer: str, ground_truths: list[str]) -> bool:
    """답변이 ground truth 중 하나라도 포함하는지 확인합니다."""
    ans_lower = answer.lower()
    for gt in ground_truths:
        if gt.lower() in ans_lower:
            return True
    return False


def main():
    init_db()

    # 벤치마크 테이블 생성 (init_db에 포함되어 있음)
    with get_conn() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS benchmark_questions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                thread_id TEXT,
                question TEXT,
                ground_truth_answers TEXT,
                dataset_name TEXT,
                source_id TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS benchmark_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_id INTEGER,
                thread_id TEXT,
                basic_rag_answer TEXT,
                basic_rag_latency_ms INTEGER,
                raptor_rag_answer TEXT,
                raptor_rag_latency_ms INTEGER,
                model_name TEXT,
                basic_correct INTEGER DEFAULT 0,
                raptor_correct INTEGER DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            );
        """)

    # 기존 스레드 확인
    with get_conn() as conn:
        existing = conn.execute(
            "SELECT id FROM threads WHERE title=?", (THREAD_TITLE,)
        ).fetchone()
        if existing:
            print(f"이미 존재: '{THREAD_TITLE}' (id={existing['id']})")
            print(f"삭제 후 재실행하거나 다른 dataset_name을 사용하세요.")
            return

    # JSONL 파일에서 직접 로드 (HuggingFace datasets 스크립트 이슈 우회)
    jsonl_path = os.path.join(DATA_DIR, f"{DATASET_NAME}.jsonl")
    if not os.path.exists(jsonl_path):
        print(f"ERROR: {jsonl_path} 파일이 없습니다.")
        print("data.zip을 /tmp/longbench/에 다운로드하고 extracted/data/에 압축을 푸세요:")
        print("  curl -L -o /tmp/longbench/data.zip https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip")
        print("  unzip /tmp/longbench/data.zip -d /tmp/longbench/extracted/")
        sys.exit(1)

    print(f"LongBench/{DATASET_NAME} 로딩 중... ({NUM_EXAMPLES}개 예제)")
    print(f"파일: {jsonl_path}")
    examples = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
                if len(examples) >= NUM_EXAMPLES:
                    break
    print(f"  {len(examples)}개 로드 완료")

    thread_id = str(uuid.uuid4())
    session_ids = []

    for i, ex in enumerate(examples):
        question = ex.get("input", "")
        context = ex.get("context", "")
        answers = ex.get("answers", [])
        if isinstance(answers, str):
            answers = [answers]

        print(f"\n  예제 {i+1}/{len(examples)}")
        print(f"    질문: {question[:80]}...")
        print(f"    컨텍스트 길이: {len(context)} chars")
        print(f"    정답: {answers[:2]}")

        # 세션 생성
        session_id = str(uuid.uuid4())
        title = f"[LongBench/{DATASET_NAME}] 예제 {i+1}"
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO sessions (id, title, mode) VALUES (?, ?, ?)",
                (session_id, title, "2person"),
            )

        # 컨텍스트를 청크로 분할하여 메시지로 저장
        chunks = split_context(context)
        print(f"    청크: {len(chunks)}개")
        speakers = ["Document", "Source"]
        with get_conn() as conn:
            for j, chunk in enumerate(chunks):
                conn.execute(
                    "INSERT INTO messages (session_id, speaker, content, timestamp) VALUES (?, ?, ?, ?)",
                    (session_id, speakers[j % 2], chunk, datetime.now().isoformat()),
                )

        session_ids.append(session_id)

        # 벤치마크 질문 저장
        with get_conn() as conn:
            conn.execute(
                """INSERT INTO benchmark_questions
                   (thread_id, question, ground_truth_answers, dataset_name, source_id)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    thread_id,
                    question,
                    json.dumps(answers, ensure_ascii=False),
                    DATASET_NAME,
                    str(ex.get("_id", i)),
                ),
            )

    # 스레드 생성
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO threads (id, title, description) VALUES (?, ?, ?)",
            (
                thread_id,
                THREAD_TITLE,
                f"LongBench {DATASET_NAME} 벤치마크 데이터셋 {len(session_ids)}개 예제. "
                f"각 예제의 컨텍스트가 세션으로 구성되며 ground truth 답변으로 정확도 측정 가능.",
            ),
        )
        for i, sid in enumerate(session_ids):
            conn.execute(
                "INSERT OR IGNORE INTO thread_sessions (thread_id, session_id, sort_order) VALUES (?, ?, ?)",
                (tid, sid, i) if False else (thread_id, sid, i),
            )

    print(f"\n스레드 생성: '{THREAD_TITLE}' (id={thread_id})")
    print(f"세션: {len(session_ids)}개")

    # 인덱싱
    print("\nBasic RAG 인덱싱 중...")
    basic_chunks = basic_rag.index_thread(thread_id)
    print(f"  Basic RAG: {basic_chunks} 청크")

    print("RAPTOR RAG 인덱싱 중...")
    raptor_nodes = raptor_rag.index_thread(thread_id)
    print(f"  RAPTOR RAG: {raptor_nodes} 노드")

    print(f"\n완료!")
    print(f"  스레드 ID: {thread_id}")
    print(f"  비교 페이지: http://localhost:8000/compare?thread={thread_id}")
    print(f"  벤치마크 실행: http://localhost:8000/compare?thread={thread_id}#benchmark")

    # 벤치마크 질문 출력
    with get_conn() as conn:
        bqs = conn.execute(
            "SELECT id, question, ground_truth_answers FROM benchmark_questions WHERE thread_id=?",
            (thread_id,),
        ).fetchall()
    print(f"\n벤치마크 질문 {len(bqs)}개:")
    for bq in bqs:
        gts = json.loads(bq["ground_truth_answers"])
        print(f"  [{bq['id']}] {bq['question'][:70]}")
        print(f"       정답: {gts[:2]}")


if __name__ == "__main__":
    main()
