import sqlite3
import os
from contextlib import contextmanager
from backend.config import SQLITE_DB_PATH

os.makedirs(os.path.dirname(SQLITE_DB_PATH), exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    mode TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    is_indexed INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    speaker TEXT,
    content TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS rag_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    query TEXT,
    basic_rag_answer TEXT,
    basic_rag_latency_ms INTEGER,
    raptor_rag_answer TEXT,
    raptor_rag_latency_ms INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS model_compare_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    query TEXT,
    rag_type TEXT,
    model_name TEXT,
    answer TEXT,
    latency_ms INTEGER,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 스레드: 여러 세션을 하나의 주제 단위로 묶는 컨테이너
CREATE TABLE IF NOT EXISTS threads (
    id TEXT PRIMARY KEY,
    title TEXT,
    description TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    basic_indexed INTEGER DEFAULT 0,
    raptor_indexed INTEGER DEFAULT 0,
    basic_chunk_count INTEGER DEFAULT 0,
    raptor_node_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS thread_sessions (
    thread_id TEXT,
    session_id TEXT,
    sort_order INTEGER DEFAULT 0,
    PRIMARY KEY (thread_id, session_id),
    FOREIGN KEY (thread_id) REFERENCES threads(id),
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE IF NOT EXISTS thread_rag_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT,
    query TEXT,
    basic_rag_answer TEXT,
    basic_rag_latency_ms INTEGER,
    raptor_rag_answer TEXT,
    raptor_rag_latency_ms INTEGER,
    model_name TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- LongBench 벤치마크 질문 (ground truth 포함)
CREATE TABLE IF NOT EXISTS benchmark_questions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    thread_id TEXT,
    question TEXT,
    ground_truth_answers TEXT,
    dataset_name TEXT,
    source_id TEXT,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 벤치마크 실행 결과 (기법별 정답 여부 포함)
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
"""


def _apply_migrations(conn):
    """신규 컬럼을 기존 DB에 추가합니다. 이미 존재하는 컬럼은 무시합니다."""
    migrations = [
        "ALTER TABLE threads ADD COLUMN roi_indexed INTEGER DEFAULT 0",
        "ALTER TABLE threads ADD COLUMN roi_eu_count INTEGER DEFAULT 0",
        "ALTER TABLE threads ADD COLUMN roi_regime TEXT DEFAULT ''",
        "ALTER TABLE thread_rag_results ADD COLUMN roi_rag_answer TEXT",
        "ALTER TABLE thread_rag_results ADD COLUMN roi_rag_latency_ms INTEGER",
        "ALTER TABLE benchmark_results ADD COLUMN roi_rag_answer TEXT",
        "ALTER TABLE benchmark_results ADD COLUMN roi_rag_latency_ms INTEGER",
        "ALTER TABLE benchmark_results ADD COLUMN roi_correct INTEGER DEFAULT 0",
    ]
    for sql in migrations:
        try:
            conn.execute(sql)
        except Exception:
            pass  # 이미 존재하는 컬럼


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
        _apply_migrations(conn)


def get_thread_text(thread_id: str) -> str:
    """스레드에 속한 모든 세션의 메시지를 시간순으로 합쳐 반환합니다."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT s.title, m.speaker, m.content, m.timestamp
               FROM thread_sessions ts
               JOIN sessions s ON ts.session_id = s.id
               JOIN messages m ON m.session_id = s.id
               WHERE ts.thread_id = ?
               ORDER BY ts.sort_order, m.timestamp""",
            (thread_id,),
        ).fetchall()
    return "\n".join(f"[{r['speaker']}] {r['content']}" for r in rows)


@contextmanager
def get_conn():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
