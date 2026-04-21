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
"""


def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_conn():
    conn = sqlite3.connect(SQLITE_DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
