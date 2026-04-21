import uuid
from datetime import datetime
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.db.database import get_conn

router = APIRouter(prefix="/api", tags=["chat"])


class SessionCreate(BaseModel):
    title: str
    mode: str = "2person"


class MessageCreate(BaseModel):
    speaker: str
    content: str


@router.get("/sessions")
def list_sessions():
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, title, mode, created_at, is_indexed FROM sessions ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("/sessions", status_code=201)
def create_session(body: SessionCreate):
    session_id = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, mode) VALUES (?, ?, ?)",
            (session_id, body.title, body.mode),
        )
    return {"id": session_id, "title": body.title, "mode": body.mode}


@router.get("/sessions/{session_id}/messages")
def get_messages(session_id: str):
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, speaker, content, timestamp FROM messages WHERE session_id = ? ORDER BY timestamp",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("/sessions/{session_id}/messages", status_code=201)
def add_message(session_id: str, body: MessageCreate):
    with get_conn() as conn:
        sess = conn.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        conn.execute(
            "INSERT INTO messages (session_id, speaker, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, body.speaker, body.content, datetime.now().isoformat()),
        )
    return {"ok": True}


@router.post("/sessions/{session_id}/index")
def index_session(session_id: str):
    from backend.rag import basic_rag, raptor_rag

    with get_conn() as conn:
        sess = conn.execute("SELECT id FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if not sess:
            raise HTTPException(status_code=404, detail="Session not found")
        count = conn.execute(
            "SELECT COUNT(*) as cnt FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()["cnt"]

    if count == 0:
        raise HTTPException(status_code=400, detail="No messages to index")

    basic_rag.index_session(session_id)
    raptor_rag.index_session(session_id)

    return {"ok": True, "indexed_messages": count}
