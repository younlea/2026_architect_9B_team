import uuid
import re
from typing import Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.db.database import get_conn
from backend.rag import basic_rag, raptor_rag, roi_rag

router = APIRouter(prefix="/api", tags=["threads"])


class ThreadCreate(BaseModel):
    title: str
    description: str = ""
    session_ids: list[str] = []


class ThreadIndexRequest(BaseModel):
    model: Optional[str] = None


@router.get("/threads")
def list_threads():
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT t.id, t.title, t.description, t.created_at,
                      t.basic_indexed, t.raptor_indexed,
                      t.basic_chunk_count, t.raptor_node_count,
                      t.roi_indexed, t.roi_eu_count, t.roi_regime,
                      COUNT(ts.session_id) AS session_count,
                      SUM(m_cnt.cnt) AS message_count
               FROM threads t
               LEFT JOIN thread_sessions ts ON ts.thread_id = t.id
               LEFT JOIN (
                   SELECT session_id, COUNT(*) as cnt FROM messages GROUP BY session_id
               ) m_cnt ON m_cnt.session_id = ts.session_id
               GROUP BY t.id
               ORDER BY t.created_at DESC""",
        ).fetchall()
    return [dict(r) for r in rows]


@router.post("/threads", status_code=201)
def create_thread(body: ThreadCreate):
    tid = str(uuid.uuid4())
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO threads (id, title, description) VALUES (?, ?, ?)",
            (tid, body.title, body.description),
        )
        for i, sid in enumerate(body.session_ids):
            conn.execute(
                "INSERT OR IGNORE INTO thread_sessions (thread_id, session_id, sort_order) VALUES (?, ?, ?)",
                (tid, sid, i),
            )
    return {"id": tid, "title": body.title}


@router.get("/threads/{thread_id}")
def get_thread(thread_id: str):
    with get_conn() as conn:
        t = conn.execute("SELECT * FROM threads WHERE id=?", (thread_id,)).fetchone()
        if not t:
            raise HTTPException(status_code=404, detail="Thread not found")
        sessions = conn.execute(
            """SELECT s.id, s.title, s.mode, s.created_at, s.is_indexed, ts.sort_order
               FROM thread_sessions ts JOIN sessions s ON ts.session_id = s.id
               WHERE ts.thread_id = ? ORDER BY ts.sort_order""",
            (thread_id,),
        ).fetchall()
    return {**dict(t), "sessions": [dict(s) for s in sessions]}


@router.get("/threads/{thread_id}/messages")
def get_thread_messages(thread_id: str):
    """스레드의 모든 메시지를 세션별로 묶어서 반환합니다."""
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT s.id as session_id, s.title as session_title, s.mode,
                      m.id, m.speaker, m.content, m.timestamp
               FROM thread_sessions ts
               JOIN sessions s ON ts.session_id = s.id
               JOIN messages m ON m.session_id = s.id
               WHERE ts.thread_id = ?
               ORDER BY ts.sort_order, m.timestamp""",
            (thread_id,),
        ).fetchall()

    # 세션별로 그룹핑
    sessions: dict = {}
    order: list = []
    for r in rows:
        sid = r["session_id"]
        if sid not in sessions:
            sessions[sid] = {
                "session_id": sid,
                "session_title": r["session_title"],
                "mode": r["mode"],
                "messages": [],
            }
            order.append(sid)
        sessions[sid]["messages"].append({
            "id": r["id"],
            "speaker": r["speaker"],
            "content": r["content"],
            "timestamp": r["timestamp"],
        })
    return [sessions[sid] for sid in order]


@router.post("/threads/{thread_id}/index")
def index_thread(thread_id: str, body: ThreadIndexRequest = ThreadIndexRequest()):
    with get_conn() as conn:
        t = conn.execute("SELECT id FROM threads WHERE id=?", (thread_id,)).fetchone()
        if not t:
            raise HTTPException(status_code=404, detail="Thread not found")
        msg_count = conn.execute(
            """SELECT COUNT(*) as cnt FROM messages m
               JOIN thread_sessions ts ON ts.session_id = m.session_id
               WHERE ts.thread_id=?""",
            (thread_id,),
        ).fetchone()["cnt"]

    if msg_count == 0:
        raise HTTPException(status_code=400, detail="No messages in thread")

    basic_chunks = basic_rag.index_thread(thread_id)
    raptor_nodes = raptor_rag.index_thread(thread_id, body.model)
    roi_result = roi_rag.index_thread(thread_id, body.model)

    return {
        "ok": True,
        "message_count": msg_count,
        "basic_chunk_count": basic_chunks,
        "raptor_node_count": raptor_nodes,
        "roi_eu_count": roi_result["eu_count"],
        "roi_regime": roi_result["regime"],
    }


@router.post("/threads/auto-group")
def auto_group_threads():
    """Day 번호 기준으로 세션을 자동 그룹핑하여 스레드를 생성합니다."""
    with get_conn() as conn:
        sessions = conn.execute(
            "SELECT id, title, created_at FROM sessions WHERE title LIKE '[Day%' ORDER BY created_at"
        ).fetchall()

    day_map: dict[int, list] = {}
    for s in sessions:
        m = re.match(r'\[Day(\d+)\]', s["title"])
        if m:
            day_map.setdefault(int(m.group(1)), []).append(dict(s))

    week_map = {
        range(1, 6):   "Week 1 — 온보딩",
        range(6, 13):  "Week 2 — 첫 실무",
        range(13, 20): "Week 3 — 팀 협업",
        range(20, 27): "Week 4 — 성과와 데모",
        range(27, 35): "Week 5+ — 자립과 성장",
    }

    def week_label(day: int) -> str:
        for r, label in week_map.items():
            if day in r:
                return label
        return f"Week {day // 5}"

    created_threads = []

    # 1. Day별 스레드 (Day에 세션이 2개 이상인 경우에만 생성, 1개면 그냥 세션으로 충분)
    for day, day_sessions in sorted(day_map.items()):
        if len(day_sessions) < 2:
            continue
        tid = str(uuid.uuid4())
        title = f"Day {day} — 전체 대화"
        desc = f"Day {day}의 {len(day_sessions)}개 세션을 하나의 컨텍스트로 통합"
        with get_conn() as conn:
            existing = conn.execute("SELECT id FROM threads WHERE title=?", (title,)).fetchone()
            if existing:
                continue
            conn.execute(
                "INSERT INTO threads (id, title, description) VALUES (?, ?, ?)",
                (tid, title, desc),
            )
            for i, s in enumerate(day_sessions):
                conn.execute(
                    "INSERT OR IGNORE INTO thread_sessions (thread_id, session_id, sort_order) VALUES (?, ?, ?)",
                    (tid, s["id"], i),
                )
        created_threads.append({"id": tid, "title": title, "sessions": len(day_sessions)})

    # 2. Week별 스레드 (전체 주차 단위 통합)
    week_sessions: dict[str, list] = {}
    for day, day_sessions in sorted(day_map.items()):
        wlabel = week_label(day)
        week_sessions.setdefault(wlabel, []).extend(day_sessions)

    for wlabel, wsessions in week_sessions.items():
        if len(wsessions) < 2:
            continue
        tid = str(uuid.uuid4())
        desc = f"{wlabel}의 {len(wsessions)}개 세션 통합 컨텍스트"
        with get_conn() as conn:
            existing = conn.execute("SELECT id FROM threads WHERE title=?", (wlabel,)).fetchone()
            if existing:
                continue
            conn.execute(
                "INSERT INTO threads (id, title, description) VALUES (?, ?, ?)",
                (tid, wlabel, desc),
            )
            for i, s in enumerate(wsessions):
                conn.execute(
                    "INSERT OR IGNORE INTO thread_sessions (thread_id, session_id, sort_order) VALUES (?, ?, ?)",
                    (tid, s["id"], i),
                )
        created_threads.append({"id": tid, "title": wlabel, "sessions": len(wsessions)})

    # 3. 전체 월간 통합 스레드
    all_sessions = [s for sl in day_map.values() for s in sl]
    if all_sessions:
        tid = str(uuid.uuid4())
        title = "전체 월간 대화 통합"
        with get_conn() as conn:
            existing = conn.execute("SELECT id FROM threads WHERE title=?", (title,)).fetchone()
            if not existing:
                conn.execute(
                    "INSERT INTO threads (id, title, description) VALUES (?, ?, ?)",
                    (tid, title, f"Day1~Day31 전체 {len(all_sessions)}개 세션 통합"),
                )
                for i, s in enumerate(all_sessions):
                    conn.execute(
                        "INSERT OR IGNORE INTO thread_sessions (thread_id, session_id, sort_order) VALUES (?, ?, ?)",
                        (tid, s["id"], i),
                    )
                created_threads.append({"id": tid, "title": title, "sessions": len(all_sessions)})

    return {"created": len(created_threads), "threads": created_threads}
