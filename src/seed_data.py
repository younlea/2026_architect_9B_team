"""샘플 JSON 데이터를 SQLite DB에 로드합니다."""
import json
import uuid
import sys
import os
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))

from backend.db.database import init_db, get_conn

DATA_FILES = [
    ("data/sample_2person.json", "2person"),
    ("data/sample_group.json", "group"),
]


def seed():
    init_db()
    total = 0
    for filepath, mode in DATA_FILES:
        path = os.path.join(os.path.dirname(__file__), filepath)
        with open(path, encoding="utf-8") as f:
            sessions = json.load(f)

        for sess in sessions:
            session_id = str(uuid.uuid4())
            with get_conn() as conn:
                conn.execute(
                    "INSERT INTO sessions (id, title, mode) VALUES (?, ?, ?)",
                    (session_id, sess["title"], sess.get("mode", mode)),
                )
                base_time = datetime.now()
                for i, msg in enumerate(sess["messages"]):
                    ts = (base_time + timedelta(minutes=i)).isoformat()
                    conn.execute(
                        "INSERT INTO messages (session_id, speaker, content, timestamp) VALUES (?, ?, ?, ?)",
                        (session_id, msg["speaker"], msg["content"], ts),
                    )
            total += 1
            print(f"  로드: [{sess.get('mode', mode)}] {sess['title']}")

    print(f"\n총 {total}개 세션 로드 완료.")


if __name__ == "__main__":
    os.chdir(os.path.dirname(__file__))
    seed()
