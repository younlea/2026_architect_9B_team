import uuid
from datetime import datetime
from backend.db.database import get_conn
from backend.rag.llm_client import get_llm_answer


def generate_conversation(mode: str, topic: str, turns: int, speakers: list[str]) -> str:
    """AI 에이전트가 주어진 설정으로 대화를 생성하고 DB에 저장합니다."""
    if mode == "2person":
        raw = _gen_2person(topic, turns, speakers)
    else:
        raw = _gen_group(topic, turns, speakers)

    session_id = str(uuid.uuid4())
    title = f"[AI생성] {topic[:30]} ({mode})"

    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, mode) VALUES (?, ?, ?)",
            (session_id, title, mode),
        )
        for msg in raw:
            conn.execute(
                "INSERT INTO messages (session_id, speaker, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, msg["speaker"], msg["content"], datetime.now().isoformat()),
            )

    return session_id


def _gen_2person(topic: str, turns: int, speakers: list[str]) -> list[dict]:
    s1, s2 = (speakers + ["화자A", "화자B"])[:2]
    prompt = f"""두 사람 {s1}과 {s2}가 '{topic}'에 대해 대화합니다.
총 {turns}턴의 자연스러운 대화를 생성해 주세요.
각 줄은 반드시 다음 형식으로 작성하세요: 화자이름: 대화내용
예시:
{s1}: 안녕하세요, 오늘 주제는 무엇인가요?
{s2}: 네, 오늘은 {topic}에 대해 이야기해 보려고 합니다.

지금 바로 시작하세요:"""

    return _parse_dialog(get_llm_answer(prompt), [s1, s2])


def _gen_group(topic: str, turns: int, speakers: list[str]) -> list[dict]:
    names = (speakers + ["참여자A", "참여자B", "참여자C"])[:max(3, len(speakers))]
    names_str = ", ".join(names)
    prompt = f"""참여자 {names_str}가 '{topic}'에 대해 회의/토론을 진행합니다.
총 {turns}턴의 현실감 있는 그룹 대화를 생성해 주세요.
각 줄은 반드시 다음 형식으로 작성하세요: 화자이름: 대화내용

지금 바로 시작하세요:"""

    return _parse_dialog(get_llm_answer(prompt), names)


def _parse_dialog(raw: str, speakers: list[str]) -> list[dict]:
    messages = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        for sp in speakers:
            if line.startswith(f"{sp}:"):
                content = line[len(sp) + 1:].strip()
                if content:
                    messages.append({"speaker": sp, "content": content})
                break
        else:
            # 화자 구분이 불명확한 경우 마지막 화자에 추가
            if messages and line:
                messages[-1]["content"] += " " + line
    return messages
