import re
import uuid
from datetime import datetime
from backend.db.database import get_conn
from backend.rag.llm_client import get_llm_answer

CHUNK_SIZE = 20  # turns per API call


def _parse_speaker_name(raw: str) -> tuple[str, str]:
    """'김윤래(시니어개발자)' -> ('김윤래', '시니어개발자')"""
    raw = raw.strip()
    m = re.match(r'^([^(（]+)[（(]([^)）]+)[)）]$', raw)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return raw, ''


def generate_conversation(mode: str, topic: str, turns: int, speakers: list[str], model: str = None) -> str:
    if mode == "2person":
        messages = _gen_2person(topic, turns, speakers, model)
    else:
        messages = _gen_group(topic, turns, speakers, model)

    session_id = str(uuid.uuid4())
    title = f"[AI생성] {topic[:30]} ({mode})"

    with get_conn() as conn:
        conn.execute(
            "INSERT INTO sessions (id, title, mode) VALUES (?, ?, ?)",
            (session_id, title, mode),
        )
        for msg in messages:
            conn.execute(
                "INSERT INTO messages (session_id, speaker, content, timestamp) VALUES (?, ?, ?, ?)",
                (session_id, msg["speaker"], msg["content"], datetime.now().isoformat()),
            )

    return session_id


def _gen_2person(topic: str, turns: int, speakers: list[str], model: str = None) -> list[dict]:
    raw_speakers = (speakers + ["화자A", "화자B"])[:2]
    parsed = [_parse_speaker_name(sp) for sp in raw_speakers]
    (s1_base, s1_role), (s2_base, s2_role) = parsed
    s1_full, s2_full = raw_speakers[0], raw_speakers[1]

    all_messages: list[dict] = []
    context_lines: list[str] = []
    remaining = turns
    stall = 0

    while remaining > 0 and stall < 3:
        chunk = min(CHUNK_SIZE, remaining)
        context_str = "\n".join(context_lines[-8:])
        continuation = f"\n앞선 대화:\n{context_str}\n\n위 대화에 이어서 계속:" if context_str else "지금 바로 시작:"

        # Determine whose turn starts this chunk (alternate from last speaker)
        if all_messages:
            next_base = s2_base if all_messages[-1]["speaker"] == s1_base else s1_base
        else:
            next_base = s1_base

        prompt = f"""두 사람이 '{topic}'에 대해 대화합니다.
화자:
- {s1_base}{f' ({s1_role})' if s1_role else ''}
- {s2_base}{f' ({s2_role})' if s2_role else ''}

규칙:
- {s1_base}와 {s2_base}가 반드시 한 줄씩 번갈아 발언합니다
- 각 줄은 반드시 "화자이름: 대화내용" 형식입니다 (콜론 앞에 공백 없음)
- 정확히 {chunk}턴을 생성합니다
- 번호, 설명, 빈 줄 없이 대화 줄만 출력합니다
- 화자 이름은 반드시 "{s1_base}" 또는 "{s2_base}"만 사용합니다
- 첫 발언자: {next_base}

{continuation}"""

        raw = get_llm_answer(prompt, model)

        chunk_msgs = _parse_dialog(raw, [s1_base, s2_base])

        if not chunk_msgs:
            stall += 1
            continue

        stall = 0
        all_messages.extend(chunk_msgs)
        remaining -= len(chunk_msgs)
        context_lines = [f"{m['speaker']}: {m['content']}" for m in all_messages[-8:]]

    # Restore full names (with role) for display
    name_map = {s1_base: s1_full, s2_base: s2_full}
    for msg in all_messages:
        msg["speaker"] = name_map.get(msg["speaker"], msg["speaker"])

    return all_messages


def _gen_group(topic: str, turns: int, speakers: list[str], model: str = None) -> list[dict]:
    raw_speakers = (speakers + ["참여자A", "참여자B", "참여자C"])[:max(3, len(speakers))]
    parsed = [_parse_speaker_name(sp) for sp in raw_speakers]
    base_names = [p[0] for p in parsed]

    role_lines = "\n".join(
        f"- {base}{f' ({role})' if role else ''}"
        for base, role in parsed
    )
    names_str = ", ".join(base_names)

    all_messages: list[dict] = []
    context_lines: list[str] = []
    remaining = turns
    stall = 0

    while remaining > 0 and stall < 3:
        chunk = min(CHUNK_SIZE, remaining)
        context_str = "\n".join(context_lines[-8:])
        continuation = f"\n앞선 대화:\n{context_str}\n\n위 대화에 이어서 계속:" if context_str else "지금 바로 시작:"

        prompt = f"""참여자들이 '{topic}'에 대해 회의/토론을 진행합니다.
참여자:
{role_lines}

규칙:
- 참여자({names_str})가 자연스럽게 번갈아 발언합니다
- 각 줄은 반드시 "화자이름: 대화내용" 형식입니다
- 정확히 {chunk}턴을 생성합니다
- 번호, 설명, 빈 줄 없이 대화 줄만 출력합니다
- 화자 이름은 반드시 다음 중 하나만 사용합니다: {names_str}

{continuation}"""

        raw = get_llm_answer(prompt, model)
        chunk_msgs = _parse_dialog(raw, base_names)

        if not chunk_msgs:
            stall += 1
            continue

        stall = 0
        all_messages.extend(chunk_msgs)
        remaining -= len(chunk_msgs)
        context_lines = [f"{m['speaker']}: {m['content']}" for m in all_messages[-8:]]

    # Restore full names for display
    name_map = {p[0]: raw_speakers[i] for i, p in enumerate(parsed)}
    for msg in all_messages:
        msg["speaker"] = name_map.get(msg["speaker"], msg["speaker"])

    return all_messages


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
    return messages
