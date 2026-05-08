from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from backend.agent.conversation_gen import generate_conversation

router = APIRouter(prefix="/api", tags=["agent"])


class GenerateRequest(BaseModel):
    mode: str = "2person"
    topic: str
    turns: int = 10
    speakers: list[str] = []


@router.post("/agent/generate")
def generate(body: GenerateRequest):
    if body.mode not in ("2person", "group"):
        raise HTTPException(status_code=400, detail="mode must be '2person' or 'group'")
    if not body.topic.strip():
        raise HTTPException(status_code=400, detail="topic is required")
    if body.turns < 2 or body.turns > 1000:
        raise HTTPException(status_code=400, detail="turns must be between 2 and 1000")

    session_id = generate_conversation(
        mode=body.mode,
        topic=body.topic,
        turns=body.turns,
        speakers=body.speakers,
    )
    return {"session_id": session_id, "ok": True}
