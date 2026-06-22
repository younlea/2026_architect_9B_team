from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from backend.cache.answer_cache import (
    run_answer_cache_query,
    setup_answer_cache_poc,
)

router = APIRouter(prefix="/api/dp3", tags=["dp3-cache-poc"])


class SetupRequest(BaseModel):
    thread_id: str
    reset: bool = False


class AnswerCacheRunRequest(BaseModel):
    thread_id: str
    query: str
    user_scope: str = "A"
    requested_version: Optional[str] = None
    model: Optional[str] = None
    route_threshold: float = 0.70
    cache_threshold: float = 0.86


@router.post("/answer-cache/setup")
def setup_answer_cache(body: SetupRequest):
    return setup_answer_cache_poc(body.thread_id, reset=body.reset)


@router.post("/answer-cache/run")
def run_answer_cache(body: AnswerCacheRunRequest):
    return run_answer_cache_query(
        thread_id=body.thread_id,
        query=body.query,
        user_scope=body.user_scope,
        requested_version=body.requested_version,
        model=body.model,
        route_threshold=body.route_threshold,
        cache_threshold=body.cache_threshold,
    )
