from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pathlib import Path
from backend.db.database import init_db
from backend.routers import chat, rag_compare, agent, threads, benchmark

app = FastAPI(title="RAG Compare POC")

# DB 초기화
init_db()

# 라우터 등록
app.include_router(chat.router)
app.include_router(rag_compare.router)
app.include_router(agent.router)
app.include_router(threads.router)
app.include_router(benchmark.router)

# 정적 파일 서빙
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")


def _no_cache_response(path: str) -> FileResponse:
    return FileResponse(path, headers={
        "Cache-Control": "no-cache, no-store, must-revalidate",
        "Pragma": "no-cache",
        "Expires": "0",
    })


@app.get("/")
def root():
    return _no_cache_response(str(FRONTEND_DIR / "index.html"))


@app.get("/chat")
def chat_page():
    return _no_cache_response(str(FRONTEND_DIR / "chat.html"))


@app.get("/compare")
def compare_page():
    return _no_cache_response(str(FRONTEND_DIR / "compare.html"))
