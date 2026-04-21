from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pathlib import Path
from backend.db.database import init_db
from backend.routers import chat, rag_compare, agent

app = FastAPI(title="RAG Compare POC")

# DB 초기화
init_db()

# 라우터 등록
app.include_router(chat.router)
app.include_router(rag_compare.router)
app.include_router(agent.router)

# 정적 파일 서빙
FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR / "static")), name="static")


@app.get("/")
def root():
    return FileResponse(str(FRONTEND_DIR / "index.html"))


@app.get("/chat")
def chat_page():
    return FileResponse(str(FRONTEND_DIR / "chat.html"))


@app.get("/compare")
def compare_page():
    return FileResponse(str(FRONTEND_DIR / "compare.html"))
