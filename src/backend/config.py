import os

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    def load_dotenv():
        return None

load_dotenv()


def _windows_user_env(name: str) -> str | None:
    if os.name != "nt":
        return None
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment") as key:
            value, _ = winreg.QueryValueEx(key, name)
            return value
    except OSError:
        return None


def _env(name: str, default: str = "") -> str:
    return os.getenv(name) or _windows_user_env(name) or default


OPENAI_API_KEY = _env("OPENAI_API_KEY")
LLM_PROVIDER = _env("LLM_PROVIDER", "openai")
OLLAMA_BASE_URL = _env("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = _env("OLLAMA_MODEL", "llama3")
EMBEDDING_MODEL = _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
CHROMA_PERSIST_DIR = _env("CHROMA_PERSIST_DIR", "./data/chroma")
SQLITE_DB_PATH = _env("SQLITE_DB_PATH", "./data/poc.db")
OPENAI_MODEL = _env("OPENAI_MODEL", "gpt-4o-mini")
DP3_MOCK_LLM = _env("DP3_MOCK_LLM", "true")
DP3_LLM_PROVIDER = _env("DP3_LLM_PROVIDER")
GROQ_API_KEY = _env("GROQ_API_KEY")
GROQ_BASE_URL = _env("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
GROQ_MODEL = _env("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_MIN_INTERVAL_SECONDS = float(_env("GROQ_MIN_INTERVAL_SECONDS", "2.2"))
GROQ_MAX_RETRIES = int(_env("GROQ_MAX_RETRIES", "5"))
GROQ_MAX_OUTPUT_TOKENS = int(_env("GROQ_MAX_OUTPUT_TOKENS", "256"))
GROQ_RATE_LIMIT_SAFETY = float(_env("GROQ_RATE_LIMIT_SAFETY", "0.85"))
