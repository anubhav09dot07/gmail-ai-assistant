import os
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKEND_ROOT = PROJECT_ROOT / "backend"
load_dotenv(BACKEND_ROOT / ".env")


BACKEND_HOST = os.getenv("BACKEND_HOST", "127.0.0.1")
BACKEND_PORT = int(os.getenv("BACKEND_PORT", "8000"))
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
SESSION_SECRET = os.getenv("SESSION_SECRET", "development-only-change-me")
TOKEN_FILE = PROJECT_ROOT / os.getenv("TOKEN_FILE", "backend/token.json")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
GOOGLE_REDIRECT_URI = os.getenv(
    "GOOGLE_REDIRECT_URI", "http://localhost:8000/api/auth/callback"
)
GOOGLE_SCOPES = [
    scope.strip()
    for scope in os.getenv(
        "GOOGLE_SCOPES", "https://www.googleapis.com/auth/gmail.readonly"
    ).split(",")
    if scope.strip()
]
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1200"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
EMBEDDING_DIMENSION = int(os.getenv("EMBEDDING_DIMENSION", "384"))
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "gmail_knowledge")
QDRANT_BATCH_SIZE = int(os.getenv("QDRANT_BATCH_SIZE", "64"))
GMAIL_EVIDENCE_THRESHOLD = float(os.getenv("GMAIL_EVIDENCE_THRESHOLD", "0.62"))
GMAIL_RETRIEVAL_LIMIT = int(os.getenv("GMAIL_RETRIEVAL_LIMIT", "5"))
GMAIL_USER_ID = os.getenv("GMAIL_USER_ID", "personal-gmail")
GMAIL_SYNC_PAGE_SIZE = int(os.getenv("GMAIL_SYNC_PAGE_SIZE", "100"))
GMAIL_SYNC_MAX_MESSAGES = int(os.getenv("GMAIL_SYNC_MAX_MESSAGES", "1000"))
SYNC_METADATA_FILE = PROJECT_ROOT / os.getenv(
    "SYNC_METADATA_FILE", "backend/sync_metadata.json"
)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_TIMEOUT_SECONDS = float(os.getenv("GROQ_TIMEOUT_SECONDS", "30"))

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "groq")
LLM_MODEL = os.getenv("LLM_MODEL", GROQ_MODEL)
RAG_DEBUG = os.getenv("RAG_DEBUG", "false").lower() in ("true", "1", "yes")
RERANKER_ENABLED = os.getenv("RERANKER_ENABLED", "true").lower() in ("true", "1", "yes")
RRF_WEIGHT_VECTOR = float(os.getenv("RRF_WEIGHT_VECTOR", "1.0"))
RRF_WEIGHT_LEXICAL = float(os.getenv("RRF_WEIGHT_LEXICAL", "0.8"))
RRF_WEIGHT_GMAIL = float(os.getenv("RRF_WEIGHT_GMAIL", "1.2"))
RRF_WEIGHT_TEMPORAL = float(os.getenv("RRF_WEIGHT_TEMPORAL", "0.5"))