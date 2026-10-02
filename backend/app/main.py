from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware

from . import config
from .gmail import fetch_messages
from .gmail_sync_service import GmailSyncService
from .oauth import clear_credentials, create_flow, load_credentials, save_credentials
from .rag_service import RAGResult, generate_gmail_answer
from .response import format_message


app = FastAPI(title="AI Gmail Knowledge Assistant - Phase 3")
app.add_middleware(SessionMiddleware, secret_key=config.SESSION_SECRET)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[config.FRONTEND_ORIGIN],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    query: str


def _authenticated_gmail_user() -> str:
    credentials = load_credentials()
    if credentials is None or not credentials.valid:
        raise HTTPException(status_code=401, detail="Connect Gmail before using chat.")
    return config.GMAIL_USER_ID


def _chat_response(result: RAGResult) -> dict:
    answer = result.answer
    if result.reason == "unrelated_to_gmail":
        answer = "I can only answer questions based on information available in your Gmail."
    elif result.reason == "insufficient_gmail_evidence":
        answer = "I couldn't find enough information in your Gmail to answer that."
    elif result.reason in {"generation_unavailable", "embedding_unavailable", "qdrant_unavailable"}:
        answer = "Something went wrong while searching your Gmail. Please try again."
    return {
        "answer": answer,
        "decision": result.decision,
        "reason": result.reason,
        "sources": [source.public_dict() for source in result.sources],
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "phase": "3"}


@app.get("/api/auth/login")
def login(request: Request):
    try:
        flow = create_flow()
    except RuntimeError as error:
        raise HTTPException(status_code=500, detail=str(error)) from error
    authorization_url, state = flow.authorization_url(
        access_type="offline", include_granted_scopes="true", prompt="consent"
    )
    request.session["oauth_state"] = state
    return RedirectResponse(authorization_url)


@app.get("/api/auth/callback")
def callback(request: Request, code: str | None = None, state: str | None = None):
    expected_state = request.session.pop("oauth_state", None)
    if not code or not state or state != expected_state:
        raise HTTPException(status_code=400, detail="Invalid OAuth callback state.")
    try:
        flow = create_flow()
        flow.fetch_token(code=code)
        save_credentials(flow.credentials)
    except Exception as error:
        raise HTTPException(status_code=502, detail="Google OAuth failed.") from error
    return RedirectResponse(f"{config.FRONTEND_ORIGIN}/?connected=1")


@app.get("/api/auth/status")
def auth_status() -> dict[str, bool]:
    credentials = load_credentials()
    return {"connected": credentials is not None and credentials.valid}


@app.post("/api/auth/logout")
def logout() -> dict[str, bool]:
    clear_credentials()
    return {"connected": False}


@app.get("/api/emails")
def emails(limit: int = Query(default=10, ge=1, le=10)) -> dict:
    _authenticated_gmail_user()
    try:
        return {"emails": [format_message(message) for message in fetch_messages(limit)]}
    except PermissionError as error:
        raise HTTPException(status_code=401, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail="Gmail API request failed.") from error


@app.post("/api/sync")
def sync_gmail() -> dict:
    user_id = _authenticated_gmail_user()
    try:
        return GmailSyncService().sync(user_id).public_dict()
    except PermissionError as error:
        raise HTTPException(status_code=401, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=502, detail="Gmail sync failed.") from error


@app.post("/api/chat")
@app.post("/api/query")
def chat(payload: ChatRequest) -> dict:
    user_id = _authenticated_gmail_user()
    try:
        return _chat_response(generate_gmail_answer(user_id, payload.query))
    except Exception as error:
        raise HTTPException(
            status_code=502,
            detail="Something went wrong while searching your Gmail. Please try again.",
        ) from error