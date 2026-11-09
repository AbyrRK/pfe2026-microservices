"""
app/routes/research.py
======================
Deep Research endpoint — Celery async tasks.
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import List, Optional
from app.research.research_task import run_research_task
from app.celery_app import celery
from app.services.document_client import ensure_indexed_remote

router = APIRouter()

# ── Sentinel values that mean "no session" ────────────────────────────────────
_BAD_SESSION_VALUES = {"null", "undefined", "none", ""}


def _sanitize_session_id(sid: Optional[str]) -> Optional[str]:
    """Return None if the client sent a non-value, otherwise the clean string."""
    if not sid or str(sid).strip().lower() in _BAD_SESSION_VALUES:
        return None
    return sid.strip()


class ResearchRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    language: str = "en"
    session_has_documents: bool = False
    doc_id: Optional[str] = None
    doc_ids: Optional[List[str]] = None


# CHANGED: sources optional, question added
class SaveSourcesPayload(BaseModel):
    sources: list = []
    answer: str
    question: Optional[str] = None


@router.post("/query")
def start_research(payload: ResearchRequest):
    """Start research — returns task_id immediately."""

    clean_session_id = _sanitize_session_id(payload.session_id)
    clean_doc_id     = _sanitize_session_id(payload.doc_id) or ""
    clean_doc_ids    = payload.doc_ids or []

    # Flush any lazy-deferred chunks BEFORE the research agents run
    if clean_session_id and payload.session_has_documents:
        flushed = ensure_indexed_remote(clean_session_id)
        print(f"[Research] ensure_indexed for session {clean_session_id[:8]}: flushed={flushed}")

    task = run_research_task.delay(
        question=payload.question,
        session_id=clean_session_id,
        language=payload.language,
        session_has_documents=payload.session_has_documents,
        doc_id=clean_doc_id,
        doc_ids=clean_doc_ids,
    )
    return {"task_id": task.id, "status": "started"}


@router.get("/status/{task_id}")
def get_research_status(task_id: str):
    result = celery.AsyncResult(task_id)

    if result.state == "PENDING":
        return {"status": "pending",  "answer": None, "sources": [], "session_id": None}

    if result.state == "PROGRESS":
        return {
            "status":     "running",
            "message":    result.info.get("status", "Processing..."),
            "answer":     None,
            "sources":    [],
            "session_id": None,
        }

    if result.state == "SUCCESS":
        return {"status": "done", **result.result}

    return {
        "status":     "error",
        "answer":     "An unexpected error occurred.",
        "sources":    [],
        "session_id": None,
    }


# CHANGED: saves the exchange in the chat history (question + answer + sources)
@router.post("/session/{session_id}/save-sources")
async def save_session_sources(session_id: str, payload: SaveSourcesPayload):
    """Persist the research exchange in the chat history so it survives a refresh."""
    from app.services.chat_memory import get_session_history, set_session_metadata

    answer = (payload.answer or "").strip()
    if not answer or answer.startswith("[System"):
        return {"saved": False, "reason": "no valid answer"}

    history = get_session_history(session_id)
    existing = history.messages_with_meta()

    # Idempotent: never save the same answer twice
    already = any(m.get("role") == "ai" and m.get("content") == answer for m in existing)
    if not already:
        q = (payload.question or "").strip()
        last_human = next((m for m in reversed(existing) if m.get("role") == "human"), None)
        if q and not (last_human and last_human.get("content") == q):
            history.add_user_message(q)
        history.add_ai_message(answer, sources=payload.sources)

    set_session_metadata(
        session_id=session_id,
        session_type="research",
        extra_metadata={
            "last_research_sources": payload.sources,
            "last_research_answer": answer,
        },
    )
    return {"saved": not already}