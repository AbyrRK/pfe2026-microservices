"""
app/services/document_client.py — Q&A Service only.

Replaces the old `from app.services.document_service import ensure_indexed`
local import. _deferred_chunks lives only inside Document Service's process,
so this MUST be an HTTP call, not a shared function.
"""
import httpx
from app.core.config import settings


def ensure_indexed_remote(session_id: str, timeout: float = 30.0) -> bool:
    """
    Calls Document Service to flush any deferred chunks for this session
    before retrieval runs. Returns True if indexing was just triggered,
    False if the session was already indexed (or has no documents yet).
    """
    url = f"{settings.DOCUMENT_SERVICE_URL}/documents/session/{session_id}/ensure-indexed"
    try:
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url)
            resp.raise_for_status()
            data = resp.json()
            return bool(data.get("indexed_now", False))
    except Exception as e:
        print(f"⚠️  ensure_indexed_remote failed for session {session_id[:8]}: {e}")
        return False