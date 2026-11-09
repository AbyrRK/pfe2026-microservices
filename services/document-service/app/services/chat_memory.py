"""
app/services/chat_memory.py
============================
Chat session persistence — Redis-backed, multi-document tracking.

Drop-in replacement for the JSON-file version. Public function signatures
are unchanged, so no other module needs to change beyond the import.

Storage layout:
    chat:{session_id}:messages   -> Redis LIST of JSON-encoded messages (RPUSH, O(1) append)
    chat:{session_id}:meta       -> Redis STRING, JSON-encoded metadata
    chat:sessions:index          -> Redis ZSET, member=session_id, score=updated_at (epoch)

TTL: sliding expiration on every write. Active sessions never expire;
abandoned ones clean themselves up automatically.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

import redis
from redis.exceptions import RedisError

from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Redis connection
# ─────────────────────────────────────────────────────────────────────────────

REDIS_URL = os.environ.get("REDIS_URL", "redis://redis:6379/0")
SESSION_TTL_SECONDS = int(os.environ.get("REDIS_SESSION_TTL_SECONDS", 60 * 60 * 24 * 30))  # 30 days
_MAX_SIDEBAR_SESSIONS = 20

_redis_client: Optional[redis.Redis] = None


def _client() -> redis.Redis:
    """Lazily create a single shared connection-pooled client."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def _messages_key(session_id: str) -> str:
    return f"chat:{session_id}:messages"


def _meta_key(session_id: str) -> str:
    return f"chat:{session_id}:meta"


_INDEX_KEY = "chat:sessions:index"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _touch(session_id: str, pipe=None) -> None:
    """Refresh TTL on a session's keys and bump its recency score in the index."""
    conn = pipe if pipe is not None else _client()
    ts = datetime.now(timezone.utc).timestamp()
    conn.expire(_messages_key(session_id), SESSION_TTL_SECONDS)
    conn.expire(_meta_key(session_id), SESSION_TTL_SECONDS)
    conn.zadd(_INDEX_KEY, {session_id: ts})


# ─────────────────────────────────────────────────────────────────────────────
# Redis-backed chat history
# ─────────────────────────────────────────────────────────────────────────────

class RedisChatMessageHistory(BaseChatMessageHistory):
    """
    Same public interface as the old JsonChatMessageHistory:
    .messages, add_message(), add_user_message(), add_ai_message(), clear()

    Reads go straight to Redis on every access (no process-local cache) so
    multiple replicas of qa_service always see consistent session state.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id

    def _decode(self, raw: list[str]) -> tuple[list[BaseMessage], list[dict]]:
        messages: list[BaseMessage] = []
        meta: list[dict] = []
        for item in raw:
            try:
                m = json.loads(item)
            except json.JSONDecodeError:
                logger.warning("Skipping corrupt message in session %s", self.session_id)
                continue
            msg = (
                HumanMessage(content=m["content"])
                if m.get("role") == "human"
                else AIMessage(content=m["content"])
            )
            messages.append(msg)
            meta.append({"is_summary": m.get("is_summary", False), "sources": m.get("sources", [])})
        return messages, meta

    @property
    def messages(self) -> list[BaseMessage]:
        try:
            raw = _client().lrange(_messages_key(self.session_id), 0, -1)
        except RedisError:
            logger.exception("Redis read failed for session %s; returning empty history", self.session_id)
            return []
        msgs, _ = self._decode(raw)
        return msgs

    def messages_with_meta(self) -> list[dict]:
        """Full message dicts (role/content/is_summary/sources), used by get_session_messages."""
        try:
            raw = _client().lrange(_messages_key(self.session_id), 0, -1)
        except RedisError:
            logger.exception("Redis read failed for session %s", self.session_id)
            return []
        out = []
        for item in raw:
            try:
                out.append(json.loads(item))
            except json.JSONDecodeError:
                continue
        return out

    def add_message(self, message: BaseMessage, is_summary: bool = False, sources: list = None) -> None:
        payload = {
            "role": "human" if isinstance(message, HumanMessage) else "ai",
            "content": message.content,
            "is_summary": is_summary,
            "sources": sources or [],
        }
        try:
            with _client().pipeline() as pipe:
                pipe.rpush(_messages_key(self.session_id), json.dumps(payload, ensure_ascii=False))
                _touch(self.session_id, pipe=pipe)
                pipe.execute()
        except RedisError:
            logger.exception("Redis write failed for session %s; message not persisted", self.session_id)

    def add_user_message(self, message: str) -> None:
        self.add_message(HumanMessage(content=message))

    def add_ai_message(self, message: str, is_summary: bool = False, sources: list = None) -> None:
        self.add_message(AIMessage(content=message), is_summary=is_summary, sources=sources)

    def clear(self) -> None:
        try:
            _client().delete(_messages_key(self.session_id))
        except RedisError:
            logger.exception("Redis clear failed for session %s", self.session_id)


# ─────────────────────────────────────────────────────────────────────────────
# Session helpers (same signatures as the JSON-file version)
# ─────────────────────────────────────────────────────────────────────────────

def get_session_history(session_id: str) -> RedisChatMessageHistory:
    return RedisChatMessageHistory(session_id)


def save_session(session_id: str):
    """Force-touch a session (messages auto-save; this refreshes metadata timestamp/TTL)."""
    try:
        meta = get_session_metadata(session_id)
        meta["updated_at"] = _now_iso()
        with _client().pipeline() as pipe:
            pipe.set(_meta_key(session_id), json.dumps(meta, ensure_ascii=False))
            _touch(session_id, pipe=pipe)
            pipe.execute()
    except RedisError:
        logger.exception("Redis save_session failed for %s", session_id)


def delete_session(session_id: str) -> bool:
    """Delete session history, metadata, and index entry."""
    try:
        removed = _client().delete(_messages_key(session_id), _meta_key(session_id))
        _client().zrem(_INDEX_KEY, session_id)
        return removed > 0
    except RedisError:
        logger.exception("Redis delete_session failed for %s", session_id)
        return False


def get_session_messages(session_id: str) -> list[dict]:
    return RedisChatMessageHistory(session_id).messages_with_meta()


def list_sessions() -> list[dict]:
    """
    Return the most recent _MAX_SIDEBAR_SESSIONS sessions for the sidebar.

    Uses the ZSET index (score = updated_at) so this is a single ZREVRANGE
    call instead of scanning and reading every session file.
    """
    try:
        session_ids = _client().zrevrange(_INDEX_KEY, 0, _MAX_SIDEBAR_SESSIONS - 1)
    except RedisError:
        logger.exception("Redis list_sessions failed")
        return []

    sessions = []
    for sid in session_ids:
        try:
            meta = get_session_metadata(sid)
            message_count = _client().llen(_messages_key(sid))
            sessions.append({
                "session_id":    sid,
                "title":         meta.get("title", sid[:8]),
                "session_type":  meta.get("session_type", "document"),
                "message_count": message_count,
                "updated_at":    meta.get("updated_at", ""),
                "file_count":    len(meta.get("files", [])),
            })
        except RedisError:
            logger.exception("Skipping session %s in list_sessions due to Redis error", sid)
            continue
    return sessions


# ─────────────────────────────────────────────────────────────────────────────
# Session metadata (title, files, etc.)
# ─────────────────────────────────────────────────────────────────────────────

def set_session_metadata(session_id: str, file_name: str = "", title: str = "",
    session_type: str = "document", research_question: str = "",
    extra_metadata: dict = None):

    data = get_session_metadata(session_id)
    data["session_id"] = session_id
    if file_name:
        data["file_name"] = file_name
    if title:
        data["title"] = title
    if session_type:
        data["session_type"] = session_type
    if research_question:
        data["research_question"] = research_question
    if "files" not in data:
        data["files"] = [file_name] if file_name else []
    if extra_metadata:
        data.update(extra_metadata)
    data["updated_at"] = _now_iso()

    try:
        with _client().pipeline() as pipe:
            pipe.set(_meta_key(session_id), json.dumps(data, ensure_ascii=False))
            _touch(session_id, pipe=pipe)
            pipe.execute()
    except RedisError:
        logger.exception("Redis set_session_metadata failed for %s", session_id)


def get_session_metadata(session_id: str) -> dict:
    try:
        raw = _client().get(_meta_key(session_id))
    except RedisError:
        logger.exception("Redis get_session_metadata failed for %s", session_id)
        return {"session_id": session_id, "files": []}
    if not raw:
        return {"session_id": session_id, "files": []}
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Corrupt metadata for session %s; resetting", session_id)
        return {"session_id": session_id, "files": []}


# ═════════════════════════════════════════════════════════════════════════════
# MULTI-DOCUMENT TRACKING
# ═════════════════════════════════════════════════════════════════════════════

def add_session_file(session_id: str, file_name: str):
    """Track a file as part of a session."""
    data = get_session_metadata(session_id)
    files = data.get("files", [])
    if file_name and file_name not in files:
        files.append(file_name)
    data["files"] = files
    data["updated_at"] = _now_iso()
    try:
        with _client().pipeline() as pipe:
            pipe.set(_meta_key(session_id), json.dumps(data, ensure_ascii=False))
            _touch(session_id, pipe=pipe)
            pipe.execute()
    except RedisError:
        logger.exception("Redis add_session_file failed for %s", session_id)


def get_session_files(session_id: str) -> list[str]:
    """Get all files associated with a session."""
    return get_session_metadata(session_id).get("files", [])


def remove_session_file(session_id: str, file_name: str):
    """Remove a file from a session's tracking."""
    data = get_session_metadata(session_id)
    files = data.get("files", [])
    if file_name in files:
        files.remove(file_name)
    data["files"] = files
    data["updated_at"] = _now_iso()
    try:
        with _client().pipeline() as pipe:
            pipe.set(_meta_key(session_id), json.dumps(data, ensure_ascii=False))
            _touch(session_id, pipe=pipe)
            pipe.execute()
    except RedisError:
        logger.exception("Redis remove_session_file failed for %s", session_id)