"""
app/services/deferred_chunks.py
=================================
Redis-backed replacement for document_service.py's in-memory
`_deferred_chunks: dict[str, list] = {}`.

Why this exists:
    The in-memory dict had the same class of bug the old JSON chat_memory
    had — it only lives in one process's RAM. That means:
      - A Document Service restart between "lazy upload" and "ensure-indexed"
        silently loses the pending chunks (session ends up permanently
        under-indexed, with no error surfaced anywhere).
      - If Document Service ever runs more than one worker/replica, a
        request can land on a worker that never saw the upload — same
        silent data loss.
      - `if session_id in d: d.pop(session_id)` is a check-then-act race:
        two threads can both pass the `in` check before either pops,
        and the second `.pop()` raises KeyError.

Fix:
    Chunks live in Redis as a LIST per session, with a TTL so abandoned
    sessions clean themselves up. "Pop everything" is done via a Lua
    script (LRANGE + DEL) so it's atomic — no other caller can observe
    a partial state or double-pop.

Setup:
    Uses the same REDIS_URL as the rest of the app. If you gave chat
    sessions their own DB index (e.g. .../1), you can point this at a
    different one too via DEFERRED_CHUNKS_REDIS_URL — not required,
    since keys are prefixed with "deferred:" and won't collide.
"""

import json
import logging
import os

import redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)

REDIS_URL = os.environ.get("DEFERRED_CHUNKS_REDIS_URL", os.environ.get("REDIS_URL", "redis://redis:6379/0"))

# Deferred chunks are meant to be indexed soon after upload (on the first
# question, or via the background thread). If a session is abandoned before
# that happens, don't let stale/wrong chunks linger forever.
DEFERRED_CHUNKS_TTL_SECONDS = int(os.environ.get("DEFERRED_CHUNKS_TTL_SECONDS", 60 * 60 * 24))  # 24h

_redis_client: redis.Redis | None = None

# Atomic "read everything, then clear it" — single round trip, single
# Redis execution slot, so no other client can interleave.
_POP_ALL_SCRIPT = """
local vals = redis.call('LRANGE', KEYS[1], 0, -1)
redis.call('DEL', KEYS[1])
return vals
"""


def _client() -> redis.Redis:
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)
    return _redis_client


def _key(session_id: str) -> str:
    return f"deferred:{session_id}:chunks"


def defer_chunks(session_id: str, chunks: list[dict]) -> None:
    """
    Append chunks to this session's deferred queue (creates it if absent).
    Equivalent to the old dict's `.extend()` / initial-assign branches —
    RPUSH naturally handles both cases.
    """
    if not chunks:
        return
    try:
        key = _key(session_id)
        payload = [json.dumps(c, ensure_ascii=False, default=str) for c in chunks]
        with _client().pipeline() as pipe:
            pipe.rpush(key, *payload)
            pipe.expire(key, DEFERRED_CHUNKS_TTL_SECONDS)
            pipe.execute()
    except RedisError:
        logger.exception(
            "Failed to defer %d chunk(s) for session %s — chunks will NOT be "
            "indexed until re-uploaded", len(chunks), session_id
        )


def peek_deferred_chunks(session_id: str) -> list[dict]:
    """
    Non-destructive read — used when we need to see what's pending
    (e.g. to build a fallback brief) without consuming it.
    """
    try:
        raw = _client().lrange(_key(session_id), 0, -1)
    except RedisError:
        logger.exception("Failed to read deferred chunks for session %s", session_id)
        return []
    return _decode_all(raw, session_id)


def pop_deferred_chunks(session_id: str) -> list[dict]:
    """
    Atomically read AND clear all deferred chunks for a session.
    Safe to call concurrently from multiple threads/workers — only one
    caller will ever see a given chunk.
    """
    try:
        raw = _client().eval(_POP_ALL_SCRIPT, 1, _key(session_id))
    except RedisError:
        logger.exception("Failed to pop deferred chunks for session %s", session_id)
        return []
    return _decode_all(raw, session_id)


def has_deferred_chunks(session_id: str) -> bool:
    try:
        return bool(_client().exists(_key(session_id)))
    except RedisError:
        logger.exception("Failed to check deferred chunks for session %s", session_id)
        return False


def _decode_all(raw: list[str], session_id: str) -> list[dict]:
    out = []
    for item in raw:
        try:
            out.append(json.loads(item))
        except json.JSONDecodeError:
            logger.warning("Skipping corrupt deferred chunk for session %s", session_id)
    return out