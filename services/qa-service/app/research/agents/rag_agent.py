import re
import hashlib

from app.research.state import ResearchState
from app.services.vector_store import vector_store
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from langchain_core.documents import Document

_SMALL_DOC_THRESHOLD = 10

# ── Module-level constants (not inside any function) ──────────────────────────
SUMMARY_SIGNALS = {
    "ttc", "tva", "total", "net à payer", "montant total",
    "hors taxes", "toutes taxes",
    "مجموع", "الإجمالي", "إجمالي", "ضريبة",
}


def _has_summary_chunk(chunks: list) -> bool:
    """
    Detects whether at least one chunk contains totals/TVA-related content.

    Hardened against formatting variants that would otherwise defeat a plain
    substring check — e.g. "T.V.A", "T V A", "tva" should all be recognized
    as the same signal. Also falls back to a bare percentage near a tax
    keyword (e.g. "19%" near "tva"/"tax") in case the exact SUMMARY_SIGNALS
    words don't appear verbatim.
    """
    for c in chunks:
        content_lower = (
            c.get("content", "").lower() if isinstance(c, dict)
            else c.page_content.lower()
        )

        # Normalize punctuation/whitespace so "t.v.a" / "t v a" / "tva" match
        normalized = re.sub(r"[.\s]+", "", content_lower)

        if any(
            sig.replace(".", "").replace(" ", "") in normalized
            for sig in SUMMARY_SIGNALS
        ):
            return True

        # Fallback: a percentage figure near a tax keyword (catches totals
        # lines that use "%" without spelling out "tva"/"ttc" nearby)
        if re.search(r"\b(1[0-9]|[0-9])\s?%", content_lower) and (
            "tax" in content_lower or "tva" in normalized
        ):
            return True

    return False


def _chunk_index_of(point) -> int:
    """
    Reads chunk_index from a Qdrant point's payload, handling both the flat
    and double-nested metadata shapes that show up across the ingestion /
    search paths (same fallback pattern used in the format loop below).
    Returns 0 only if chunk_index is genuinely missing in both shapes —
    NOT as a silent default that makes every point tie during sort.
    """
    meta = point.payload.get("metadata", {}) or {}
    nested = meta.get("metadata", {}) or {}

    idx = meta.get("chunk_index")
    if idx is None:
        idx = nested.get("chunk_index")
    if idx is None:
        idx = 0

    return idx


def _dedup_key_for(doc) -> str:
    """
    Strong dedup key for a Document. Prefers the chunk_hash already stored
    in metadata at indexing time (see chunking_service.py — it's an MD5 of
    the FULL chunk text, unique per chunk). Falls back to hashing the full
    page_content ourselves if chunk_hash is missing for any reason.

    IMPORTANT: this must never be a truncated prefix of the content. Two
    genuinely different chunks (e.g. a table chunk and a totals/text chunk)
    can easily share the same page header / first ~120 chars — a prefix key
    collides on them and silently drops one during merge, which is exactly
    what happened to the TVA/TTC chunk here.
    """
    meta = doc.metadata or {}
    nested = meta.get("metadata", {}) or {}
    chunk_hash = meta.get("chunk_hash") or nested.get("chunk_hash")
    if chunk_hash:
        return chunk_hash
    return hashlib.md5(doc.page_content.encode()).hexdigest()


def _build_scope_filter(doc_id: str = "", session_id: str = ""):
    if doc_id:
        return Filter(must=[FieldCondition(
            key="metadata.doc_id", match=MatchValue(value=doc_id)
        )])
    elif session_id:
        return Filter(must=[FieldCondition(
            key="metadata.session_id", match=MatchValue(value=session_id)
        )])
    return None


def _count_chunks(doc_id: str = "", session_id: str = "") -> int:
    try:
        filt = _build_scope_filter(doc_id=doc_id, session_id=session_id)
        if filt is None:
            return 0

        points, _ = vector_store.client.scroll(
            collection_name=vector_store.collection_name,
            scroll_filter=filt,
            limit=500,
            with_payload=False,
            with_vectors=False,
        )
        return len(points)
    except Exception as e:
        print(f"📄 RAG: chunk count failed: {e}")
        return 999


def _get_all_chunks(doc_id: str = "", session_id: str = "") -> list:
    """
    Fetch EVERY chunk for a doc/session, unranked. Used for small documents
    where semantic top-k search can silently drop the one chunk that
    contains the answer (e.g. the totals/TVA line) just because it doesn't
    score highest for a given query. When you already have all the chunks,
    ranking only adds risk — so for small docs we skip it entirely.
    """
    try:
        filt = _build_scope_filter(doc_id=doc_id, session_id=session_id)
        if filt is None:
            return []

        points, _ = vector_store.client.scroll(
            collection_name=vector_store.collection_name,
            scroll_filter=filt,
            limit=200,
            with_payload=True,
            with_vectors=False,
        )

        sorted_points = sorted(points, key=_chunk_index_of)

        # DEBUG — shows exactly what the small-doc path had available
        for p in sorted_points:
            preview = p.payload.get("page_content", "")[:200].replace("\n", " ")
            print(f"📄 RAG: [all-chunks] idx={_chunk_index_of(p)} → '{preview}'")

        results = []
        for p in sorted_points:
            doc = Document(
                page_content=p.payload.get("page_content", ""),
                metadata=p.payload.get("metadata", {})
            )
            results.append((doc, _dedup_key_for(doc)))
        return results
    except Exception as e:
        print(f"📄 RAG: fetch-all failed: {e}")
        return []


def _get_tail_chunks(doc_id: str = "", session_id: str = "", n: int = 5) -> list:
    try:
        filt = _build_scope_filter(doc_id=doc_id, session_id=session_id)
        if filt is None:
            return []

        all_points, _ = vector_store.client.scroll(
            collection_name=vector_store.collection_name,
            scroll_filter=filt,
            limit=200,
            with_payload=True,
            with_vectors=False,
        )

        # DEBUG — temporary, remove once you've confirmed real indices show up
        print(f"📄 RAG: tail sort keys → {[_chunk_index_of(p) for p in all_points]}")

        sorted_points = sorted(all_points, key=_chunk_index_of)

        tail = sorted_points[-n:] if len(sorted_points) >= n else sorted_points

        results = []
        for p in tail:
            doc = Document(
                page_content=p.payload.get("page_content", ""),
                metadata=p.payload.get("metadata", {})
            )
            results.append((doc, _dedup_key_for(doc)))
        return results
    except Exception as e:
        print(f"📄 RAG: tail fetch failed: {e}")
        return []


def _wrap_search_results(docs: list) -> list:
    return [(doc, _dedup_key_for(doc)) for doc in docs]


def rag_node(state: ResearchState) -> dict:
    session_id = state.get("session_id", "")
    doc_id     = state.get("doc_id", "")
    query      = state.get("reformulated_query") or state["question"]

    print(f"📄 RAG: query='{query[:50]}...' | session={session_id[:8] if session_id else 'NONE'} | doc_id={doc_id[:8] if doc_id else 'NONE'}")

    if not session_id and not doc_id:
        print("📄 RAG: ⚠️  No session_id or doc_id — skipping document search")
        return {"rag_results": [], "rag_confidence": 0.0}

    total_chunks = _count_chunks(
        doc_id=doc_id,
        session_id=session_id if not doc_id else ""
    )

    if total_chunks <= _SMALL_DOC_THRESHOLD:
        # Small doc: fetch ALL chunks unconditionally instead of ranking via
        # semantic search. With so few chunks, top-k similarity search can
        # rank the answer-bearing chunk (e.g. totals/TVA line) below the cut
        # and silently drop it. "All chunks" is a strict superset of what
        # search would return, so this can only help.
        print(f"📄 RAG: small doc ({total_chunks} chunks) → fetching ALL chunks (no ranking)")
        paired = _get_all_chunks(
            doc_id=doc_id,
            session_id=session_id if not doc_id else ""
        )
        print(f"📄 RAG: fetched {len(paired)} chunks")

        # Fallback in case the scope filter above returns nothing (e.g. only
        # doc_id was set but points are indexed under session_id, or vice
        # versa) — mirrors the original doc_id → session_id fallback.
        if not paired and doc_id:
            paired = _get_all_chunks(session_id=session_id)
            print(f"📄 RAG: fallback session scope (all chunks) → {len(paired)} chunks")

        tail_paired = []

    else:
        if doc_id:
            raw = vector_store.search_by_doc(query, doc_id=doc_id, k=15)
            print(f"📄 RAG: used doc_id scope → {len(raw)} chunks")
            if not raw:
                raw = vector_store.search_by_session(query, session_id=session_id, k=15)
                print(f"📄 RAG: fallback session scope → {len(raw)} chunks")
        else:
            raw = vector_store.search_by_session(query, session_id=session_id, k=15)
            print(f"📄 RAG: used session scope → {len(raw)} chunks")

        paired      = _wrap_search_results(raw)
        tail_paired = _get_tail_chunks(doc_id=doc_id, session_id=session_id, n=5)
        print(f"📄 RAG: tail chunks fetched → {len(tail_paired)}")

    # ── Deduplicate ───────────────────────────────────────────────────────────
    seen   = set()
    merged = []
    for doc, dedup_key in paired + tail_paired:
        if dedup_key and dedup_key not in seen:
            seen.add(dedup_key)
            merged.append(doc)

    print(f"📄 RAG: {len(paired)} primary + {len(tail_paired)} tail → {len(merged)} after dedup")

    if not merged:
        print("📄 RAG: ⚠️  No chunks found")
        return {"rag_results": [], "rag_confidence": 0.0}

    # ── Totals safety net — runs BEFORE format loop ───────────────────────────
    # Still useful for the >10-chunk path (semantic search can still miss the
    # totals chunk there). For the small-doc path this should now almost
    # always already be satisfied since we fetched everything above — if it
    # still fires there, it means the totals/TVA text genuinely isn't in any
    # indexed chunk for this document.
    if not _has_summary_chunk(merged):
        print("📄 RAG: ⚠️  No totals/TVA chunk found — force-fetching tail chunks")
        force_tail = _get_tail_chunks(
            doc_id=doc_id,
            session_id=session_id if not doc_id else "",
            n=2
        )
        for doc, dedup_key in force_tail:
            if dedup_key not in seen:
                seen.add(dedup_key)
                merged.append(doc)
                print(f"📄 RAG: ✅ force-added tail chunk ({len(doc.page_content)} chars)")
            else:
                print("📄 RAG: tail chunk already in merged — skipping")

    # ── Format results ────────────────────────────────────────────────────────
    formatted = []
    for r in merged:
        meta   = r.metadata or {}
        nested = meta.get("metadata", {})

        file_name    = meta.get("file_name")   or nested.get("file_name")   or "Document"
        page         = meta.get("page_number") or meta.get("page") or nested.get("page_number") or nested.get("page") or "?"
        chunk_doc_id = meta.get("doc_id")      or nested.get("doc_id")      or doc_id
        chunk_index  = meta.get("chunk_index") or nested.get("chunk_index") or 0
        score        = meta.get("score", 0.7)

        formatted.append({
            "content":      r.page_content,
            "file_name":    file_name,
            "page":         page,
            "doc_id":       chunk_doc_id,
            "chunk_index":  chunk_index,
            "session_id":   session_id,
            "source_type":  "document",
            "score":        score,
            "content_type": meta.get("content_type") or nested.get("content_type", "text"),
        })

    confidence = min(len(formatted) / 10, 1.0)
    print(f"📄 RAG: {len(formatted)} chunks total, confidence={confidence:.2f}")
    return {"rag_results": formatted, "rag_confidence": confidence}