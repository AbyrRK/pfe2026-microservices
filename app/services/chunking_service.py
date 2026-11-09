"""
app/services/chunking_service.py
"""

import re
import hashlib
from typing import List, Dict
from datetime import datetime
from langchain_text_splitters import RecursiveCharacterTextSplitter


def _split_table_with_header(text: str, chunk_size: int) -> List[str]:
    """
    Splits a large Markdown table on row boundaries, WITHOUT ever breaking
    mid-row, AND re-prepends the header + separator row to every resulting
    chunk (not just the first).

    Why this matters: without re-prepending the header, only the first
    chunk of a split table has column labels (e.g. "| Item | Qty | Price |").
    Every later chunk is bare data rows with no way for the LLM to know
    which number is which column — a silent correctness bug that only
    shows up on large tables, since small tables never hit this path.
    """
    lines = text.split("\n")

    # Markdown table shape: line 0 = header row, line 1 = "|---|---|" separator
    has_separator = len(lines) >= 2 and lines[1].strip().startswith("|---")
    header_lines  = lines[:2] if has_separator else lines[:1]
    header_block  = "\n".join(header_lines).strip()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size * 2,
        chunk_overlap=0,
        length_function=len,
        separators=["\n|", "\n"]  # only break between rows, never mid-row
    )
    raw_chunks = splitter.split_text(text)

    if not raw_chunks:
        return []

    result = [raw_chunks[0]]  # first chunk already contains the header
    for c in raw_chunks[1:]:
        stripped = c.strip()
        if header_block and not stripped.startswith(header_block.split("\n")[0]):
            result.append(f"{header_block}\n{stripped}")
        else:
            # already starts with the header (can happen depending on
            # where the splitter cut) — don't duplicate it
            result.append(stripped)

    return result


def _chunk_table_block(text: str, chunk_size: int) -> List[str]:
    """
    Chunks a block of text known to be a Markdown table, keeping rows atomic.
    Mirrors the "table" branch of create_chunks so the same protection
    applies whether content_type was declared by the caller or detected
    automatically from mixed text.
    """
    if len(text) <= chunk_size * 3:
        return [text]
    return _split_table_with_header(text, chunk_size)


def _segment_by_tables(text: str) -> List[Dict]:
    """
    Splits raw mixed text (prose + embedded Markdown tables) into ordered
    segments, tagging each as content_type "table" or "text".

    Why this exists: extraction pipelines often return one blob of text
    containing both prose and one or more "| ... |" tables (e.g. a DOCX
    with a paragraph followed by a pricing table). Without this, a caller
    has to pre-know which parts are tables and pass content_type per call
    — in practice nobody does that for a whole-document blob, so tables
    inside mixed text silently lose row protection and get chopped mid-row
    by the generic word/character splitter.
    """
    lines = text.split("\n")
    segments: List[Dict] = []
    buffer: List[str] = []
    buffer_is_table = None  # None = empty buffer

    def flush():
        if buffer:
            joined = "\n".join(buffer).strip("\n")
            if joined.strip():
                segments.append({
                    "text": joined,
                    "content_type": "table" if buffer_is_table else "text",
                })

    for line in lines:
        is_table_line = line.strip().startswith("|")
        if buffer_is_table is None:
            buffer_is_table = is_table_line
        if is_table_line != buffer_is_table:
            flush()
            buffer = []
            buffer_is_table = is_table_line
        buffer.append(line)
    flush()

    return segments


def create_chunks(
    text: str,
    metadata: dict,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[Dict]:
    if not text or not text.strip():
        return []

    declared_content_type = metadata.get("content_type", "text")

    # If the caller already declared this whole call as a table (existing
    # behavior, unchanged), keep the original single-path logic.
    if declared_content_type == "table":
        if len(text) <= chunk_size * 3:
            chunks = [text]
        else:
            chunks = _split_table_with_header(text, chunk_size)
        chunk_types = ["table"] * len(chunks)
        table_total_parts = len(chunks) if len(chunks) > 1 else 1
        table_flags = [declared_content_type == "table" and table_total_parts > 1] * len(chunks)
    else:
        # Mixed / plain text: auto-detect embedded table blocks so they
        # get row-safe chunking even though the caller didn't pre-classify
        # them. Prose segments still go through the normal splitter.
        segments = _segment_by_tables(text)

        # No table lines detected at all -> behave exactly as before.
        if len(segments) <= 1 and all(s["content_type"] == "text" for s in segments):
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
            )
            chunks = splitter.split_text(text)
            chunk_types = ["text"] * len(chunks)
            table_flags = [False] * len(chunks)
        else:
            chunks = []
            chunk_types = []
            table_flags = []
            for seg in segments:
                if seg["content_type"] == "table":
                    seg_chunks = _chunk_table_block(seg["text"], chunk_size)
                    is_multi = len(seg_chunks) > 1
                    chunks.extend(seg_chunks)
                    chunk_types.extend(["table"] * len(seg_chunks))
                    table_flags.extend([is_multi] * len(seg_chunks))
                else:
                    splitter = RecursiveCharacterTextSplitter(
                        chunk_size=chunk_size,
                        chunk_overlap=chunk_overlap,
                        length_function=len,
                        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
                    )
                    seg_chunks = splitter.split_text(seg["text"])
                    chunks.extend(seg_chunks)
                    chunk_types.extend(["text"] * len(seg_chunks))
                    table_flags.extend([False] * len(seg_chunks))

    # Recompute table_total_parts per contiguous table run for metadata
    # (only meaningful when a table was actually split into multiple parts).
    enriched_chunks = []
    # Figure out total_parts per table-run for accurate table_part numbering
    run_total = {}
    run_id = -1
    run_ids = []
    prev_type = None
    for t in chunk_types:
        if t == "table" and prev_type != "table":
            run_id += 1
        run_ids.append(run_id if t == "table" else None)
        prev_type = t
    for rid in set(r for r in run_ids if r is not None):
        run_total[rid] = run_ids.count(rid)

    run_counter = {}
    for idx, (chunk, ctype, rid) in enumerate(zip(chunks, chunk_types, run_ids)):
        chunk_hash = hashlib.md5(chunk.encode()).hexdigest()
        page_match = re.search(r"\[Page\s+(\d+)\]", chunk)
        page_number = int(page_match.group(1)) if page_match else metadata.get("page_number", idx + 1)

        chunk_meta = {
            "file_name":    metadata.get("file_name", ""),
            "doc_id":       metadata.get("doc_id", ""),
            "session_id":   metadata.get("session_id", ""),
            "language":     metadata.get("language", ""),
            "content_type": ctype,
            "chunk_index":  idx,
            "chunk_hash":   chunk_hash,
            "page_number":  page_number,
            "created_at":   datetime.utcnow().isoformat(),
        }

        if ctype == "table" and rid is not None and run_total.get(rid, 1) > 1:
            run_counter[rid] = run_counter.get(rid, 0) + 1
            chunk_meta["table_part"]        = run_counter[rid]
            chunk_meta["table_total_parts"] = run_total[rid]

        enriched_chunks.append({"content": chunk, "metadata": chunk_meta})

    return enriched_chunks