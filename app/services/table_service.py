"""
app/services/table_service.py
==============================
Table extraction pipeline for invoices, contracts, and any tabular document.

Strategy (same philosophy as OCR service):
    Native PDF  → pdfplumber  (fast, accurate, no dependencies)
         ↓ no tables found AND text looks tabular (cheap heuristic gate)
    Scanned PDF / Image → img2table + Tesseract
         ↓ still no tables
    Gemini Vision fallback (handles borderless / complex tables)

Output per table:
    - JSON  : structured rows/columns for RAG pipeline
    - Markdown : human-readable for display

CHANGE (perf fixes — table extraction now runs on EVERY upload, not just
mode=="table", so cost here matters a lot more than it used to):

  1. Native-PDF path: pdfplumber runs first as before. If it finds nothing,
     we no longer blindly cascade into img2table (full page rasterization +
     Tesseract OCR) and then Gemini Vision (an API call per page) for EVERY
     plain-text document. A cheap heuristic (`_looks_tabular`) now gates
     whether that expensive fallback chain runs at all — most prose-only
     documents never pay for it.

  2. Scanned-PDF / image path: previously ran TWO independent OCR passes —
     once inside img2table (for table cell detection) and once again via
     ocr_service (for surrounding body text). Body text is now extracted
     ONCE via ocr_service and passed in / reused, instead of triggering a
     second full OCR pass through `_extract_text_for_pages`.
"""

import io
import json
import re
import tempfile
from pathlib import Path
from typing import Optional

import fitz                         # PyMuPDF — PDF → image rasterization
import pdfplumber
from PIL import Image

from app.core.config import settings


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _page_to_image(pdf_path: str, page_index: int, dpi: int = 200) -> Image.Image:
    """Render one PDF page to a PIL Image."""
    doc  = fitz.open(pdf_path)
    page = doc[page_index]
    mat  = fitz.Matrix(dpi / 72, dpi / 72)
    pix  = page.get_pixmap(matrix=mat)
    doc.close()
    return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def _clean_cell(cell) -> str:
    """
    Stringify a table cell, treating both None and pandas/numpy NaN as an
    empty string. img2table returns cells from a pandas DataFrame, and a
    bare `str(cell)` on a float NaN produces the literal text "nan" — which
    was showing up verbatim in extracted Arabic tables (e.g. "| nan | nan |").
    NaN != NaN is the standard (if slightly odd-looking) way to detect it
    without importing pandas/numpy just for this check.
    """
    if cell is None:
        return ""
    if isinstance(cell, float) and cell != cell:  # NaN check
        return ""
    text = str(cell)
    if text.strip().lower() == "nan":
        return ""
    return text


def _table_to_markdown(table: list[list]) -> str:
    """Convert a list-of-rows table into a Markdown table string."""
    if not table:
        return ""

    # Clean None/NaN values
    cleaned = [
        [_clean_cell(cell) for cell in row]
        for row in table
    ]

    header    = cleaned[0]
    separator = ["---"] * len(header)
    rows      = cleaned[1:]

    lines = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(separator) + " |")
    for row in rows:
        # Pad row if it has fewer columns than header
        padded = row + [""] * (len(header) - len(row))
        lines.append("| " + " | ".join(padded) + " |")

    return "\n".join(lines)


def _table_to_json(table: list[list]) -> list[dict]:
    """
    Convert a list-of-rows into a list of dicts using the first row as keys.
    [ {"col1": "val", "col2": "val"}, ... ]
    """
    if not table or len(table) < 2:
        return []

    headers = [
        _clean_cell(h) or f"col_{i}"
        for i, h in enumerate(table[0])
    ]
    result  = []
    for row in table[1:]:
        row_dict = {}
        for i, header in enumerate(headers):
            cell = row[i] if i < len(row) else None
            row_dict[header] = _clean_cell(cell)
        result.append(row_dict)
    return result


def _looks_tabular(text: str) -> bool:
    """
    Cheap heuristic used ONLY to decide whether it's worth paying for the
    expensive img2table/Gemini fallback chain on a native PDF that
    pdfplumber already failed to find tables in.

    This is deliberately conservative in favor of running the fallback
    (false positives just mean we still do the expensive check; false
    negatives mean we silently miss a real table) — but it stops the
    common case of a plain prose document paying full OCR + Vision API
    cost on every single upload just to conclude "no tables here."

    Signals:
      - financial/table-ish keywords (multilingual: FR/EN/AR-adjacent)
      - multiple lines with wide internal whitespace gaps, which is how
        tab/space-aligned columns usually look once flattened to text
    """
    if not text or not text.strip():
        return False

    keyword_hit = bool(re.search(
        r"\b(HT|TTC|TVA|total|montant|prix|qty|quantit|amount|price|subtotal)\b",
        text, re.IGNORECASE
    ))
    aligned_lines = sum(1 for line in text.splitlines() if re.search(r"\s{3,}", line))

    return keyword_hit or aligned_lines >= 3


# ─────────────────────────────────────────────────────────────────────────────
# Level 1 — pdfplumber (native PDFs only)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_tables_pdfplumber(pdf_path: str, page_indices: list[int]) -> list[dict]:
    """
    Extract tables from native PDF pages using pdfplumber.
    Returns list of table dicts with page info, JSON and Markdown.
    """
    tables_found = []

    with pdfplumber.open(pdf_path) as pdf:
        for idx in page_indices:
            if idx >= len(pdf.pages):
                continue
            page        = pdf.pages[idx]
            page_tables = page.extract_tables()

            for t_idx, raw_table in enumerate(page_tables):
                if not raw_table:
                    continue
                tables_found.append({
                    "page":     idx + 1,
                    "table_index": t_idx + 1,
                    "source":   "pdfplumber",
                    "json":     _table_to_json(raw_table),
                    "markdown": _table_to_markdown(raw_table),
                    "row_count": len(raw_table) - 1,          # minus header
                    "col_count": len(raw_table[0]) if raw_table else 0,
                })

    return tables_found


# ─────────────────────────────────────────────────────────────────────────────
# Level 2 — img2table (scanned PDFs and images)
# ─────────────────────────────────────────────────────────────────────────────

def _extract_tables_img2table_image(image: Image.Image, page_num: int) -> list[dict]:
    """
    Extract tables from a PIL Image using img2table + Tesseract.
    """
    from img2table.document import Image as Img2TableImage
    from img2table.ocr import TesseractOCR

    # img2table needs the image saved to a temp file or passed as bytes
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    buf.seek(0)

    ocr     = TesseractOCR(lang="fra+eng+ara")
    doc     = Img2TableImage(src=buf)
    # implicit_rows=True helps detect borderless tables
    extracted = doc.extract_tables(
        ocr=ocr,
        implicit_rows=True,
        borderless_tables=True,
        min_confidence=50,
    )

    tables_found = []
    for t_idx, table in enumerate(extracted):
        raw = table.df.values.tolist()
        # Prepend column names as first row
        cols     = list(table.df.columns)
        raw_full = [cols] + raw

        tables_found.append({
            "page":        page_num,
            "table_index": t_idx + 1,
            "source":      "img2table",
            "json":        _table_to_json(raw_full),
            "markdown":    _table_to_markdown(raw_full),
            "row_count":   len(raw),
            "col_count":   len(cols),
        })

    return tables_found


def _extract_tables_img2table_pdf(pdf_path: str, page_indices: list[int]) -> list[dict]:
    """
    Rasterize each selected page and run img2table on it.
    Used for scanned PDFs.
    """
    tables_found = []
    for idx in page_indices:
        image  = _page_to_image(pdf_path, idx, dpi=200)
        tables = _extract_tables_img2table_image(image, page_num=idx + 1)
        tables_found.extend(tables)
    return tables_found


# ─────────────────────────────────────────────────────────────────────────────
# Level 3 — Gemini Vision fallback
# ─────────────────────────────────────────────────────────────────────────────

def _extract_tables_gemini(image: Image.Image, page_num: int) -> list[dict]:
    """
    Ask Gemini Vision to extract tables as JSON.
    Used when pdfplumber and img2table find nothing.
    """
    import base64
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=settings.GEMINI_API_KEY)

    buf = io.BytesIO()
    image.save(buf, format="PNG")
    img_bytes = buf.getvalue()

    prompt = """
    Look at this document image and extract ALL tables you can see.
    For each table, return a JSON array of objects where keys are column headers.
    
    Return ONLY a valid JSON array like this:
    [
      {
        "table_index": 1,
        "data": [
          {"col1": "val", "col2": "val"},
          {"col1": "val", "col2": "val"}
        ]
      }
    ]
    
    If no tables are found, return an empty array: []
    Return ONLY the JSON, no explanation.
    """

    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=[
            prompt,
            types.Part.from_bytes(data=img_bytes, mime_type="image/png")
        ]
    )

    raw_text = response.text.strip()
    # Strip markdown fences if Gemini wraps in ```json
    if raw_text.startswith("```"):
        raw_text = raw_text.split("```")[1]
        if raw_text.startswith("json"):
            raw_text = raw_text[4:]
    raw_text = raw_text.strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        return []

    tables_found = []
    for item in parsed:
        data = item.get("data", [])
        if not data:
            continue

        # Convert back to row format for markdown
        headers  = list(data[0].keys())
        rows     = [[row.get(h, "") for h in headers] for row in data]
        raw_full = [headers] + rows

        tables_found.append({
            "page":        page_num,
            "table_index": item.get("table_index", len(tables_found) + 1),
            "source":      "gemini",
            "json":        data,
            "markdown":    _table_to_markdown(raw_full),
            "row_count":   len(rows),
            "col_count":   len(headers),
        })

    return tables_found


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction alongside tables
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text_for_pages(pdf_path: str, page_indices: list[int], is_scanned: bool) -> str:
    """
    Extract surrounding text (non-table content) for context.
    Native → pdfplumber text. Scanned → reuse OCR service.

    NOTE: for the scanned/image path, callers of `extract_tables()` should
    prefer passing already-extracted OCR text via the `existing_text`
    parameter instead of relying on this function to OCR a second time.
    This function remains as the fallback path when no text was supplied.
    """
    if is_scanned:
        from app.services.ocr_service import ocr_pdf_pages
        return ocr_pdf_pages(pdf_path, page_indices)

    parts = []
    with pdfplumber.open(pdf_path) as pdf:
        for idx in page_indices:
            if idx >= len(pdf.pages):
                continue
            text = pdf.pages[idx].extract_text() or ""
            if text.strip():
                parts.append(f"[Page {idx + 1}]\n{text.strip()}")
    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Main entry point — called by document_service.py
# ─────────────────────────────────────────────────────────────────────────────

def extract_tables(
    file_path: str,
    page_indices: list[int],
    is_scanned: bool,
    existing_text: Optional[str] = None,
) -> dict:
    """
    Extract tables + surrounding text from a PDF or image.

    Parameters
    ----------
    file_path     : path to the file
    page_indices  : 0-based list of pages to process (from _select_pages)
    is_scanned    : True → use img2table, False → try pdfplumber first
    existing_text : OPTIONAL — if the caller already ran OCR / native text
                    extraction for this document (e.g. document_service.py
                    already calls ocr_pdf_pages for scanned PDFs), pass that
                    text here to avoid a second, redundant OCR pass. If not
                    provided, this function extracts text itself as before.

    Returns
    -------
    dict with:
        tables       : list of table dicts (page, json, markdown, row/col counts)
        text         : surrounding text content
        tables_found : total number of tables detected
        engines_used : which extraction engines were invoked
    """
    suffix       = Path(file_path).suffix.lower()
    is_image     = suffix in {".png", ".jpg", ".jpeg", ".gif", ".bmp", ".tiff", ".webp"}
    tables_found = []
    engines_used = []

    # ── Image file ────────────────────────────────────────────────────────────
    if is_image:
        image = Image.open(file_path).convert("RGB")

        # img2table first
        tables_found = _extract_tables_img2table_image(image, page_num=1)
        engines_used.append("img2table")

        # Gemini fallback if nothing found
        if not tables_found and settings.GEMINI_API_KEY:
            tables_found = _extract_tables_gemini(image, page_num=1)
            engines_used.append("gemini")

        # Text — reuse caller-supplied OCR text if available, otherwise OCR once
        if existing_text is not None:
            text = existing_text
        else:
            from app.services.ocr_service import ocr_image_file
            text = ocr_image_file(file_path)

    # ── Native PDF ────────────────────────────────────────────────────────────
    elif not is_scanned:
        # Level 1 — pdfplumber
        tables_found = _extract_tables_pdfplumber(file_path, page_indices)
        engines_used.append("pdfplumber")

        # Surrounding text — reuse if supplied, otherwise extract now.
        # Native-PDF text extraction via pdfplumber is cheap (no OCR), so
        # doing it here even before deciding on fallbacks is fine cost-wise.
        text = existing_text if existing_text is not None else \
            _extract_text_for_pages(file_path, page_indices, is_scanned=False)

        # Level 2/3 — only pay for the expensive fallback chain if pdfplumber
        # found nothing AND the text actually looks like it might contain a
        # table. Without this gate, EVERY plain prose document would trigger
        # a full-page rasterization + Tesseract pass, and potentially a
        # Gemini Vision API call per page, just to confirm "no tables here."
        if not tables_found and _looks_tabular(text):
            tables_found = _extract_tables_img2table_pdf(file_path, page_indices)
            engines_used.append("img2table")

            if not tables_found and settings.GEMINI_API_KEY:
                for idx in page_indices:
                    image  = _page_to_image(file_path, idx, dpi=200)
                    tables = _extract_tables_gemini(image, page_num=idx + 1)
                    tables_found.extend(tables)
                if tables_found:
                    engines_used.append("gemini")

    # ── Scanned PDF ───────────────────────────────────────────────────────────
    else:
        # img2table directly (already works on images) — this is its OWN
        # internal OCR pass for table CELL detection, separate from the
        # body-text OCR below. We can't avoid this one without losing table
        # detection, but we CAN avoid a second, redundant body-text OCR pass
        # by reusing existing_text when the caller already has it.
        tables_found = _extract_tables_img2table_pdf(file_path, page_indices)
        engines_used.append("img2table")

        # Gemini fallback
        if not tables_found and settings.GEMINI_API_KEY:
            for idx in page_indices:
                image  = _page_to_image(file_path, idx, dpi=200)
                tables = _extract_tables_gemini(image, page_num=idx + 1)
                tables_found.extend(tables)
            if tables_found:
                engines_used.append("gemini")

        # Surrounding text — reuse caller-supplied OCR text if available.
        # This is the fix for the double-OCR issue: document_service.py
        # already runs ocr_pdf_pages() for scanned PDFs before calling here;
        # passing that result in via existing_text skips a second full pass.
        if existing_text is not None:
            text = existing_text
        else:
            text = _extract_text_for_pages(file_path, page_indices, is_scanned=True)

    return {
        "tables":        tables_found,
        "text":          text,
        "tables_found":  len(tables_found),
        "engines_used":  engines_used,
    }