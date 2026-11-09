"""
.py
======================
Two modes, one tool — everything needed for the OCR section screenshots:

1) SPEED mode — sequential vs. parallel timing on a LARGE scanned PDF.
   Demonstrates the parallelism speedup (needs many pages to show up).

       python .py speed path/to/large_scanned.pdf

2) ROUTE mode — per-file diagnostic showing which OCR level/engine was
   selected, confidence, handwriting flag, Arabic ratio, and timing.
   Works on images (.png/.jpg/...) and PDFs (native or scanned), and
   accepts multiple files at once for a side-by-side table.

       python .py route facture_arabic.png scanned_english.pdf

Both modes reuse your real pipeline (app.services.ocr_service) — no
logic is duplicated here, so the numbers you screenshot are the real
production behaviour, not a simulation.
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import fitz  # PyMuPDF

from app.services.ocr_service import (
    process_single_page,
    ocr_image,
    ocr_scanned_pdf,
    is_scanned_pdf,
)


def ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


# ─────────────────────────────────────────────────────────────────────────
# MODE 1 — Speed benchmark (sequential vs parallel)
# ─────────────────────────────────────────────────────────────────────────

def run_sequential(pdf_path: str) -> float:
    doc = fitz.open(pdf_path)
    total = len(doc)
    args = [(i, doc[i], "fra+eng+ara", False) for i in range(total)]

    print(f"\n[{ts()}] === SEQUENTIAL RUN START — {total} pages ===")
    start = time.perf_counter()
    for idx, arg in enumerate(args):
        process_single_page(arg)
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            print(f"[{ts()}]   {idx + 1}/{total} pages processed (sequential)")
    elapsed = time.perf_counter() - start
    print(f"[{ts()}] === SEQUENTIAL RUN END — {elapsed:.1f}s total ===\n")
    doc.close()
    return elapsed


def run_parallel(pdf_path: str, max_workers: int = 4) -> float:
    doc = fitz.open(pdf_path)
    total = len(doc)
    args = [(i, doc[i], "fra+eng+ara", False) for i in range(total)]

    print(f"[{ts()}] === PARALLEL RUN START — {total} pages, {max_workers} workers ===")
    start = time.perf_counter()
    completed = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_single_page, a): i for i, a in enumerate(args)}
        for future in as_completed(futures):
            future.result()
            completed += 1
            if completed % 10 == 0 or completed == total:
                print(f"[{ts()}]   {completed}/{total} pages processed (parallel)")
    elapsed = time.perf_counter() - start
    print(f"[{ts()}] === PARALLEL RUN END — {elapsed:.1f}s total ===\n")
    doc.close()
    return elapsed


def mode_speed(pdf_path: str):
    if not is_scanned_pdf(pdf_path):
        print(f"⚠️  Warning: '{pdf_path}' looks like a NATIVE PDF (not scanned).")
        print("   This mode needs a large SCANNED PDF to show the OCR parallelism gain.")

    seq = run_sequential(pdf_path)
    par = run_parallel(pdf_path)
    speedup = seq / par if par > 0 else float("inf")

    print("=" * 55)
    print(" SPEED BENCHMARK SUMMARY")
    print("=" * 55)
    print(f" Sequential : {seq:.1f}s  ({seq / 60:.2f} min)")
    print(f" Parallel   : {par:.1f}s  ({par / 60:.2f} min)")
    print(f" Speedup    : {speedup:.2f}x")
    print("=" * 55)


# ─────────────────────────────────────────────────────────────────────────
# MODE 2 — Routing diagnostic (which engine handled each file, and why)
# ─────────────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}


def diagnose_file(path: str) -> dict:
    p = Path(path)
    ext = p.suffix.lower()

    print(f"\n[{ts()}] --- Processing '{p.name}' ---")
    start = time.perf_counter()

    if ext == ".pdf":
        scanned = is_scanned_pdf(path)
        if not scanned:
            doc = fitz.open(path)
            text = doc[0].get_text().strip()
            doc.close()
            elapsed = time.perf_counter() - start
            print(f"[{ts()}] '{p.name}' → NATIVE PDF, OCR skipped ({elapsed:.2f}s)")
            return {
                "file": p.name, "type": "native_pdf", "engine": "native_extraction",
                "confidence": "n/a", "handwritten": "n/a",
                "char_count": len(text), "elapsed_s": round(elapsed, 2),
            }
        else:
            result = ocr_scanned_pdf(path)
            elapsed = time.perf_counter() - start
            engines_used = {p["ocr_engine"] for p in result["metadata"]["pages"]}
            print(f"[{ts()}] '{p.name}' → SCANNED PDF, engines used: {engines_used} ({elapsed:.2f}s)")
            return {
                "file": p.name, "type": "scanned_pdf",
                "engine": "/".join(sorted(engines_used)),
                "confidence": "see per-page", "handwritten": "see per-page",
                "char_count": result["metadata"]["char_count"], "elapsed_s": round(elapsed, 2),
            }

    elif ext in IMAGE_EXTS:
        result = ocr_image(path)
        elapsed = time.perf_counter() - start
        meta = result["metadata"]
        print(
            f"[{ts()}] '{p.name}' → engine={meta['ocr_engine']}, "
            f"confidence={meta['ocr_confidence']:.1f}%, "
            f"handwritten={meta['is_handwritten']} ({elapsed:.2f}s)"
        )
        return {
            "file": p.name, "type": "image", "engine": meta["ocr_engine"],
            "confidence": f"{meta['ocr_confidence']:.1f}%",
            "handwritten": meta["is_handwritten"],
            "char_count": meta["char_count"], "elapsed_s": round(elapsed, 2),
        }

    else:
        print(f"[{ts()}] '{p.name}' → unsupported extension '{ext}', skipped")
        return {"file": p.name, "type": "unsupported", "engine": "-", "confidence": "-",
                "handwritten": "-", "char_count": 0, "elapsed_s": 0.0}


def mode_route(paths: list):
    rows = [diagnose_file(p) for p in paths]

    print("\n" + "=" * 90)
    print(" ROUTING DIAGNOSTIC SUMMARY")
    print("=" * 90)
    header = f"{'File':<25} {'Type':<14} {'Engine':<20} {'Confidence':<12} {'Handwritten':<12} {'Time (s)'}"
    print(header)
    print("-" * 90)
    for r in rows:
        print(
            f"{r['file']:<25} {r['type']:<14} {r['engine']:<20} "
            f"{str(r['confidence']):<12} {str(r['handwritten']):<12} {r['elapsed_s']}"
        )
    print("=" * 90)


# ─────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)

    mode = sys.argv[1].lower()
    args = sys.argv[2:]

    if mode == "speed":
        mode_speed(args[0])
    elif mode == "route":
        mode_route(args)
    else:
        print(f"Unknown mode '{mode}'. Use 'speed' or 'route'.")
        sys.exit(1)


if __name__ == "__main__":
    main()