# tests/test_chunking_strategies.py

import pytest
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.services.chunking_service import create_chunks
from app.services.document_service import process_document

SAMPLE_PATH = "tests/samples/Devis 32 (1).pdf"
DEVIS_DOCX_PATH = "tests/samples/Devis 32.docx"


@pytest.mark.skipif(not os.path.exists(DEVIS_DOCX_PATH), reason="Devis_32.docx not found in tests/samples/")
class TestDevisDocxTableChunking:

    def test_real_extraction_and_table_chunking(self):
        result = process_document(file_path=DEVIS_DOCX_PATH, mode="full", index=False)
        assert result["status"] == "success"

        full_text = result["extraction"]["full_text"]
        print(f"\n📄 {result['file_name']} — {result['extraction'].get('word_count')} words, "
              f"{result['extraction'].get('char_count')} chars")
        print(f"\n--- RAW EXTRACTED TEXT ---\n{full_text}\n--- END ---\n")

        # Detect table rows generically — works whether the format is
        # Markdown pipes ("| ... |") or something else once we see it above
        table_lines = [line for line in full_text.splitlines() if line.strip().startswith("|")]
        print(f"Detected {len(table_lines)} Markdown-style table lines")

        # --- Naive recursive chunking ---
        splitter = RecursiveCharacterTextSplitter(chunk_size=60, chunk_overlap=0)
        naive_chunks = splitter.split_text(full_text)

        # --- Real project chunking ---
        aware_raw = create_chunks(
            full_text,
            {"file_name": result["file_name"], "doc_id": result["doc_id"]},
            chunk_size=60,
            chunk_overlap=0,
        )
        aware_chunks = [c["content"] for c in aware_raw]

        def broken_rows(chunks):
            if not table_lines:
                return None  # can't evaluate — no table lines detected
            original = set(table_lines)
            broken = 0
            for c in chunks:
                for line in c.splitlines():
                    line = line.strip()
                    if line.startswith("|") and line not in original:
                        broken += 1
            return broken

        naive_broken = broken_rows(naive_chunks)
        aware_broken = broken_rows(aware_chunks)

        print(f"\n--- Naive chunks ({len(naive_chunks)}) ---")
        for i, c in enumerate(naive_chunks):
            print(f"chunk_{i}: {c!r}")

        print(f"\n--- create_chunks output ({len(aware_chunks)}) ---")
        for i, c in enumerate(aware_chunks):
            print(f"chunk_{i}: {c!r}")

        print(f"\n📊 Broken table rows — naive: {naive_broken}, create_chunks: {aware_broken}")

        with open("tests/DEVIS_DOCX_TABLE_REPORT.md", "w", encoding="utf-8") as f:
            f.write("# Devis_32.docx — table chunking comparison\n\n")
            f.write(f"Table lines detected: {len(table_lines)}\n\n")
            f.write("| Strategy | Total chunks | Broken table rows |\n|---|---|---|\n")
            f.write(f"| Naive recursive | {len(naive_chunks)} | {naive_broken} |\n")
            f.write(f"| create_chunks | {len(aware_chunks)} | {aware_broken} |\n\n")
            f.write("## Naive chunks\n\n")
            for c in naive_chunks:
                f.write(f"- `{c!r}`\n")
            f.write("\n## create_chunks output\n\n")
            for c in aware_chunks:
                f.write(f"- `{c!r}`\n")

        print("\n✅ Report written to tests/DEVIS_DOCX_TABLE_REPORT.md")