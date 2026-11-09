"""
tests/benchmark_vectordb.py
============================
Qdrant vs ChromaDB — fair, reproducible comparison (PFE 2026)

What this script does:
  1. Embeds the SAME texts once (Gemini embeddings) and loads them into BOTH engines
     (in-memory, in-process), so the data and the conditions are identical.
  2. Times SEARCH ONLY (query vectors are pre-computed, so Gemini network time is
     excluded for both). Each query is repeated and the median is reported.
  3. Runs REAL functional tests on both engines (nothing is hardcoded):
       - filter by session
       - filter by language
       - combined filter (session AND language)
       - delete by file name
       - expiry cleanup (delete by timestamp filter)

What it does NOT show:
  - Behaviour at large scale (the dataset is tiny, so latency differences are noise).
  - The production Qdrant server over HTTP (this tests the engines in-process).
  - Which database is "better". The choice of Qdrant is a judgment about production fit.

Run:
    python tests/benchmark_vectordb.py
"""

import sys
import os
import time
import statistics

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

QUERIES = [
    ("English", "what are the main technical requirements"),
    ("English", "architecture and system design"),
    ("French",  "délai de livraison et budget"),
    ("French",  "description des ouvrages"),
    ("Arabic",  "المتطلبات التقنية"),
]

PDF_PATH = "tests/samples/AI_Deep_Research_Module_Architecture.pdf"

FALLBACK_TEXTS = [
    "The system architecture is composed of an ingestion service, a vector store and a research agent.",
    "The main technical requirements are scalability, multilingual support and low latency.",
    "Le délai de livraison est de trois mois et le budget total est de quarante mille dinars.",
    "La description des ouvrages comprend la conduite PVC, les regards de visite et le branchement.",
    "المتطلبات التقنية تشمل دعم اللغة العربية والفرنسية والإنجليزية وسرعة الاستجابة.",
    "The deep research agent combines document retrieval with web search to answer complex questions.",
]

N_MIN_ITEMS = 24      # minimum number of items loaded in each engine
MAX_CHUNKS = 60       # cap on PDF chunks to embed (keeps embedding quota small)
N_REPEATS = 20        # repetitions per query for timing
K = 5

LANGS = ["en", "fr", "ar"]
SESSIONS = ["A", "B"]
FILES = ["file_0.pdf", "file_1.pdf", "file_2.pdf", "file_3.pdf"]


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def load_texts():
    """Return (unique_texts, source_description)."""
    if os.path.exists(PDF_PATH):
        import fitz
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        doc = fitz.open(PDF_PATH)
        text = "\n\n".join(page.get_text("text") for page in doc)
        doc.close()
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = [c for c in splitter.split_text(text) if c.strip()][:MAX_CHUNKS]
        if chunks:
            return chunks, PDF_PATH
    return FALLBACK_TEXTS, "built-in sample sentences (PDF not found)"


def make_items(unique_texts, unique_vectors):
    """Expand to at least N_MIN_ITEMS items and attach identical metadata for both engines."""
    n = max(N_MIN_ITEMS, len(unique_texts))
    texts, vectors, payloads = [], [], []
    for i in range(n):
        j = i % len(unique_texts)
        texts.append(unique_texts[j])
        vectors.append(unique_vectors[j])
        payloads.append({
            "session_id": SESSIONS[i % len(SESSIONS)],
            "language":   LANGS[i % len(LANGS)],
            "file_name":  FILES[i % len(FILES)],
            "created_at": i,            # numeric timestamp stand-in, used for expiry test
        })
    return texts, vectors, payloads


# ─────────────────────────────────────────────────────────────────────────────
# Qdrant (in-memory)
# ─────────────────────────────────────────────────────────────────────────────

class QdrantEngine:
    name = "Qdrant"

    def __init__(self, vectors, texts, payloads):
        from qdrant_client import QdrantClient, models
        self.models = models
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name="bench",
            vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE),
        )
        self.client.upsert(
            collection_name="bench",
            points=[
                models.PointStruct(id=i, vector=vectors[i], payload={**payloads[i], "text": texts[i]})
                for i in range(len(vectors))
            ],
        )

    def _filter(self, conds):
        """conds: list of (key, value) equality conditions, or (key, ('lt', n)) for range."""
        m = self.models
        must = []
        for key, val in conds:
            if isinstance(val, tuple) and val[0] == "lt":
                must.append(m.FieldCondition(key=key, range=m.Range(lt=val[1])))
            else:
                must.append(m.FieldCondition(key=key, match=m.MatchValue(value=val)))
        return m.Filter(must=must)

    def search(self, vec, k=K, conds=None):
        flt = self._filter(conds) if conds else None
        if hasattr(self.client, "query_points"):
            res = self.client.query_points(
                collection_name="bench", query=vec, limit=k, query_filter=flt
            ).points
        else:
            res = self.client.search(
                collection_name="bench", query_vector=vec, limit=k, query_filter=flt
            )
        return [r.payload for r in res]

    def count(self, conds):
        return self.client.count(
            collection_name="bench", count_filter=self._filter(conds), exact=True
        ).count

    def delete(self, conds):
        self.client.delete(
            collection_name="bench",
            points_selector=self.models.FilterSelector(filter=self._filter(conds)),
        )


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB (in-memory)
# ─────────────────────────────────────────────────────────────────────────────

class ChromaEngine:
    name = "ChromaDB"

    def __init__(self, vectors, texts, payloads):
        import chromadb
        self.client = chromadb.Client()
        try:
            self.client.delete_collection("bench")
        except Exception:
            pass
        self.col = self.client.create_collection(name="bench", metadata={"hnsw:space": "cosine"})
        self.col.add(
            ids=[f"c{i}" for i in range(len(vectors))],
            embeddings=vectors,
            documents=texts,
            metadatas=payloads,
        )

    @staticmethod
    def _where(conds):
        clauses = []
        for key, val in conds:
            if isinstance(val, tuple) and val[0] == "lt":
                clauses.append({key: {"$lt": val[1]}})
            else:
                clauses.append({key: val})
        return clauses[0] if len(clauses) == 1 else {"$and": clauses}

    def search(self, vec, k=K, conds=None):
        kwargs = {"query_embeddings": [vec], "n_results": k}
        if conds:
            kwargs["where"] = self._where(conds)
        res = self.col.query(**kwargs)
        return res["metadatas"][0] if res.get("metadatas") else []

    def count(self, conds):
        return len(self.col.get(where=self._where(conds))["ids"])

    def delete(self, conds):
        self.col.delete(where=self._where(conds))


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

def time_search(engine, qvecs):
    """Median search time (ms) per query, search only."""
    medians = []
    for v in qvecs:
        engine.search(v)  # warm-up
        runs = []
        for _ in range(N_REPEATS):
            t0 = time.perf_counter()
            engine.search(v)
            runs.append((time.perf_counter() - t0) * 1000)
        medians.append(statistics.median(runs))
    return medians


def run_functional_tests(engine, qvec):
    """Return a list of (test_name, passed, detail). Order matters: deletes come last."""
    results = []

    def record(name, fn):
        try:
            ok, detail = fn()
        except Exception as exc:
            ok, detail = False, f"error: {str(exc)[:120]}"
        results.append((name, ok, detail))

    def filter_test(conds, keys):
        def run():
            hits = engine.search(qvec, k=K, conds=conds)
            ok = len(hits) > 0 and all(
                all(h.get(k_) == v_ for k_, v_ in zip(keys, [c[1] for c in conds])) for h in hits
            )
            return ok, f"{len(hits)} results, all matching" if ok else f"{len(hits)} results, mismatch"
        return run

    record("Filter by session",
           filter_test([("session_id", "A")], ["session_id"]))
    record("Filter by language",
           filter_test([("language", "fr")], ["language"]))
    record("Combined filter (session AND language)",
           filter_test([("session_id", "A"), ("language", "en")], ["session_id", "language"]))

    def delete_by_file():
        target = [("file_name", "file_0.pdf")]
        before = engine.count(target)
        engine.delete(target)
        after = engine.count(target)
        return (before > 0 and after == 0), f"{before} chunks before, {after} after"
    record("Delete chunks by file name", delete_by_file)

    def expiry_cleanup():
        old = [("created_at", ("lt", 5))]
        before = engine.count(old)
        engine.delete(old)
        after = engine.count(old)
        return (after == 0), f"{before} expired chunks before, {after} after (timestamp filter + delete)"
    record("Expiry cleanup (timestamp filter + delete)", expiry_cleanup)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def run_benchmark(unique_texts, unique_vectors, qvecs, source):
    texts, vectors, payloads = make_items(unique_texts, unique_vectors)

    print("=" * 70)
    print("  VECTOR DATABASE BENCHMARK — Qdrant vs ChromaDB (in-memory, same data)")
    print("  PFE 2026")
    print("=" * 70)
    print(f"\n  Data source : {source}")
    print(f"  Items loaded: {len(texts)} (same vectors + same metadata in both engines)")
    print(f"  Timing      : search only, median of {N_REPEATS} runs per query")

    engines = [QdrantEngine(vectors, texts, payloads), ChromaEngine(vectors, texts, payloads)]

    # ── Part 1: latency ──────────────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  PART 1 — Search latency (query embedding excluded for both)")
    print("─" * 70)
    timings = {e.name: time_search(e, qvecs) for e in engines}
    print(f"\n  {'Query':<42} {'Lang':<8} " + " ".join(f"{e.name + ' ms':>11}" for e in engines))
    for i, (lang, q) in enumerate(QUERIES):
        row = " ".join(f"{timings[e.name][i]:>11.2f}" for e in engines)
        print(f"  {q[:40]:<42} {lang:<8} {row}")
    avgs = {n: sum(t) / len(t) for n, t in timings.items()}
    print("\n  Average of medians: " + "   ".join(f"{n}: {a:.2f} ms" for n, a in avgs.items()))

    # ── Part 2: functional tests ─────────────────────────────────────────────
    print("\n" + "─" * 70)
    print("  PART 2 — Functional tests (run for real on both engines)")
    print("─" * 70)
    all_results = {e.name: run_functional_tests(e, qvecs[0]) for e in engines}
    test_names = [r[0] for r in all_results[engines[0].name]]
    print(f"\n  {'Test':<46} " + " ".join(f"{e.name:^12}" for e in engines))
    for idx, name in enumerate(test_names):
        marks = " ".join(f"{('PASS' if all_results[e.name][idx][1] else 'FAIL'):^12}" for e in engines)
        print(f"  {name:<46} {marks}")
    print("\n  Details:")
    for e in engines:
        for name, ok, detail in all_results[e.name]:
            print(f"    [{e.name}] {name}: {detail}")

    # ── Conclusion: based on what was measured, nothing hardcoded ────────────
    print("\n" + "=" * 70)
    print("  WHAT THIS SHOWS")
    print("=" * 70)
    passed = {e.name: sum(1 for r in all_results[e.name] if r[1]) for e in engines}
    total = len(test_names)
    for e in engines:
        print(f"  - {e.name}: {passed[e.name]}/{total} functional tests passed, "
              f"avg search {avgs[e.name]:.2f} ms")
    print("""
  Reading these results honestly:
  - Both engines can filter by metadata and delete by filter, so these features
    alone do not decide the choice.
  - With this few items, both engines answer in about a millisecond or less. Any gap
    is practically irrelevant for a chatbot (the LLM call takes seconds), and it says
    nothing about performance at large scale.
  - Neither engine was given a built-in TTL here: expiry cleanup is done the same
    way in both (a timestamp field + delete by filter).
  - The choice of Qdrant is a judgment about production fit (server mode, payload
    indexes, scaling options, LangChain integration), not a case of ChromaDB being
    unable to do the same operations.
  - This does not test the production Qdrant server over HTTP or large data volumes.
""")


def main():
    from app.core.config import settings
    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=settings.GEMINI_API_KEY,
    )
    unique_texts, source = load_texts()
    print(f"Embedding {len(unique_texts)} texts and {len(QUERIES)} queries (one-time)...")
    unique_vectors = embeddings.embed_documents(unique_texts)
    qvecs = [embeddings.embed_query(q) for _, q in QUERIES]
    run_benchmark(unique_texts, unique_vectors, qvecs, source)


if __name__ == "__main__":
    main()