"""
compare_llms.py
─────────────────────────────────────────────────────────────────────────────
Compare multiple LLM backends on the same retrieved context.

Backend name formats (all of these have a free tier / are free to run):
  • gemini                      → Gemini 2.5 Flash (Google AI Studio free tier)
  • gemini:<model>              → another Google model, e.g. gemini:gemini-2.5-flash-lite
  • groq:<model>                → Groq,     e.g. groq:llama-3.3-70b-versatile
  • mistral:<model>             → Mistral,  e.g. mistral:mistral-small-latest
  • cerebras:<model>            → Cerebras, e.g. cerebras:llama-3.3-70b
  • ollama                      → local Ollama, default model
  • ollama:<model>              → local Ollama, e.g. ollama:gemma2:2b
  • or:<provider>/<model>       → OpenRouter (optional; free ones end with :free)

Key design points:
  • Retrieval runs ONCE per case; every model gets the same context.
  • Failed answers (error / no chunks / rate limit / no credits) score 0.
  • 429 (rate limit) is retried automatically; 402 (no credits) is reported as such.
  • Free backends get an automatic minimum delay between calls.
  • A backend with any failed call is marked INCOMPLETE and is NOT ranked.
  • The script aborts early if the session has no chunks (wrong SESSION_ID).

Keys (in .env):  GEMINI_API_KEY, GROQ_API_KEY, MISTRAL_API_KEY, CEREBRAS_API_KEY,
                 OPENROUTER_API_KEY (only for or: backends)

Usage:
    python tests/compare_llms.py --backends gemini groq:llama-3.3-70b-versatile \
        mistral:mistral-small-latest ollama:gemma2:2b \
        --cases 17 --report --output LLM_COMPARISON_REPORT_free.md
─────────────────────────────────────────────────────────────────────────────
"""

import time
import sys
import os
import json
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser

from app.services.vector_store import vector_store
from app.core.config import settings
from eval_scorer import score_answer


# ── CONFIG ──────────────────────────────────────────────────────────────────

# Session that contains Devis 32.pdf + facture_arabic.pdf + P_L.xlsx.
# Override with:  $env:EVAL_SESSION_ID = "..."
SESSION_ID = os.environ.get("EVAL_SESSION_ID", "42b6cb20-fa7e-4735-ad96-910be6b1b8f4")
K = 5
DELAY = float(os.environ.get("EVAL_DELAY", "1.0"))
MAX_RETRIES = 3

DEFAULT_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma2:2b")

NAME_W = 50

# OpenAI-compatible providers: prefix → (base_url, env var for the key, min delay in s)
OPENAI_COMPATIBLE = {
    "groq":     ("https://api.groq.com/openai/v1", "GROQ_API_KEY",     2.5),
    "mistral":  ("https://api.mistral.ai/v1",      "MISTRAL_API_KEY",  2.5),
    "cerebras": ("https://api.cerebras.ai/v1",     "CEREBRAS_API_KEY", 2.5),
}

BENCHMARK_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a multilingual document assistant supporting Arabic, French, and English.
Rules:
- Detect the language of the user's question.
- Always respond in that same language.
- Answer ONLY from the context below. Cite as [Excerpt X].
- If the answer is not in the context, say so in the user's language.
- Never use outside knowledge.

Context:
{context}"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])


# ── LLM FACTORY ─────────────────────────────────────────────────────────────

def build_llm(backend: str):
    backend = backend.strip()
    key = backend.lower()
    prefix, _, model = backend.partition(":")
    prefix = prefix.lower()

    # Google: "gemini" (2.5 Flash) or "gemini:<model>"
    if key == "gemini" or prefix == "gemini":
        return ChatGoogleGenerativeAI(
            model=model if (prefix == "gemini" and model) else "gemini-2.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.3,
        )

    # Ollama (local): "ollama" or "ollama:<model>"
    if key.startswith("ollama"):
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            raise ImportError("pip install langchain-ollama")
        return ChatOllama(model=model or DEFAULT_OLLAMA_MODEL, temperature=0.3)

    # Direct OpenAI-compatible providers: groq / mistral / cerebras
    if prefix in OPENAI_COMPATIBLE:
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("pip install langchain-openai")
        base_url, env_var, _ = OPENAI_COMPATIBLE[prefix]
        api_key = os.environ.get(env_var)
        if not api_key:
            raise RuntimeError(f"{env_var} is missing from .env")
        if not model:
            raise ValueError(f"Model name missing, use {prefix}:<model>")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0.3,
            max_tokens=1024,
        )

    # OpenRouter (optional): or:<provider>/<model>
    if prefix == "or":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError:
            raise ImportError("pip install langchain-openai")
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing from .env")
        return ChatOpenAI(
            model=backend[3:],
            api_key=api_key,
            base_url="https://openrouter.ai/api/v1",
            temperature=0.3,
            max_tokens=1024,
        )

    raise ValueError(f"Unknown backend: {backend}")


def backend_delay(backend: str) -> float:
    """Minimum delay between calls, to respect free-tier rate limits."""
    key = backend.lower()
    prefix = key.split(":", 1)[0]
    if prefix in OPENAI_COMPATIBLE:
        return max(DELAY, OPENAI_COMPATIBLE[prefix][2])
    if prefix == "or" and key.endswith(":free"):
        return max(DELAY, 4.0)
    return DELAY


def classify_error(err: str) -> str:
    low = err.lower()
    if "402" in err or "credits" in low:
        return "no_credits"
    if "429" in err or "rate limit" in low or "exhausted" in low or "quota" in low:
        return "rate_limited"
    return "error"


# ── RETRIEVAL (once per case) ────────────────────────────────────────────────

def build_context(question: str, session_id: str, k: int) -> Tuple[str, int]:
    docs = vector_store.search_by_session(question, session_id=session_id, k=k)
    if not docs:
        return "", 0
    context = "\n\n".join(
        f"[Excerpt {i+1}]\n{d.page_content}"
        for i, d in enumerate(docs)
    )
    return context, len(docs)


def retrieve_with_retry(question: str, session_id: str, k: int, retries: int = 3) -> Tuple[str, int]:
    """Retrieve context; if the Gemini embedding quota (429) is hit, wait and retry."""
    for attempt in range(1, retries + 1):
        try:
            ctx, n = build_context(question, session_id, k)
            if n > 0:
                return ctx, n
            # 0 chunks is always abnormal here (top-k search always returns something),
            # so retry: it is usually a transient "Collection doesn't exist" glitch
            if attempt < retries:
                print(f"\n  ⏳ 0 chunks returned, retrying ({attempt}/{retries - 1})…", flush=True)
                time.sleep(3)
                continue
            return ctx, n
        except Exception as exc:
            err = str(exc)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                print(f"\n  ⏳ Gemini embedding quota hit. Waiting 65s (attempt {attempt}/{retries})…", flush=True)
                time.sleep(65)
                continue
            print(f"\n  ❌ Retrieval error: {err[:200]}")
            return "", 0
    print("\n  ❌ Retrieval failed after retries (quota).")
    return "", 0


# ── RUNNER ────────────────────────────────────────────────────────────────────

FAILED_SCORE = {"total": 0.0}


def run_backend(backend: str, case: Dict[str, Any], context: str, n_chunks: int) -> Dict[str, Any]:
    """Run one backend on one case using an already-retrieved context."""
    if not context:
        return {
            "answer": "[NO CHUNKS RETRIEVED]",
            "elapsed": 0.0,
            "status": "no_chunks",
            "chunks": 0,
            "scores": FAILED_SCORE,
        }

    delay = backend_delay(backend)
    answer, elapsed, status = "", 0.0, "error"

    try:
        llm = build_llm(backend)
        chain = BENCHMARK_PROMPT | llm | StrOutputParser()
    except Exception as exc:
        return {
            "answer": f"[ERROR: {str(exc)[:180]}]",
            "elapsed": 0.0,
            "status": "error",
            "chunks": n_chunks,
            "scores": FAILED_SCORE,
        }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            time.sleep(delay)
            t0 = time.time()
            answer = chain.invoke({
                "context": context,
                "question": case["question"],
                "history": [],
            })
            elapsed = round(time.time() - t0, 2)
            status = "ok"
            break
        except Exception as exc:
            err = str(exc)
            status = classify_error(err)
            answer = f"[{status.upper()}: {err[:180]}]"
            elapsed = 0.0
            # Retry only on rate limit (429); no-credits or other errors won't fix themselves
            if status == "rate_limited" and attempt < MAX_RETRIES:
                wait = 30 * attempt
                print(f"\n     ⏳ rate limited, waiting {wait}s (retry {attempt}/{MAX_RETRIES - 1})…", end=" ", flush=True)
                time.sleep(wait)
                continue
            break

    return {
        "answer": answer,
        "elapsed": elapsed,
        "status": status,
        "chunks": n_chunks,
        "scores": score_answer(answer, case) if status == "ok" else FAILED_SCORE,
    }


# ── ORCHESTRATOR ──────────────────────────────────────────────────────────────

def compare_backends(backends: List[str], max_cases: int = 12) -> Dict[str, Any]:
    cases_path = os.path.join(os.path.dirname(__file__), "eval_cases.json")
    with open(cases_path, "r", encoding="utf-8") as f:
        all_cases = json.load(f)

    cases = all_cases[:max_cases]

    print("\n")
    print("═" * 72)
    print("  LLM BACKEND COMPARISON — PFE2026")
    print("═" * 72)
    print(f"  Backends   : {', '.join(backends)}")
    print(f"  Cases      : {len(cases)}")
    print(f"  Session    : {SESSION_ID}")
    print(f"  Prompt     : Multilingual (Variant D)")
    print("═" * 72)

    # ── Sanity check: the session must contain chunks, or abort early ────────
    print("\n  Checking that the session has indexed chunks…", flush=True)
    _, n = retrieve_with_retry("Devis", SESSION_ID, 1)
    if n == 0:
        # first call can fail once ("Collection doesn't exist" quirk) → try again
        _, n = retrieve_with_retry("Devis", SESSION_ID, 1)
    if n == 0:
        print("\n  ❌ No chunks found for this session. Aborting.")
        print("     → Set the right session:  $env:EVAL_SESSION_ID = \"<your-session-id>\"")
        print("     → Or re-upload the 3 documents in one session and use its ID.")
        sys.exit(1)
    print("  ✅ Session OK.")

    # ── Preflight: drop backends that cannot even be built (missing key, bad name) ──
    usable = []
    for b in backends:
        try:
            build_llm(b)
            usable.append(b)
        except Exception as exc:
            print(f"  ❌ Skipping {b}: {str(exc)[:150]}")
    if not usable:
        print("\n  ❌ No usable backend. Check your keys / backend names. Aborting.")
        sys.exit(1)
    backends = usable

    results: Dict[str, Dict[str, Any]] = {b: {} for b in backends}
    contexts: Dict[str, str] = {}

    for case in cases:
        print(f"\n\n{'━'*72}")
        print(f"  CASE: {case['id']} — {case['description']}")
        print(f"  Q: {case['question']}")
        print(f"{'━'*72}")

        # Retrieve ONCE per case; every model gets the same context
        context, n_chunks = retrieve_with_retry(case["question"], SESSION_ID, K)
        print(f"  Retrieved {n_chunks} chunks")
        contexts[case["id"]] = context

        for backend in backends:
            print(f"\n  ▶ {backend} …", end=" ", flush=True)
            res = run_backend(backend, case, context, n_chunks)
            results[backend][case["id"]] = res

            flag = {"ok": "✅", "no_chunks": "⚠️", "rate_limited": "⛔",
                    "no_credits": "💳", "error": "❌"}.get(res["status"], "?")
            print(f"{flag} {res['status']} ({res['elapsed']}s, {res['chunks']} chunks) | score: {res['scores']['total']:.0%}")
            short = res["answer"].replace("\n", " ")[:300]
            print(f"  {short}{'…' if len(res['answer']) > 300 else ''}")

    # ── Summary ──────────────────────────────────────────────────────────────
    summary: Dict[str, Dict[str, Any]] = {}
    for backend in backends:
        scores = [results[backend][c["id"]]["scores"]["total"] for c in cases]
        statuses = [results[backend][c["id"]]["status"] for c in cases]
        times = [results[backend][c["id"]]["elapsed"] for c in cases if results[backend][c["id"]]["elapsed"] > 0]

        rl = statuses.count("rate_limited")
        nc = statuses.count("no_credits")
        er = statuses.count("error")
        nch = statuses.count("no_chunks")

        summary[backend] = {
            "avg_score": sum(scores) / len(scores),
            "ok": statuses.count("ok"),
            "rate_limited": rl,
            "no_credits": nc,
            "error": er,
            "avg_time": sum(times) / len(times) if times else 0.0,
            "complete": (rl + nc + er + nch) == 0,   # no failed calls or empty retrievals → can be ranked
        }

    print(f"\n\n{'═'*72}")
    print("  COMPARISON SUMMARY")
    print(f"{'═'*72}")
    print(f"  {'Backend':<{NAME_W}} {'Avg Score':>9} {'OK':>4} {'RateLim':>8} {'NoCred':>7} {'Err':>4} {'Time':>7}")
    print(f"  {'─'*NAME_W} {'─'*9} {'─'*4} {'─'*8} {'─'*7} {'─'*4} {'─'*7}")
    for backend in backends:
        s = summary[backend]
        mark = "" if s["complete"] else "  ⚠️ INCOMPLETE"
        print(f"  {backend:<{NAME_W}} {s['avg_score']:>8.0%} {s['ok']:>4} {s['rate_limited']:>8} {s['no_credits']:>7} {s['error']:>4} {s['avg_time']:>6.1f}s{mark}")

    # Only complete backends are ranked
    complete = [b for b in backends if summary[b]["complete"]]
    winner = None
    if complete:
        winner = max(complete, key=lambda b: summary[b]["avg_score"])
        print(f"\n  🏆 Best by score (complete runs only): {winner} ({summary[winner]['avg_score']:.0%})")
    else:
        print("\n  ⚠️  No backend completed all cases — nothing is ranked. Rerun the incomplete ones.")
    incomplete = [b for b in backends if not summary[b]["complete"]]
    if incomplete:
        print(f"  ⚠️  Incomplete (not ranked): {', '.join(incomplete)}")

    return {
        "results": results,
        "summary": summary,
        "winner": winner,
        "backends": backends,
        "cases": cases,
        "contexts": contexts,
        "timestamp": datetime.now().isoformat(),
    }


def generate_comparison_report(data: Dict[str, Any], path: str = "LLM_COMPARISON_REPORT_multi.md"):
    summary = data["summary"]
    winner = data["winner"]
    incomplete = [b for b in data["backends"] if not summary[b]["complete"]]

    lines = [
        "# LLM Backend Comparison Report",
        f"**Date:** {data['timestamp']}  ",
        f"**Backends:** {', '.join(data['backends'])}  ",
        f"**Cases:** {len(data['cases'])}  ",
        f"**Session:** {SESSION_ID}  ",
        "",
        "## Executive Summary",
        "",
    ]
    if winner:
        lines.append(f"**Best by average score (complete runs only):** `{winner}` ({summary[winner]['avg_score']:.0%})  ")
    else:
        lines.append("**No backend completed all cases, so nothing is ranked.**  ")
    if incomplete:
        lines.append(f"**Incomplete (not ranked):** {', '.join(f'`{b}`' for b in incomplete)}  ")

    lines += [
        "",
        "## Scoreboard",
        "",
        "| Backend | Avg Score | OK | Rate-Limited | No Credits | Error | Avg Latency | Status |",
        "|---------|-----------|----|--------------|------------|-------|-------------|--------|",
    ]

    for backend in sorted(summary.keys(), key=lambda b: summary[b]["avg_score"], reverse=True):
        s = summary[backend]
        status = "complete" if s["complete"] else "⚠️ incomplete"
        lines.append(
            f"| {backend} | {s['avg_score']:.0%} | {s['ok']} | {s['rate_limited']} | {s['no_credits']} | {s['error']} | {s['avg_time']:.1f}s | {status} |"
        )

    lines += ["", "## Score Matrix (cases × models)", ""]
    lines.append("| Case | " + " | ".join(data["backends"]) + " |")
    lines.append("|---|" + "---|" * len(data["backends"]))
    for case in data["cases"]:
        cells = [
            f"{data['results'][b][case['id']]['scores']['total']:.0%}"
            for b in data["backends"]
        ]
        lines.append(f"| {case['id']} | " + " | ".join(cells) + " |")

    lines += ["", "## Per-Case Results", ""]

    for case in data["cases"]:
        lines.append(f"### {case['id']}: {case['description']}")
        lines.append(f"**Question:** {case['question']}")
        lines.append("")
        lines.append("| Backend | Score | Status | Time | Answer Preview |")
        lines.append("|---------|-------|--------|------|----------------|")

        for backend in data["backends"]:
            entry = data["results"][backend][case["id"]]
            ans = entry["answer"].replace("\n", " ").replace("|", "/")[:100]
            if len(entry["answer"]) > 100:
                ans += "…"
            lines.append(
                f"| {backend} | {entry['scores']['total']:.0%} | {entry['status']} | {entry['elapsed']}s | {ans} |"
            )
        lines.append("")

    # ── Head-to-head against the first backend listed (put gemini first) ─────
    base = data["backends"][0]
    if len(data["backends"]) > 1:
        lines += [
            "## Head-to-head (per case)", "",
            f"Baseline: `{base}`. Number of cases where each other backend scored higher / equal / lower.", "",
            "| Backend | Higher | Equal | Lower |", "|---|---|---|---|",
        ]
        for b in data["backends"][1:]:
            hi = eq = lo = 0
            for case in data["cases"]:
                a = data["results"][b][case["id"]]["scores"]["total"]
                c = data["results"][base][case["id"]]["scores"]["total"]
                if a > c + 1e-9:
                    hi += 1
                elif abs(a - c) <= 1e-9:
                    eq += 1
                else:
                    lo += 1
            lines.append(f"| {b} | {hi} | {eq} | {lo} |")
        lines.append("")

    # ── Retrieved context per case (lets a reader check retrieval vs model) ──
    lines += [
        "## Appendix: retrieved context per case", "",
        "If the expected answer is missing from the context, a low score is a retrieval limitation, not a model failure.", "",
    ]
    for case in data["cases"]:
        ctx = data.get("contexts", {}).get(case["id"], "")
        lines.append(f"### {case['id']}")
        lines.append("```")
        lines.append(ctx[:1500] if ctx else "[no context]")
        lines.append("```")
        lines.append("")

    lines += [
        "## Note",
        "",
        "Failed answers (error / no chunks / rate limit / no credits) count as 0%.",
        "A backend with any failed call is marked incomplete and is not ranked.",
        "All models received the same retrieved context for each case.",
        "The highest average score is only one criterion. Also weigh latency, cost per query,",
        "language behaviour and refusal on the hallucination-trap cases. Cases where every model",
        "scores low are usually retrieval limitations, not model failures.",
        "",
        "---",
        "*Generated by compare_llms.py*",
    ]

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"\n📄 Comparison report written to: {path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", nargs="+", default=["gemini", "ollama"],
                        help="gemini | gemini:<model> | groq:<model> | mistral:<model> | "
                             "cerebras:<model> | ollama[:<model>] | or:<provider>/<model>")
    parser.add_argument("--cases", type=int, default=12,
                        help="Number of test cases to run")
    parser.add_argument("--report", action="store_true",
                        help="Generate Markdown report")
    parser.add_argument("--output", default="LLM_COMPARISON_REPORT_multi.md",
                        help="Report file name")
    parser.add_argument("--ollama-model", default=DEFAULT_OLLAMA_MODEL,
                        help=f"Default Ollama model for plain 'ollama' (default: {DEFAULT_OLLAMA_MODEL})")
    args = parser.parse_args()

    DEFAULT_OLLAMA_MODEL = args.ollama_model

    data = compare_backends(args.backends, max_cases=args.cases)

    if args.report:
        generate_comparison_report(data, path=args.output)