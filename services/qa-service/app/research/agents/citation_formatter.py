import re
from urllib.parse import urlparse

from app.research.state import ResearchState
from app.research.source_validity import (
    web_validity,
    judge_sources,
    document_relevance_pct,
)

# Max chars of chunk content to expose as excerpt for frontend highlighting
_EXCERPT_LEN = 400


def _claims_for(answer: str, sid: int) -> str:
    """All answer sentences that cite [sid], with citation markers removed."""
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    hits = [s for s in sentences if re.search(rf"\[{sid}\]", s)]
    return " ".join(re.sub(r"\[\d+\]", "", s) for s in hits).strip()


def citation_formatter_node(state: ResearchState) -> dict:
    answer = state.get("final_answer", "") or ""
    source_map = state.get("cited_sources") or []

    # ── Short-circuit 1: guardrail refused ───────────────────────────────
    if state.get("guardrail_decision") == "refuse":
        return {"validated_answer": answer, "sources": []}

    # ── Short-circuit 2: nothing to format ───────────────────────────────
    if not answer or not source_map:
        return {"validated_answer": answer, "sources": []}

    # ── Which citation numbers appear in the answer ──────────────────────
    cited_order = [int(n) for n in re.findall(r"\[(\d+)\]", answer)]
    used_ids = set(cited_order)

    # ── One batched LLM call to judge all cited web sources ──────────────
    web_for_judge = [
        s for s in source_map
        if s["id"] in used_ids and s["type"] == "web"
    ]
    # FIXED: judge_sources(question, answer, web_sources) needs the question
    judged = judge_sources(state.get("question", ""), answer, web_for_judge)

    sources = []
    for s in source_map:
        if s["id"] not in used_ids:
            continue

        full_text = s.get("content", "") or ""

        if s["type"] == "document":
            sources.append({
                "id":            s["id"],
                "type":          "document",
                "title":         s["title"],
                "subtitle":      f"Page {s['page']}" if s.get("page") else "",
                "url":           None,
                "icon":          "📄",
                "doc_id":        s.get("doc_id", ""),
                "chunk_index":   s.get("chunk_index", 0),
                "page":          s.get("page", ""),
                "excerpt":       full_text[:_EXCERPT_LEN],
                "relevance_pct": document_relevance_pct(s.get("score")),  # NEW
            })
        else:
            url = s.get("url", "") or ""
            domain = urlparse(url).netloc.replace("www.", "") if url else ""
            domain = domain or "Web"

            v = web_validity(
                tavily_score=s.get("score", 0.5),
                url=url,
                published=s.get("published_date"),
                claim=_claims_for(answer, s["id"]),
                page_text=full_text,  # full content, not the excerpt
                judge=judged.get(s["id"]),
            )

            sources.append({
                "id":       s["id"],
                "type":     "web",
                "title":    (s.get("title") or domain)[:60],
                "subtitle": domain,
                "url":      url,
                "icon":     "🌐",
                "excerpt":  full_text[:_EXCERPT_LEN],
                "validity_pct":       v["validity_pct"],
                "validity_label":     v["label"],
                "validity_verified":  v["verified"],
                "validity_breakdown": v["breakdown"],
                "validity_method":    v["method"],
                "is_best":            False,
            })

    # ── Sort by first appearance in the answer ───────────────────────────
    order = {}
    for i, n in enumerate(cited_order):
        order.setdefault(n, i)
    sources.sort(key=lambda s: order.get(s["id"], 999))

    # ── Mark the best web source (verified ones only) ────────────────────
    web = [s for s in sources if s["type"] == "web" and s["validity_verified"]]
    if web:
        max(web, key=lambda s: s["validity_pct"])["is_best"] = True

    n_llm = sum(1 for s in sources if s.get("validity_method") == "llm")
    print(f"📎  CITATION FORMATTER: {len(used_ids)} citations used, "
          f"{len(sources)} sources attached, {n_llm} judged by LLM")

    return {
        "validated_answer": answer,
        "sources": sources,
    }