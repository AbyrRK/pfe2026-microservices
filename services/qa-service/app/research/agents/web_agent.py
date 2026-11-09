import json
import os
import re

import httpx
from tavily import TavilyClient
from app.research.state import ResearchState

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

ACADEMIC_DOMAINS = [
    "arxiv.org",
    "semanticscholar.org",
    "proceedings.neurips.cc",
    "aclanthology.org",
    "openreview.net",
    "dl.acm.org",
    "ieeexplore.ieee.org",
    "springer.com",
    "researchgate.net",
]

# ─────────────────────────────────────────────
# LLM search planner (replaces the keyword list)
# ─────────────────────────────────────────────

_PLANNER_PROMPT = """You plan a web search for a business-document assistant.

Question: "{question}"

Decide:
- "search_type": "academic" ONLY if the user explicitly wants scientific papers,
  authors, citations or research literature. Otherwise "general".
  (A question that merely contains words like "paper", "article" or "research",
  for example "the TVA of the paper", is "general".)
- "recent": true if the answer depends on current facts (prices, rates, laws,
  regulations, news, "this year", "these days", "currently"), otherwise false.

Reply ONLY with JSON: {{"search_type": "general", "recent": false}}"""


def _plan_search(question: str) -> dict:
    """Returns {"search_type": ..., "recent": ...}. Safe defaults on any failure."""
    default = {"search_type": "general", "recent": False}
    try:
        from app.services.llm_client import llm

        safe_q = question.replace('"', "'").replace("{", "").replace("}", "")[:400]
        raw = llm.generate_sync(
            prompt=_PLANNER_PROMPT.format(question=safe_q),
            temperature=0,
            max_tokens=60,
            use_cache=True,
        )
        if not raw or raw.startswith("[System"):
            return default
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return default
        data = json.loads(match.group(0))
        return {
            "search_type": "academic" if data.get("search_type") == "academic" else "general",
            "recent": bool(data.get("recent", False)),
        }
    except Exception as e:
        print(f"🌐 WEB: search planner failed, using defaults: {e}")
        return default


# ─────────────────────────────────────────────
# Semantic Scholar enrichment (free, no key)
# ─────────────────────────────────────────────

def _enrich_with_semantic_scholar(title: str) -> dict | None:
    """
    Given a paper title, fetch its canonical arXiv URL,
    PDF link, authors and year from Semantic Scholar API.
    Returns None if not found or on error.
    """
    try:
        resp = httpx.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={
                "query": title,
                "fields": "title,authors,year,externalIds,openAccessPdf",
                "limit": 1,
            },
            timeout=5.0,
        )
        data = resp.json().get("data", [])
        if not data:
            return None

        paper = data[0]
        arxiv_id = paper.get("externalIds", {}).get("ArXiv")
        pdf_info = paper.get("openAccessPdf") or {}

        return {
            "canonical_url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else None,
            "pdf_url":       pdf_info.get("url"),
            "authors":       [a["name"] for a in paper.get("authors", [])],
            "year":          paper.get("year"),
            "ss_title":      paper.get("title"),
        }
    except Exception as e:
        print(f"🔬 SemanticScholar enrichment failed: {e}")
        return None


# ─────────────────────────────────────────────
# Tavily search helpers
# ─────────────────────────────────────────────

def _tavily_search(query: str, domains: list[str] | None = None,
                   recent: bool = False) -> list[dict]:
    """
    Run a Tavily search. If `domains` is provided, restrict results to those.
    If `recent` is True, ask Tavily for results from the last year only.
    Returns a list of {title, url, content, score, published_date, source_type}.
    """
    kwargs = dict(
        query=query,
        search_depth="advanced",
        max_results=5,
        include_answer=False,
    )
    if domains:
        kwargs["include_domains"] = domains
    if recent:
        kwargs["time_range"] = "year"

    try:
        resp = tavily.search(**kwargs)
    except TypeError:
        # Older client without time_range support: retry without it
        kwargs.pop("time_range", None)
        resp = tavily.search(**kwargs)

    return [
        {
            "title":          r.get("title"),
            "url":            r.get("url"),
            "content":        r.get("content"),
            "score":          r.get("score", 0.5),
            "published_date": r.get("published_date"),
            "source_type":    "academic" if domains else "web",
        }
        for r in resp.get("results", [])
    ]


# ─────────────────────────────────────────────
# Main node
# ─────────────────────────────────────────────

def web_node(state: ResearchState) -> dict:
    query    = state.get("reformulated_query") or state["question"]
    question = state["question"]

    print(f"🌐 WEB: searching for '{query[:60]}...'")

    try:
        plan     = _plan_search(question)
        academic = plan["search_type"] == "academic"
        recent   = plan["recent"]
        print(f"🌐 WEB: plan = {plan}")

        if academic:
            print("🔬 WEB: academic query → using domain-restricted search")
            results = _tavily_search(query, domains=ACADEMIC_DOMAINS)

            for r in results:
                enrichment = _enrich_with_semantic_scholar(r["title"] or query)
                if enrichment:
                    if enrichment["canonical_url"]:
                        r["url"] = enrichment["canonical_url"]
                    r["pdf_url"] = enrichment.get("pdf_url")
                    r["authors"] = enrichment.get("authors", [])
                    r["year"]    = enrichment.get("year")

            if not results:
                print("🌐 WEB: academic search empty → fallback to general")
                results = _tavily_search(query, recent=recent)
        else:
            results = _tavily_search(query, recent=recent)
            # Recency filter too strict? Retry without it rather than return nothing.
            if not results and recent:
                print("🌐 WEB: no recent results → retry without time filter")
                results = _tavily_search(query)

        print(f"🌐 WEB: found {len(results)} results "
              f"({'academic' if academic else 'general'}{', recent' if recent else ''})")
        return {
            "web_results": results,
            "web_confidence": min(len(results) / 5, 1.0),
        }

    except Exception as e:
        print(f"🌐 WEB: ⚠️ failed ({e}), returning empty")
        return {"web_results": []}