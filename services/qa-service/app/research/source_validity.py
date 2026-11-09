"""Validity score for WEB sources only (documents are the user's own files).

Two layers:
  1. LLM judge (one batched call): classifies each source type, rates its
     relevance to the QUESTION, and checks whether the page supports the answer.
  2. Heuristic fallback (domain lists, number/word matching, Tavily score)
     used whenever the LLM is unavailable, so the score never breaks.
"""
import json
import re
from datetime import date, datetime
from email.utils import parsedate_to_datetime
from urllib.parse import urlparse

# ── Heuristic fallback: domain tiers ─────────────────────────────────────────
_HIGH_SUFFIXES = (".gov", ".gov.tn", ".edu", ".int")
_HIGH_DOMAINS = (
    "who.int", "worldbank.org", "imf.org", "oecd.org", "jort.gov.tn",
    "finances.gov.tn", "douane.gov.tn",
    "arxiv.org", "semanticscholar.org", "aclanthology.org", "openreview.net",
    "dl.acm.org", "ieeexplore.ieee.org", "springer.com", "nature.com",
)
_MED_DOMAINS = (
    "reuters.com", "bbc.com", "lemonde.fr",
    "businessnews.com.tn", "ilboursa.com", "lapresse.tn",
    "pwc.com", "kpmg.com", "ey.com", "deloitte.com",
    "researchgate.net",
)

# ── LLM judge: source type -> authority score ────────────────────────────────
AUTHORITY_BY_TYPE = {
    "government": 95,
    "institution": 90,
    "academic": 85,
    "major_press": 75,
    "commercial": 55,
    "blog_forum": 35,
    "unknown": 45,
}

WEIGHTS = {"relevance": 0.35, "authority": 0.25, "grounding": 0.25, "freshness": 0.15}
UNVERIFIED_BELOW = 40   # grounding under this => source marked unverified
HIGH_FROM = 75
MEDIUM_FROM = 55


# ─────────────────────────────────────────────────────────────────────────────
# Heuristics
# ─────────────────────────────────────────────────────────────────────────────
def _host(url: str) -> str:
    host = urlparse(url).netloc.lower().split(":")[0]
    return host[4:] if host.startswith("www.") else host


def _matches(host: str, domain: str) -> bool:
    return host == domain or host.endswith("." + domain)


def authority_pct(url: str) -> int:
    host = _host(url)
    if not host:
        return 40
    if host.endswith(_HIGH_SUFFIXES) or any(_matches(host, d) for d in _HIGH_DOMAINS):
        return 95
    if any(_matches(host, d) for d in _MED_DOMAINS):
        return 75
    return 50  # unknown site, blog, forum


def _to_date(value) -> date | None:
    """Tavily may give None, a date, or a string (ISO or RFC 2822)."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    try:
        return parsedate_to_datetime(text).date()
    except (TypeError, ValueError):
        return None


def _year_hint(text: str):
    years = [int(y) for y in re.findall(r"\b(20[12]\d)\b", text or "")]
    return max(years) if years else None


def freshness_pct(published, hint_text: str = "") -> int:
    d = _to_date(published)
    if d is None:
        y = _year_hint(hint_text)
        if y is None:
            return 60  # neutral when unknown
        age_years = date.today().year - y
        return 100 if age_years <= 0 else 80 if age_years == 1 else 40
    age = (date.today() - d).days
    if age < 0:
        return 100
    return 100 if age < 180 else 80 if age < 730 else 50


def _normalize_numbers(text: str) -> str:
    """'4,200' / '4.200' / '4 200' -> '4200' so formats compare equal."""
    text = re.sub(r"(?<=\d)[.,](?=\d)", "", text)
    return re.sub(r"(?<=\d)[ \u00a0\u202f](?=\d{3}\b)", "", text)


def grounding_pct(claim: str, page_text: str, sim: float | None = None) -> int:
    """How well the answer's claim is supported by the page text (0-100)."""
    if not claim.strip() or not page_text.strip():
        return 0

    claim_n = _normalize_numbers(claim)
    page_n = _normalize_numbers(page_text)

    nums = [n for n in re.findall(r"\d+", claim_n) if len(n) >= 2]
    num_score = (sum(n in page_n for n in nums) / len(nums)) if nums else 1.0

    if sim is not None:
        sim_score = max(0.0, min(1.0, (sim - 0.5) / 0.4))
    else:  # cheap fallback, no embedding quota used
        words = {w.lower() for w in re.findall(r"\w{4,}", claim)}
        low = page_text.lower()
        sim_score = (sum(w in low for w in words) / len(words)) if words else 0.0

    return round(100 * (0.5 * num_score + 0.5 * sim_score))


def document_relevance_pct(score) -> int | None:
    """Retrieval (cosine) score of a document chunk -> 0-100 for display.

    Same 0.5-0.9 window as grounding_pct; adjust if your Qdrant scores
    usually fall in a different range. Returns None when there is no score.
    """
    try:
        return round(100 * max(0.0, min(1.0, (float(score) - 0.5) / 0.4)))
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# LLM judge (one batched call for all web sources of an answer)
# ─────────────────────────────────────────────────────────────────────────────
_JUDGE_PROMPT = """You verify web sources used in an answer.

QUESTION: {question}

ANSWER:
{answer}

SOURCES:
{sources}

For each source return:
- "type": one of government, institution, major_press, academic, commercial, blog_forum, unknown
- "relevance": integer 0-100, how directly the source addresses the QUESTION
  (a page about import tariffs is NOT relevant to a VAT-rate question)
- "support": integer 0-100, how well the source text supports the claims that cite it

Reply ONLY with JSON, no explanation:
{{"results": [{{"id": 1, "type": "government", "relevance": 80, "support": 80}}]}}"""


def judge_sources(question: str, answer: str, web_sources: list[dict]) -> dict:
    """Returns {source_id: {"type", "relevance", "support"}} or {} on any failure."""
    if not web_sources:
        return {}
    try:
        from app.services.llm_client import llm  # lazy: keeps this module testable

        blocks = "\n\n".join(
            f"[{s['id']}] {s.get('url', '')}\n{(s.get('content') or '')[:700]}"
            for s in web_sources
        )
        raw = llm.generate_sync(
            prompt=_JUDGE_PROMPT.format(
                question=(question or "")[:400],
                answer=(answer or "")[:1500],
                sources=blocks,
            ),
            temperature=0,
            max_tokens=700,
            use_cache=True,
        )
        if not raw or raw.startswith("[System"):
            return {}
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return {}
        data = json.loads(match.group(0))
        return {int(r["id"]): r for r in data.get("results", []) if "id" in r}
    except Exception as e:
        print(f"Source judge failed, using heuristics: {e}")
        return {}


def _pct(value) -> int | None:
    try:
        return max(0, min(100, int(value)))
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Final score
# ─────────────────────────────────────────────────────────────────────────────
def web_validity(tavily_score, url, published, claim, page_text,
                 sim=None, judge: dict | None = None) -> dict:
    """judge = {"type", "relevance", "support"} from the LLM, or None."""
    try:
        relevance = round(max(0.0, min(1.0, float(tavily_score))) * 100)
    except (TypeError, ValueError):
        relevance = 50

    authority = authority_pct(url or "")
    grounding = grounding_pct(claim or "", page_text or "", sim)
    method = "heuristic"

    if judge:
        src_type = str(judge.get("type", "")).strip().lower()
        if src_type in AUTHORITY_BY_TYPE:
            authority = AUTHORITY_BY_TYPE[src_type]
            method = "llm"

        llm_rel = _pct(judge.get("relevance"))
        if llm_rel is not None:
            relevance = llm_rel
            method = "llm"

        llm_sup = _pct(judge.get("support"))
        if llm_sup is not None:
            grounding = llm_sup
            method = "llm"

    parts = {
        "relevance": relevance,
        "authority": authority,
        "grounding": grounding,
        "freshness": freshness_pct(published, f"{url or ''} {(page_text or '')[:300]}"),
    }
    pct = round(sum(parts[k] * w for k, w in WEIGHTS.items()))
    verified = parts["grounding"] >= UNVERIFIED_BELOW

    if not verified:
        label = "Unverified"
    else:
        label = "High" if pct >= HIGH_FROM else "Medium" if pct >= MEDIUM_FROM else "Low"

    return {
        "validity_pct": pct,
        "label": label,
        "verified": verified,
        "breakdown": parts,
        "method": method,
    }