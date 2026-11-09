"""
app/research/agents/guardrail_agent.py
=======================================
Agent 1 — Domain Gatekeeper

Refuses off-topic questions immediately, before any expensive LLM/API call.
Replies in the SAME language as the user's question.

CHANGE (false-positive refusal fix):
  Two real financial-analysis questions were incorrectly refused tonight:
    - "Quelle a été la marge brute (Gross Profit) totale pour l'année ?"
    - "This cafe spent about $9,550 on marketing... is that in line with
       typical marketing budgets for small coffee shops?"
  Root causes, both fixed below:
    1. BUSINESS_KEYWORDS was missing common financial-analysis terms
       (margin/marge, profit/bénéfice, marketing, COGS, YTD, etc.) in
       both English and French.
    2. The LLM slow-path prompt never told the classifier that a business
       document is already in scope for this session — so it judged
       analytical/comparative questions ("is this typical for a small
       business") in isolation and leaned toward refusing them as general
       opinion questions rather than document-grounded analysis.
    3. Keyword matching was exact-word only (`\\b[a-zA-Z0-9_]+\\b` set
       intersection), so plural/inflected forms like "budgets" didn't match
       "budget". Switched to substring-contains matching for Latin keywords
       too, mirroring what Arabic already did.
"""

import re
from functools import lru_cache

from app.core.language import detect_language, get_guardrail_refusal

# ---------------------------------------------------------------------------
# Business keyword lists (multilingual) for fast path check
# ---------------------------------------------------------------------------
BUSINESS_KEYWORDS = {
    # English
    "invoice", "contract", "report", "quote", "quotation", "offer",
    "payment", "balance", "tax", "vat", "revenue", "expense", "budget",
    "supplier", "vendor", "client", "customer", "order", "purchase",
    "agreement", "clause", "penalty", "deadline", "delivery", "amount",
    "financial", "statement", "accounting", "audit", "fiscal", "bill",
    "receipt", "document", "file", "company", "business", "firm",
    # Financial-analysis terms — added, previously missing
    "margin", "profit", "profitability", "income", "net income",
    "gross profit", "marketing", "cogs", "cost of goods", "ytd",
    "year-to-date", "p&l", "sales", "spending", "spend", "cash flow",
    "roi", "return on investment", "overhead", "operating cost",

    # French
    "facture", "contrat", "rapport", "devis", "offre", "paiement",
    "solde", "tva", "taxe", "revenu", "depense", "budget", "fournisseur",
    "client", "commande", "achat", "accord", "clause", "penalite",
    "livraison", "montant", "financier", "comptabilite", "audit",
    "fiscal", "recu", "document", "societe", "entreprise",
    # Financial-analysis terms — added, previously missing
    "marge", "brute", "benefice", "rentabilite", "marketing",
    "chiffre d'affaires", "ventes", "depenses", "resultat", "tresorerie",

    # Arabic
    "فاتورة", "عقد", "تقرير", "عرض", "دفع", "رصيد", "ضريبة",
    "إيراد", "مصروف", "ميزانية", "مورد", "عميل", "طلب", "شراء",
    "اتفاقية", "بند", "غرامة", "تسليم", "مبلغ", "مالي", "محاسبة",
    "تدقيق", "ضريبي", "وصل", "وثيقة", "شركة", "مؤسسة",
    "ربح", "هامش", "تسويق", "مبيعات",

    # Khaliji / Gulf Arabic
    "فلوس", "حساب", "مستحقات", "معاملة", "شيك", "قسط",
    "مؤسسة", "شركة", "زبون", "عميل", "استحقاق", "كفالة",
}

# Split once for fast substring checks (Arabic vs Latin handled the same way
# now — both use "is this keyword a substring of the question" rather than
# exact word-boundary matching, so plurals/inflections like "budgets" or
# "marges" still match "budget"/"marge").
ARABIC_BUSINESS_KEYWORDS = {kw for kw in BUSINESS_KEYWORDS if "\u0600" <= kw[0] <= "\u06FF"}
LATIN_BUSINESS_KEYWORDS = BUSINESS_KEYWORDS - ARABIC_BUSINESS_KEYWORDS

# Hard off-topic signals — if any match, refuse immediately
OFF_TOPIC_SIGNALS = {
    # Nature / geography
    "flower", "plant", "garden", "animal", "weather", "climate",
    "mountain", "beach", "ocean", "river", "lake", "forest",
    "fleur", "plante", "jardin", "animal", "météo", "montagne",
    "زهرة", "نبتة", "حيوان", "طقس", "جبل", "شاطئ", "نهر",

    # Entertainment / pop culture
    "movie", "film", "song", "music", "game", "sport", "recipe",
    "film", "chanson", "musique", "jeu", "sport", "recette",
    "فيلم", "أغنية", "موسيقى", "لعبة", "رياضة", "وصفة",
    "batman", "superhero", "marvel", "dc comics", "comic", "anime",
    "cartoon", "netflix", "spotify", "celebrity", "actor", "singer",

    # Personal / medical
    "diet", "health", "doctor", "medicine", "love", "relationship",
    "santé", "médecin", "amour", "relation",
    "صحة", "طبيب", "دواء", "حب", "علاقة",

    # Coding / tech (unrelated to documents)
    "python tutorial", "javascript", "html css", "algorithm",
    "sort array", "machine learning tutorial",
}

# General-knowledge phrasing patterns — catch "who is X" / "how is X" / "tell
# me about X" style questions that won't hit a keyword list but are almost
# never about business documents.
GENERAL_KNOWLEDGE_PATTERNS = [
    re.compile(r"\bwho is\b"), re.compile(r"\bhow is\b"),
    re.compile(r"\btell me about\b"), re.compile(r"\bwhat is\b.*\?$"),
    re.compile(r"\bqui est\b"), re.compile(r"\bcomment va\b"),
    re.compile(r"\bparle[- ]moi de\b"),
    re.compile(r"\bمن هو\b"), re.compile(r"\bكيف حال\b"),
]


def _fast_keyword_check(question: str) -> str:
    """
    Fast path: keyword scan before any LLM call.
    Returns: "proceed" | "refuse" | "uncertain"
    """
    q_lower = question.lower()

    # Check hard off-topic signals first (substring match)
    for signal in OFF_TOPIC_SIGNALS:
        if signal in q_lower:
            return "refuse"

    # Latin business keywords: substring match (not exact word-boundary),
    # so "budgets", "marges", "profitable" etc. still match "budget",
    # "marge", "profit" without needing every inflected form listed.
    for kw in LATIN_BUSINESS_KEYWORDS:
        if kw in q_lower:
            return "proceed"

    # Arabic tokens — substring match directly (no reliable word boundaries)
    for kw in ARABIC_BUSINESS_KEYWORDS:
        if kw in question:
            return "proceed"

    # General-knowledge phrasing with no business keyword present is a strong
    # off-topic signal (e.g. "how is batman", "who is the CEO of Apple" would
    # still hit BUSINESS_KEYWORDS via "company"/"CEO" isn't listed — extend
    # BUSINESS_KEYWORDS rather than special-casing here if that case matters).
    for pattern in GENERAL_KNOWLEDGE_PATTERNS:
        if pattern.search(q_lower):
            return "refuse"

    return "uncertain"


def run_guardrail(state: dict) -> dict:
    """
    Guardrail agent node for LangGraph.

    Reads:  state["question"], state["language"], state.get("session_has_documents")
    Writes: state["guardrail_decision"], state["final_answer"] (if refused)

    The refusal message is always in the SAME language as the question.
    """
    question: str = state.get("question", "")
    language: str = state.get("language", "en")
    has_docs: bool = bool(state.get("session_has_documents", False))

    # Always detect from the actual question text (more reliable than frontend hint)
    try:
        detected_lang = detect_language(question)
    except Exception:
        detected_lang = language  # fallback to state hint if detector fails

    effective_lang = detected_lang if detected_lang != "en" else language

    # --- Fast path ---
    fast = _fast_keyword_check(question)

    if fast == "proceed":
        return {
            **state,
            "guardrail_decision": "proceed",
            "guardrail_reason": "Business keyword matched (fast path)",
            "language": effective_lang,
        }

    if fast == "refuse":
        return {
            **state,
            "guardrail_decision": "refuse",
            "guardrail_reason": "Off-topic signal matched (fast path)",
            "final_answer": get_guardrail_refusal(effective_lang),
            "language": effective_lang,
        }

    # --- Slow path: LLM disambiguation ---
    try:
        llm_decision = _llm_guardrail_check(question, effective_lang, has_docs)
        if llm_decision == "proceed":
            return {
                **state,
                "guardrail_decision": "proceed",
                "guardrail_reason": "LLM classified as business-related",
                "language": effective_lang,
            }
        else:
            return {
                **state,
                "guardrail_decision": "refuse",
                "guardrail_reason": "LLM classified as off-topic",
                "final_answer": get_guardrail_refusal(effective_lang),
                "language": effective_lang,
            }
    except Exception as e:
        # Fail CLOSED: if the LLM classifier is unavailable, refuse rather than
        # silently let an unclassified question bypass domain scoping. A flaky
        # API call should not be able to disable the guardrail.
        return {
            **state,
            "guardrail_decision": "refuse",
            "guardrail_reason": f"LLM guardrail failed ({type(e).__name__}) — defaulting to refuse",
            "final_answer": get_guardrail_refusal(effective_lang),
            "language": effective_lang,
        }


@lru_cache(maxsize=1)
def _get_gemini_client():
    """Lazy-load and cache the Gemini client."""
    from google import genai
    from app.core.config import settings
    return genai.Client(api_key=settings.GEMINI_API_KEY)


def _llm_guardrail_check(question: str, language: str, has_docs: bool = False) -> str:
    """
    Use Gemini to classify ambiguous questions.
    Returns "proceed" or "refuse".
    """
    # Sanitize to avoid prompt injection via quotes
    safe_question = question.replace('"', "'").replace("{", "").replace("}", "")

    context_note = (
        "The user currently has a business document uploaded in this session "
        "(e.g. an invoice, contract, report, or financial statement such as a "
        "P&L). They may ask analytical, comparative, or benchmarking questions "
        "about it — not just simple lookups.\n\n"
        if has_docs else ""
    )

    prompt = f"""You are a business document assistant domain classifier.
The user's question is written in language code "{language}".
{context_note}Classify this question as either:
- "proceed" → the question relates to business/financial documents, data, or
  analysis — including simple lookups (amounts, dates, terms), calculations,
  AND analytical or comparative questions that reference or could reasonably
  relate to figures from a business document (e.g. "is this marketing spend
  typical for a small business", "what's a healthy gross margin", "does this
  tax rate comply with current regulations", "quelle a été la marge brute
  totale"). Treat benchmarking/typical-range questions tied to a business
  figure as IN-SCOPE, not as general opinion questions.
- "refuse" → the question is clearly unrelated to business/documents (nature,
  entertainment, personal/medical topics, general trivia, unrelated tech)

Question: "{safe_question}"

Reply with ONLY one word: proceed  OR  refuse"""

    client = _get_gemini_client()
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    result = response.text.strip().lower()
    return "proceed" if "proceed" in result else "refuse"


guardrail_node = run_guardrail