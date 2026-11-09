"""
eval_scorer.py
─────────────────────────────────────────────────────────────────────────────
Reusable scoring module for prompt evaluation. CORRECTED VERSION.
 
Changes vs. original:
  1. must_contain / must_not_contain are now actually populated in eval_cases.json,
     so `contains` and `no_hallucination` measure real things instead of always
     returning 1.0.
  2. French language marker list is now generic function words only (no more
     document-specific content words like "budget"/"euros" that overfit to one
     particular document).
  3. New `non_answer` dimension: penalizes the model dodging the question
     ("please specify", "veuillez préciser"...) instead of answering OR refusing.
     Previously this scored a free 1.0 on every dimension.
  4. Numeric matching in `contains`/`must_not_contain` normalizes common
     thousand-separator variants (12.500 / 12,500 / 12 500 / 12500) so ground
     truth doesn't have to enumerate every formatting variant by hand.
 
Usage:
    from eval_scorer import score_answer, is_refusal
    scores = score_answer(answer_text, test_case_dict)
"""
 
import re
from typing import Dict, Any
 
 
# ── Refusal detection (multilingual) ────────────────────────────────────────
 
REFUSAL_PHRASES = [
    # English
    "not found in document", "don't know", "doesn't contain", "not in the context",
    "cannot find", "not available in the context", "not present in the context",
    "not mentioned in the context", "not contained in", "no information",
    "not provided in", "not included in", "does not mention", "is not mentioned",
    "i don't have", "i do not have", "no relevant information",
    "does not contain", "text does not", "provided text does not",
    # French
    "je ne sais pas", "pas dans le document", "pas dans le contexte",
    "n'est pas dans", "ne contient pas", "pas disponible", "pas mentionné",
    "n'est pas présent", "introuvable", "aucune information", "pas présente",
    "pas fourni", "n'est pas mentionné", "je n'ai pas", "ne mentionne pas",
    "n'est pas spécifié", "aucun délai", "aucune mention",
    # Arabic
    "لا أعرف", "غير متوفرة", "غير موجود", "لا يحتوي", "لا يوجد",
    "ليس في", "غير متاح", "لا تتوفر", "غير موجودة", "لا أملك",
    "ليست متوفرة", "غير مذكور", "لا توجد معلومات", "لم يتم ذكر",
]
 
# ── Non-answer detection (model dodges instead of answering or refusing) ───
 
NON_ANSWER_PHRASES = [
    # English
    "please provide", "please specify", "could you clarify", "what would you like",
    "which information", "please let me know what",
    # French
    "veuillez préciser", "pourriez-vous préciser", "merci de préciser",
    "quelle information souhaitez",
    # Arabic
    "يرجى تحديد", "هل يمكنك التوضيح", "أي معلومات تريد",
]
 
 
def is_refusal(answer_lower: str) -> bool:
    """Return True if the answer appears to be a genuine refusal in any supported language."""
    return any(phrase in answer_lower for phrase in REFUSAL_PHRASES)
 
 
def is_non_answer(answer_lower: str) -> bool:
    """Return True if the model dodged the question instead of answering or refusing."""
    return any(phrase in answer_lower for phrase in NON_ANSWER_PHRASES)
 
 
def _normalize_numbers(text: str) -> str:
    """
    Normalize common thousand-separator variants so ground truth doesn't need
    to enumerate every formatting style: '12.500,000' / '12,500' / '12 500' / '12500'
    all become '12500' for matching purposes. Keeps the original text too
    (this function is only used to build a secondary matching string).
    """
    return re.sub(r"[.,\s](?=\d{3}\b)", "", text)
 
 
def _keyword_found(word: str, answer_lower: str, answer_normalized: str) -> bool:
    w = word.lower()
    if w in answer_lower:
        return True
    # Try normalized-number matching for keywords that are mostly digits
    if re.search(r"\d", w):
        w_norm = _normalize_numbers(w)
        if w_norm and w_norm in answer_normalized:
            return True
    return False
 
 
# ── Scoring function ────────────────────────────────────────────────────────
 
def score_answer(answer: str, case: Dict[str, Any]) -> Dict[str, float]:
    """
    Score a single answer against a test case.
 
    Returns a dict with keys:
        contains, no_hallucination, citation, refusal, non_answer, language, total
    All values are 0.0–1.0.
    """
    answer_lower = answer.lower()
    answer_normalized = _normalize_numbers(answer_lower)
    scores: Dict[str, float] = {}
 
    refused = is_refusal(answer_lower)
    dodged = is_non_answer(answer_lower)
 
    # 1. must_contain — partial credit per keyword found
    #    If the case expects a refusal and the model correctly refused,
    #    we don't penalize for missing content keywords (there's nothing to cite).
    must = case.get("must_contain", [])
    if case.get("expect_refusal") and refused:
        scores["contains"] = 1.0
    elif must:
        hits = sum(1 for w in must if _keyword_found(w, answer_lower, answer_normalized))
        scores["contains"] = round(hits / len(must), 2)
    else:
        scores["contains"] = 1.0  # no requirement → full marks
 
    # 2. must_not_contain — binary penalty (real hallucination check)
    must_not = case.get("must_not_contain", [])
    if must_not:
        bad_hits = sum(1 for w in must_not if _keyword_found(w, answer_lower, answer_normalized))
        scores["no_hallucination"] = 1.0 if bad_hits == 0 else 0.0
    else:
        scores["no_hallucination"] = 1.0
 
    # 3. citation
    if case.get("expect_citation"):
        # If the answer is a correct refusal, we don't penalise missing citation
        if case.get("expect_refusal") and refused:
            scores["citation"] = 1.0
        else:
            has_cite = (
                "[page" in answer_lower
                or "[excerpt" in answer_lower
                or "page " in answer_lower
            )
            scores["citation"] = 1.0 if has_cite else 0.0
    else:
        scores["citation"] = 1.0
 
    # 4. refusal
    if case.get("expect_refusal"):
        scores["refusal"] = 1.0 if refused else 0.0
    else:
        # If the case expects a real answer but the model refused instead,
        # that's also wrong — penalize a false refusal.
        scores["refusal"] = 0.0 if refused else 1.0
 
    # 5. non_answer — dodging the question is neither an answer nor a refusal
    scores["non_answer"] = 0.0 if dodged else 1.0
 
    # 6. language match (generic function words only — not tuned to any one document)
    lang = case.get("expect_lang")
    if lang == "ar":
        arabic_chars = sum(1 for c in answer if "\u0600" <= c <= "\u06FF")
        scores["language"] = 1.0 if arabic_chars > 15 else 0.0
    elif lang == "fr":
        words = set(re.findall(r"\b\w+\b", answer_lower))
        # Generic French function words only — no document-specific content words.
        french_markers = {
            "le", "la", "les", "des", "est", "sont", "pour", "dans",
            "je", "pas", "ne", "que", "qui", "ce", "cette", "un", "une",
            "et", "en", "au", "aux", "du", "de", "sur", "avec", "mais",
            "ou", "donc", "car", "ni", "son", "sa", "ses", "leur", "leurs",
            "nous", "vous", "il", "elle", "ils", "elles", "être", "avoir",
        }
        hits = len(words & french_markers)
        scores["language"] = 1.0 if hits >= 2 else 0.5 if hits >= 1 else 0.0
    else:
        scores["language"] = 1.0
 
    # total = average of all dimensions
    scores["total"] = round(sum(scores.values()) / len(scores), 3)
    return scores
 