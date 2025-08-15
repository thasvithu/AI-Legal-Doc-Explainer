import os
from typing import List, Dict, Any, Optional
from langchain.schema import Document
import google.generativeai as genai
from dotenv import load_dotenv
from utils.exception import CustomException

load_dotenv()
if os.getenv("GEMINI_API_KEY"):
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

_model = None
try:
    _model = genai.GenerativeModel("gemini-1.5-flash") if os.getenv("GEMINI_API_KEY") else None
except Exception:
    _model = None

RISK_KEYWORDS = {
    "penalty": "Potential penalty clause – check amounts and triggers.",
    "terminate": "Termination terms – verify notice periods and conditions.",
    "auto-renew": "Auto-renewal – ensure you know how to opt out.",
    "renew": "Renewal terms – look for automatic extensions.",
    "indemnify": "Indemnification – check scope of liability.",
    "liability": "Liability limitation – confirm caps and exclusions.",
    "warranty": "Warranty/guarantee terms – confirm duration and scope.",
    "confidential": "Confidentiality obligations – check duration & carve-outs.",
    "governing law": "Jurisdiction – ensure acceptable governing law.",
    "exclusive": "Exclusivity – may restrict other partnerships.",
    "non-compete": "Non-compete – evaluate scope & duration.",
}

# Clause categories (simple mapping; can be expanded)
CLAUSE_CATEGORIES = {
    "terminate": "Termination",
    "termination": "Termination",
    "penalty": "Penalty",
    "auto-renew": "Renewal",
    "renew": "Renewal",
    "indemnify": "Indemnification",
    "indemnification": "Indemnification",
    "liability": "Liability",
    "warranty": "Warranty",
    "confidential": "Confidentiality",
    "governing law": "Jurisdiction",
    "jurisdiction": "Jurisdiction",
    "exclusive": "Exclusivity",
    "non-compete": "Non-Compete",
    "payment": "Payment",
    "fee": "Payment",
    "intellectual property": "Intellectual Property",
    "ownership": "Intellectual Property",
}


def summarize_documents(docs: List[Document], max_chars: int = 6000) -> str:
    try:
        combined = "\n\n".join(d.page_content for d in docs)[:max_chars]
        if not _model:
            # Fallback simple heuristic summary
            first_bits = combined.split(". ")[:8]
            return "Simple summary (offline): " + ". ".join(first_bits)
        prompt = f"""
        Summarize the following legal document in clear, simple language. Avoid legal jargon.
        Focus on: parties, purpose, key obligations, payment/consideration, duration & termination, risks.
        Keep it under 250 words.
        Document Text:\n{combined}
        """
        resp = _model.generate_content(prompt)
        return resp.text
    except Exception as e:
        return str(CustomException(e))


def extract_key_clauses(docs: List[Document], top_n: int = 8) -> List[Dict[str, Any]]:
    try:
        clauses = []
        for d in docs:
            text = d.page_content
            for kw in RISK_KEYWORDS.keys():
                if kw in text.lower():
                    snippet_idx = text.lower().index(kw)
                    start = max(0, snippet_idx - 120)
                    end = min(len(text), snippet_idx + 180)
                    snippet = text[start:end].replace("\n", " ")
                    category = CLAUSE_CATEGORIES.get(kw, "General")
                    clauses.append({
                        "keyword": kw,
                        "category": category,
                        "snippet": snippet,
                        "page": d.metadata.get("page", d.metadata.get("page_number")),
                        "note": RISK_KEYWORDS[kw]
                    })
        # Deduplicate by keyword+page
        seen = set()
        unique = []
        for c in clauses:
            key = (c["keyword"], c.get("page"))
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique[:top_n]
    except Exception as e:
        return [{"error": str(CustomException(e))}]


def detect_red_flags(clauses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    flags = []
    for c in clauses:
        severity = "medium"
        kw = c["keyword"]
        if kw in {"penalty", "indemnify", "liability", "exclusive", "non-compete"}:
            severity = "high"
        elif kw in {"auto-renew", "renew", "terminate"}:
            severity = "medium"
        flags.append({**c, "severity": severity})
    return flags


def extract_entities(docs: List[Document]) -> Dict[str, Optional[str]]:
    """Very lightweight regex / heuristic entity extraction for key contract metadata.
    Returns dict with possible None values if not found.
    """
    import re
    combined = "\n".join(d.page_content for d in docs)
    entities: Dict[str, Optional[str]] = {
        "effective_date": None,
        "parties": None,
        "governing_law": None,
        "term_length": None,
    }
    # Effective Date
    m = re.search(r"(Effective Date|Commencement Date)[^\n]{0,40}?\b(on|:)?\s*([A-Z][a-z]+\s+\d{1,2},\s+\d{4})", combined)
    if m:
        entities["effective_date"] = m.group(3)
    # Parties
    m = re.search(r"This (Agreement|Contract|Lease) (is made|made and entered) (?:on[^\n]{0,60}? between|between)?\s*(.+?)\s+(and|&)\s+(.+?)\.", combined, re.IGNORECASE)
    if m:
        entities["parties"] = f"{m.group(4).strip()} & {m.group(6).strip()}" if len(m.groups()) >= 6 else m.group(0)
    # Governing law
    m = re.search(r"governed by the laws? of ([A-Z][A-Za-z &]+)", combined, re.IGNORECASE)
    if m:
        entities["governing_law"] = m.group(1).strip()
    # Term length (simple)
    m = re.search(r"(initial term|term of this (agreement|contract))[^\n]{0,100}? (\d+\s+(months?|years?))", combined, re.IGNORECASE)
    if m:
        entities["term_length"] = m.group(3)
    return entities


def answer_with_confidence(answer: str) -> Dict[str, Any]:
    # Simple heuristic confidence: shorter answers with decisive language ranked higher
    lower = answer.lower()
    score = 0.5
    if any(w in lower for w in ["cannot", "not provided", "unsure", "uncertain"]):
        score -= 0.2
    if len(answer) < 400:
        score += 0.1
    if any(w in lower for w in ["must", "shall", "requires"]):
        score += 0.1
    return {"answer": answer, "confidence": round(max(0.05, min(0.95, score)), 2)}


def compute_similarity_confidence(similarities: List[float], answer_text: str) -> float:
    """Calculate confidence from similarity scores and answer length.
    similarities: raw cosine scores (assumed in [-1,1])
    """
    if not similarities:
        return 0.1
    norm = [ (s + 1)/2 for s in similarities ]  # map to [0,1]
    top3 = norm[:3]
    base = sum(top3)/len(top3)
    length_factor = min(1.0, len(answer_text)/180)
    confidence = base * 0.7 + length_factor * 0.3
    return round(max(0.05, min(0.95, confidence)), 2)


def compute_risk_index(red_flags: List[Dict[str, Any]], total_chars: int) -> Dict[str, Any]:
    weights = {"high": 3, "medium": 2, "low": 1}
    raw = sum(weights.get(f.get("severity","low"),1) for f in red_flags)
    import math
    denom = max(1.0, math.log10(max(50, total_chars)/1000 + 1))
    scaled = min(100, raw/denom * 14)
    level = ("Low" if scaled < 26 else "Moderate" if scaled < 56 else "Elevated" if scaled < 76 else "High")
    return {"index": round(scaled), "level": level}


def refine_plain_language(summary: str) -> str:
    """Attempt to further simplify summary for non-legal audience.
    If model not available, fallback heuristic simplification.
    """
    if not _model:
        # Heuristic: shorten long sentences
        parts = summary.split('. ')
        trimmed = [p.strip() for p in parts if p.strip()][:12]
        return ' '.join(trimmed)
    prompt = f"Rewrite the following summary in plain language for a non-legal reader. Use short sentences, avoid jargon. Keep meaning.\n\nSUMMARY:\n{summary}"
    try:
        resp = _model.generate_content(prompt)
        return resp.text
    except Exception:
        return summary
