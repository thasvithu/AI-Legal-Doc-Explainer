from __future__ import annotations
from typing import List
from src.utils.types import Chunk, ClauseResult
from src.utils.config import AppConfig
from src.llm.gemini import GeminiClient
from src.llm.fallback import LocalLLM
import re

CLAUSE_PROMPT_PATH = "src/prompts/clauses.txt"
with open(CLAUSE_PROMPT_PATH, "r", encoding="utf-8") as f:
    CLAUSE_TEMPLATE = f.read()

TARGET_CLAUSES = [
    "Term/Duration", "Termination", "Payment", "Late fees/penalties", "Confidentiality", "IP ownership", "Liability", "Indemnity", "Arbitration/Jurisdiction", "Auto-renewal", "Unusual obligations"
]

IMPORTANCE_RULES = {
    "Indemnity": "High",
    "Liability": "High",
    "Auto-renewal": "Medium",
    "Late fees/penalties": "Medium",
}

def _get_llm(config: AppConfig):
    if config.use_gemini:
        try:
            return GeminiClient(config)
        except Exception:
            return LocalLLM(config)
    return LocalLLM(config)

CLAUSE_LINE_RE = re.compile(r"^CLAUSE:(.*?)\|EXPLANATION:(.*?)\|SNIPPET:(.*?)\|PAGE:(\d+)$")


def extract_clauses(config: AppConfig, chunks: List[Chunk]) -> List[ClauseResult]:
    """Extract clauses using LLM template format; if only stub fallback available, use heuristic regex/keyword scanning.

    Heuristic mode triggers when LocalLLM has no underlying transformers pipeline (offline / deps missing).
    """
    llm = _get_llm(config)
    is_stub = isinstance(llm, LocalLLM) and getattr(llm, 'pipe', None) is None
    results: List[ClauseResult] = []

    if is_stub:  # heuristic extraction (improved scoring)
        import re, math, hashlib
        # Define weighted keyword sets per target clause
        KEYWORDS = {
            "Term/Duration": ["term", "duration", "renew", "renewal", "expiration", "expiry", "initial subscription term", "renewal period"],
            "Termination": ["terminate", "termination", "expire", "early termination", "notice period"],
            "Payment": ["payment", "fee", "fees", "charge", "charges", "invoice", "billing", "payable"],
            "Late fees/penalties": [("late", "fee"), ("late", "payment"), ("overdue", "interest"), "penalt", "liquidated damages"],
            "Confidentiality": ["confidential", "non-disclosure", "confidential information"],
            "IP ownership": ["intellectual property", "ip rights", "ownership", "retain ownership", "license", "licence"],
            "Liability": ["liability", "liable", "limitation of liability", "limit liability", "liability cap"],
            "Indemnity": ["indemnify", "indemnification", "hold harmless"],
            "Arbitration/Jurisdiction": ["jurisdiction", "governing law", "arbitration", "venue", "court"],
            "Auto-renewal": ["auto-renew", "automatic renewal"],
            "Unusual obligations": ["sole discretion", "audit rights", "beta services", "unlimited liability", "exclusive remedy"],
        }
        # Precompile simple patterns
        SIMPLE_PATTERNS = {k: [re.compile(re.escape(kw), re.I) if isinstance(kw, str) else kw for kw in v] for k, v in KEYWORDS.items()}

        def score_sentence(clause_type: str, sent: str) -> int:
            low = sent.lower()
            s = 0
            for kw in SIMPLE_PATTERNS[clause_type]:
                if isinstance(kw, re.Pattern):
                    if kw.search(low):
                        s += 2  # direct hit
                else:  # tuple requirement (all terms present)
                    if all(t in low for t in kw):
                        s += 3
            return s

        # Track best sentences per (clause_type, page)
        best: dict[tuple[str, int], list[tuple[int, str]]] = {}
        snippet_seen = set()
        for ch in chunks:
            # Skip binary-like garbage fragments
            if ch.content.count("\uFFFD") > 5:  # many replacement chars
                continue
            # Split sentences; also split on newlines that look like headings
            raw = re.split(r'(?<=[.!?])\s+|\n{1,2}', ch.content)
            for sent in raw:
                s_clean = sent.strip()
                if not (30 <= len(s_clean) <= 450):
                    continue
                # Avoid definitional noise ("X: means") appearing repeatedly
                if re.match(r'^[A-Z][A-Za-z0-9\s]{0,40}:\s*(means|the)', s_clean):
                    continue
                for clause_type in TARGET_CLAUSES:
                    sc = score_sentence(clause_type, s_clean)
                    if sc < 3:  # threshold
                        continue
                    key = (clause_type, ch.page)
                    best.setdefault(key, []).append((sc, s_clean))
    # Reduce to top 2 per clause/page
        for (clause_type, page), lst in best.items():
            lst.sort(key=lambda x: (-x[0], len(x[1])))
            take = lst[:2]
            for sc, sent in take:
                snippet = sent[:350]
                h = hashlib.md5((clause_type + str(page) + snippet.lower()).encode()).hexdigest()
                if h in snippet_seen:
                    continue
                snippet_seen.add(h)
                explanation = snippet.split('. ')[0][:180]
                importance = IMPORTANCE_RULES.get(clause_type, 'Low')
                # Promote importance if high score and in critical types
                if clause_type in ("Indemnity", "Liability") and sc >= 5:
                    importance = "High"
                elif sc >= 6 and importance == 'Low':
                    importance = 'Medium'
                results.append(ClauseResult(
                    clause_type=clause_type,
                    explanation=explanation,
                    snippet=snippet,
                    page=page,
                    importance=importance,
                ))
        # Relaxed fallback if nothing found
        if not results:  # relaxed secondary pass
            for ch in chunks:
                raw_sents = re.split(r'(?<=[.!?])\s+|\n{1,2}', ch.content)
                for sent in raw_sents:
                    s_clean = sent.strip()
                    if not (20 <= len(s_clean) <= 500):
                        continue
                    low = s_clean.lower()
                    for clause_type, kws in KEYWORDS.items():
                        hits = 0
                        for kw in kws:
                            if isinstance(kw, tuple):
                                if all(t in low for t in kw):
                                    hits += 1
                            elif isinstance(kw, str):
                                if kw in low:
                                    hits += 1
                        if hits >= 1:
                            explanation = s_clean.split('. ')[0][:160]
                            results.append(ClauseResult(
                                clause_type=clause_type,
                                explanation=explanation,
                                snippet=s_clean[:350],
                                page=ch.page,
                                importance=IMPORTANCE_RULES.get(clause_type,'Low')
                            ))
                            break
    else:  # LLM-driven extraction
        for batch_start in range(0, len(chunks), 10):
            batch = chunks[batch_start: batch_start + 10]
            prompt = CLAUSE_TEMPLATE.format(text="\n\n".join(c.content for c in batch), target_clauses=", ".join(TARGET_CLAUSES))
            raw = llm.generate(prompt)
            for line in raw.splitlines():
                m = CLAUSE_LINE_RE.match(line.strip())
                if not m:
                    continue
                clause_type, explanation, snippet, page = m.groups()
                importance = IMPORTANCE_RULES.get(clause_type.strip(), "Low")
                results.append(ClauseResult(
                    clause_type=clause_type.strip(),
                    explanation=explanation.strip()[:400],
                    snippet=snippet.strip()[:400],
                    page=int(page),
                    importance=importance,
                ))

    # deduplicate
    seen = set()
    deduped: List[ClauseResult] = []
    for r in results:
        key = (r.clause_type, r.page, r.snippet[:60])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    # Merge near-identical explanations per clause_type & page (retain shortest)
    merged: dict[tuple[str,int], ClauseResult] = {}
    for r in deduped:
        k = (r.clause_type, r.page)
        cur = merged.get(k)
        if not cur:
            merged[k] = r
            continue
        # If new explanation shorter and not subset of existing, keep shorter
        if len(r.explanation) < len(cur.explanation) and r.explanation.lower() not in cur.explanation.lower():
            merged[k] = r
    final_list = list(merged.values())
    # Sort by importance then page
    ord_map = {"High":0, "Medium":1, "Low":2}
    final_list.sort(key=lambda x: (ord_map.get(x.importance, 3), x.page, x.clause_type))
    return final_list
