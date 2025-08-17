from __future__ import annotations
from typing import List
from src.utils.types import ClauseResult, RedFlagResult
from src.utils.config import AppConfig
from src.llm.gemini import GeminiClient
from src.llm.fallback import LocalLLM
import re

REDFLAG_PROMPT_PATH = "src/prompts/redflags.txt"
with open(REDFLAG_PROMPT_PATH, "r", encoding="utf-8") as f:
    REDFLAG_TEMPLATE = f.read()

RISK_KEYWORDS = [
    (re.compile(r"sole discretion", re.I), 15, "Unilateral discretion"),
    (re.compile(r"indemnif", re.I), 20, "Broad indemnity"),
    (re.compile(r"automatic renewal|auto-renew", re.I), 10, "Auto-renewal"),
    (re.compile(r"liquidated damages", re.I), 15, "Penalties"),
]

def _get_llm(config: AppConfig):
    if config.use_gemini:
        try:
            return GeminiClient(config)
        except Exception:
            return LocalLLM(config)
    return LocalLLM(config)

LINE_RE = re.compile(r"^RISK:(.*?)\|REASON:(.*?)\|SNIPPET:(.*?)\|PAGE:(\d+)\|SCORE:(\d+)$")


def detect_redflags(config: AppConfig, clauses: List[ClauseResult]) -> List[RedFlagResult]:
    """Detect red flags; if only stub LLM available, use heuristic scoring without prompt round-trip."""
    llm = _get_llm(config)
    is_stub = isinstance(llm, LocalLLM) and getattr(llm, 'pipe', None) is None
    results: List[RedFlagResult] = []

    if is_stub:  # pure heuristic mode
        for c in clauses:
            base_score = 30
            reasons = []
            for pattern, add, reason in RISK_KEYWORDS:
                if pattern.search(c.snippet):
                    base_score += add
                    reasons.append(reason)
            if base_score < config.confidence_threshold:
                continue
            reason_text = "; ".join(reasons) if reasons else f"Potential {c.clause_type.lower()} exposure"
            results.append(RedFlagResult(
                risk_type=c.clause_type,
                reason=reason_text[:300],
                snippet=c.snippet[:400],
                page=c.page,
                confidence=float(min(base_score, 95)),
            ))
    else:
        for batch_start in range(0, len(clauses), 12):
            batch = clauses[batch_start: batch_start + 12]
            heuristic_lines = []
            for c in batch:
                base_score = 30
                for pattern, add, reason in RISK_KEYWORDS:
                    if pattern.search(c.snippet):
                        base_score += add
                heuristic_lines.append(f"CLAUSE:{c.clause_type}|SNIPPET:{c.snippet}|PAGE:{c.page}|BASE:{base_score}")
            prompt = REDFLAG_TEMPLATE.format(clauses="\n".join(heuristic_lines))
            raw = llm.generate(prompt)
            for line in raw.splitlines():
                m = LINE_RE.match(line.strip())
                if not m:
                    continue
                risk_type, reason, snippet, page, score = m.groups()
                results.append(RedFlagResult(
                    risk_type=risk_type.strip(),
                    reason=reason.strip()[:300],
                    snippet=snippet.strip()[:400],
                    page=int(page),
                    confidence=float(score),
                ))

    # Normalize raw confidence (cap 100)
    for r in results:
        if r.confidence > 100:
            r.confidence = 100.0
        if r.confidence < 0:
            r.confidence = 0.0
    filtered = [r for r in results if r.confidence >= config.confidence_threshold]

    # Adaptive fallback: if nothing surfaced, run a broader heuristic scan directly on clauses
    if not filtered and clauses:
        broadened: List[RedFlagResult] = []
        have_liability = any(c.clause_type.lower().startswith('liability') for c in clauses)
        have_indemnity = any(c.clause_type.lower().startswith('indemn') for c in clauses)
        patterns = [
            (re.compile(r"sole discretion.*terminate|terminate.*sole discretion", re.I), "Unilateral termination right", 65),
            (re.compile(r"auto[- ]?renew", re.I), "Automatic renewal (check opt-out window)", 60),
            (re.compile(r"indemnif.*any and all|indemnif.*all claims", re.I), "Broad indemnity scope", 70),
            (re.compile(r"unlimited liability|without (any )?limit", re.I), "Potential unlimited liability", 72),
            (re.compile(r"liquidated damages", re.I), "Liquidated damages / penalty", 68),
            (re.compile(r"use .*data for any purpose", re.I), "Broad data usage rights", 62),
        ]
        for c in clauses:
            for pat, desc, base in patterns:
                if pat.search(c.snippet):
                    broadened.append(RedFlagResult(
                        risk_type=desc,
                        reason=f"Detected pattern in {c.clause_type} clause.",
                        snippet=c.snippet[:400],
                        page=c.page,
                        confidence=float(min(100, base)),
                    ))
                    break
        if have_indemnity and not have_liability:
            ind_clause = next((c for c in clauses if c.clause_type.lower().startswith('indemn')), clauses[0])
            broadened.append(RedFlagResult(
                risk_type="No explicit liability cap located",
                reason="Indemnity clause present but no separate liability limitation clause detected.",
                snippet=ind_clause.snippet[:400],
                page=ind_clause.page,
                confidence=67.0,
            ))
        seen = set()
        dedup = []
        for r in broadened:
            key = (r.risk_type, r.page)
            if key in seen:
                continue
            seen.add(key)
            dedup.append(r)
        filtered = dedup if dedup else filtered
    return filtered
