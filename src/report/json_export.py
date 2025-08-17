from __future__ import annotations
import json
from typing import Any, Dict, List
from src.utils.types import Document, ClauseResult, RedFlagResult, QAHistory

def build_analysis_json(
    docs: List[Document],
    summaries: Dict[str, Dict[str, str]],
    clauses: List[ClauseResult],
    redflags: List[RedFlagResult],
    qa_history: QAHistory,
    meta: Dict[str, Any],
) -> str:
    """Return a structured JSON snapshot of the analysis suitable for downstream evaluation.

    meta can include build/version timestamps, model info, etc.
    """
    payload = {
        "meta": meta,
        "documents": [
            {"name": d.name, "pages": d.pages} for d in docs
        ],
        "summaries": summaries,
        "clauses": [
            {
                "type": c.clause_type,
                "importance": c.importance,
                "page": c.page,
                "explanation": c.explanation,
                "snippet": c.snippet,
            } for c in clauses
        ],
        "red_flags": [
            {
                "risk_type": r.risk_type,
                "confidence": r.confidence,
                "page": r.page,
                "reason": r.reason,
                "snippet": r.snippet,
            } for r in redflags
        ],
        "qa_history": qa_history,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
