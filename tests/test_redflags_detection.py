from src.analysis.redflags import detect_redflags
from src.utils.types import ClauseResult


class TinyConfig:
    confidence_threshold = 0  # allow all
    use_gemini = False
    local_llm_small = True
    local_llm_model = "distilgpt2"
    temperature = 0.0
    max_tokens = 64


def clause(snippet: str, ctype: str, page: int = 1):
    return ClauseResult(clause_type=ctype, explanation="", snippet=snippet, page=page, importance="High")


def test_redflags_pattern_detection():
    clauses = [
        clause("The Provider may in its sole discretion terminate this Agreement at any time.", "Termination"),
        clause("Customer shall indemnify for any and all claims arising out of use.", "Indemnity"),
    ]
    flags = detect_redflags(TinyConfig(), clauses)
    risk_types = {f.risk_type for f in flags}
    assert any("termination" in r.lower() or "unilateral" in r.lower() for r in risk_types)
    assert any("indemn" in r.lower() for r in risk_types)
