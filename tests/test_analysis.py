import pytest
from langchain.schema import Document
from modules.analysis import (
    compute_risk_index,
    bulletize_summary,
    extract_obligations,
    extract_key_clauses,
    compute_similarity_confidence,
)
from modules.constants import RISK_KEYWORDS


def make_doc(text: str, page: int = 0):
    return Document(page_content=text, metadata={"page": page})


def test_bulletize_summary_numbered():
    summary = "This contract sets payment terms. It defines parties. It sets termination rights. It includes confidentiality."  # noqa
    out = bulletize_summary(summary, max_items=4, numbered=True)
    lines = [l for l in out.splitlines() if l.strip()]
    assert len(lines) <= 4
    assert all(l[0].isdigit() for l in lines)


def test_extract_obligations_simple():
    text = "The Supplier shall deliver goods monthly. The Customer must pay within 30 days."
    docs = [make_doc(text)]
    obligations = extract_obligations(docs)
    parties = {o['party'] for o in obligations}
    assert 'Supplier' in parties or 'Customer' in parties


def test_compute_risk_index_scaling():
    # High severity items should push index upward but stay <=100
    red_flags = [
        {"severity": "high"}, {"severity": "high"}, {"severity": "medium"}, {"severity": "low"}
    ]
    res = compute_risk_index(red_flags, total_chars=5000)
    assert 0 <= res["index"] <= 100
    assert res["level"] in {"Low", "Moderate", "Elevated", "High"}


def test_extract_key_clauses_dedup():
    kw = list(RISK_KEYWORDS.keys())[0]
    text = f"This clause covers {kw}. Another {kw} appears later. {kw} again."  # multiple occurrences
    docs = [make_doc(text, page=1)]
    clauses = extract_key_clauses(docs)
    # Should only have one entry for that keyword+page
    assert len([c for c in clauses if c['keyword'] == kw]) <= 1


def test_similarity_confidence_empty():
    conf = compute_similarity_confidence([], "answer")
    assert conf == 0.1
