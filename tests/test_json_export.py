from src.report.json_export import build_analysis_json


def test_json_export_structure():
    blob = build_analysis_json(
        docs=[], summaries={}, clauses=[], redflags=[], qa_history=[], meta={"app": "test"}
    )
    assert '"app": "test"' in blob
    assert '"clauses": []' in blob
