"""Lightweight health check utilities for the AI Legal Doc Explainer.

Avoid heavy model downloads: we intentionally DO NOT instantiate large HF
pipelines here. The goal is a fast readiness signal for CI / demo scripts.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any, List


@dataclass
class HealthStatus:
    component: str
    ok: bool
    detail: str

    def as_dict(self) -> Dict[str, Any]:
        return {"component": self.component, "ok": self.ok, "detail": self.detail}


def _check_import(module: str) -> HealthStatus:
    try:
        __import__(module)
        return HealthStatus(module, True, "import ok")
    except Exception as e:  # pragma: no cover - diagnostic path
        return HealthStatus(module, False, f"import failed: {e}")


CORE_IMPORTS = [
    "streamlit",
    "langchain",
    "langchain_community.vectorstores.faiss",
    "faiss",
    "pypdf",
    "reportlab.pdfgen",
]


def run_health_check(light: bool = True) -> Dict[str, Any]:
    """Run a series of lightweight checks.

    light=True skips any network / model downloads.
    """
    results: List[HealthStatus] = []
    for mod in CORE_IMPORTS:
        results.append(_check_import(mod))

    # Minimal FAISS sanity (vector add + search) without embeddings pipeline
    faiss_ok = False
    faiss_detail = ""
    try:
        from langchain_community.vectorstores import FAISS
        from langchain.schema import Document

        class DummyEmb:
            def embed_documents(self, texts):  # noqa: D401
                return [[0.1] * 8 for _ in texts]
            def embed_query(self, text):
                return [0.1] * 8

        docs = [Document(page_content="Sample contract clause about termination and liability.")]
        vs = FAISS.from_documents(docs, DummyEmb())
        _ = vs.similarity_search("termination", k=1)
        faiss_ok = True
        faiss_detail = "faiss mini index ok"
    except Exception as e:  # pragma: no cover - rare path
        faiss_detail = f"faiss test failed: {e}"
    results.append(HealthStatus("faiss-mini", faiss_ok, faiss_detail))

    aggregate = all(r.ok for r in results)
    return {
        "ok": aggregate,
        "components": [r.as_dict() for r in results],
    }


if __name__ == "__main__":  # Manual invocation helper
    import json, sys
    report = run_health_check()
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["ok"] else 1)
