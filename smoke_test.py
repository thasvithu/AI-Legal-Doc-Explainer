"""Quick smoke test for core pipeline (no heavy model downloads).

Run with:  python smoke_test.py
"""
from __future__ import annotations
from src.rag.qa_chain import QAChain
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


class TinyConfig:
    use_gemini = False  # Force local stub path


class StubLLM:
    pipe = None  # signals heuristic stub in QAChain definition branch
    def generate(self, prompt: str) -> str:  # noqa: D401
        return "(stub)"


class DummyEmb:
    def embed_documents(self, texts):
        return [[0.05] * 12 for _ in texts]
    def embed_query(self, text):
        return [0.05] * 12


def build_stub_chain():
    docs = [
        Document(page_content="Software as a Service (SaaS) means a subscription model for hosted software.", metadata={"page": 1}),
        Document(page_content="The Provider may terminate this Agreement upon material breach.", metadata={"page": 2}),
    ]
    vs = FAISS.from_documents(docs, DummyEmb())
    chain = QAChain(TinyConfig(), vs)
    chain.llm = StubLLM()
    return chain


def main():
    chain = build_stub_chain()
    q = "What is SaaS?"
    ans = chain.ask(q)
    print("Question:", q)
    print("Answer:", ans.get("answer"))
    print("Citations:", ans.get("citations"))
    assert "subscription" in ans.get("answer", "").lower(), "Definition fast-path appears not to have triggered"
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
