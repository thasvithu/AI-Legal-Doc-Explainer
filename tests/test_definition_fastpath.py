from src.rag.qa_chain import QAChain
from langchain_community.vectorstores import FAISS
from langchain.schema import Document


class TinyConfig:
    use_gemini = False
    # Prevent LocalLLM attribute errors if accidentally invoked
    temperature = 0.0
    local_llm_model = "distilgpt2"
    local_llm_small = True


class StubLLM:
    pipe = None
    def generate(self, prompt: str) -> str:  # pragma: no cover
        return "(stub)"


class DummyEmb:
    def embed_documents(self, texts):
        return [[0.1]*8 for _ in texts]
    def embed_query(self, text):
        return [0.1]*8


def test_definition_fast_path():
    docs = [
        Document(page_content="Software as a Service (SaaS) means a subscription based hosted software delivery model.", metadata={"page": 3}),
        Document(page_content="Other filler text about termination and liability.", metadata={"page": 4}),
    ]
    vs = FAISS.from_documents(docs, DummyEmb())
    chain = QAChain(TinyConfig(), vs, llm=StubLLM())
    out = chain.ask("What is SaaS?")
    assert "subscription" in out["answer"].lower()
    assert out["citations"], "Citations should be present"
