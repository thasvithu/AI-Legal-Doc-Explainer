from src.utils.config import AppConfig
from src.utils.types import Document
from src.ingest.chunker import chunk_documents
from src.embeddings.embeddings import get_embedding_model
from src.vectorstore.faiss_store import FaissStoreManager
from src.rag.qa_chain import build_qa_chain

class DummyLLM:
    def generate(self, prompt: str) -> str:
        return "Answer. Citations: p1"

def test_qa_chain_basic(monkeypatch):
    # create config with Gemini disabled by env override simulation
    config = AppConfig(use_gemini=False)
    docs = [Document(name="a.pdf", text="The agreement may be terminated early with 30 days notice.", pages=1)]
    chunks = chunk_documents(docs, chunk_size=60, chunk_overlap=0)
    embed = get_embedding_model(config)
    manager = FaissStoreManager(config)
    vs = manager.build_index(chunks, embed, force_rebuild=True)
    chain = build_qa_chain(config, vs)
    # monkeypatch the llm
    chain.llm = DummyLLM()
    out = chain.ask("Can I terminate early?")
    assert 'answer' in out and out['citations']
