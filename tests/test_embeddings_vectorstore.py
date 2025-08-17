from src.utils.types import Document
from src.ingest.chunker import chunk_documents
from src.embeddings.embeddings import get_embedding_model
from src.vectorstore.faiss_store import FaissStoreManager
from src.utils.config import AppConfig

def test_vectorstore_roundtrip():
    config = AppConfig()
    docs = [Document(name="a.pdf", text="Payment shall be made within 30 days. Term is one year.", pages=1)]
    chunks = chunk_documents(docs, chunk_size=50, chunk_overlap=0)
    embed = get_embedding_model(config)
    manager = FaissStoreManager(config)
    vs = manager.build_index(chunks, embed, force_rebuild=True)
    retriever = vs.as_retriever(search_kwargs={"k":1})
    docs = retriever.get_relevant_documents("payment term")
    assert docs
