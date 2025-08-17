from __future__ import annotations
import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document as LCDocument
from langchain.embeddings.base import Embeddings
from src.utils.config import AppConfig
from src.utils.types import Chunk

class FaissStoreManager:
    def __init__(self, config: AppConfig):
        self.config = config
        os.makedirs(config.workspace_dir, exist_ok=True)
        self.index_path = os.path.join(config.workspace_dir, "faiss_index")

    def build_index(self, chunks: List[Chunk], embed: Embeddings, force_rebuild: bool = False):
        if os.path.exists(self.index_path) and not force_rebuild:
            try:
                return FAISS.load_local(self.index_path, embed, allow_dangerous_deserialization=True)
            except Exception:
                pass
        docs = [LCDocument(page_content=c.content, metadata={"chunk_id": c.id, "doc": c.document_name, "page": c.page}) for c in chunks]
        vs = FAISS.from_documents(docs, embed)
        vs.save_local(self.index_path)
        return vs
