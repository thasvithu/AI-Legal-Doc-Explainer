from __future__ import annotations
from langchain_community.vectorstores import FAISS

def get_retriever(vs: FAISS, k: int = 5):
    return vs.as_retriever(search_kwargs={"k": k})
