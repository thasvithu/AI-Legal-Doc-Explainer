from __future__ import annotations
from __future__ import annotations
from src.utils.config import AppConfig
from functools import lru_cache
import os
import math
import hashlib
from langchain_core.embeddings import Embeddings

try:  # attempt light import; may fail on some Windows envs
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
    _HF_NEW = True
except Exception:  # pragma: no cover
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
        _HF_NEW = False
    except Exception:  # pragma: no cover
        HuggingFaceEmbeddings = None  # type: ignore


class HashingEmbedding(Embeddings):
    """Very lightweight fallback embedding (bag-of-hashed tokens) for emergencies.

    Produces deterministic vectors without external ML deps; supports FAISS similarity.
    Not semantic; only for degraded mode when HF models unavailable.
    """

    def __init__(self, dim: int = 384):
        self.dim = dim

    def _vectorize(self, text: str):
        vec = [0.0] * self.dim
        tokens = [t for t in text.lower().split() if t]
        if not tokens:
            return vec
        for tok in tokens:
            h = int(hashlib.sha1(tok.encode()).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0
        # l2 normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts):  # type: ignore[override]
        return [self._vectorize(t) for t in texts]

    def embed_query(self, text):  # type: ignore[override]
        return self._vectorize(text)


@lru_cache(maxsize=4)
def _load_embedding(model_name: str):  # pragma: no cover (cache wrapper)
    if os.getenv("DISABLE_HF_EMBED", "false").lower() == "true" or HuggingFaceEmbeddings is None:
        return HashingEmbedding()
    try:
        return HuggingFaceEmbeddings(model_name=model_name)
    except Exception:
        return HashingEmbedding()


def get_embedding_model(config: AppConfig):
    return _load_embedding(config.embed_model)
