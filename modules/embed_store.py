import warnings
try:
    # Preferred (no deprecation warning if package installed)
    from langchain_huggingface import HuggingFaceEmbeddings  
except ImportError:  
    # Fallback for environments where migration not yet done
    from langchain_community.embeddings import HuggingFaceEmbeddings
    warnings.filterwarnings(
        "ignore",
        message=r"The class `HuggingFaceEmbeddings` was deprecated",
        category=Warning,
    )
from pathlib import Path
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List, Tuple, Optional, Callable
from utils.exception import CustomException
from utils.logger import logger
import os
import atexit
import shutil
from threading import Lock


def _project_root() -> Path:
    """Return project root (directory containing this file's parent)."""
    return Path(__file__).resolve().parent.parent


def get_index_dir(name: str = "faiss_legal_index") -> Path:
    """Return (and ensure) the persistent directory used to store FAISS index files."""
    idx_dir = _project_root() / name
    idx_dir.mkdir(parents=True, exist_ok=True)
    return idx_dir


def embed_and_store_documents(
    chunks: List[Document],
    model_name: str = "BAAI/bge-small-en-v1.5",
    index_dir_name: str = "faiss_legal_index",
    ephemeral: bool = False,
) -> Optional[Tuple[FAISS, Path, Optional[Callable[[], None]]]]:
    """Generate embeddings for text chunks and store them.

    Args:
        chunks: Documents to embed.
        model_name: HF embedding model name.
        index_dir_name: Folder name (ignored if ephemeral=True).
        ephemeral: If True, store index in a temporary directory that can be deleted after session.

    Returns:
        (db, path, cleanup_fn) or None. cleanup_fn deletes the directory when called (only set if ephemeral True).
    """
    try:
        if not chunks:
            raise ValueError("No chunks provided to embed.")

        embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
        db = FAISS.from_documents(chunks, embeddings_model)

        if ephemeral:
            import tempfile
            tmp_dir = Path(tempfile.mkdtemp(prefix="faiss_idx_"))
            db.save_local(str(tmp_dir))
            logger.debug("Created ephemeral FAISS index at %s", tmp_dir)

            # Track ephemeral directory globally for guaranteed cleanup on process exit
            _EphemeralIndexRegistry.register(tmp_dir)

            def _cleanup():  # closes over tmp_dir
                _EphemeralIndexRegistry.unregister_and_delete(tmp_dir)

            return db, tmp_dir, _cleanup
        else:
            index_dir = get_index_dir(index_dir_name)
            db.save_local(str(index_dir))
            logger.info("Saved FAISS index to %s", index_dir)
            return db, index_dir, None
    except Exception as e:
    logger.exception("Failed to build/save embeddings: %s", e)
    return None


# ----------------- Ephemeral registry -----------------
class _EphemeralIndexRegistry:
    _dirs: set[Path] = set()
    _lock = Lock()
    _atexit_registered = False

    @classmethod
    def register(cls, path: Path):
        with cls._lock:
            cls._dirs.add(path)
            if not cls._atexit_registered:
                atexit.register(cls.cleanup_all)
                cls._atexit_registered = True

    @classmethod
    def unregister_and_delete(cls, path: Path):
        with cls._lock:
            if path in cls._dirs:
                cls._dirs.remove(path)
        shutil.rmtree(path, ignore_errors=True)
    logger.debug("Deleted ephemeral index %s", path)

    @classmethod
    def cleanup_all(cls):  # pragma: no cover
        with cls._lock:
            dirs = list(cls._dirs)
            cls._dirs.clear()
        for d in dirs:
            shutil.rmtree(d, ignore_errors=True)
            logger.debug("Ephemeral index cleaned at exit: %s", d)

