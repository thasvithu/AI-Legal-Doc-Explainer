import warnings
from langchain_community.vectorstores import FAISS
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  
    from langchain_community.embeddings import HuggingFaceEmbeddings
    warnings.filterwarnings(
        "ignore",
        message=r"The class `HuggingFaceEmbeddings` was deprecated",
        category=Warning,
    )
from langchain.schema import Document
from langchain.vectorstores.base import VectorStoreRetriever
from utils.exception import CustomException
from utils.logger import logger

def load_faiss_index(path: str, model_name: str = "BAAI/bge-small-en-v1.5") -> FAISS:
    """Load a locally saved FAISS index (newer LangChain requires embeddings arg).

    Args:
        path: Directory containing `index.faiss` & `index.pkl`.
        model_name: HuggingFace embedding model (must match the one used to build the index).

    Returns:
        FAISS vector store or None on failure.
    """
    try:
        embeddings_model = HuggingFaceEmbeddings(model_name=model_name)
    db = FAISS.load_local(path, embeddings_model, allow_dangerous_deserialization=True)
    logger.debug("Loaded FAISS index from %s", path)
        return db
    except Exception as e:
    logger.error("FAISS load failed (%s): %s", path, e)
        return None


def create_retriever(faiss_db: FAISS, k: int = 5) -> VectorStoreRetriever:
    """
    Create a retriever object from FAISS for RAG queries.
    
    Args:
        faiss_db (FAISS): FAISS vector store object.
        k (int): Number of relevant chunks to retrieve.
    
    Returns:
        VectorStoreRetriever: Retriever object to use in QA chains.
    """
    try:
    return faiss_db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    except Exception as e:
    logger.error("Retriever creation failed: %s", e)
    return None