import warnings
try:
    # Preferred (no deprecation warning if package installed)
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except ImportError:  # pragma: no cover
    # Fallback for environments where migration not yet done
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
    warnings.filterwarnings(
        "ignore",
        message=r"The class `HuggingFaceEmbeddings` was deprecated",
        category=Warning,
    )
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from typing import List
from utils.exception import CustomException
import os


def embed_and_store_documents(chunks: List[Document], model_name: str = "BAAI/bge-small-en-v1.5"):
    """
    Generate embeddings for text chunks and store them in a FAISS vector store.

    Args:
        chunks (List[Document]): List of Document objects (text chunks).
        model_name (str): HuggingFace embedding model to use.

    Returns:
        FAISS: FAISS vector store object.
    """
    try:
        # Initialize embedding model
        embeddings_model = HuggingFaceEmbeddings(model_name=model_name)

        # Create FAISS vector store from documents
        db = FAISS.from_documents(chunks, embeddings_model)

        # Save FAISS index locally
        save_path = os.path.join(os.getcwd(), "faiss_legal_index")
        db.save_local(save_path)
        print("FAISS index saved successfully!")

        return db

    except Exception as e:
        print(CustomException(str(e)))
        return None

