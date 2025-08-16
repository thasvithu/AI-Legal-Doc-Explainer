import os
from pathlib import Path
from modules.retriever import create_retriever, load_faiss_index
from dotenv import load_dotenv
from utils.exception import CustomException
from modules.embed_store import get_index_dir
from utils.logger import logger

load_dotenv()

# Gemini optional; allow offline mode.
_GEMINI_AVAILABLE = False
model = None
try:
    import google.generativeai as genai  # type: ignore
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            _GEMINI_AVAILABLE = True
        except Exception as e:
            logger.warning("Gemini init failed: %s", e)
except Exception as e:
    logger.debug("Gemini library not available: %s", e)

def answer_query_with_retriever(query, faiss_index_path: str | None = None, k=5):
    """Retrieve relevant context and answer.

    Returns dict with keys: answer, docs, similarities.
    """
    # Resolve path: if not provided use standard project index dir
    if faiss_index_path is None:
        faiss_index_path = str(get_index_dir())
    # Load FAISS vector store
    faiss_db = load_faiss_index(faiss_index_path)
    if not faiss_db:
        logger.warning("Could not load FAISS index at %s", faiss_index_path)
        return {"answer": "Error: Could not load FAISS index.", "docs": [], "similarities": []}

    # Create retriever
    retriever = create_retriever(faiss_db, k=k)
    
    # Retrieve relevant docs (invoke avoids deprecation of get_relevant_documents)
    relevant_docs = retriever.invoke(query)
    # Attempt to approximate similarities if store supports similarity_search_with_score
    similarities = []
    try:
        sim_results = faiss_db.similarity_search_with_score(query, k=k)
        # sim_results: List[ (Document, score) ] where score lower distance or higher similarity depending backend
        # Convert distances to similarity heuristic (invert if positive distance)
        for doc, score in sim_results:
            # If score > 1 assume distance; transform via 1/(1+score)
            sim = 1/(1+score) if score > 1 else score
            similarities.append(sim)
    except Exception as e:
        logger.debug("similarity_search_with_score failed: %s", e)
        similarities = [0.5]*len(relevant_docs)

    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    if _GEMINI_AVAILABLE and model is not None:
        prompt = f"""You are a legal document assistant. Answer the question using ONLY the context. If absent, say you cannot find it.\n\nContext:\n{context}\n\nQuestion: {query}\n"""
        try:
            response = model.generate_content(prompt)
            return {"answer": response.text, "docs": relevant_docs, "similarities": similarities}
        except Exception as e:
            logger.warning("Gemini generation failed: %s", e)

    # Fallback: extractive snippet
    q_terms = [t for t in query.lower().split() if len(t) > 2]
    chosen = None
    for d in relevant_docs:
        low = d.page_content.lower()
        if any(t in low for t in q_terms):
            chosen = d.page_content[:600] + ("..." if len(d.page_content) > 600 else "")
            break
    if not chosen and relevant_docs:
        chosen = relevant_docs[0].page_content[:600] + ("..." if len(relevant_docs[0].page_content) > 600 else "")
    answer_text = chosen or "I cannot find relevant information in the document."
    return {"answer": answer_text + "\n\n(Offline heuristic answer)", "docs": relevant_docs, "similarities": similarities}