import os
from pathlib import Path
import google.generativeai as genai
from modules.retriever import create_retriever, load_faiss_index
from dotenv import load_dotenv
from utils.exception import CustomException
from modules.embed_store import get_index_dir

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

def answer_query_with_retriever(query, faiss_index_path: str | None = None, k=5):
    """Retrieve relevant context and answer.

    Returns dict with keys: answer, docs, similarities (list[float]).
    """
    # Resolve path: if not provided use standard project index dir
    if faiss_index_path is None:
        faiss_index_path = str(get_index_dir())
    # Load FAISS vector store
    faiss_db = load_faiss_index(faiss_index_path)
    if not faiss_db:
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
    except Exception:
        similarities = [0.5]*len(relevant_docs)

    context = "\n\n".join(doc.page_content for doc in relevant_docs)

    # Build prompt
    prompt = f"""
    You are a legal document assistant.
    Answer the following question based on the context below.
    If the answer is not present, say "I cannot find relevant information in the document."

    Context:
    {context}

    Question:
    {query}
    """

    # Call Gemini
    response = model.generate_content(prompt)
    return {"answer": response.text, "docs": relevant_docs, "similarities": similarities}