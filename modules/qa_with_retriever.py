import os
import google.generativeai as genai
from modules.retriever import create_retriever, load_faiss_index
from dotenv import load_dotenv
from utils.exception import CustomException

# Load environment variables
load_dotenv()

# Configure Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

def answer_query_with_retriever(query, faiss_index_path="../faiss_legal_index", k=5):
    """
    Retrieve relevant context from FAISS.
    """
    # Load FAISS vector store
    faiss_db = load_faiss_index(faiss_index_path)
    if not faiss_db:
        return "Error: Could not load FAISS index."

    # Create retriever
    retriever = create_retriever(faiss_db, k=k)
    
    # Retrieve relevant docs
    relevant_docs = retriever.invoke(query)
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
    return response.text