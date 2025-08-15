from modules.pdf_reader import get_pdf_text
from modules.splitter import split_pdf_text
from modules.embed_store import embed_and_store_documents
from modules.qa_with_retriever import answer_query_with_retriever   

from app import get_docs

docs = get_docs()

text_chunks = split_pdf_text(docs, chunk_size=800, chunk_overlap=100)

db = embed_and_store_documents(text_chunks)

query = "What is SaaS?"
result = answer_query_with_retriever(query)
print(result)
