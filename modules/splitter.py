from langchain.text_splitter import RecursiveCharacterTextSplitter
from utils.exception import CustomException

def split_pdf_text(docs, chunk_size=800, chunk_overlap=100):
    """
    Split a list of LangChain Document objects into smaller text chunks for embeddings.
    
    Args:
        docs (List[Document]): List of Document objects from a PDF.
        chunk_size (int): Maximum characters per chunk.
        chunk_overlap (int): Number of overlapping characters between chunks.
    
    Returns:
        List[Document]: List of Document objects representing text chunks.
                        Returns an empty list if an error occurs.
    """
    try:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", " "]
        )
        text_chunks = text_splitter.split_documents(docs)
        return text_chunks
    except Exception as e:
        print(CustomException(str(e)))
        return []
