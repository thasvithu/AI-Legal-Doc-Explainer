from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from typing import List
from utils.exception import CustomException

def get_pdf_text(path: str) -> List[Document]:
    """
    Load a PDF file and return its contents as a list of LangChain Document objects.

    Each Document object contains:
        - page_content: the text content of the page or chunk
        - metadata: dictionary with information such as page number and source file

    Args:
        path (str): Path to the PDF file to be loaded.

    Returns:
        List[Document]: A list of Document objects representing the PDF content.
                        Returns an empty list if an error occurs.
    """
    try:
        loader = PyMuPDFLoader(path)
        docs = loader.load()
        return docs
    except Exception as e:
        print(CustomException(str(e)))
        return []
