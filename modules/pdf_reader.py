from langchain_community.document_loaders import PyMuPDFLoader
from langchain.schema import Document
from typing import List, Union, IO
from utils.exception import CustomException
import tempfile
import os

def get_pdf_text(source: Union[str, "UploadedFile", IO[bytes]]) -> List[Document]: 
    """Load a PDF (path or Streamlit UploadedFile) and return list of Documents.

    Args:
        source (str | UploadedFile | IO[bytes]): Path, uploaded file, or file-like object.

    Returns:
        List[Document]: Parsed documents from the PDF.
    """
    tmp_path = None
    try:
        # String path
        if isinstance(source, str):
            pdf_path = source

        # Streamlit UploadedFile
        elif hasattr(source, "read") and hasattr(source, "name"):
            suffix = os.path.splitext(source.name)[1] or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(source.getbuffer())
                tmp_path = tmp.name
            pdf_path = tmp_path

        # Generic file-like
        elif hasattr(source, "read"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(source.read())
                tmp_path = tmp.name
            pdf_path = tmp_path

        else:
            raise ValueError("Unsupported source type for PDF loading")

        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"File path {pdf_path} does not exist")

        loader = PyMuPDFLoader(pdf_path)
        return loader.load()

    except Exception as e:
        print(CustomException(f"PDF loading failed: {e}"))
        return []

    finally:
        if tmp_path and os.path.isfile(tmp_path):
            os.remove(tmp_path)
