from __future__ import annotations
from typing import List
import io
import re
from src.utils.types import Document

try:  # primary fast lib
    from pypdf import PdfReader  # type: ignore
    _PDF_IMPL = 'pypdf'
except Exception:  # pragma: no cover
    try:
        from PyPDF2 import PdfReader  # type: ignore
        _PDF_IMPL = 'PyPDF2'
    except Exception:  # pragma: no cover
        PdfReader = None  # type: ignore
        _PDF_IMPL = 'none'

WHITESPACE_RE = re.compile(r"\s+")

def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = WHITESPACE_RE.sub(" ", text)
    return text.strip()

def load_pdfs(uploaded_files) -> List[Document]:
    """Load PDFs with resilient text extraction.

    Strategy:
      1. Try standard extract_text per page.
      2. If page yields little/no text but has many characters in raw / or looks scanned, mark for optional OCR (placeholder).
      3. Concatenate cleaned text. Store per-page count for downstream heuristics.
    """
    documents: List[Document] = []
    for f in uploaded_files:
        data = f.read()
        if not PdfReader:  # hard failure fallback
            documents.append(Document(name=f.name, text="", pages=0))
            continue
        try:
            reader = PdfReader(io.BytesIO(data))
        except Exception:
            documents.append(Document(name=f.name, text="", pages=0))
            continue
        pages_text: List[str] = []
        for idx, page in enumerate(reader.pages):
            txt = ""
            try:
                txt = page.extract_text() or ""
            except Exception:
                txt = ""
            # Heuristic: if extremely short, maybe layout issue => try layout-based merge of extracted content objects
            if len(txt.strip()) < 15:
                # Attempt access to raw / objects (best-effort; if fails, ignore)
                try:  # pragma: no cover - depends on underlying parser
                    raw = " ".join(str(o) for o in page.extract_text().split("\n")) if hasattr(page, 'extract_text') else ''
                    if len(raw) > len(txt):
                        txt = raw
                except Exception:
                    pass
            pages_text.append(clean_text(txt))
    combined = "\n".join(p for p in pages_text if p)
    documents.append(Document(name=f.name, text=combined, pages=len(pages_text), pages_text=pages_text))
    return documents
