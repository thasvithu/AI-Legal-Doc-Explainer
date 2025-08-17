from __future__ import annotations
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.types import Document, Chunk
import hashlib

def chunk_documents(documents: List[Document], chunk_size: int = 1100, chunk_overlap: int = 150) -> List[Chunk]:
    """Chunk documents while attempting to preserve original PDF page numbers.

    We insert a marker "\n---PAGE_BREAK---\n" between pages during PDF load (future enhancement) or infer
    page splits by splitting on form-feed / large newline groups. For now, we assume pages are separated by '\n'.
    Each produced chunk is assigned the page number of the first originating page segment it overlaps.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks: List[Chunk] = []
    for doc in documents:
        if getattr(doc, 'pages_text', None):
            segments = [(i+1, t) for i, t in enumerate(doc.pages_text)]
            page_map = []
            builder = []
            for page_no, seg in segments:
                builder.append(seg)
                page_map.extend([page_no]*len(seg))
            full = ''.join(builder)
        else:  # fallback approximate split
            approx_pages = max(doc.pages, 1)
            per_len = max(len(doc.text)//approx_pages, 1)
            page_boundaries = [(i*per_len) for i in range(approx_pages)] + [len(doc.text)]
            segments = []
            for i in range(approx_pages):
                seg = doc.text[page_boundaries[i]:page_boundaries[i+1]]
                segments.append((i+1, seg))
            page_map = []
            builder = []
            for page_no, seg in segments:
                builder.append(seg)
                page_map.extend([page_no]*len(seg))
            full = ''.join(builder)
        doc_splits = splitter.split_text(full)
        cursor = 0
        for i, text in enumerate(doc_splits):
            digest = hashlib.sha1(f"{doc.name}-{i}".encode()).hexdigest()[:12]
            # Determine page: majority vote of first 200 chars indices
            sample_range = range(cursor, min(cursor+len(text), cursor+200, len(page_map)))
            pages = {}
            for idx in sample_range:
                p = page_map[idx] if idx < len(page_map) else 1
                pages[p] = pages.get(p,0)+1
            page_assigned = sorted(pages.items(), key=lambda x: (-x[1], x[0]))[0][0] if pages else 1
            chunks.append(Chunk(id=digest, document_name=doc.name, page=page_assigned, content=text))
            cursor += len(text)
    return chunks
