# AI Legal Document Explainer (Round 1 Prototype)

Goal: Upload a legal PDF (contract / lease) and get:
- Plain-language summary
- Highlighted clauses & basic risk flags (prototype)
- Simple clause-level retrieval-powered Q&A

## Quick Start

1. Create virtual environment
2. Install dependencies `pip install -r requirements.txt`
3. Run Streamlit UI: `streamlit run app/app_streamlit.py`

## Current Architecture (Prototype)
```
PDF -> Text Extraction (pdfplumber) -> Clause Split (regex heuristic) -> Embeddings (sentence-transformers) -> Vector Store (Chroma) -> Simple Explain (placeholder) -> Q&A (nearest clause)
```

## Next Enhancements
- Replace placeholder explanation with LLM reasoning + structured outputs
- Add obligations extraction & suggestion generation
- Add citation-based verification & confidence scoring
- Improve clause splitter (heading detection, bullet handling)
- Reranking & hybrid retrieval

## Testing
`pytest -q`

## Disclaimer
Prototype stage; not legal advice.
