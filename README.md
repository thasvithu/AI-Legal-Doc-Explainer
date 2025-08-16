---
title: AI Legal Doc Explainer
emoji: ⚖️
colorFrom: indigo
colorTo: gray
sdk: streamlit
app_file: app.py
license: mit
pinned: false
---

# AI Legal Document Explainer

This is a small, focused tool for reading a contract / agreement PDF and quickly answering: “What’s in here and where are the sharp edges?” It keeps things private (ephemeral in‑memory index), keeps the UI simple, and falls back gracefully if you don’t provide an LLM key.

## What You Get

* Plain language summary (also viewable as bullets or numbered points)
* Key clause snippets grouped by category with a simple Typical / Atypical hint
* Red flag list with severity and an overall risk index (0–100)
* Basic contract metadata (effective date, parties, governing law, term length) via lightweight heuristics
* Party obligations heuristic (modal verb scan; early prototype)
* Retrieval‑augmented Q&A with source citations and a confidence bar
* Downloadable JSON / text exports (full report, clauses, risks, summary)
* No persistent storage by default – temp FAISS index auto‑cleans after short inactivity

## How It Works (Short Version)

1. PDF is parsed with PyMuPDF into per‑page documents.  
2. Pages are split into overlapping chunks (configurable size/overlap).  
3. We embed chunks with a compact sentence‑transformer (bge-small).  
4. Chunks live in a FAISS index created in a temp folder.  
5. Summaries & clause/risk heuristics run over the raw pages.  
6. Q&A retrieves top chunks and (if a Gemini key is present) asks the model to answer using only those; otherwise a simple extractive fallback kicks in.  
7. Session inactivity (default ~2 min) triggers cleanup of the temp index.

## Risk Index (Why a Number?)

We just weight flags (High=3, Medium=2, Low=1), normalize by document size with a log factor, and scale to 0–100. Buckets: Low / Moderate / Elevated / High. It’s a heuristic, not a legal risk score – meant for quick triage only.

## Confidence Bar

Average the top few similarity scores + a bit of length sanity, blend them, clamp. If it’s low – or the risk level is elevated – the UI nudges you to double‑check with a professional. It’s intentionally conservative.

## Typical vs Atypical

Just a small keyword pattern list for now (e.g. “perpetual”, “unlimited liability”). Future version can swap this for an LLM classification with a rationale. Treat it as a hint, not a verdict.

## Project Layout

```
app.py                # Streamlit UI (tabs: Summary, Clauses, Risks, Q&A, Export, Obligations)
modules/
	pdf_reader.py       # PDF -> pages (PyMuPDF)
	splitter.py         # Recursive chunking
	embed_store.py      # Embeddings + FAISS (ephemeral/persistent)
	retriever.py        # Load + retriever factory
	qa_with_retriever.py# Retrieval + LLM / fallback answer
	analysis.py         # Summary, bullets, clauses, risks, entities, scoring
	session_manager.py  # Inactivity cleanup
utils/
	logger.py, exception.py
requirements.txt
runtime.txt
LICENSE
```

## Run It Locally

```bash
python -m venv .venv
./.venv/Scripts/activate  # Windows PowerShell
pip install -r requirements.txt
streamlit run app.py
```

Optional: create a `.env` file with:
```
GEMINI_API_KEY=your_key_here
```
Without it you still get summaries (heuristic) and Q&A (extractive fallback).

## Hugging Face Space / Demo

1. Create a Space (SDK = streamlit).  
2. Push this repo (include `runtime.txt` with `python-3.10`).  
3. Add a secret `GEMINI_API_KEY` for richer answers (optional).  
4. Open the Space, upload a sample PDF, test tabs, done.  

Live Space: (placeholder – add URL)

Demo Video: (placeholder – link to short walkthrough)

## Roadmap (Condensed)

- [x] Core ingestion, embeddings, RAG
- [x] Risk index + severity
- [x] Bulleted summary toggle
- [x] Citations + similarity-based confidence
- [ ] Consolidated PDF export
- [ ] LLM clause “standardness” with rationale
- [ ] Remediation / negotiation suggestions
- [ ] Test suite + CI
- [ ] Multi‑document comparison view

## Limitations & Notes

* Heuristics can miss or mis‑label edge cases. Always read the actual text.
* Similarity ≠ legal certainty – treat answers as directional pointers.
* Entity extraction is intentionally lightweight: no full NLP pipeline yet.
* No persistence by default; add a persistent path only if you handle privacy correctly.

## Contributing

Feel free to open a PR or issue. Improvements most welcome around retrieval quality, richer clause taxonomy, test coverage, or UI ergonomics.

## License

MIT (see `LICENSE`).

## Disclaimer (Important)

This tool provides AI‑generated assistance for exploring legal documents. It is **not** legal advice. Always consult a qualified lawyer before relying on any output.

—

If this helped you: star it, fork it, or plug it into a workflow and tell me what broke. That feedback helps prioritize the next round of polish.
