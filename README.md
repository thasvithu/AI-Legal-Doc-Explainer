<<<<<<< HEAD
<div align="center">

# ⚖️ AI Legal Document Explainer

Smart, private, explainable RAG assistant for contracts & legal PDFs. Upload a document and instantly get:

✅ Plain‑language summary  
✅ Key clause & category extraction (with Typical / Atypical heuristic)  
✅ Red flag & severity detection + overall Risk Index  
✅ Structured entities (effective date, parties, governing law, term)  
✅ Grounded Q&A with source citations & confidence bar  
✅ Downloadable structured JSON + text reports  
✅ Ephemeral vector indexing (auto‑purged) for privacy

</div>

---

## ✨ Feature Overview

| Area | What It Does | Notes |
|------|---------------|-------|
| Ingestion | Parses PDF via PyMuPDF | Supports Streamlit uploads & file paths |
| Chunking | Recursive splitter (size & overlap configurable) | Preserves semantic boundaries using multi‑separator fallback |
| Embeddings | `BAAI/bge-small-en-v1.5` (HuggingFace) | Balanced speed vs quality |
| Vector Store | FAISS (in‑memory / ephemeral temp dir) | Auto clean after inactivity (120s) & on process exit |
| Summarization | Gemini (LLM) or offline heuristic fallback | Refined into simpler “plain language” version |
| Clause Extraction | Keyword + category mapping | Adds Typical / Atypical heuristic badge |
| Risk Detection | Severity tagging (High / Medium / Low) | Weighted Risk Index (0–100) computed |
| Entity Extraction | Regex heuristics | Effective date, parties, governing law, term length |
| Q&A | Retrieval‑augmented generation (RAG) | Citations + similarity‑based confidence |
| Confidence Scoring | Hybrid similarity + length heuristic | Visual progress bar in UI |
| Exports | Summary TXT, clauses JSON, risks JSON, full report JSON | PDF export planned |
| Privacy | Ephemeral FAISS + session janitor | No persistent storage unless extended |

---

## 🔧 Tech Stack

| Layer | Tools |
|-------|-------|
| UI | Streamlit + custom CSS |
| LLM | Google Gemini 1.5 Flash (optional; graceful fallback) |
| Embeddings | HuggingFace `sentence-transformers` (bge-small) |
| Vector Store | FAISS (`langchain_community.vectorstores`) |
| Orchestration | LangChain components (loader, splitter, retriever) |
| Environment | Python 3.10+ (`requirements.txt` pinned) |
| Logging / Errors | Lightweight custom exception wrapper |

---

## 🧠 High-Level Architecture

```
┌──────────┐   Upload   ┌──────────────┐   Pages -> Docs   ┌──────────────┐   Chunk -> Chunks   ┌────────────────┐
│  Client  │──────────▶│  PDF Loader  │──────────────────▶│  Documents    │────────────────────▶│  Splitter       │
└──────────┘            └──────────────┘                   └──────────────┘                     └────────────────┘
																																																					│
																																																		Chunks │
																																																					▼
																																																 ┌────────────────┐
																																																 │ Embeddings (HF)│
																																																 └────────────────┘
																																																					│
																																																					▼
																																																 ┌────────────────┐
																																																 │  FAISS Index   │ (ephemeral)
																																																 └────────────────┘
																																																					│
														 ┌───────────────────────────────┬───────────────────────────────┬───────────────┘
														 ▼                               ▼                               ▼
										Summarization / Refinement    Clause & Risk Analysis          Retrieval for Q&A
														 │                               │                               │
														 └──────────────┬────────────────┴──────────────┬────────────────┘
																						▼                               ▼
																					UI Panels (Summary • Clauses • Risks • Q&A • Export)
```

---

## 🔐 Privacy & Ephemeral Design

All embeddings & FAISS index files are stored in a randomly named temp directory per session. A lightweight registry + background janitor thread deletes them after a period of inactivity (`TTL_SECONDS = 120`) or automatically at process exit. No document text is retained once the session ends (unless you adapt persistence intentionally).

---

## 📊 Risk Index Formula

Let:  
`weights = { high:3, medium:2, low:1 }`  
`raw = Σ weights(severity_i)`  
`denom = log10( max(50, total_chars)/1000 + 1 )`  
`scaled = min(100, (raw / denom) * 14 )`  

The resulting integer `index` is bucketed:  
`<26 = Low`, `26–55 = Moderate`, `56–75 = Elevated`, `>75 = High`.

Rationale: contract length normalization prevents long documents from inflating risk unfairly; severity weighting emphasizes concentrated high‑impact clauses.

---

## 🎯 Confidence Scoring (Answer Panel)

1. Retrieve top-k chunks & collect raw similarity scores (cosine or distance heuristic).  
2. Normalize: `norm = (score + 1)/2` (map to [0,1]).  
3. Base similarity = mean of top 3 normalized scores.  
4. Length factor = `min(1, answer_tokens/≈180 chars)`.  
5. Combined: `confidence = base*0.7 + length_factor*0.3`, clamped to `[0.05, 0.95]`.  
6. Displayed as Streamlit progress bar + advisory warning if low or risk elevated.

---

## 🧩 Typical vs Atypical Heuristic

A clause is tagged Atypical if it contains trigger phrases (e.g. `perpetual`, `unlimited liability`, `automatic penalty`). This is a lightweight interim signal; roadmap includes LLM‑powered standardness scoring with contextual rationale & market comparables.

---

## 📁 Project Structure (Key Parts)

```
app.py                     # Streamlit application (tabs, metrics, exports)
modules/
	pdf_reader.py            # Robust PDF ingestion (file path or upload)
	splitter.py              # Recursive chunking
	embed_store.py           # Embedding + FAISS (ephemeral/persistent)
	retriever.py             # Load + create retriever
	qa_with_retriever.py     # RAG answer + similarities + citations
	analysis.py              # Summary, clauses, risks, entities, risk index, refinement
	session_manager.py       # TTL-based ephemeral cleanup
utils/
	logger.py, exception.py  # Minimal diagnostics helpers
requirements.txt           # Pinned dependencies
LICENSE                    # MIT license
```

---

## 🚀 Quick Start

```bash
python -m venv .venv
./.venv/Scripts/activate   # Windows PowerShell
pip install --upgrade pip
pip install -r requirements.txt

# Create .env with your Gemini key (optional for enhanced LLM features)
echo GEMINI_API_KEY=your_key_here > .env  # (Windows: use a text editor instead)

streamlit run app.py
```

Then open the provided local URL, upload a PDF, and explore the tabs.

### Environment Variables
| Name | Purpose | Required |
|------|---------|----------|
| `GEMINI_API_KEY` | Enables LLM summarization & high‑quality Q&A. Fallback heuristics used if absent. | Optional |

---

## 🖥️ Usage Flow

1. Upload PDF (contract / lease / agreement).  
2. App extracts pages → splits text into overlapping chunks.  
3. Chunks embedded & stored in ephemeral FAISS index.  
4. Summaries + clause scan + risk computation + entities displayed.  
5. Ask questions; system retrieves top chunks → constructs grounded answer → shows citations + confidence.  
6. Export structured outputs (JSON/TXT) for downstream analysis or compliance review.  
7. Inactivity → automatic secure cleanup.

---

## 🧪 Testing (Planned)

Planned lightweight tests to cover:  
* Clause extraction deduplication  
* Risk index scaling edge cases (tiny vs large docs)  
* Confidence score boundaries  
* Ephemeral cleanup behavior (mocking registry)  

> Current prototype does not yet ship with automated tests; roadmap includes PyTest + GitHub Actions CI.

---

## 🚧 Roadmap

| Phase | Item | Status |
|-------|------|--------|
| 1 | Core ingestion + RAG + risks | ✅ |
| 2 | Plain‑language refinement | ✅ |
| 3 | Citations & similarity confidence | ✅ |
| 4 | Risk index & severity buckets | ✅ |
| 5 | Clause categorization + typicality heuristic | ✅ |
| 6 | Multi‑doc comparison dashboard | ⏳ |
| 7 | Atypicality via LLM + rationale | ⏳ |
| 8 | PDF consolidated export (report) | ⏳ |
| 9 | Remediation suggestions per risk | ⏳ |
| 10 | Actionable obligations extraction | ⏳ |
| 11 | Automated tests + CI | ⏳ |
| 12 | Docker & cloud deployment template | ⏳ |
| 13 | Advanced retrieval (hybrid / rerank) | ⏳ |
| 14 | Multi‑language support | ⏳ |

---

## ⚠️ Limitations

* Regex-based entity extraction may miss nuanced phrasing.  
* Clause detection relies on keyword heuristics (recall can be improved).  
* Typical/Atypical classification is not yet model-driven.  
* Confidence scoring is heuristic; not a calibrated probability.  
* No legal liability—this is an assistive research tool only.

---

## 🔒 Security / Privacy Notes

* No persistent storage of document content by default.  
* Ephemeral directories scrubbed after inactivity or shutdown.  
* If you deploy remotely, review hosting provider temp storage policies.  
* Add encryption-at-rest & access controls if enabling persistence.

---

## 🤝 Contributing

Contributions welcome! Ideas: smarter clause taxonomy, ML classifiers, better UI micro‑interactions, retrieval metrics. Feel free to open an issue or submit a PR following conventional commits.

---

## 📜 License

MIT – see `LICENSE` for full text.

---

## 🙏 Acknowledgments

* HuggingFace & SentenceTransformers for embedding models.  
* LangChain ecosystem for composable retrieval components.  
* Google Gemini for optional generative augmentation.  
* FAISS for blazing fast vector similarity search.

---

## 🛡️ Disclaimer

This application provides AI‑generated assistance and risk heuristics. It does NOT constitute legal advice. Always consult qualified legal counsel before executing or relying on contractual terms.

---

### ⭐ If you find this useful
Bookmark, share, or extend it for your own compliance / contract intelligence workflows!

</div>

---

## 🚀 Deploy on Hugging Face Spaces

You can deploy this Streamlit app as a Space in a few minutes.

### 1. Prepare Repository
Ensure the following files are present (already included):
* `app.py` (entrypoint)
* `requirements.txt` (pinned dependencies)
* `runtime.txt` (optional – specify Python version, e.g. `python-3.10`)
* `README.md` (project description – shows on Space page)

Create a `runtime.txt` (example):
```
python-3.10
```

### 2. Create a New Space
1. Go to https://huggingface.co/spaces and click New Space.  
2. Select SDK = Streamlit.  
3. Choose a name (e.g. `your-username/legal-doc-explainer`).  
4. Set visibility (Public or Private).  
5. Create Space.

### 3. Push Code
If using Git locally:
```bash
git remote add hf https://huggingface.co/spaces/<username>/legal-doc-explainer
git push hf main
```

Or upload files directly in the Space UI.

### 4. Configure Gemini Key (Optional)
In the Space page: Settings → Variables & secrets → Add:
* Key: `GEMINI_API_KEY`
* Value: your API key

Without this key the app still works (heuristic summaries + extractive Q&A fallback).

### 5. Build & Run
The Space will auto-build: install requirements → launch Streamlit.  
Entry command defaults to:
```
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```
If needed, set this explicitly in the Space Settings > App File (point to `app.py`).

### 6. Test
Upload a sample PDF and verify:
* Summary renders
* Key clauses populate
* Risk index appears
* Q&A works (with or without Gemini)
* Downloads succeed

### 7. (Optional) Persistence
Current design uses ephemeral FAISS indexes. To persist across sessions, adapt `embed_store.py` to disable `ephemeral=True` and commit the `faiss_legal_index/` directory (not recommended for sensitive docs).

### 8. Troubleshooting
| Symptom | Fix |
|---------|-----|
| Space stuck on "Building" | Ensure Python version in `runtime.txt` is supported; keep dependency versions pinned |
| Module import error for `google.generativeai` | Ensure it’s in `requirements.txt` (already included) |
| App shows offline heuristic message | Add `GEMINI_API_KEY` secret |
| Memory errors on large PDFs | Reduce chunk size / increase overlap modestly |

### Minimal Fileset Example
```
app.py
modules/ (code)
utils/
requirements.txt
runtime.txt
README.md
LICENSE
```
=======
---
title: Legal Doc Explainer
emoji: 📚
colorFrom: yellow
colorTo: gray
sdk: gradio
sdk_version: 5.42.0
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
>>>>>>> 2dcd28af771df9f497e1fd81067d872fdd6df673
