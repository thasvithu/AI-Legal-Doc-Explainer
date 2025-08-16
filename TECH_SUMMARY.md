# Technical Summary

## Overview
AI Legal Doc Explainer ingests uploaded PDF contracts, produces an accessible summary, surfaces key / risky clauses, extracts lightweight entities, answers user questions with source citations, and now heuristically lists party obligations. Privacy is reinforced by using ephemeral FAISS vector indices that are deleted after session end or inactivity.

## Tech Stack
Language: Python 3.10  
UI: Streamlit  
Vector Store: FAISS  
Embeddings: HuggingFace `BAAI/bge-small-en-v1.5`  
LLM (optional): Google Gemini 1.5 Flash (fallback heuristics if absent)  
Libraries: LangChain (docs, embeddings, retrieval), PyMuPDF (PDF), dotenv, google-generativeai  
Testing: Pytest (initial heuristics tests)  
Infra: Local / Hugging Face Space (stateless ephemeral indices)  

## Design & Architecture (Text Diagram)
Upload PDF -> Parse Pages -> Split Chunks -> Embed -> FAISS (temp, session-scoped) -> Retriever -> (LLM or heuristic) Answer + Citations  
								   |-> Clause/Risk/Obligation Heuristics -> Risk Index / Flags  
								   |-> Summary -> Bulletization / Plain-Language Refinement  
								   |-> Entity / Obligation Extraction (regex heuristics)  
Ephemeral Index Registry -> Cleanup Thread (TTL)  

## How AI Is Used
1. Semantic Embeddings: Dense vectors for retrieval powering context-grounded Q&A.  
2. Optional LLM Generation: Summaries, answer refinement, bullet formatting (when API key present).  
3. Heuristic NLP: Regex + keyword taxonomies for clauses, risks, obligations, and entities when model absent.  
4. Confidence & Risk Scoring: Lightweight statistical + heuristic blend instead of opaque LLM scoring.  

## Key Features
* Plain-language contract summary + bullet/numbered toggle.  
* Clause extraction with category + typicality hint.  
* Red flags with severity + global risk index (0–100).  
* Retrieval-augmented Q&A with citations & confidence score.  
* Party obligations heuristic (modal verb scan).  
* Entity extraction (effective date, parties, governing law, term length).  
* JSON / text export.  
* Ephemeral privacy-preserving index (auto cleanup).  

## Challenges Faced
| Challenge | Mitigation |
|-----------|------------|
| Mixed clause density & long pages | Recursive chunking tuned for balance (size/overlap) |
| Absence of LLM key for some users | Heuristic summaries + extractive Q&A fallback |
| Over-counting risks in long docs | Log-length normalization in risk index |
| Clause duplication noise | (keyword, page) dedupe set |
| Privacy concerns (uploaded contracts) | Ephemeral temp FAISS indices + TTL cleanup |
| Slow cold start on large PDFs | Small embedding model (bge-small) to keep latency low |

## Unique Value Proposition
1. Works fully offline (summaries/Q&A heuristics) – graceful degradation.  
2. Transparent, explainable risk index (documented formula + comments).  
3. Ephemeral, privacy-first vector storage (no silent persistence).  
4. Early obligations extraction guiding negotiation prep.  
5. Lean footprint: minimal model size & dependencies for rapid deploy.  

## Roles & Contributions
If solo: Responsible for end-to-end architecture, feature design, heuristics, UI/UX, deployment, and documentation. (Replace with team member role breakdown if multi-person.)  

## Submission Packaging Notes
* Convert this Markdown to PDF (target 2–4 pages) for submission.  
* Ensure demo URL + video link inserted in README placeholders.  
* Include CHANGELOG.md with evolution timeline.  


## Pipeline
1. PDF Load: `modules/pdf_reader.py` (PyMuPDF) -> LangChain `Document` objects (per page).
2. Splitting: `modules/splitter.py` recursive character splitter tuned for legal text length balance.
3. Embedding: HuggingFace `BAAI/bge-small-en-v1.5` via LangChain embedding wrapper.
4. Vector Store: FAISS (in-memory + temp dir). New indices tracked for cleanup.
5. Retrieval: Similarity search k=4 (configurable) feeding QA.
6. Generation: If `GEMINI_API_KEY` present, Gemini 1.5 Flash for summaries & formatting; else heuristic fallbacks.
7. Analytics: Risk keyword scan, clause categorization, severity, risk index scaling (log length normalization), confidence scoring from similarity + answer length, obligation heuristics.
8. UI: Streamlit multi-tab interface (Summary, Clauses, Risks, Q&A, Export, Obligations) with styling, toggles (bullets vs numbered), JSON export.

## Key Heuristics
- Risk Keywords & Categories: Centralized in `modules/constants.py` for maintainability.
- Clause Deduplication: Unique by (keyword, page) pair to avoid repetition noise.
- Risk Index: Weighted (high=3, medium=2, low=1) scaled by document length denominator = log10(chars/1000 + 1) ensuring longer docs don't inflate risk unfairly.
- Similarity Confidence: Uses top3 cosine scores normalized + length factor to bound 0.05–0.95.
- Obligation Extraction: Regex over modal verbs (shall/must/agrees to/responsible for). Simple party role inference (Supplier/Customer) pending NLP role tagging upgrade.
- Bulletization: LLM-first attempt; heuristic sentence split fallback with de-dup and min length filter.

## Privacy & Ephemerality
Each upload session builds a fresh FAISS index under a temp path. Registry + TTL mechanism cleans unused directories to reduce data persistence concerns.

## Error Handling & Logging
Custom exception wrapper (`utils/exception.py`) and structured logging scaffold (`utils/logger.py`). Printing replaced by logging calls in most modules (continuing to expand coverage).

## Extensibility Points
- Swap Embedding Model: central config in `embed_store`.
- Add Negotiation Suggestions: extend obligations function or add new post-processing pass referencing risk categories.
- Multi-Doc Workspace: iterate uploaded PDFs, merge indices (future: per-document selectors).
- Advanced NER: replace regex with lightweight spaCy model (kept minimal now to reduce dependency weight).
- Auth / Persistence: Add user auth & (optional) encrypted long-term index storage.

## Testing Strategy (Initial)
`tests/test_analysis.py` covers: bullet formatting, obligation extraction, risk index scaling bounds, clause dedupe, similarity confidence baseline. Future: integration test for end-to-end upload -> retrieval; mock LLM responses.

## Deployment
- Local: `streamlit run app.py` after installing `requirements.txt`.
- Hugging Face Space: runtime.txt pin + ephemeral index path creation; ensure no secrets baked into repo.

## Known Gaps / TODO
- Negotiation suggestion engine (map risks -> mitigating proposals).
- More granular severity scoring (TF-IDF weighting by clause length, context).
- LLM caching layer for repeated queries.
- Expand category taxonomy (SaaS SLAs, Data Protection, Employment clauses).
- Add CHANGELOG.md and richer logging contexts (session id, index path).

## Version
Application version constant: `APP_VERSION` in `app.py` (displayed in footer).
