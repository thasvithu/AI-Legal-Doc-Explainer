# Technical Summary: AI Legal Doc Explainer

## 1. Stack & Goals
- Streamlit UI for rapid, reactive legal document exploration.
- LangChain components for chunking, embeddings, retrieval (FAISS) and prompt chaining.
- Hybrid LLM strategy: Gemini (primary) with open-source fallback (transformers local pipeline).
- CPU-friendly embedding model (`intfloat/e5-small-v2`).
- Privacy: all parsing + vectorization local; only model calls leave host.

## 2. Architecture
High-level flow:
Upload PDFs -> Parse & Clean -> Chunk -> Embed -> FAISS -> (Summaries / Clauses / Red Flags / QA) -> Report Export

Modules:
- ingest: `pdf_loader`, `chunker`
- embeddings: HF embedding wrapper
- vectorstore: FAISS index lifecycle
- llm: `gemini` client + `fallback` local model
- summarize: iterative bullet summarization
- analysis: clause extraction + risk scoring
- rag: retriever + QA chain with citation packaging
- report: PDF generator (ReportLab)
- ui: Streamlit component helpers
- prompts: editable templates
- utils: config, logging, types

## 3. RAG & Prompts
- Chunking: recursive splitter (1200 chars, 180 overlap) balances semantic cohesion & token limits.
- Retrieval: FAISS cosine similarity via LangChain; top-k=5 default.
- QA prompt enforces strict grounding & citation requirement.
- Summarization uses map (batch chunk bullets) then reduce (aggregate) strategy to stay within context limits.
- Clause & Red Flag prompts emit structured line-oriented outputs easy to parse deterministically.

## 4. Clause Extraction & Red Flags
- Patterns instruct model to emit canonical clause categories.
- Importance heuristic assigns High to indemnity/liability, Medium to auto-renewal & penalties.
- Red flag scoring: blend of regex keyword boosts (broad indemnity, unilateral discretion, auto-renewal, liquidated damages) and LLM adjustment.
- Configurable confidence threshold filters noise.

## 5. Report Generation
- Single-pass PDF assembly: metadata, summaries, clause table lines, red flags, Q&A history.
- Uses ReportLab for deterministic, offline PDF creation.

## 6. Testing & Quality
- Unit tests: ingestion cleaning, vectorstore retrieval roundtrip, QA chain citation presence.
- Lint: `ruff` for style & imports.
- Deterministic defaults: low temperature (0.3) lowers hallucination risk.

## 7. Performance Considerations
- Small embedding model cached; single FAISS index reused unless rebuild.
- Local LLM fallback may be slow on CPU for large models; recommendation: use Gemini where available or switch to a smaller instruct model.
- Batch sizes keep prompt under typical free-tier context limits.

## 8. Security & Privacy
- Only model API calls leave environment; raw docs not persisted beyond session workspace.
- Temporary workspace directory can be reset; user-controlled.

## 9. Trade-offs & Future Enhancements
- Simple regex + LLM hybrid risk scoring could evolve into learned classifier.
- Clause pagination approximated by chunk index; could map actual PDF page references with positional extraction.
- More granular confidence calibration (embedding similarity + model self-estimate) future work.
- Multi-language detection & translation pipeline not yet implemented.

## 10. Unique Value Proposition
- Editable prompt templates in-repo empower rapid iteration.
- Structured outputs -> reliable downstream tables & report export.
- Integrated risk heuristics + mitigation guidance (reason field) beyond plain summarization.

## 11. Limitations & Disclaimer
- Not a substitute for legal advice; model may miss nuances or mis-rank risks.
- Large PDFs (>50 pages each) may slow on CPU; consider streaming chunk processing.

(See `ARCHITECTURE.png` for the diagram.)
