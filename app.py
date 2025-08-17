import os
import streamlit as st
from dotenv import load_dotenv
from src.utils.config import AppConfig
from src.ui.components import sidebar, overview_tab, clauses_tab, redflags_tab, qa_tab, report_tab
from src.ingest.pdf_loader import load_pdfs
from src.ingest.chunker import chunk_documents
from src.embeddings.embeddings import get_embedding_model
from src.vectorstore.faiss_store import FaissStoreManager
from src.summarize.summarizer import summarize_documents
from src.analysis.clauses import extract_clauses
from src.analysis.redflags import detect_redflags
from src.rag.qa_chain import build_qa_chain
from src.utils.types import ClauseResult, RedFlagResult
from src.report.report import build_report
from src.report.json_export import build_analysis_json

load_dotenv()
config = AppConfig.from_env()

st.set_page_config(page_title="AI Legal Doc Explainer", layout="wide", page_icon="⚖️")

# Global aesthetic enhancements injected once
if 'theming_loaded' not in st.session_state:
    st.session_state.theming_loaded = True
    st.markdown(
                """
                                <style>
                                /* Root palette defaults (dark) */
                                :root {
                                        --pri:#6a5acd; --pri-grad:linear-gradient(135deg,#6a5acd,#8f7bff);
                                        --accent:#ffb347; --danger:#ff4b4b;
                                        --bg1:#0f1117; --bg2:#161c23; --panel:#1d2530; --border:#263040;
                                        --text:#e6e9ef; --text-soft:#b5c1d1;
                                }
                                @media (prefers-color-scheme: light) {
                                    :root {
                                        --bg1:#f7f9fc; --bg2:#ffffff; --panel:#ffffff; --border:#dbe2ec;
                                        --text:#1c2330; --text-soft:#5a6675;
                                    }
                                    body.light .hero { background:radial-gradient(circle at 25% 15%, #ffffff, #eef2f8 70%) !important; }
                                }
                                body, .stApp { background: var(--bg1); color: var(--text); }
                                header[data-testid="stHeader"] { background: rgba(0,0,0,0); }
                                /* Hero banner (will adapt via media query overrides) */
                                .hero { padding: 1.1rem 1.5rem 0.9rem 1.5rem; margin:-1rem -1rem 1rem -1rem; background:radial-gradient(circle at 25% 15%, #223045, #0f1117 70%); border-bottom:1px solid var(--border); }
                                .hero h1 { font-size:1.9rem; background:var(--pri-grad); -webkit-background-clip:text; color:transparent; margin:0; font-weight:700; letter-spacing:.5px; }
                                .hero p { color:var(--text-soft); margin:.4rem 0 0 0; font-size:.85rem; }
                                /* Buttons */
                                div[data-testid="stSidebar"] button, .stButton>button { background:var(--pri-grad)!important; border:0!important; color:#fff!important; font-weight:600!important; box-shadow:0 2px 6px -2px #000!important; }
                                .stButton>button:hover { filter:brightness(1.07); }
                                /* File uploader tweak */
                                .stFileUploader { background:var(--panel); padding:.5rem .75rem; border:1px solid var(--border); border-radius:10px; }
                                /* Tabs active underline */
                                button[data-baseweb="tab"] { font-weight:600; }
                                button[aria-selected="true"][data-baseweb="tab"] { border-bottom:3px solid var(--accent); }
                                /* Spinner */
                                .stSpinner > div { border-top-color: var(--accent); }
                                /* Download button */
                                .stDownloadButton>button { background:var(--pri-grad); border:0; }
                                /* Legal notice */
                                .legal-footer { text-align:center; padding:.75rem 0 2rem 0; font-size:.65rem; color:var(--text-soft); }
                                /* Glass panels */
                                .glass { background:rgba(255,255,255,0.03); backdrop-filter: blur(8px); }
                                </style>
                                <script>
                                // Add class for light/dark detection so we can scope extra overrides if needed
                                const prefersLight = window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches;
                                if (prefersLight) { document.body.classList.add('light'); }
                                </script>
                                <div class='hero'>
                                        <h1>⚖️ AI Legal Doc Explainer</h1>
                                        <p>Contract intelligence: summaries • key clauses • risk flags • grounded Q&A • exportable audit report.</p>
                                </div>
                                """,
                unsafe_allow_html=True,
    )
sidebar_state = sidebar(config)

if 'documents' not in st.session_state:
    st.session_state.documents = []
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'summaries' not in st.session_state:
    st.session_state.summaries = {}
if 'clauses' not in st.session_state:
    st.session_state.clauses = []
if 'redflags' not in st.session_state:
    st.session_state.redflags = []
if 'vectorstore' not in st.session_state:
    st.session_state.vectorstore = None
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'qa_history' not in st.session_state:
    st.session_state.qa_history = []

st.markdown("<h2 style='margin-top:0;'>Workspace</h2>", unsafe_allow_html=True)
uploaded_files = st.file_uploader("Upload legal PDFs", type=["pdf"], accept_multiple_files=True, help="You can add multiple contracts before analyzing.")

home_cols = st.columns([1,1,1,1])
with home_cols[0]:
    process_clicked = st.button("🚀 Full Analyze", type="primary", use_container_width=True, help="Run summaries, clauses, red flags & QA preparation")
with home_cols[1]:
    export_report = st.button("📥 PDF Report", use_container_width=True)
with home_cols[2]:
    export_json = st.button("🗂️ Export JSON", use_container_width=True, help="Download structured JSON for competition submission")
with home_cols[3]:
    reset_workspace = st.button("♻️ Reset", use_container_width=True)

# Ensure Export JSON button (third column) gets gradient background if theme overrides generic rule
st.markdown(
    """
    <style>
    div[data-testid="column"]:nth-of-type(3) .stButton>button {background:var(--pri-grad)!important;border:0!important;color:#fff!important;box-shadow:0 2px 6px -2px #000!important;}
    </style>
    """,
    unsafe_allow_html=True,
)

if reset_workspace:
    st.session_state.clear()
    st.rerun()

if 'uploaded_file_names' not in st.session_state:
    st.session_state.uploaded_file_names = []

# Auto quick index build on new upload (for immediate chat)
if uploaded_files:
    current_names = sorted([f.name for f in uploaded_files])
    if current_names != st.session_state.uploaded_file_names:
        with st.spinner("Auto-indexing uploaded documents for chat..."):
            docs = load_pdfs(uploaded_files)
            st.session_state.documents = docs
            chunks = chunk_documents(docs)
            st.session_state.chunks = chunks
            if chunks:
                embed = get_embedding_model(config)
                manager = FaissStoreManager(config)
                vs = manager.build_index(chunks, embed, force_rebuild=True)
                st.session_state.vectorstore = vs
                st.session_state.qa_chain = build_qa_chain(config, vs)
            # Quick heuristic summaries (fast) so overview isn't empty
            if docs and chunks:
                from src.summarize.summarizer import heuristic_document_summary
                quick_sums = {}
                by_doc = {}
                for ch in chunks:
                    by_doc.setdefault(ch.document_name, []).append(ch.content)
                for d in docs:
                    quick_sums[d.name] = {"bullets": heuristic_document_summary(by_doc.get(d.name, []))}
                st.session_state.summaries = quick_sums
            # Quick clause + red flag extraction (heuristic will run if no full model)
            if chunks:
                try:
                    from src.analysis.clauses import extract_clauses
                    from src.analysis.redflags import detect_redflags
                    if not st.session_state.get('clauses'):
                        st.session_state.clauses = extract_clauses(config, chunks)
                    if st.session_state.clauses and not st.session_state.get('redflags'):
                        st.session_state.redflags = detect_redflags(config, st.session_state.clauses)
                except Exception as e:
                    st.warning(f"Quick clause extraction skipped: {e}")
        st.session_state.uploaded_file_names = current_names
        if st.session_state.qa_chain:
            st.success("Chat ready. You can start asking questions now or run Full Analyze for deeper insights.")

if process_clicked and uploaded_files:
    with st.spinner("Loading PDFs..."):
        docs = load_pdfs(uploaded_files)
        st.session_state.documents = docs
    with st.spinner("Chunking documents..."):
        chunks = chunk_documents(docs)
        st.session_state.chunks = chunks
    with st.spinner("Embedding and indexing..."):
        embed = get_embedding_model(config)
        manager = FaissStoreManager(config)
        vs = manager.build_index(chunks, embed)
        st.session_state.vectorstore = vs
    with st.spinner("Summarizing documents..."):
        summaries = summarize_documents(config, docs, chunks)
        st.session_state.summaries = summaries
    with st.spinner("Extracting clauses..."):
        st.session_state.clauses = extract_clauses(config, chunks)
    with st.spinner("Detecting red flags..."):
        st.session_state.redflags = detect_redflags(config, st.session_state.clauses)
    with st.spinner("Preparing QA chain..."):
        st.session_state.qa_chain = build_qa_chain(config, st.session_state.vectorstore)
    st.success("Analysis complete.")

# Removed manual rebuild button; index rebuild happens automatically on new upload or full analyze

if export_report and st.session_state.documents:
    with st.spinner("Generating PDF report..."):
        pdf_bytes = build_report(
            docs=st.session_state.documents,
            summaries=st.session_state.summaries,
            clauses=st.session_state.clauses,
            redflags=st.session_state.redflags,
            qa_history=st.session_state.qa_history,
            config=config,
        )
        st.download_button("Download Report", data=pdf_bytes, file_name="legal_report.pdf", mime="application/pdf")

if 'export_json_count' not in st.session_state:
    st.session_state.export_json_count = 0
if export_json and st.session_state.documents:
    with st.spinner("Preparing JSON export..."):
        meta = {
            "app": "AI Legal Doc Explainer",
            "version": "0.1.0",
            "use_gemini": config.use_gemini,
            "embed_model": config.embed_model,
            "documents": len(st.session_state.documents),
            "clauses": len(st.session_state.clauses),
            "red_flags": len(st.session_state.redflags),
        }
        blob = build_analysis_json(
            docs=st.session_state.documents,
            summaries=st.session_state.summaries,
            clauses=st.session_state.clauses,
            redflags=st.session_state.redflags,
            qa_history=st.session_state.qa_history,
            meta=meta,
        )
        st.session_state.export_json_count += 1
        st.download_button(
            label="Download JSON", data=blob, file_name="analysis_snapshot.json", mime="application/json", key=f"jsondl{st.session_state.export_json_count}"
        )

# Tabs
overview, clauses_tab_ui, redflags_tab_ui, qa_tab_ui, report_tab_ui = st.tabs([
    "Overview", "Clauses", "Red Flags", "Ask Questions", "Report"
])

with overview:
    overview_tab(config, st.session_state.documents, st.session_state.summaries)
with clauses_tab_ui:
    clauses_tab(st.session_state.clauses)
with redflags_tab_ui:
    redflags_tab(st.session_state.redflags, config)
with qa_tab_ui:
    qa_tab(config, st.session_state.qa_chain, st.session_state.qa_history)
with report_tab_ui:
    report_tab(st.session_state)

st.markdown("<div class='legal-footer'>Not legal advice. For informational purposes only.</div>", unsafe_allow_html=True)
