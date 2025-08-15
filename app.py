import streamlit as st
from modules.pdf_reader import get_pdf_text
from modules.splitter import split_pdf_text
from modules.embed_store import embed_and_store_documents
from modules.qa_with_retriever import answer_query_with_retriever
from modules.analysis import (
    summarize_documents,
    extract_key_clauses,
    detect_red_flags,
    answer_with_confidence,
    extract_entities,
    compute_similarity_confidence,
    compute_risk_index,
    refine_plain_language,
)
from modules.session_manager import (
    register as register_session_cleanup,
    touch as session_touch,
    ensure_started as ensure_session_manager,
)
import uuid
import json
from io import StringIO

st.set_page_config(page_title="AI Legal Doc Explainer", page_icon="⚖️", layout="wide")

# -------------------- Custom CSS Theme --------------------
st.markdown(
    """
    <style>
    :root {
        --primary-color: #0A2342; /* deep navy */
        --accent-color: #D4AF37;  /* gold */
        --bg-alt: #f5f7fa;
        --risk-high: #d93025;
        --risk-med: #ff9500;
        --risk-low: #6c757d;
        --card-border: #e2e8f0;
    }
    .main > div {padding-top: 1rem;}
    .app-header h1 {font-size: 2.1rem; margin-bottom: .2rem; color: var(--primary-color);}    
    .app-sub {color: #334155; font-size: 0.95rem; margin-bottom: 1.2rem;}
    .legal-card {background: white; border: 1px solid var(--card-border); border-radius: 10px; padding: 0.9rem 1.1rem; margin-bottom: .8rem; box-shadow: 0 1px 2px rgba(0,0,0,0.05);}    
    .badge {display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.65rem; font-weight:600; letter-spacing:.5px; text-transform:uppercase;}
    .badge-high {background: var(--risk-high); color:#fff;}
    .badge-medium {background: var(--risk-med); color:#fff;}
    .badge-low {background: var(--risk-low); color:#fff;}
    .metric-box {background:linear-gradient(135deg,#0A2342 0%, #123a6d 90%); color:white; padding:14px 16px; border-radius:12px; font-size:0.85rem;}
    .divider {height:1px; background:#e2e8f0; margin:18px 0;}
    .qabox {background:var(--bg-alt); border:1px solid var(--card-border); padding:10px 14px; border-radius:10px;}
    .footer-note {font-size:0.7rem; color:#64748b; margin-top:2rem; text-align:center;}
    .download-buttons button {margin-right: .5rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='app-header'>
      <h1>⚖️ AI Legal Document Explainer</h1>
      <div class='app-sub'>Summarize contracts, surface key clauses, flag risks, and ask precise questions – fast.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

ensure_session_manager()

# Derive a stable pseudo session id (once per connected browser session)
if "_session_id" not in st.session_state:
    st.session_state["_session_id"] = uuid.uuid4().hex
session_id = st.session_state["_session_id"]

with st.sidebar:
    st.markdown("### 📤 Upload & Settings")
    uploaded_file = st.file_uploader("PDF Document", type=["pdf"])
    st.markdown("---")
    chunk_size = st.number_input("Chunk Size", 200, 2000, 800, 50, help="Characters per text chunk for embeddings")
    chunk_overlap = st.number_input("Chunk Overlap", 0, 800, 100, 10, help="Overlap to preserve context between chunks")
    st.markdown("### 💡 Sample Questions")
    st.caption("Try:")
    st.write("• What are the termination conditions?\n• Are there auto-renewal terms?\n• What penalties apply?\n• What are my payment obligations?")
    st.markdown("### ⚠️ Disclaimer")
    st.caption("This tool provides AI-generated assistance, not legal advice. For critical matters consult a qualified lawyer.")

    st.markdown("---")
    st.caption("Session ID: `" + session_id + "`")

if uploaded_file:
    with st.spinner("Reading PDF..."):
        docs = get_pdf_text(uploaded_file)

    if not docs:
        st.error("Could not read the PDF.")
    else:
        # If a previous ephemeral index exists, clean it up before creating a new one
        # If prior cleanup exists, invoke it (user uploaded new file)
        if "_faiss_cleanup" in st.session_state:
            try:
                st.session_state["_faiss_cleanup"]()
            finally:
                st.session_state.pop("_faiss_cleanup", None)

        with st.spinner("Analyzing & indexing (ephemeral)..."):
            text_chunks = split_pdf_text(docs, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
            result_embed = embed_and_store_documents(text_chunks, ephemeral=True)

        if result_embed is None:
            st.error("Failed to build embeddings.")
        else:
            # embed_and_store_documents now returns (db, path, cleanup_fn)
            if len(result_embed) == 3:
                db, index_path, cleanup_fn = result_embed
            else:  # backward safety
                db, index_path = result_embed  # type: ignore
                cleanup_fn = None
            st.success("Index ready (ephemeral) – auto-cleans after inactivity.")
            # Keep cleanup function in session (for optional later deletion)
            if cleanup_fn:
                st.session_state["_faiss_cleanup"] = cleanup_fn
                register_session_cleanup(session_id, cleanup_fn)

            # Prepare data
            raw_summary = summarize_documents(docs)
            summary = refine_plain_language(raw_summary)
            clauses = extract_key_clauses(docs)
            red_flags = detect_red_flags(clauses) if clauses else []
            entities = extract_entities(docs)

            high_ct = sum(1 for r in red_flags if r['severity'] == 'high')
            med_ct = sum(1 for r in red_flags if r['severity'] == 'medium')
            low_ct = sum(1 for r in red_flags if r['severity'] == 'low')

            # Tabs for organized navigation
            tab_summary, tab_clauses, tab_risks, tab_qa, tab_export = st.tabs(
                ["📘 Summary", "🔍 Clauses", "⚠️ Risks", "💬 Q&A", "📥 Export"]
            )

            with tab_summary:
                st.markdown("### Overview Summary")
                st.markdown(f"<div class='legal-card'>{summary}</div>", unsafe_allow_html=True)
                st.markdown("#### Metrics")
                risk_meta = compute_risk_index(red_flags, sum(len(d.page_content) for d in docs))
                c1, c2, c3, c4, c5 = st.columns(5)
                with c1:
                    st.markdown(f"<div class='metric-box'><b>{len(docs)}</b><br/>Pages</div>", unsafe_allow_html=True)
                with c2:
                    st.markdown(f"<div class='metric-box'><b>{len(text_chunks)}</b><br/>Chunks</div>", unsafe_allow_html=True)
                with c3:
                    st.markdown(f"<div class='metric-box'><b>{len(clauses)}</b><br/>Key Clauses</div>", unsafe_allow_html=True)
                with c4:
                    st.markdown(f"<div class='metric-box'><b>{high_ct}/{med_ct}/{low_ct}</b><br/>Risks H/M/L</div>", unsafe_allow_html=True)
                with c5:
                    st.markdown(f"<div class='metric-box'><b>{risk_meta['index']}</b><br/>Risk Index</div>", unsafe_allow_html=True)
                st.markdown("#### Extracted Entities")
                ent_cols = st.columns(4)
                ent_cols[0].markdown(f"<div class='legal-card'><b>Effective Date</b><br/>{entities.get('effective_date') or '—'}</div>", unsafe_allow_html=True)
                ent_cols[1].markdown(f"<div class='legal-card'><b>Parties</b><br/>{entities.get('parties') or '—'}</div>", unsafe_allow_html=True)
                ent_cols[2].markdown(f"<div class='legal-card'><b>Governing Law</b><br/>{entities.get('governing_law') or '—'}</div>", unsafe_allow_html=True)
                ent_cols[3].markdown(f"<div class='legal-card'><b>Term Length</b><br/>{entities.get('term_length') or '—'}</div>", unsafe_allow_html=True)

            with tab_clauses:
                st.markdown("### Key Clauses & Context")
                if not clauses:
                    st.info("No notable clauses detected.")
                else:
                    for c in clauses:
                        category = c.get('category','General')
                        # Placeholder standardness classification (simple heuristic + could be replaced by LLM)
                        snippet_lower = c['snippet'].lower()
                        atypical = any(term in snippet_lower for term in ["perpetual", "unlimited liability", "automatic penalty"])
                        standardness = "Atypical" if atypical else "Typical"
                        badge_color = "#d97706" if atypical else "#15803d"
                        st.markdown(
                            f"""
                            <div class='legal-card'>
                                <div><strong>Keyword:</strong> <code>{c['keyword']}</code> • Page: {c.get('page','?')} • <span style='background:#1e293b;color:#e2e8f0;padding:2px 6px;border-radius:8px;font-size:0.65rem'>{category}</span>
                                <span style='background:{badge_color};color:#fff;padding:2px 6px;border-radius:8px;font-size:0.65rem;margin-left:6px'>{standardness}</span></div>
                                <div style='margin-top:6px;font-size:0.85rem;line-height:1.35'>{c['snippet']}</div>
                                <div style='margin-top:8px;font-size:0.7rem;color:#475569'><em>{c['note']}</em></div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            with tab_risks:
                st.markdown("### Red Flags / Risk Indicators")
                if not red_flags:
                    st.success("No risk indicators found.")
                else:
                    for rf in red_flags:
                        severity = rf["severity"].lower()
                        badge_cls = {
                            "high": "badge-high",
                            "medium": "badge-medium",
                            "low": "badge-low",
                        }.get(severity, "badge-low")
                        st.markdown(
                            f"""
                            <div class='legal-card'>
                                <span class='badge {badge_cls}'>{severity.upper()}</span>
                                <strong style='margin-left:6px'>{rf['keyword']}</strong> – {rf['note']}
                                <div style='margin-top:6px;font-size:0.8rem;color:#475569'>{rf['snippet']}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

            with tab_qa:
                st.markdown("### Ask Questions")
                st.caption("Answers are grounded in the indexed document.")
                user_q = st.text_area("Your question", value="What are the termination terms?", height=120)
                if st.button("Get Answer", key="ask_btn") and user_q.strip():
                    with st.spinner("Retrieving answer..."):
                        qa_result = answer_query_with_retriever(user_q, faiss_index_path=str(index_path))
                    answer_text = qa_result.get('answer','') if isinstance(qa_result, dict) else str(qa_result)
                    similarities = qa_result.get('similarities', []) if isinstance(qa_result, dict) else []
                    docs_used = qa_result.get('docs', []) if isinstance(qa_result, dict) else []
                    conf = compute_similarity_confidence(similarities, answer_text)
                    st.markdown(f"<div class='qabox'><b>Answer:</b><br/>{answer_text}</div>", unsafe_allow_html=True)
                    st.progress(conf)
                    if conf < 0.45 or high_ct > 0 or risk_meta['level'] in ['Elevated','High']:
                        st.warning("Low confidence and/or elevated risk. Consult qualified legal counsel before acting.")
                    # Citations
                    if docs_used:
                        st.markdown("#### Citations")
                        for i, dref in enumerate(docs_used):
                            page = dref.metadata.get('page') or dref.metadata.get('page_number')
                            score_disp = f"{similarities[i]:.3f}" if i < len(similarities) else "—"
                            excerpt = dref.page_content[:280].replace('\n',' ')
                            st.markdown(f"**{i+1}. Page {page} (sim {score_disp})** – {excerpt}...")
                    session_touch(session_id)

            with tab_export:
                st.markdown("### Export & Downloads")
                # Prepare export content
                summary_txt = summary
                clauses_json = json.dumps(clauses, indent=2, ensure_ascii=False)
                risks_json = json.dumps(red_flags, indent=2, ensure_ascii=False)

                st.download_button(
                    "⬇️ Download Summary (TXT)",
                    data=summary_txt.encode("utf-8"),
                    file_name="summary.txt",
                    mime="text/plain",
                )
                st.download_button(
                    "⬇️ Download Clauses (JSON)",
                    data=clauses_json.encode("utf-8"),
                    file_name="clauses.json",
                    mime="application/json",
                )
                st.download_button(
                    "⬇️ Download Risks (JSON)",
                    data=risks_json.encode("utf-8"),
                    file_name="red_flags.json",
                    mime="application/json",
                )

                # Combined report
                risk_meta = compute_risk_index(red_flags, sum(len(d.page_content) for d in docs))
                combined_report = {
                    "summary_raw": raw_summary,
                    "summary_plain": summary_txt,
                    "clauses": clauses,
                    "red_flags": red_flags,
                    "risk_index": risk_meta,
                    "entities": entities,
                    "stats": {
                        "pages": len(docs),
                        "chunks": len(text_chunks),
                        "key_clauses": len(clauses),
                    },
                }
                st.download_button(
                    "⬇️ Download Full Report (JSON)",
                    data=json.dumps(combined_report, indent=2).encode("utf-8"),
                    file_name="legal_ai_report.json",
                    mime="application/json",
                )

            # Heartbeat to keep session active while interacting
            session_touch(session_id)

            st.markdown(
                "<div class='footer-note'>© 2025 AI Legal Doc Explainer – Prototype for CodeStorm.AI. Not legal advice.</div>",
                unsafe_allow_html=True,
            )