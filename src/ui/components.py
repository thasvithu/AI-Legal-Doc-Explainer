from __future__ import annotations
import streamlit as st
import pandas as pd
from typing import List, Dict, Any
from src.utils.config import AppConfig
from src.utils.types import ClauseResult, RedFlagResult

PRIMARY_COLOR = "#6A5ACD"  # slate purple
ACCENT_COLOR = "#FFB347"
RISK_COLORS = {"High": "#FF4B4B", "Medium": "#FFB347", "Low": "#4CAF50"}

_CSS_TEMPLATE = r"""
<style>
html, body, [class*="css"]  { font-family: 'Inter', 'Segoe UI', sans-serif; }
section.main > div { padding-top: 1rem; }
div[data-testid="stSidebar"] { background: radial-gradient(circle at 30% 10%, #18212d, #0f141a 70%); border-right:1px solid #1d252e; padding-top:.4rem; }
@media (prefers-color-scheme: light) {
    div[data-testid="stSidebar"] { background: radial-gradient(circle at 30% 10%, #ffffff, #e9eef4 70%); border-right:1px solid #d8e0ea; }
    div[data-testid="stSidebar"] .status-pill { background:#f2f5f9; border-color:#dfe5ec; }
    div[data-testid="stSidebar"] .nav-links a { background:#f4f7fa; border-color:#dfe5ec; color:#3a4552; }
    div[data-testid="stSidebar"] .nav-links a:hover { background:#e9eff5; }
    .brand-badge { box-shadow:0 4px 10px -2px rgba(130,140,160,.35); }
    .progress-bar { background:#e0e7ef; }
    .progress-bar > div { box-shadow:none; }
    .glassy { background:rgba(255,255,255,0.55); }
}
div[data-testid="stSidebar"] h1, div[data-testid="stSidebar"] h2, div[data-testid="stSidebar"] h3 { color: __ACCENT__; }
div[data-testid="stSidebar"] .sidebar-brand {display:flex;align-items:center;gap:.55rem;margin:.2rem 0 1rem 0;}
.brand-badge {width:38px;height:38px;display:flex;align-items:center;justify-content:center;background:linear-gradient(135deg,#6a5acd,#8f7bff);color:#fff;font-weight:700;border-radius:12px;font-size:.9rem;letter-spacing:.5px;box-shadow:0 4px 10px -2px rgba(0,0,0,.5);}
/* Status pills */
.status-grid {display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:6px;margin-top:.25rem;}
.status-pill {background:#1c232e;border:1px solid #2d3441;padding:6px 8px;border-radius:10px;font-size:.55rem;line-height:1.05;letter-spacing:.4px;text-transform:uppercase;display:flex;flex-direction:column;gap:2px;position:relative;overflow:hidden;}
.status-pill:before {content:"";position:absolute;inset:0;opacity:0;transition:opacity .3s;background:linear-gradient(135deg,rgba(255,255,255,.05),rgba(255,255,255,0));}
.status-pill:hover:before {opacity:1;}
.status-pill span.value {font-size:.74rem;font-weight:600;color:#e2e6ee;letter-spacing:0;}
/* Progress */
.progress-cluster {display:flex;align-items:center;gap:10px;margin-top:.6rem;}
.radial { position:relative;width:70px;height:70px; }
.radial-ring {width:70px;height:70px;border-radius:50%;background:conic-gradient(#6a5acd calc(var(--p)*1%), #2a323d 0);display:flex;align-items:center;justify-content:center;position:relative;}
.radial-ring:after {content:"";position:absolute;inset:6px;background:#111922;border-radius:50%;}
.radial-label {position:absolute;inset:0;display:flex;align-items:center;justify-content:center;font-size:.7rem;font-weight:600;color:#dde3ec;}
.progress-meta {flex:1;}
.progress-meta h6 {margin:0;font-size:.6rem;letter-spacing:.5px;text-transform:uppercase;opacity:.65;}
.progress-meta .bar {height:6px;border-radius:4px;background:#2a323d;overflow:hidden;margin-top:4px;}
.progress-meta .bar > div {height:100%;background:linear-gradient(90deg,#6a5acd,#ffb347);width:0%;transition:width .6s ease;}
/* Nav */
.nav-links {display:flex;flex-direction:column;gap:4px;margin-top:10px;}
.nav-links a {text-decoration:none;font-size:.68rem;padding:6px 10px;border:1px solid #252d38;border-radius:10px;color:#c9d2dd;background:#141a22;display:flex;justify-content:space-between;align-items:center;transition:background .15s,border .15s,transform .15s;}
.nav-links a:hover {background:#1d2530;border-color:#2e3642;transform:translateY(-1px);}
.nav-links a span.badge {background:#252f3a;color:#aeb8c4;padding:2px 6px;border-radius:12px;font-size:.55rem;}
.glassy {background:rgba(255,255,255,0.04);border:1px solid #2a3140;padding:.75rem .85rem .9rem;border-radius:16px;box-shadow:0 4px 14px -6px rgba(0,0,0,.55);}
.divider-line {height:1px;background:linear-gradient(90deg,transparent,#2d3642,transparent);margin:.9rem 0 .7rem 0;border:none;}
button[data-baseweb="tab"]:hover { color:__ACCENT__; }
/* Metrics, content styles unchanged below */
.metric-grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:0.75rem; margin:0.75rem 0 1.25rem 0; }
.metric { background:#1f2430; border:1px solid #2d3441; border-radius:10px; padding:0.75rem 0.9rem; }
.metric h4 { font-size:0.70rem; letter-spacing:1px; text-transform:uppercase; color:#8892a0; margin:0 0 4px 0; }
.metric p { font-weight:600; font-size:1.05rem; margin:0; color:#e2e6ee; }
.badge { display:inline-block; padding:2px 8px; border-radius:12px; font-size:0.65rem; font-weight:600; letter-spacing:.5px; margin-right:6px; background:#2d3441; color:#cfd5df; }
.risk-chip { padding:2px 8px; border-radius:12px; font-size:0.65rem; font-weight:600; color:#111; }
.chat-q { background:#252d3a; padding:10px 14px; border-radius:12px; margin-bottom:4px; font-weight:600; }
.chat-a { background:#1d2330; padding:10px 14px; border-left:3px solid __PRIMARY__; border-radius:0 12px 12px 12px; margin-bottom:6px; }
.citation { font-size:0.65rem; opacity:.75; }
.scroll-table { max-height:520px; overflow:auto; border:1px solid #2d3441; border-radius:10px; padding:4px; background:#14191f; }
h2.section-title { position:relative; padding-left:12px; font-size:1.15rem; margin-top:1rem; }
h2.section-title:before { content:""; position:absolute; left:0; top:4px; width:5px; height:70%; background:linear-gradient(180deg,__PRIMARY__,#8f7bff); border-radius:4px; }
.clause-card { border:1px solid #2e3442; border-radius:12px; padding:.65rem .85rem; margin-bottom:.6rem; background:#1a202b; transition: border .15s, background .15s; }
.clause-card:hover { border-color:#3a4354; background:#202836; }
.clause-head { display:flex; gap:.5rem; align-items:center; }
.imp-tag { font-size:.55rem; letter-spacing:.5px; text-transform:uppercase; padding:2px 6px; border-radius:6px; background:#283041; color:#cfd6e1; }
.imp-High { background:#ff4b4b; color:#111; }
.imp-Medium { background:#ffb347; color:#222; }
.imp-Low { background:#455263; color:#d7dee8; }
.rf-card { border:1px solid #3a2a2a; background:#271d1d; border-radius:12px; padding:.7rem .85rem; margin-bottom:.6rem; }
.rf-head { font-weight:600; color:#ffbfbf; }
</style>
"""

GLOBAL_CSS = _CSS_TEMPLATE.replace("__ACCENT__", ACCENT_COLOR).replace("__PRIMARY__", PRIMARY_COLOR)


def sidebar(config: AppConfig):
    st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
    st.sidebar.markdown(
        "<div class='sidebar-brand'><div class='brand-badge'>AI</div><div><div style='font-weight:600;font-size:.85rem;letter-spacing:.5px;color:#fff;'>Legal Doc Explainer</div><div style='font-size:.55rem;color:#8a96a6;'>RAG • Clauses • Risk</div></div></div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("<div class='glassy'>", unsafe_allow_html=True)
    gemini_status = "Gemini" if config.use_gemini else "Local"
    docs_count = len(st.session_state.get('documents', []))
    chunks_count = len(st.session_state.get('chunks', []))
    clause_count = len(st.session_state.get('clauses', []))
    risk_count = len(st.session_state.get('redflags', []))
    progress_pct = 0
    if docs_count:
        stages = [docs_count>0, chunks_count>0, clause_count>0, risk_count>0]
        progress_pct = int((sum(stages)/4)*100)
    st.sidebar.markdown(
        f"<div class='status-grid'>"
        f"<div class='status-pill'><span>Mode</span><span class='value'>{gemini_status}</span></div>"
        f"<div class='status-pill'><span>Embed</span><span class='value'>{config.embed_model.split('/')[-1][:10]}</span></div>"
        f"<div class='status-pill'><span>Docs</span><span class='value'>{docs_count}</span></div>"
        f"<div class='status-pill'><span>Chunks</span><span class='value'>{chunks_count}</span></div>"
        f"<div class='status-pill'><span>Clauses</span><span class='value'>{clause_count}</span></div>"
        f"<div class='status-pill'><span>Risks</span><span class='value'>{risk_count}</span></div>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        f"<div class='progress-cluster'><div class='radial'><div class='radial-ring' style='--p:{progress_pct};'></div><div class='radial-label'>{progress_pct}%</div></div><div class='progress-meta'><h6>Pipeline Progress</h6><div class='bar'><div style='width:{progress_pct}%;'></div></div></div></div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<div style='margin-top:.6rem;font-size:.55rem;letter-spacing:.5px;text-transform:uppercase;opacity:.55;'>Navigate</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        "<div class='nav-links'>"
        "<a href='#overview'><span>Overview</span><span class='badge'>Docs</span></a>"
        "<a href='#clauses'><span>Clauses</span><span class='badge'>Extract</span></a>"
        "<a href='#red-flags'><span>Red Flags</span><span class='badge'>Risk</span></a>"
        "<a href='#ask-questions'><span>Q&A</span><span class='badge'>RAG</span></a>"
        "<a href='#report'><span>Report</span><span class='badge'>PDF</span></a>"
        "</div>",
        unsafe_allow_html=True,
    )
    # Quick info / tips
    st.sidebar.markdown(
        "<div style='margin-top:10px;font-size:.6rem;line-height:1.15;opacity:.6;'>Tip: Run 'Full Analyze' for enriched summaries & risk scoring. Definitions (e.g., 'What is indemnity?') use a fast heuristic path.</div>",
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<hr class='divider-line'>", unsafe_allow_html=True)
    # Use markdown with unsafe_allow_html so styling span is rendered (caption escapes HTML)
    st.sidebar.markdown("<div style='color:#7d8896;font-size:.6rem;margin-top:2px;'>Config locked server-side • No user keys needed • Not legal advice</div>", unsafe_allow_html=True)
    st.sidebar.markdown("<div style='font-size:0.6rem; line-height:1.05; color:#54606e;'>© 2025 Legal Doc AI</div>", unsafe_allow_html=True)
    return {}


def overview_tab(config: AppConfig, documents, summaries):
    if not documents:
        st.info("Upload PDFs and click 'Analyze Documents' to begin.")
        return
    st.markdown("<h2 class='section-title'>Documents</h2>", unsafe_allow_html=True)
    col_wrap = st.columns(min(4, max(1, len(documents))))
    for idx, d in enumerate(documents):
        with col_wrap[idx % len(col_wrap)]:
            st.markdown(f"<div class='metric'><h4>{d.name[:18]}</h4><p>{d.pages} pages</p></div>", unsafe_allow_html=True)
    st.markdown("<h2 class='section-title'>Summaries</h2>", unsafe_allow_html=True)
    for name, data in summaries.items():
        with st.expander(f"📄 {name}", expanded=True):
            bullets = data.get("bullets") or "- (No summary generated yet)"
            st.markdown(bullets)
    if not summaries:
        st.warning("No summaries yet. Quick summaries are generated automatically after upload; run Full Analyze for enriched version.")


def clauses_tab(clauses: List[ClauseResult]):
    if not clauses:
        st.info("No clauses extracted yet.")
        return
    search_col, importance_col = st.columns([2,1])
    with search_col:
        search = st.text_input("Search clauses", placeholder="keyword or type...")
    with importance_col:
        imp_filter = st.multiselect("Importance", ["High","Medium","Low"], default=["High","Medium","Low"], label_visibility="collapsed")
    count = 0
    for c in clauses:
        if search:
            blob = f"{c.clause_type} {c.explanation} {c.snippet} {c.importance}".lower()
            if search.lower() not in blob:
                continue
        if c.importance not in imp_filter:
            continue
        count += 1
        st.markdown(
            f"""
            <div class='clause-card'>
              <div class='clause-head'>
                 <span class='imp-tag imp-{c.importance}'>{c.importance}</span>
                 <strong>{c.clause_type}</strong>
                 <span style='opacity:.55;font-size:.6rem;'>p{c.page}</span>
              </div>
              <div style='margin-top:4px;font-size:.73rem;line-height:1.15;'>{c.explanation}</div>
              <div style='margin-top:6px;font-size:.6rem;opacity:.55;white-space:pre-wrap;'>{c.snippet}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    st.caption(f"{count} clause(s) shown")


def redflags_tab(redflags: List[RedFlagResult], config: AppConfig):
    if not redflags:
        st.info("No red flags above threshold.")
        return
    st.write("Higher scores = higher estimated risk (hybrid heuristic + LLM).")
    for r in redflags:
        if r.confidence >= 80:
            color = RISK_COLORS['High']
        elif r.confidence >= 65:
            color = RISK_COLORS['Medium']
        else:
            color = RISK_COLORS['Low']
        st.markdown(
            f"""
            <div class='rf-card' style='border-left:5px solid {color};'>
              <div class='rf-head'>{r.risk_type} <span style='font-size:.6rem;opacity:.6;'>p{r.page}</span></div>
              <div style='font-size:.7rem; margin-top:4px;'>{r.reason}</div>
              <div style='font-size:.55rem; opacity:.55; margin-top:6px;'>{r.snippet[:360]}</div>
              <div style='margin-top:4px; font-size:.6rem;'>Confidence: <strong>{r.confidence:.0f}</strong></div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def qa_tab(config: AppConfig, qa_chain, qa_history: List[Dict[str, Any]]):
    st.write("Ask document-grounded questions. Answers cite pages.")
    if not qa_chain:
        st.info("Build an index first by running analysis.")
        return
    # Check if vectorstore appears empty (no internal docs)
    try:
        if hasattr(qa_chain.vs, 'docstore') and not getattr(qa_chain.vs.docstore, '_dict', {}):
            st.warning("Index contains 0 chunks. Upload PDFs and click Analyze first.")
    except Exception:
        pass
    with st.form("qa_form", clear_on_submit=True):
        question = st.text_input("Enter your question", placeholder="e.g., Can I terminate early?", key="qa_input")
        colq1, colq2 = st.columns([1,1])
        with colq1:
            submitted = st.form_submit_button("Ask", use_container_width=True)
        with colq2:
            clear_hist = st.form_submit_button("Clear History", use_container_width=True)
    if clear_hist:
        qa_history.clear()
    if submitted and question:
        with st.spinner("Retrieving & generating answer..."):
            try:
                result = qa_chain.ask(question)
                qa_history.append({
                    "q": question,
                    "a": result.get("answer","(no answer)"),
                    "citations": result.get("citations",[]),
                    "confidence": result.get("confidence")
                })
            except Exception as e:
                st.error(f"QA failure: {e}")
    for item in reversed(qa_history):
        st.markdown(f"<div class='chat-q'>Q: {item['q']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='chat-a'>{item['a']}</div>", unsafe_allow_html=True)
        if item.get('confidence') is not None:
            st.markdown(f"<div style='font-size:.55rem;opacity:.6;margin-top:-4px;margin-bottom:2px;'>Confidence: {item['confidence']:.0f}</div>", unsafe_allow_html=True)
        if item['citations']:
            cits = []
            for c in item['citations']:
                raw_snip = c.get('snippet','')[:160]
                snippet = raw_snip.replace('<','&lt;').replace('>','&gt;').replace('"','&quot;')
                cits.append(f"<span class='citation' title=\"{snippet}\">p{c['page']}</span>")
            st.markdown(' '.join(cits), unsafe_allow_html=True)
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)


def report_tab(state):
    if not state.get('documents'):
        st.info("Report will appear after analysis.")
        return
    st.write("Report ready. Use sidebar to export.")
    st.info("Export includes summaries, clauses, red flags, and Q&A trail.")
