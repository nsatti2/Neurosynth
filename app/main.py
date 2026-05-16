import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

import streamlit as st
import tempfile

from src.pdf_parser import parse_paper
from src.vector_store import add_chunks, list_papers
from src.rag_pipeline import answer_question, extract_methodology


st.set_page_config(
    page_title="NeuroSynth",
    page_icon="🧠",
    layout="wide",
)

# Animated header
st.markdown("""
<style>
@keyframes pulse { 0%, 100% { opacity: 0.2; r: 3; } 50% { opacity: 0.8; r: 4.5; } }
@keyframes dash { from { stroke-dashoffset: 20; } to { stroke-dashoffset: 0; } }
.n1 { animation: pulse 2.1s ease-in-out infinite; }
.n2 { animation: pulse 2.1s ease-in-out infinite 0.3s; }
.n3 { animation: pulse 2.1s ease-in-out infinite 0.6s; }
.n4 { animation: pulse 2.1s ease-in-out infinite 0.9s; }
.n5 { animation: pulse 2.1s ease-in-out infinite 1.2s; }
.n6 { animation: pulse 2.1s ease-in-out infinite 1.5s; }
.n7 { animation: pulse 2.1s ease-in-out infinite 1.8s; }
.edge { stroke-dasharray: 4 3; animation: dash 1.5s linear infinite; }
.header-graphic { position: relative; overflow: hidden; margin-bottom: 1.5rem; padding: 24px 28px 18px; background: #0D0D0D; border: 0.5px solid #1E1E1E; border-radius: 8px; }
.header-logo { font-family: 'Syne', sans-serif; font-size: 24px; font-weight: 700; color: #F5F0E8; letter-spacing: -0.03em; }
.header-logo span { color: #5DCAA5; }
.header-tag { font-family: 'DM Mono', monospace; font-size: 10px; color: #444; margin-top: 3px; }
.neural-svg { position: absolute; right: 0; top: 0; bottom: 0; opacity: 0.55; }
</style>
<div class="header-graphic">
  <div style="position:relative;z-index:2;">
    <div class="header-logo">Neuro<span>Synth</span></div>
    <div class="header-tag">research intelligence tool · v1.0</div>
  </div>
  <svg class="neural-svg" width="220" height="100" viewBox="0 0 220 100">
    <line x1="20" y1="15" x2="80" y2="35" stroke="#1D9E75" stroke-width="0.5" class="edge"/>
    <line x1="20" y1="50" x2="80" y2="35" stroke="#1D9E75" stroke-width="0.5" class="edge"/>
    <line x1="20" y1="50" x2="80" y2="70" stroke="#1D9E75" stroke-width="0.5" class="edge"/>
    <line x1="20" y1="85" x2="80" y2="70" stroke="#1D9E75" stroke-width="0.5" class="edge"/>
    <line x1="80" y1="35" x2="150" y2="25" stroke="#1D9E75" stroke-width="0.5" class="edge"/>
    <line x1="80" y1="35" x2="150" y2="55" stroke="#1D9E75" stroke-width="0.5" class="edge"/>
    <line x1="80" y1="70" x2="150" y2="55" stroke="#1D9E75" stroke-width="0.5" class="edge"/>
    <line x1="80" y1="70" x2="150" y2="80" stroke="#1D9E75" stroke-width="0.5" class="edge"/>
    <line x1="150" y1="25" x2="210" y2="50" stroke="#1D9E75" stroke-width="0.5" class="edge"/>
    <line x1="150" y1="55" x2="210" y2="50" stroke="#1D9E75" stroke-width="0.5" class="edge"/>
    <line x1="150" y1="80" x2="210" y2="50" stroke="#1D9E75" stroke-width="0.5" class="edge"/>
    <circle cx="20" cy="15" r="3" fill="#5DCAA5" class="n1"/>
    <circle cx="20" cy="50" r="3" fill="#5DCAA5" class="n2"/>
    <circle cx="20" cy="85" r="3" fill="#5DCAA5" class="n3"/>
    <circle cx="80" cy="35" r="3" fill="#5DCAA5" class="n4"/>
    <circle cx="80" cy="70" r="3" fill="#5DCAA5" class="n5"/>
    <circle cx="150" cy="25" r="3" fill="#5DCAA5" class="n6"/>
    <circle cx="150" cy="55" r="3" fill="#5DCAA5" class="n7"/>
    <circle cx="150" cy="80" r="3" fill="#5DCAA5" class="n1"/>
    <circle cx="210" cy="50" r="4" fill="#5DCAA5" class="n4"/>
  </svg>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:ital,wght@0,400;0,500;1,400&family=Syne:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {
    background-color: #0D0D0D !important;
    color: #C8C2B8 !important;
    font-family: 'DM Mono', monospace !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem !important; max-width: 100% !important; }
[data-testid="stSidebar"] { background: #0A0A0A !important; border-right: 0.5px solid #1E1E1E !important; }
[data-testid="stSidebar"] * { color: #C8C2B8 !important; font-family: 'DM Mono', monospace !important; }
[data-testid="stSidebar"] .stMarkdown h2 { font-family: 'Syne', sans-serif !important; font-size: 18px !important; font-weight: 700 !important; color: #F5F0E8 !important; letter-spacing: -0.02em !important; }
[data-testid="stSidebar"] hr { border-color: #1E1E1E !important; }
[data-testid="stSidebar"] .stMarkdown strong { color: #888 !important; font-size: 10px !important; letter-spacing: 0.08em !important; text-transform: uppercase !important; }
[data-testid="stFileUploader"] { background: #111 !important; border: 0.5px dashed #2A2A2A !important; border-radius: 6px !important; }
.stButton > button { background: transparent !important; border: 0.5px solid #2A2A2A !important; color: #888 !important; font-family: 'DM Mono', monospace !important; font-size: 11px !important; border-radius: 4px !important; }
.stButton > button:hover { border-color: #5DCAA5 !important; color: #5DCAA5 !important; background: transparent !important; }
.stButton > button[kind="primary"] { border-color: #5DCAA5 !important; color: #5DCAA5 !important; }
[data-testid="stMultiSelect"] > div { background: #111 !important; border: 0.5px solid #2A2A2A !important; border-radius: 4px !important; }
.stMultiSelect span { background: #1A2E26 !important; color: #5DCAA5 !important; font-size: 10px !important; border-radius: 3px !important; }
.stTabs [data-baseweb="tab-list"] { background: transparent !important; border-bottom: 0.5px solid #1E1E1E !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #444 !important; font-family: 'DM Mono', monospace !important; font-size: 12px !important; padding: 10px 20px !important; }
.stTabs [aria-selected="true"] { color: #F5F0E8 !important; border-bottom: 1.5px solid #5DCAA5 !important; background: transparent !important; }
[data-testid="stChatMessage"] { background: #111 !important; border: 0.5px solid #1E1E1E !important; border-radius: 6px !important; margin-bottom: 8px !important; }
[data-testid="stChatMessage"] p { color: #C8C2B8 !important; font-family: 'DM Mono', monospace !important; font-size: 13px !important; line-height: 1.7 !important; }
[data-testid="stChatInput"] { background: #111 !important; border: 0.5px solid #2A2A2A !important; border-radius: 6px !important; }
[data-testid="stChatInput"] textarea { background: transparent !important; color: #C8C2B8 !important; font-family: 'DM Mono', monospace !important; font-size: 13px !important; }
h1, h2, h3 { font-family: 'Syne', sans-serif !important; color: #F5F0E8 !important; letter-spacing: -0.02em !important; }
[data-testid="stExpander"] { background: #0D0D0D !important; border: 0.5px solid #1E1E1E !important; border-radius: 4px !important; }
[data-testid="stExpander"] summary { color: #555 !important; font-size: 11px !important; }
[data-testid="stSelectbox"] > div { background: #111 !important; border: 0.5px solid #2A2A2A !important; border-radius: 4px !important; }
[data-testid="stMetric"] { background: #111 !important; border: 0.5px solid #1E1E1E !important; border-radius: 6px !important; padding: 12px 16px !important; }
[data-testid="stMetricLabel"] { color: #555 !important; font-size: 10px !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; }
[data-testid="stMetricValue"] { color: #F5F0E8 !important; font-size: 16px !important; }
[data-testid="stInfo"] { background: #0D1F18 !important; border-color: #1A3D2B !important; color: #5DCAA5 !important; }
[data-testid="stError"] { background: #1F0D0D !important; border-color: #5A1A1A !important; color: #E87070 !important; }
[data-testid="stSuccess"] { background: #0D1F18 !important; border-color: #5DCAA5 !important; color: #5DCAA5 !important; }
hr { border-color: #1E1E1E !important; }
.stCaption { color: #444 !important; font-size: 11px !important; }
.source-card { background: #0A0A0A; border: 0.5px solid #1E1E1E; border-radius: 4px; padding: 10px 14px; margin: 5px 0; font-family: 'DM Mono', monospace; font-size: 11px; color: #888; line-height: 1.6; }
.source-card b { color: #5DCAA5; font-weight: 500; }
.cite-badge { background: #1A2E26; color: #5DCAA5; padding: 1px 7px; border-radius: 3px; font-size: 10px; font-weight: 500; margin-left: 6px; font-family: 'DM Mono', monospace; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🧠 NeuroSynth")
    st.caption("Research Intelligence Tool")

    st.divider()

    # PDF upload
    st.markdown("**Upload papers**")
    uploaded_files = st.file_uploader(
        "Drop PDFs here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            paper_id = Path(uploaded_file.name).stem.replace(" ", "_")
            if st.button(f"Index: {uploaded_file.name[:30]}", key=f"idx_{paper_id}"):
                with st.spinner(f"Parsing {uploaded_file.name}..."):
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        tmp.write(uploaded_file.read())
                        tmp_path = tmp.name
                    chunks = parse_paper(tmp_path, paper_id=paper_id, original_name=uploaded_file.name)
                    add_chunks(chunks)
                    os.unlink(tmp_path)
                st.success(f"Indexed {len(chunks)} chunks")

    st.divider()

    # Paper list
    st.markdown("**Loaded papers**")
    available_papers = list_papers()
if available_papers:
    colors = ["#5DCAA5", "#378ADD", "#D85A30", "#7F77DD", "#EF9F27"]
    cards_html = ""
    for i, paper in enumerate(available_papers):
        dot_color = colors[i % len(colors)]
        short = paper.replace("_", " ")[:35]
        cards_html += f"""
        <div style="background:#111;border:0.5px solid #1E1E1E;border-radius:6px;
             padding:8px 12px;margin-bottom:6px;display:flex;align-items:center;gap:10px;
             font-family:'DM Mono',monospace;cursor:pointer;">
          <div style="width:7px;height:7px;border-radius:50%;background:{dot_color};flex-shrink:0;"></div>
          <div style="flex:1;min-width:0;">
            <div style="font-size:11px;color:#C8C2B8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{short}</div>
          </div>
        </div>"""
    st.markdown(cards_html, unsafe_allow_html=True)

    selected_papers = st.multiselect(
        "Filter",
        options=available_papers,
        default=available_papers,
        label_visibility="collapsed",
    )
else:
    selected_papers = []
    st.caption("No papers indexed yet")


# ── Main tabs ──────────────────────────────────────────────────────────────────

tab_ask, tab_methods, tab_gaps = st.tabs(["💬 Ask papers", "🔬 Extract methods", "🔍 Find gaps"])


# Tab 1: Ask questions
with tab_ask:
    st.markdown("### Ask your papers")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📄 {len(msg['sources'])} sources cited"):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="source-card">'
                            f'<b>{src["source"]}</b> · {src["section"]} '
                            f'<span class="cite-badge">score: {src["relevance_score"]}</span><br>'
                            f'<small>{src["text"][:200]}...</small>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    if prompt := st.chat_input("Ask anything about your papers..."):
        if not os.environ.get("GROQ_API_KEY"):
            st.error("Groq API key not found. Check your .env file.")
        elif not selected_papers:
            st.error("Upload and index at least one paper first.")
        else:
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching papers..."):
                    result = answer_question(
                        prompt,
                        history=[
                            {"role": m["role"], "content": m["content"]}
                            for m in st.session_state.messages
                        ],
                        paper_ids=selected_papers if selected_papers != available_papers else None,
)
                st.markdown(result["answer"])
                if result["sources"]:
                    with st.expander(f"📄 {len(result['sources'])} sources cited"):
                        for src in result["sources"]:
                            st.markdown(
                                f'<div class="source-card">'
                                f'<b>{src["source"]}</b> · {src["section"]} '
                                f'<span class="cite-badge">score: {src["relevance_score"]}</span><br>'
                                f'<small>{src["text"][:200]}...</small>'
                                f'</div>',
                                unsafe_allow_html=True,
                            )

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result["sources"],
            })


# Tab 2: Extract methodology
with tab_methods:
    st.markdown("### Extract methodology")
    st.caption("Automatically extract study design, sample sizes, brain regions, and statistical methods.")

    if not available_papers:
        st.info("Index some papers first using the sidebar.")
    else:
        paper_to_extract = st.selectbox("Select a paper", available_papers)
        if st.button("Extract methodology", type="primary"):
            if not os.environ.get("GROQ_API_KEY"):
                st.error("Groq API key not found. Check your .env file.")
            else:
                with st.spinner("Extracting..."):
                    methods = extract_methodology(paper_to_extract)

                if "error" in methods:
                    st.error(methods["error"])
                else:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Study design", methods.get("study_design", "—"))
                        st.metric("N participants", methods.get("n_participants") or "—")
                        st.metric("Species", methods.get("species", "—"))
                    with col2:
                        st.markdown("**Brain regions**")
                        regions = methods.get("brain_regions") or []
                        st.markdown(", ".join(regions) if regions else "—")

                        st.markdown("**Statistical methods**")
                        stats = methods.get("statistical_methods") or []
                        st.markdown(", ".join(stats) if stats else "—")

                        st.markdown("**Key measures**")
                        measures = methods.get("key_measures") or []
                        st.markdown(", ".join(measures) if measures else "—")


# Tab 3: Gap finder (V2 feature stub)
with tab_gaps:
    st.markdown("### Find research gaps")
    st.info("🚧 Coming in V2 — will analyze your corpus and surface unanswered questions and contradictions between papers.")
    st.markdown("""
**Planned features:**
- Contradiction detection across papers
- Unanswered questions surfaced from discussion sections
- Research gap summary across your full corpus
    """)
