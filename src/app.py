"""
app.py — Streamlit entry point for Chat-with-PDF.

Run with:
    streamlit run src/app.py

Features:
  - Upload a PDF → auto-embed → stored in usefulcomponents/vectorstores/<pdf_name>/
  - Sidebar shows all previously processed PDFs (reload without re-processing)
  - On question: page numbers appear immediately, then the AI answer loads
  - If the LLM fails, page numbers are still shown so the user isn't left empty-handed
  - Manage stored PDFs: delete individual vector stores from the sidebar
"""

import os
import sys
import tempfile
import warnings

# Suppress harmless LibreSSL / urllib3 warning
warnings.filterwarnings("ignore", message=".*urllib3.*", category=Warning)

# Ensure src/ is importable regardless of where streamlit is launched from
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
from dotenv import load_dotenv

from config import (
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_EMBEDDING_MODEL,
    MODEL_CONFIG,
)
from embedding_model import get_embedding_model
from ingestion import ingest_pdf
from vectore_store import get_vectorstore, list_vectorstores, delete_vectorstore
from llm import get_llm
from retriever import retrieve_pages, get_answer


load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Chat with your PDF",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


# css
st.markdown(
    """
<style>
    .stApp { background-color: #0f1117; }
    .main-title { font-size: 2rem; font-weight: 700; color: #e2e8f0; }
    .page-badge {
        display: inline-block;
        background: #1e3a5f;
        color: #60a5fa;
        border: 1px solid #2563eb;
        border-radius: 6px;
        padding: 2px 10px;
        margin: 2px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .pages-header {
        color: #60a5fa;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 4px;
    }
    .answer-block {
        background: #1a1f2e;
        border-left: 3px solid #2563eb;
        border-radius: 4px;
        padding: 12px 16px;
        margin-top: 8px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state defaults ────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []  # [{role, content}]
if "active_pdfs" not in st.session_state:
    st.session_state.active_pdfs = []  # names of the currently loaded PDFs
if "embedding_model" not in st.session_state:
    st.session_state.embedding_model = None  # cached embedding model instance

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📄 Chat with your PDF")
    st.divider()

    # ── 1. Upload new PDF ──────────────────────────────────────────────────────
    st.markdown("### Upload PDF(s)")
    uploaded_files = st.file_uploader(
        "Choose PDF file(s)",
        type=["pdf"],
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    if uploaded_files:
        if st.button(" Process & Embed All", use_container_width=True, type="primary"):
            new_pdfs = []
            for uploaded_file in uploaded_files:
                pdf_name = uploaded_file.name
                new_pdfs.append(pdf_name)

                with st.spinner(f"Extracting text from **{pdf_name}**…"):
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=".pdf"
                    ) as tmp:
                        tmp.write(uploaded_file.getvalue())
                        tmp_path = tmp.name

                    try:
                        chunks = ingest_pdf(tmp_path, pdf_name=pdf_name)
                    finally:
                        os.unlink(tmp_path)

                    if not chunks:
                        st.error(
                            f"⚠️ No text could be extracted from {pdf_name}. The PDF may be empty or corrupted."
                        )
                        continue

                    st.success(f" Extracted **{len(chunks)} chunks** from *{pdf_name}*")

                with st.spinner(f"Building vector store for {pdf_name}…"):
                    # Load (and cache) the embedding model once per session
                    if st.session_state.embedding_model is None:
                        st.session_state.embedding_model = get_embedding_model(
                            DEFAULT_EMBEDDING_MODEL
                        )

                    get_vectorstore(
                        pdf_name=pdf_name,
                        documents=chunks,
                        db_type="chroma",
                        embedding_model=st.session_state.embedding_model,
                    )
                    st.success(f" Vector store saved for *{pdf_name}*")

            if uploaded_files:
                st.session_state.active_pdfs = new_pdfs
                st.session_state.messages = []  # reset chat for new docs
                st.rerun()

    st.divider()

    # ── 2. Load previously processed PDFs ────────────────────────────────────
    st.markdown("### Previously Processed PDFs")
    existing_stores = list_vectorstores()

    if existing_stores:
        selected = st.multiselect(
            "Load stored PDFs",
            options=existing_stores,
            default=[p for p in st.session_state.active_pdfs if p in existing_stores],
            label_visibility="collapsed",
        )
        col1, col2 = st.columns([2, 1])
        with col1:
            st.divider()
            if st.button("📂 Load Selected", use_container_width=True):
                st.session_state.active_pdfs = selected
                st.session_state.messages = []
                st.rerun()
        with col2:
            st.divider()
            if st.button("🗑️ Delete", use_container_width=True):
                if selected:
                    for s in selected:
                        delete_vectorstore(s)
                        if s in st.session_state.active_pdfs:
                            st.session_state.active_pdfs.remove(s)
                    st.session_state.messages = []
                    st.rerun()
    else:
        st.info("No PDFs processed yet.")

    st.divider()

    # ── 3. Model configuration ─────────────────────────────────────────────────
    st.markdown("### Model Settings")
    llm_provider = st.selectbox(
        "LLM Provider", ["Groq", "HuggingFace", "Google"], key="llm_provider"
    )
    if llm_provider == "Groq":
        llm_model = st.selectbox(
            "Model",
            MODEL_CONFIG["Groq"],
            key="llm_model",
        )
    elif llm_provider == "Google":
        llm_model = st.selectbox("Model", MODEL_CONFIG["Google"], key="llm_model")
    else:
        llm_model = st.selectbox(
            "HuggingFace Repo ID",
            MODEL_CONFIG["HuggingFace_models"],
            key="llm_model_hf",
        )

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">📄 Chat with your PDF</p>', unsafe_allow_html=True)

if st.session_state.active_pdfs:
    pdfs_str = ", ".join([f"`{p}`" for p in st.session_state.active_pdfs])
    st.markdown(
        f"**Active PDF(s):** {pdfs_str} — ask anything below.",
        help="Change / delete via the sidebar.",
    )
else:
    st.info(
        "👈 Upload PDF(s) or load previously processed ones from the sidebar to begin."
    )
    st.stop()

st.divider()

# ── Chat history ──────────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"], unsafe_allow_html=True)

# ── Chat input ────────────────────────────────────────────────────────────────
user_query = st.chat_input("Ask a question about your PDF…")

if user_query:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})

    # Load embedding model (cached in session)
    if st.session_state.embedding_model is None:
        with st.spinner("Loading embedding model…"):
            st.session_state.embedding_model = get_embedding_model(
                DEFAULT_EMBEDDING_MODEL
            )

    # Load vector stores
    try:
        vector_stores = []
        for pdf_name in st.session_state.active_pdfs:
            vs = get_vectorstore(
                pdf_name=pdf_name,
                documents=None,
                db_type="chroma",
                embedding_model=st.session_state.embedding_model,
            )
            vector_stores.append(vs)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    with st.chat_message("assistant"):
        response_container = st.empty()

        # ── STEP 1: Showing page numbers immediately ──────────────────────────────
        pages = retrieve_pages(user_query, vector_stores)

        if pages:
            pages_html = "".join(f'<span class="page-badge">{p}</span>' for p in pages)
            step1_html = (
                f'<p class="pages-header">📑 Relevant pages:</p>'
                f"{pages_html}<br><br>"
                f'<span style="color:#94a3b8; font-size:0.85rem;">⏳ Generating AI answer…</span>'
            )
        else:
            step1_html = '<span style="color:#94a3b8; font-size:0.85rem;">🔍 Searching… generating answer…</span>'

        response_container.markdown(step1_html, unsafe_allow_html=True)

        # ── STEP 2: Run LLM and get full answer ────────────────────────────────
        answer, answer_pages = get_answer(
            query=user_query,
            vector_stores=vector_stores,
            llm=get_llm(
                provider=llm_provider.lower(),
                model_name=(llm_model.lower()),
            ),
        )

        # Merge page sets (retriever + QA chain may differ slightly)
        all_pages = sorted(set(pages) | set(answer_pages))

        if all_pages:
            pages_html = "".join(
                f'<span class="page-badge">{p}</span>' for p in all_pages
            )
            citation_block = f'<p class="pages-header">📑 Sources:</p>{pages_html}'
        else:
            citation_block = '<p style="color:#64748b; font-size:0.85rem;">*(No source citations found)*</p>'

        final_html = f"{citation_block}" f'<div class="answer-block">{answer}</div>'

        response_container.markdown(final_html, unsafe_allow_html=True)

    pages_text = (
        f"**📑 Sources: {', '.join(map(str, all_pages))}**\n\n" if all_pages else ""
    )
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": pages_text + answer,
        }
    )
