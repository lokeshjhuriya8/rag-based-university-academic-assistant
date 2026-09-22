"""Streamlit interface for the incremental University Academic AI Assistant."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from rag import (
    DOCUMENTS_DIR,
    answer,
    collection,
    enqueue_unindexed_documents,
    indexing_in_progress,
    indexing_snapshot,
)

st.set_page_config(
    page_title="University Academic AI Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)


def add_styles() -> None:
    """High contrast only. Native Streamlit controls remain visible and usable."""
    st.markdown(
        """
        <style>
        .stApp { background: #f7f9fc; color: #142b3d; }
        .block-container { max-width: 1120px; padding-top: 1.8rem; }
        h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] { color: #142b3d !important; }
        h1 { color: #0b2d4d !important; }
        [data-testid="stSidebar"] { background: #ffffff; border-right: 1px solid #a9bac7; }
        [data-testid="stSidebar"] * { color: #142b3d; opacity: 1; }
        [data-testid="stSidebar"] .stButton > button,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
        .stFormSubmitButton > button {
            background: #075b70 !important; color: #ffffff !important;
            border: 1px solid #075b70 !important; border-radius: 5px !important;
            font-weight: 700 !important; opacity: 1 !important;
        }
        [data-testid="stSidebar"] .stButton > button *,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button *,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] svg,
        .stFormSubmitButton > button * {
            color: #ffffff !important; fill: currentColor !important; opacity: 1 !important;
        }
        [data-testid="stSidebar"] .stButton > button:hover,
        [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button:hover,
        .stFormSubmitButton > button:hover { background: #063f4e !important; border-color: #063f4e !important; }
        button[kind="header"],
        button[data-testid="stSidebarCollapseButton"],
        [data-testid="stSidebarCollapsedControl"] button {
            background: #0b2d4d !important; color: #ffffff !important;
            border: 2px solid #075b70 !important; border-radius: 5px !important; opacity: 1 !important;
        }
        button[kind="header"] svg,
        button[data-testid="stSidebarCollapseButton"] svg,
        [data-testid="stSidebarCollapsedControl"] button svg {
            color: #ffffff !important; fill: currentColor !important; opacity: 1 !important;
        }
        .stTextInput input { color: #142b3d !important; background: #ffffff !important;
            border: 1px solid #56758a !important; border-radius: 5px !important; }
        .stTextInput input::placeholder { color: #39586c !important; opacity: 1 !important; }
        [data-testid="stExpander"] { background: #ffffff; border: 1px solid #c4d1da; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def documents() -> list[Path]:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(DOCUMENTS_DIR.glob("*.pdf"), key=lambda item: item.name.lower())


def chunk_count() -> int:
    try:
        return collection().count()
    except Exception:
        return 0


def show_sources(sources: list[dict[str, object]]) -> None:
    with st.expander("Sources used for this answer"):
        for source in sources:
            st.markdown(f"**{source['source']} — page {source['page']}**")
            st.write(source["text"])


add_styles()
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
if "conversation" not in st.session_state:
    st.session_state.conversation = []

pdfs = documents()
# This returns immediately. It only queues PDFs not recorded with their current
# SHA-256 fingerprint; all extraction and embedding happens in the worker.
enqueue_unindexed_documents(pdfs)
jobs = indexing_snapshot()
active_jobs = [job for job in jobs if job.get("state") in {"queued", "indexing"}]

with st.sidebar:
    st.title("Knowledge Base")
    st.write("Upload official university PDF documents.")
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    if uploaded_files:
        for uploaded in uploaded_files:
            safe_name = uploaded.name.replace("/", "_").replace("\\", "_")
            (DOCUMENTS_DIR / safe_name).write_bytes(uploaded.getbuffer())
        pdfs = documents()
        queued = enqueue_unindexed_documents(pdfs)
        if queued:
            st.success(f"{queued} PDF(s) added to the indexing queue.")
        else:
            st.info("These PDF versions are already indexed or queued.")

    if st.button("Check and index new PDFs", use_container_width=True):
        queued = enqueue_unindexed_documents(documents())
        if queued:
            st.success(f"Started background indexing for {queued} PDF(s).")
        else:
            st.info("No new or changed PDFs need indexing.")

    if st.button("Refresh indexing status", use_container_width=True):
        st.rerun()

    st.divider()
    st.subheader("Status")
    st.write(f"Documents: **{len(pdfs)}**")
    st.write(f"Indexed chunks: **{chunk_count()}**")
    if active_jobs:
        st.warning("New PDFs are being indexed. Existing documents remain searchable.")
        for job in active_jobs:
            st.write(f"**{job['filename']}**")
            st.caption(str(job.get("message", "Indexing…")))
    else:
        completed = [job for job in jobs if job.get("state") == "complete"]
        failed = [job for job in jobs if job.get("state") == "failed"]
        if completed:
            st.success(f"{completed[-1]['filename']} is available in the knowledge base.")
        if failed:
            st.error(f"Indexing failed for {failed[-1]['filename']}: {failed[-1].get('message', 'Unknown error')}")

st.title("University Academic AI Assistant")
st.write("Ask about university policies, attendance, examinations, courses, or schedules. Every answer is based on retrieved PDF passages.")

summary_a, summary_b, summary_c = st.columns(3)
summary_a.write(f"**Documents**\n\n{len(pdfs)}")
summary_b.write(f"**Indexed chunks**\n\n{chunk_count()}")
summary_c.write("**Answer mode**\n\nRetrieved sources only")

st.divider()
st.subheader("Ask a question")
with st.form("question_form", clear_on_submit=True):
    question = st.text_input("Your question", placeholder="Example: What is the minimum attendance requirement?")
    submitted = st.form_submit_button("Search and answer")

if submitted:
    if not question.strip():
        st.warning("Please type a question first.")
    else:
        history: list[dict[str, str]] = []
        for previous in st.session_state.conversation:
            history.append({"role": "user", "content": previous["question"]})
            history.append({"role": "assistant", "content": previous["answer"]})
        try:
            if indexing_in_progress():
                st.info("A new document is currently being indexed. This answer uses documents already available in the knowledge base.")
            with st.spinner("Searching indexed university documents…"):
                response, sources = answer(question.strip(), history)
            st.session_state.conversation.append({"question": question.strip(), "answer": response, "sources": sources})
        except Exception as error:
            st.error(str(error))

if st.session_state.conversation:
    st.divider()
    st.subheader("Answers")
    for item in reversed(st.session_state.conversation):
        st.markdown(f"**Question:** {item['question']}")
        st.markdown("**Answer:**")
        st.markdown(item["answer"])
        show_sources(item["sources"])
        st.divider()

with st.expander("Documents currently in the library"):
    if pdfs:
        for pdf in pdfs:
            st.write(f"• {pdf.name}")
    else:
        st.write("No PDFs have been uploaded yet.")
