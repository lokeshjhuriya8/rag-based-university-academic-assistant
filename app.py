"""Streamlit interface for the incremental University Academic AI Assistant."""
from __future__ import annotations

import hashlib
from pathlib import Path

import streamlit as st

from rag import (DOCUMENTS_DIR, answer, delete_document, document_inventory,
                 document_metrics, enqueue_unindexed_documents, generate_quiz,
                 indexing_in_progress, indexing_snapshot)

st.set_page_config(page_title="University Academic AI Assistant", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")


def add_styles() -> None:
    st.markdown("""<style>
    :root { --ink:#142b3d; --muted:#3f5b6d; --panel:#ffffff; --border:#b9c9d3; }
    .stApp { background:#f7f9fc; color:var(--ink); } .block-container { max-width:1120px; padding-top:1.8rem; }
    /* Leave ordinary Streamlit text to inherit its component-specific colours. */
    h1,h2,h3 { color:#0b2d4d !important; }
    [data-testid="stSidebar"] { background:var(--panel); border-right:1px solid #a9bac7; }
    [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] label, [data-testid="stSidebar"] [data-testid="stWidgetLabel"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color:var(--ink) !important; opacity:1 !important; }
    .stButton > button,.stFormSubmitButton > button,[data-testid="stFileUploaderDropzone"] button { background:#075b70 !important; color:#fff !important; border:1px solid #075b70 !important; border-radius:5px !important; font-weight:700 !important; opacity:1 !important; }
    .stButton > button *, .stFormSubmitButton > button *, [data-testid="stFileUploaderDropzone"] button *, [data-testid="stFileUploaderDropzone"] button svg { color:#fff !important; fill:currentColor !important; opacity:1 !important; }
    .stButton > button:hover,.stFormSubmitButton > button:hover,[data-testid="stFileUploaderDropzone"] button:hover { background:#063f4e !important; border-color:#063f4e !important; }
    button[kind="header"],button[data-testid="stSidebarCollapseButton"],[data-testid="stSidebarCollapsedControl"] button { background:#0b2d4d !important; color:#fff !important; border:2px solid #075b70 !important; border-radius:5px !important; opacity:1 !important; }
    button[kind="header"] svg,button[data-testid="stSidebarCollapseButton"] svg,[data-testid="stSidebarCollapsedControl"] button svg { color:#fff !important; fill:currentColor !important; opacity:1 !important; }
    .stTextInput input { color:var(--ink) !important; background:var(--panel) !important; border:1px solid #56758a !important; border-radius:5px !important; }
    [data-testid="stExpander"] { background:var(--panel); border:1px solid #c4d1da; }
    [data-testid="stMetric"] { background:var(--panel); border:1px solid #c4d1da; border-radius:8px; padding:.7rem; }
    /* These widgets have their own theme tokens, so set foregrounds on the light cards explicitly. */
    [data-testid="stMetricLabel"], [data-testid="stMetricLabel"] *,
    [data-testid="stMetricValue"], [data-testid="stMetricValue"] * { color:var(--ink) !important; opacity:1 !important; }
    [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
    [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * { color:var(--muted) !important; opacity:1 !important; }
    [data-testid="stTextInput"] [data-testid="stWidgetLabel"], [data-testid="stTextInput"] [data-testid="stWidgetLabel"] *,
    [data-testid="stSelectbox"] [data-testid="stWidgetLabel"], [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] *,
    [data-testid="stRadio"] [data-testid="stWidgetLabel"], [data-testid="stRadio"] [data-testid="stWidgetLabel"] * { color:var(--ink) !important; }
    [data-testid="stRadio"] [data-testid="stRadioOption"],
    [data-testid="stRadio"] [data-testid="stRadioOption"] *,
    [data-testid="stRadio"] label { color:var(--ink) !important; opacity:1 !important; }
    /* Selectbox values are rendered in an internal input by current Streamlit versions. */
    [data-testid="stSelectbox"] input, [data-testid="stSelectbox"] [role="combobox"] {
        color:var(--ink) !important; -webkit-text-fill-color:var(--ink) !important;
        background-color:var(--panel) !important; caret-color:var(--ink) !important; opacity:1 !important;
    }
    [data-testid="stSelectbox"] button, [data-testid="stSelectboxVirtualDropdown"] [role="option"],
    [data-testid="stSelectboxVirtualDropdown"] [role="option"] * { color:var(--ink) !important; opacity:1 !important; }
    [data-testid="stSelectbox"] svg { fill:var(--ink) !important; }
    [data-testid="stExpander"] summary, [data-testid="stExpander"] summary *,
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"],
    [data-testid="stExpander"] [data-testid="stMarkdownContainer"] * { color:var(--ink) !important; opacity:1 !important; }
    /* Uploaded-file rows are separate Streamlit components, not ordinary sidebar text. */
    [data-testid="stFileUploaderFile"], [data-testid="stFileUploaderFile"] *,
    [data-testid="stFileUploaderFileName"], [data-testid="stFileUploaderFileName"] *,
    [data-testid="stFileUploaderFileData"], [data-testid="stFileUploaderFileData"] *,
    [data-testid="stFileUploaderFile"] svg { color:var(--ink) !important; fill:var(--ink) !important; opacity:1 !important; }
    /* Current and legacy Streamlit alerts need their own foregrounds. */
    [data-testid="stAlert"], .stAlertContainer { border-radius:6px; }
    [data-testid="stAlert"] *, .stAlertContainer * { opacity:1 !important; }
    [data-testid="stAlert"] [data-testid="stMarkdownContainer"],
    .stAlertContainer [data-testid="stMarkdownContainer"], .stAlertContainer p { color:inherit !important; }
    [data-testid="stAlert"] svg, .stAlertContainer svg { fill:currentColor !important; }
    .stAlertContainer:has([data-testid="stAlertContentSuccess"]) { background:#eaf7ef !important; border:1px solid #74b98b !important; color:#135c32 !important; }
    .stAlertContainer:has([data-testid="stAlertContentWarning"]) { background:#fff7df !important; border:1px solid #d7a72c !important; color:#694d00 !important; }
    .stAlertContainer:has([data-testid="stAlertContentError"]) { background:#fdecec !important; border:1px solid #d58989 !important; color:#8a1f1f !important; }
    .stAlertContainer:has([data-testid="stAlertContentInfo"]) { background:#eef5fb !important; border:1px solid #8db5d3 !important; color:#164d72 !important; }
    </style>""", unsafe_allow_html=True)


def documents() -> list[Path]:
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(DOCUMENTS_DIR.glob("*.pdf"), key=lambda item: item.name.lower())


def file_size(size_bytes: int) -> str:
    return f"{size_bytes / 1024:.1f} KB" if size_bytes < 1024 * 1024 else f"{size_bytes / (1024 * 1024):.2f} MB"


def show_sources(sources: list[dict[str, object]]) -> None:
    with st.expander("Sources used for this answer"):
        for source in sources:
            st.markdown(f"**{source['source']} — page {source['page']}**")
            st.write(source["text"])


def show_job_notice() -> None:
    active = [job for job in indexing_snapshot() if job.get("state") in {"queued", "indexing"}]
    if active:
        st.info("New PDFs are indexing in the background. Existing indexed documents remain searchable.")
        for job in active:
            st.caption(f"{job['filename']}: {job.get('message', 'Indexing…')}")


def upload_documents() -> None:
    with st.sidebar:
        st.title("Knowledge Base")
        st.write("Upload approved university PDF documents.")
        uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
        if uploaded_files:
            added = 0
            for uploaded in uploaded_files:
                safe_name = Path(uploaded.name).name
                payload = uploaded.getvalue()
                destination = DOCUMENTS_DIR / safe_name
                if not destination.exists() or hashlib.sha256(destination.read_bytes()).digest() != hashlib.sha256(payload).digest():
                    destination.write_bytes(payload)
                    added += 1
            queued = enqueue_unindexed_documents(documents())
            if queued:
                st.success(f"{queued} PDF(s) added to the indexing queue.")
            elif added:
                st.info("The PDFs were saved; matching versions are already indexed or queued.")
        if st.button("Check and index new PDFs", use_container_width=True):
            queued = enqueue_unindexed_documents(documents())
            st.success(f"Started background indexing for {queued} PDF(s).") if queued else st.info("No new or changed PDFs need indexing.")
        if st.button("Refresh indexing status", use_container_width=True):
            st.rerun()
        st.divider()
        metrics = document_metrics()
        st.subheader("Status")
        st.write(f"Documents: **{metrics['total_documents']}**")
        st.write(f"Indexed chunks: **{metrics['total_chunks']}**")
        show_job_notice()


def home_page(metrics: dict[str, int]) -> None:
    st.title("University Academic AI Assistant")
    st.write("Ask about university policies, attendance, examinations, courses, or schedules. Every answer is based on retrieved PDF passages.")
    first, second, third = st.columns(3)
    first.metric("Documents", metrics["total_documents"])
    second.metric("Indexed chunks", metrics["total_chunks"])
    third.metric("Answer mode", "Retrieved sources only")
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
                history.extend(({"role": "user", "content": previous["question"]}, {"role": "assistant", "content": previous["answer"]}))
            try:
                if indexing_in_progress():
                    st.info("A new document is currently being indexed. This answer uses documents already available in the knowledge base.")
                with st.spinner("Searching indexed university documents…"):
                    response, sources = answer(question.strip(), history)
                st.session_state.conversation.append({"question": question.strip(), "answer": response, "sources": sources})
            except Exception as error:
                st.error(str(error))
    if st.session_state.conversation:
        st.divider(); st.subheader("Answers")
        for item in reversed(st.session_state.conversation):
            st.markdown(f"**Question:** {item['question']}"); st.markdown("**Answer:**"); st.markdown(item["answer"]); show_sources(item["sources"]); st.divider()


def document_dashboard(metrics: dict[str, int]) -> None:
    st.title("Document Dashboard")
    st.write("Review uploaded PDFs, indexing results and document-level actions.")
    columns = st.columns(4)
    columns[0].metric("Total Documents", metrics["total_documents"]); columns[1].metric("Total Indexed Chunks", metrics["total_chunks"])
    columns[2].metric("Successfully Indexed", metrics["indexed_documents"]); columns[3].metric("Pending or Failed", metrics["pending_or_failed"])
    show_job_notice()
    inventory = document_inventory()
    if not inventory:
        st.info("No PDFs are stored yet. Upload one from the Knowledge Base sidebar.")
        return
    st.subheader("Document library")
    for item in inventory:
        left, center, action = st.columns([4, 3, 3])
        with left:
            st.markdown(f"**{item['name']}**"); st.caption(f"{file_size(int(item['size_bytes']))} · {item['chunks']} indexed chunks")
        with center:
            st.write(f"Status: **{item['status']}**")
            if item["message"]: st.caption(str(item["message"]))
        with action:
            if st.button("View information", key=f"info_{item['name']}", use_container_width=True): st.session_state.document_info = item["name"]
            if st.button("Delete document", key=f"delete_{item['name']}", use_container_width=True): st.session_state.pending_delete = item["name"]
        if st.session_state.get("document_info") == item["name"]:
            st.info(f"{item['name']} — {file_size(int(item['size_bytes']))}, {item['chunks']} chunks, status: {item['status']}.")
        if st.session_state.get("pending_delete") == item["name"]:
            st.warning(f"Delete **{item['name']}**? This removes its PDF and all matching ChromaDB chunks. Other PDFs are not affected.")
            confirm, cancel = st.columns(2)
            if confirm.button("Yes, delete permanently", key=f"confirm_{item['name']}"):
                try:
                    result = delete_document(item["name"]); st.session_state.pending_delete = None
                    st.success(f"Deleted {result['filename']} and {result['removed_chunks']} indexed chunk(s)."); st.rerun()
                except Exception as error: st.error(f"Could not delete {item['name']}: {error}")
            if cancel.button("Cancel", key=f"cancel_{item['name']}"):
                st.session_state.pending_delete = None; st.rerun()
        st.divider()
    if st.button("Refresh document dashboard"): st.rerun()


def show_quiz_results(quiz: list[dict[str, object]]) -> None:
    answers = st.session_state.get("quiz_answers", {})
    correct = sum(answers.get(index) == item["correct_answer"] for index, item in enumerate(quiz))
    total = len(quiz); columns = st.columns(4)
    columns[0].metric("Total questions", total); columns[1].metric("Correct", correct); columns[2].metric("Incorrect", total - correct); columns[3].metric("Score", f"{correct / total * 100:.0f}%")
    st.subheader("Review answers")
    for index, item in enumerate(quiz, start=1):
        selected = answers.get(index - 1, "No answer"); correct_answer = str(item["correct_answer"])
        with st.expander(f"Question {index}: {'Correct' if selected == correct_answer else 'Review'}"):
            st.write(item["question"]); st.write(f"Your answer: **{selected}**"); st.write(f"Correct answer: **{correct_answer}. {item['options'][correct_answer]}**")
            st.write(f"Explanation: {item['explanation']}"); st.caption(f"Source: {item['source']} — page {item['page']}")


def quiz_page() -> None:
    st.title("AI Quiz Generator")
    st.write("Create source-grounded MCQs from indexed academic documents. Answers stay hidden until submission.")
    items = [item for item in document_inventory() if item["status"] == "Indexed" and int(item["chunks"]) > 0]
    if not items:
        st.info("Upload and finish indexing at least one text-based PDF before generating a quiz."); return
    source_names = [str(item["name"]) for item in items]
    with st.form("quiz_settings"):
        source_choice = st.selectbox("Documents for this quiz", ["All indexed documents", *source_names])
        question_count = st.selectbox("Number of questions", [5, 10, 15, 20]); difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1)
        create_quiz = st.form_submit_button("Generate quiz")
    if create_quiz:
        try:
            with st.spinner("Creating source-grounded questions…"):
                questions = generate_quiz(None if source_choice == "All indexed documents" else [source_choice], question_count, difficulty)
            st.session_state.quiz = questions; st.session_state.quiz_answers = {}; st.session_state.quiz_index = 0; st.session_state.quiz_submitted = False; st.session_state.confirm_submit = False
            st.success("Quiz generated. Select an answer for each question, then submit when ready.")
        except Exception as error: st.error(str(error))
    quiz = st.session_state.get("quiz")
    if not quiz: return
    if st.session_state.get("quiz_submitted"):
        show_quiz_results(quiz)
        if st.button("Generate New Quiz"):
            for key in ("quiz", "quiz_answers", "quiz_index", "quiz_submitted", "confirm_submit"): st.session_state.pop(key, None)
            st.rerun()
        return
    index = int(st.session_state.get("quiz_index", 0)); question = quiz[index]
    st.divider(); st.subheader(f"Question {index + 1} of {len(quiz)}"); st.write(question["question"])
    labels = [f"{letter}. {question['options'][letter]}" for letter in ("A", "B", "C", "D")]
    previous = st.session_state.quiz_answers.get(index)
    selected = st.radio("Choose one answer", labels, index=("ABCD".index(previous) if previous else None), key=f"quiz_choice_{index}")
    if selected: st.session_state.quiz_answers[index] = selected[0]
    back, forward, submit = st.columns(3)
    if back.button("Previous", disabled=index == 0): st.session_state.quiz_index = index - 1; st.rerun()
    if forward.button("Next", disabled=index == len(quiz) - 1): st.session_state.quiz_index = index + 1; st.rerun()
    if submit.button("Submit Quiz"):
        unanswered = len(quiz) - len(st.session_state.quiz_answers)
        if unanswered: st.session_state.confirm_submit = True; st.warning(f"You have {unanswered} unanswered question(s). Confirm submission to score it anyway.")
        else: st.session_state.quiz_submitted = True; st.rerun()
    if st.session_state.get("confirm_submit") and st.button("Confirm submit with unanswered questions"):
        st.session_state.quiz_submitted = True; st.session_state.confirm_submit = False; st.rerun()


add_styles(); DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
for key, value in (("conversation", []), ("pending_delete", None), ("quiz", None), ("quiz_answers", {}), ("quiz_index", 0), ("quiz_submitted", False), ("confirm_submit", False)):
    if key not in st.session_state: st.session_state[key] = value
enqueue_unindexed_documents(documents()); upload_documents(); metrics = document_metrics()
with st.sidebar:
    st.divider(); page = st.radio("Navigate", ["Home / Ask a Question", "Document Dashboard", "AI Quiz Generator"])
if page == "Home / Ask a Question": home_page(metrics)
elif page == "Document Dashboard": document_dashboard(metrics)
else: quiz_page()
