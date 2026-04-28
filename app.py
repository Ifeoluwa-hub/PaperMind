"""
PaperMind – Streamlit Chatbot UI
"""

import os
import sys

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="PaperMind – AI Research Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stChatMessage { border-radius: 10px; }
    .source-box {
        background: #f0f2f6;
        border-left: 4px solid #4c8bf5;
        padding: 8px 12px;
        border-radius: 4px;
        margin-bottom: 6px;
        font-size: 0.85em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


if not os.getenv("OPENAI_API_KEY"):
    st.error(
        "**OPENAI_API_KEY not set.**\n\n"
        "Create a `.env` file (copy `.env.example`) and add your key, "
        "then restart the app."
    )
    st.stop()

import config
from src.vector_store import load_vector_store, vector_store_exists, load_chunks
from src.retriever import get_retriever
from src.chatbot import PaperMindChatbot


@st.cache_resource(show_spinner="Loading vector store and embedding model …")
def init_system(strategy: str) -> PaperMindChatbot:
    """Initialise and cache the RAG pipeline for a given retrieval strategy."""
    vector_store = load_vector_store()
    chunks = load_chunks() if strategy == "hybrid" else None
    retriever = get_retriever(vector_store, strategy=strategy, chunks=chunks)
    return PaperMindChatbot(retriever)


with st.sidebar:
    st.title("📚 PaperMind")
    st.caption("AI Research Assistant for Generative AI Literature")
    st.divider()

    st.subheader("Retrieval Strategy")
    strategy_options = {
        "mmr":        "MMR – Diverse Results (default)",
        "similarity": "Semantic Similarity",
        "hybrid":     "Hybrid – BM25 + Semantic",
    }
    selected_strategy = st.selectbox(
        label="Strategy",
        options=list(strategy_options.keys()),
        format_func=lambda k: strategy_options[k],
        index=0,
        help=(
            "**MMR** (Maximal Marginal Relevance): balances relevance and "
            "diversity — great for broad questions.\n\n"
            "**Semantic Similarity**: pure vector similarity — best for "
            "focused, conceptual questions.\n\n"
            "**Hybrid**: fuses keyword (BM25) + semantic search — best for "
            "short, term-heavy queries."
        ),
    )

    st.divider()

    st.subheader("Available Papers")
    for title in config.PAPER_TITLES.values():
        st.markdown(f"- {title}")

    st.divider()

    st.subheader("Example Questions")
    examples = [
        "What is multi-head attention and why is it useful?",
        "How does InstructGPT use RLHF to align language models?",
        "What are the key differences between GPT-4 and Mistral 7B?",
        "Explain the Gemini model architecture.",
        "What are the advantages of the Transformer over RNNs?",
    ]
    for q in examples:
        if st.button(q, use_container_width=True):
            st.session_state["prefill_question"] = q

    st.divider()

    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        if "chatbot" in st.session_state:
            st.session_state.chatbot.clear_history()
        st.rerun()


st.title("PaperMind")
st.markdown(
    "*Ask questions about Generative AI research papers — "
    "Transformers, GPT-4, InstructGPT, Gemini, and Mistral.*"
)
st.divider()


if not vector_store_exists():
    st.warning(
        "**Vector store not found.**\n\n"
        "Run the ingestion script first to index the papers:\n\n"
        "```\npython ingest.py\n```"
    )
    st.stop()


if (
    "chatbot" not in st.session_state
    or st.session_state.get("active_strategy") != selected_strategy
):
    st.session_state.chatbot = init_system(selected_strategy)
    st.session_state.active_strategy = selected_strategy


def _show_sources(sources):
    """Render source document expander."""
    with st.expander(f"View {len(sources)} source(s) used"):
        for i, doc in enumerate(sources, 1):
            paper_name = doc.metadata.get("paper_name", "unknown")
            paper_title = doc.metadata.get(
                "paper_title",
                config.PAPER_TITLES.get(paper_name, paper_name),
            )
            page_num = doc.metadata.get("page", 0)
            snippet = doc.page_content[:350].replace("\n", " ").strip()
            st.markdown(
                f'<div class="source-box">'
                f"<b>Source {i}:</b> {paper_title} &nbsp;·&nbsp; "
                f"Page {page_num + 1}<br>"
                f'<i>\u201c{snippet}\u2026\u201d</i>'
                f"</div>",
                unsafe_allow_html=True,
            )


if "messages" not in st.session_state:
    st.session_state.messages = []

# Render existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            _show_sources(msg["sources"])


prefill = st.session_state.pop("prefill_question", None)

user_input = st.chat_input("Ask about the research papers …") or prefill

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Searching papers and generating response …"):
            answer, sources = st.session_state.chatbot.chat(user_input)

        st.markdown(answer)
        if sources:
            _show_sources(sources)

    # Persist assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
